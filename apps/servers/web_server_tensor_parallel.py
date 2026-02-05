"""
Tensor Parallel Web Server for Multi-GPU Inference

This server uses tensor parallelism to split the model across multiple GPUs,
reducing per-block latency by distributing computation.

Architecture:
- Each GPU runs a portion of each attention head and FFN
- All-reduce operations synchronize results after attention and FFN
- Only rank 0 runs the web server; other ranks are workers

Usage:
    # Launch with torchrun for multi-GPU
    uv run torchrun --nproc_per_node=2 --module apps.servers.web_server_tensor_parallel \
        --config_path configs/inference_yaml/inference_universal.yaml \
        --checkpoint_path Matrix-Game-2.0/base_distilled_model/base_distill.safetensors \
        --pretrained_model_path Matrix-Game-2.0 \
        --port 8888

    # Or use the launcher script (recommended)
    uv run python -m apps.launchers.launch_tensor_parallel --num_gpus 2 --port 8888

Requires: 2+ GPUs with sufficient VRAM (~16GB each with TP=2)
"""

import argparse
import asyncio
import base64
import io
import os
import time
import warnings
from typing import Optional, List
from threading import Thread
import queue

import numpy as np
import torch
import torch.distributed as dist
import torch._dynamo

torch._dynamo.config.cache_size_limit = 64

from PIL import Image
from einops import rearrange
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torchvision.transforms import v2

# Must set these before importing TP modules
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))

torch.cuda.set_device(LOCAL_RANK)

from wan.vae.wanx_vae import get_wanx_vae_wrapper
from demo_utils.vae_block3 import VAEDecoderWrapper
from demo_utils.constant import ZERO_VAE_CACHE
from utils.misc import set_seed
from utils.conditions import Bench_actions_universal
from pipeline.causal_inference import cond_current


def print_rank0(*args, **kwargs):
    """Print only on rank 0."""
    if RANK == 0:
        print(*args, **kwargs)


class TensorParallelInferenceEngine:
    """
    Tensor-parallel inference engine.

    Distributes model computation across multiple GPUs using tensor parallelism.
    Each GPU handles a portion of attention heads and FFN dimensions.
    """

    def __init__(self, args):
        self.args = args
        self.weight_dtype = torch.bfloat16
        self.rank = RANK
        self.world_size = WORLD_SIZE
        self.local_rank = LOCAL_RANK
        self.device = torch.device(f"cuda:{LOCAL_RANK}")

        print_rank0(f"Initializing Tensor Parallel Engine with {WORLD_SIZE} GPUs")
        print(f"[Rank {RANK}] Using device: {self.device}")

        self._init_distributed()
        self._init_config()
        self._init_model()

        if self.rank == 0:
            self.frame_process = v2.Compose([
                v2.Resize(size=(352, 640), antialias=True),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            self.action_queue = asyncio.Queue()
            self.is_running = False
            self.should_stop = False

    def _init_distributed(self):
        """Initialize distributed process group."""
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                world_size=WORLD_SIZE,
                rank=RANK,
            )

        # Initialize tensor parallel state
        from wan.distributed.tensor_parallel import set_tensor_parallel_state
        set_tensor_parallel_state(WORLD_SIZE, RANK, dist.group.WORLD)

        print(f"[Rank {RANK}] Distributed initialized")

    def _init_config(self):
        """Load configuration."""
        self.config = OmegaConf.load(self.args.config_path)
        self.mode = self.config.get('mode', 'universal')

        if getattr(self.args, 'fast', False):
            print_rank0("[fast mode] Using 2 denoising steps")
            self.config.denoising_step_list = [1000, 500]

    def _init_model(self):
        """Initialize model with tensor parallelism."""
        print_rank0("Loading model weights...")
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")

        # Load checkpoint
        main_checkpoint = None
        if self.args.checkpoint_path:
            print_rank0(f"Loading checkpoint: {self.args.checkpoint_path}")
            main_checkpoint = load_file(self.args.checkpoint_path)

        # Create standard model first, then convert to TP
        from utils.wan_wrapper import WanDiffusionWrapper
        from wan.modules.causal_model_tp import TPCausalWanModel
        from wan.distributed.tensor_parallel import get_tensor_parallel_world_size

        # Create base generator to get config
        print_rank0("Creating base model...")
        base_generator = WanDiffusionWrapper(
            **getattr(self.config, "model_kwargs", {}),
            is_causal=True
        )

        if main_checkpoint:
            base_generator.load_state_dict(main_checkpoint)

        base_generator = base_generator.to(device=self.device, dtype=self.weight_dtype)

        # Convert to tensor parallel
        print_rank0(f"Converting to tensor parallel (world_size={WORLD_SIZE})...")
        self.generator = TPCausalWanModel.from_pretrained_model(
            base_generator.model,
            world_size=WORLD_SIZE,
            rank=RANK
        )
        self.generator = self.generator.to(device=self.device, dtype=self.weight_dtype)
        self.generator.eval()

        # Free base model memory
        del base_generator
        torch.cuda.empty_cache()

        # Load VAE (only on rank 0 for encoding, all ranks for decoding context)
        print_rank0("Loading VAE...")
        self.vae = get_wanx_vae_wrapper(self.args.pretrained_model_path, torch.float16)
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae = self.vae.to(self.device, self.weight_dtype)

        # Load VAE decoder
        vae_path = os.path.join(self.args.pretrained_model_path, "Wan2.1_VAE.pth")
        vae_state_dict = torch.load(vae_path, map_location="cpu", mmap=True)
        decoder_state_dict = {k: v for k, v in vae_state_dict.items()
                              if 'decoder.' in k or 'conv2' in k}

        self.vae_decoder = VAEDecoderWrapper()
        self.vae_decoder.load_state_dict(decoder_state_dict)
        self.vae_decoder.to(self.device, torch.float16)
        self.vae_decoder.requires_grad_(False)
        self.vae_decoder.eval()

        print_rank0("Compiling VAE decoder...")
        self.vae_decoder.compile(mode="max-autotune-no-cudagraphs")

        # Store config values
        self.denoising_step_list = self.config.denoising_step_list
        self.num_frame_per_block = self.config.get('num_frame_per_block', 3)
        self.frame_seq_length = 880  # 44 * 80 / (2 * 2)
        self.context_noise = self.config.get('context_noise', 900)

        # Set generator's num_frame_per_block to match config (overrides default of 1)
        self.generator.num_frame_per_block = self.num_frame_per_block

        # Synchronize all ranks
        dist.barrier(device_ids=[self.local_rank])
        print_rank0("Model initialization complete!")

        if not getattr(self.args, 'no_warmup', False):
            self._warmup()
        else:
            print_rank0("Skipping warmup (--no-warmup). Kernels will compile on first inference.")

    def _warmup(self):
        """Warmup the model to compile kernels."""
        print_rank0("Warming up model...")

        device = self.device
        num_frame_per_block = self.num_frame_per_block
        max_output_frames = self.args.max_num_output_frames

        # Dummy inputs
        test_image = torch.randn([1, 3, 1, 352, 640], device=device, dtype=self.weight_dtype)
        with torch.no_grad():
            visual_context_shape = self.vae.clip.encode_video(test_image).shape

        num_frames = (max_output_frames - 1) * 4 + 1

        cond_concat = torch.randn([1, 20, max_output_frames, 44, 80], device=device, dtype=self.weight_dtype)
        visual_context = torch.randn(visual_context_shape, device=device, dtype=self.weight_dtype)
        keyboard = torch.zeros([1, num_frames, 4], device=device, dtype=self.weight_dtype)
        mouse = torch.zeros([1, num_frames, 2], device=device, dtype=self.weight_dtype)
        noise = torch.randn([1, 16, max_output_frames, 44, 80], device=device, dtype=self.weight_dtype)

        conditional_dict = {
            "cond_concat": cond_concat,
            "visual_context": visual_context,
            "keyboard_cond": keyboard,
            "mouse_cond": mouse,
        }

        # Initialize caches
        kv_cache1 = self._initialize_kv_cache(1, self.weight_dtype, device)
        kv_cache_mouse = self._initialize_kv_cache_action(1, self.weight_dtype, device, 880)
        kv_cache_keyboard = self._initialize_kv_cache_action(1, self.weight_dtype, device, 1)
        crossattn_cache = self._initialize_crossattn_cache(1, self.weight_dtype, device)

        # Prepare attention masks
        self.generator.block_mask = self.generator._prepare_blockwise_causal_attn_mask(
            device, num_frames=num_frame_per_block, frame_seqlen=self.frame_seq_length,
            num_frame_per_block=num_frame_per_block, local_attn_size=self.generator.local_attn_size
        )
        self.generator.num_frame_per_block = num_frame_per_block

        cond = cond_current(conditional_dict, 0, num_frame_per_block)
        noise_slice = noise[:, :, :num_frame_per_block]

        with torch.no_grad():
            output = noise_slice
            for current_timestep in self.denoising_step_list:
                timestep = torch.ones([1, num_frame_per_block], device=device, dtype=torch.int64) * current_timestep
                output = self.generator(
                    x=output,
                    t=timestep,
                    visual_context=cond["visual_context"],
                    cond_concat=cond["cond_concat"],
                    mouse_cond=cond.get("mouse_cond"),
                    keyboard_cond=cond.get("keyboard_cond"),
                    kv_cache=kv_cache1,
                    kv_cache_mouse=kv_cache_mouse,
                    kv_cache_keyboard=kv_cache_keyboard,
                    crossattn_cache=crossattn_cache,
                    current_start=0
                )

            # VAE warmup (only meaningful output on rank 0)
            if self.rank == 0:
                vae_cache = [None] * len(ZERO_VAE_CACHE)
                dummy_latent = output.transpose(1, 2)
                _, vae_cache = self.vae_decoder(dummy_latent.half(), *vae_cache)

        dist.barrier(device_ids=[self.local_rank])
        torch.cuda.empty_cache()
        print_rank0("Warmup complete!")

    def _initialize_kv_cache(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> List[dict]:
        """Initialize KV cache for self-attention."""
        num_layers = self.generator.num_layers
        num_heads = self.generator.num_heads // WORLD_SIZE  # Sharded heads
        head_dim = self.generator.dim // self.generator.num_heads
        max_seq_len = 15 * self.frame_seq_length

        kv_cache = []
        for _ in range(num_layers):
            kv_cache.append({
                "k": torch.zeros([batch_size, max_seq_len, num_heads, head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, max_seq_len, num_heads, head_dim], dtype=dtype, device=device),
                "global_end_index": torch.zeros([1], dtype=torch.long, device=device),
                "local_end_index": torch.zeros([1], dtype=torch.long, device=device),
            })
        return kv_cache

    def _initialize_kv_cache_action(self, batch_size: int, dtype: torch.dtype, device: torch.device, spatial_size: int) -> List[dict]:
        """Initialize KV cache for action modules."""
        num_layers = self.generator.num_layers
        num_heads = 16  # Action module heads (not sharded)
        head_dim = 64
        max_seq_len = 360

        kv_cache = []
        for _ in range(num_layers):
            kv_cache.append({
                "k": torch.zeros([batch_size * spatial_size, max_seq_len, num_heads, head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size * spatial_size, max_seq_len, num_heads, head_dim], dtype=dtype, device=device),
                "global_end_index": torch.zeros([1], dtype=torch.long, device=device),
                "local_end_index": torch.zeros([1], dtype=torch.long, device=device),
            })
        return kv_cache

    def _initialize_crossattn_cache(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> List[dict]:
        """Initialize cross-attention cache."""
        num_layers = self.generator.num_layers
        num_heads = self.generator.num_heads // WORLD_SIZE  # Sharded heads
        head_dim = self.generator.dim // self.generator.num_heads

        cache = []
        for _ in range(num_layers):
            cache.append({
                "k": torch.zeros([batch_size, 257, num_heads, head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 257, num_heads, head_dim], dtype=dtype, device=device),
                "is_init": False,
            })
        return cache

    def _reset_caches(self, kv_cache: List[dict], kv_cache_mouse: List[dict],
                      kv_cache_keyboard: List[dict], crossattn_cache: List[dict]):
        """Reset all caches to initial state."""
        for cache in kv_cache:
            cache["k"].zero_()
            cache["v"].zero_()
            cache["global_end_index"].zero_()
            cache["local_end_index"].zero_()

        for cache in kv_cache_mouse:
            cache["k"].zero_()
            cache["v"].zero_()
            cache["global_end_index"].zero_()
            cache["local_end_index"].zero_()

        for cache in kv_cache_keyboard:
            cache["k"].zero_()
            cache["v"].zero_()
            cache["global_end_index"].zero_()
            cache["local_end_index"].zero_()

        for cache in crossattn_cache:
            cache["k"].zero_()
            cache["v"].zero_()
            cache["is_init"] = False

    def _parse_action_to_tensor(self, action: dict, device: torch.device) -> dict:
        """Parse action dict to tensor format."""
        CAM_VALUE = 0.1

        keyboard_key = action.get("keyboard", "q").lower()
        mouse_key = action.get("mouse", "u").lower()

        CAMERA_VALUE_MAP = {
            "i": [CAM_VALUE, 0], "k": [-CAM_VALUE, 0],
            "j": [0, -CAM_VALUE], "l": [0, CAM_VALUE], "u": [0, 0]
        }
        KEYBOARD_IDX = {
            "w": [1, 0, 0, 0], "s": [0, 1, 0, 0],
            "a": [0, 0, 1, 0], "d": [0, 0, 0, 1], "q": [0, 0, 0, 0]
        }

        mouse_cond = torch.tensor(CAMERA_VALUE_MAP.get(mouse_key, [0, 0]),
                                   device=device, dtype=self.weight_dtype)
        keyboard_cond = torch.tensor(KEYBOARD_IDX.get(keyboard_key, [0, 0, 0, 0]),
                                      device=device, dtype=self.weight_dtype)

        return {"keyboard": keyboard_cond, "mouse": mouse_cond}

    def _broadcast_tensor(self, tensor: Optional[torch.Tensor], src: int = 0) -> torch.Tensor:
        """Broadcast tensor from src rank to all ranks."""
        if tensor is not None:
            dist.broadcast(tensor, src=src)
        return tensor

    def _broadcast_action(self, action: Optional[dict]) -> dict:
        """Broadcast action dict from rank 0 to all ranks."""
        if self.rank == 0:
            # Encode action as tensor
            keyboard_key = action.get("keyboard", "q")
            mouse_key = action.get("mouse", "u")
            key_map = {"w": 0, "a": 1, "s": 2, "d": 3, "q": 4}
            mouse_map = {"i": 0, "j": 1, "k": 2, "l": 3, "u": 4}
            action_tensor = torch.tensor(
                [key_map.get(keyboard_key.lower(), 4), mouse_map.get(mouse_key.lower(), 4)],
                dtype=torch.long, device=self.device
            )
        else:
            action_tensor = torch.zeros(2, dtype=torch.long, device=self.device)

        dist.broadcast(action_tensor, src=0)

        # Decode back to dict
        key_map_inv = {0: "w", 1: "a", 2: "s", 3: "d", 4: "q"}
        mouse_map_inv = {0: "i", 1: "j", 2: "k", 3: "l", 4: "u"}
        return {
            "keyboard": key_map_inv[action_tensor[0].item()],
            "mouse": mouse_map_inv[action_tensor[1].item()]
        }

    def _resizecrop(self, image, th, tw):
        """Resize and crop image to target size."""
        w, h = image.size
        if h / w > th / tw:
            new_w = int(w)
            new_h = int(new_w * th / tw)
        else:
            new_h = int(h)
            new_w = int(new_h * tw / th)
        left = (w - new_w) / 2
        top = (h - new_h) / 2
        right = (w + new_w) / 2
        bottom = (h + new_h) / 2
        image = image.crop((left, top, right, bottom))
        return image

    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert frame to base64 encoded JPEG."""
        img = Image.fromarray(frame.astype(np.uint8))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    async def run_inference_rank0(self, img_input: str, websocket):
        """Run inference loop (rank 0 only - coordinates with other ranks)."""
        assert self.rank == 0, "run_inference_rank0 should only be called on rank 0"

        self.is_running = True
        self.should_stop = False

        # Clear action queue
        while not self.action_queue.empty():
            try:
                self.action_queue.get_nowait()
            except:
                break

        try:
            # Load and process image
            if img_input.startswith('data:image'):
                header, b64_data = img_input.split(',', 1)
                img_bytes = base64.b64decode(b64_data)
                image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            else:
                from diffusers.utils import load_image
                image = load_image(img_input)

            image = self._resizecrop(image, 352, 640)
            image_tensor = self.frame_process(image)[None, :, None, :, :].to(
                dtype=self.weight_dtype, device=self.device)

            # Encode image
            max_output_frames = self.args.max_num_output_frames
            padding_video = torch.zeros_like(image_tensor).repeat(1, 1, 4 * (max_output_frames - 1), 1, 1)
            img_cond = torch.concat([image_tensor, padding_video], dim=2)
            tiler_kwargs = {"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}
            img_cond = self.vae.encode(img_cond, device=self.device, **tiler_kwargs).to(self.device)

            mask_cond = torch.ones_like(img_cond)
            mask_cond[:, :, 1:] = 0
            cond_concat = torch.cat([mask_cond[:, :4], img_cond], dim=1)
            visual_context = self.vae.clip.encode_video(image_tensor)

            sampled_noise = torch.randn(
                [1, 16, max_output_frames, 44, 80],
                device=self.device, dtype=self.weight_dtype
            )

            num_frames = (max_output_frames - 1) * 4 + 1

            conditional_dict = {
                "cond_concat": cond_concat.to(device=self.device, dtype=self.weight_dtype),
                "visual_context": visual_context.to(device=self.device, dtype=self.weight_dtype)
            }

            cond_data = Bench_actions_universal(num_frames)
            conditional_dict['mouse_cond'] = cond_data['mouse_condition'].unsqueeze(0).to(
                device=self.device, dtype=self.weight_dtype)
            conditional_dict['keyboard_cond'] = cond_data['keyboard_condition'].unsqueeze(0).to(
                device=self.device, dtype=self.weight_dtype)

            # Send initial frame
            init_frame = np.array(image.resize((640, 352)))
            await websocket.send_json({
                "type": "frame",
                "data": self._frame_to_base64(init_frame),
                "block": 0
            })

            await websocket.send_json({
                "type": "ready",
                "mode": self.mode,
                "message": f"Ready for input. [TENSOR PARALLEL: {WORLD_SIZE} GPUs]"
            })

            # Signal other ranks to start
            start_signal = torch.ones(1, device=self.device, dtype=torch.long)
            dist.broadcast(start_signal, src=0)

            # Broadcast initial tensors to all ranks
            dist.broadcast(sampled_noise, src=0)
            dist.broadcast(conditional_dict["cond_concat"], src=0)
            dist.broadcast(conditional_dict["visual_context"], src=0)
            dist.broadcast(conditional_dict["mouse_cond"], src=0)
            dist.broadcast(conditional_dict["keyboard_cond"], src=0)

            with torch.no_grad():
                await self._inference_loop(
                    noise=sampled_noise,
                    conditional_dict=conditional_dict,
                    websocket=websocket
                )

        except Exception as e:
            import traceback
            traceback.print_exc()
            await websocket.send_json({"type": "error", "message": str(e)})

            # Signal error to other ranks
            stop_signal = torch.zeros(1, device=self.device, dtype=torch.long)
            dist.broadcast(stop_signal, src=0)
        finally:
            self.is_running = False

    def run_inference_worker(self):
        """Run inference loop (worker ranks - wait for signals from rank 0)."""
        assert self.rank != 0, "run_inference_worker should not be called on rank 0"

        while True:
            # Wait for start signal
            start_signal = torch.zeros(1, device=self.device, dtype=torch.long)
            dist.broadcast(start_signal, src=0)

            if start_signal.item() == 0:
                # Stop signal or error
                continue

            # Receive initial tensors
            max_output_frames = self.args.max_num_output_frames
            num_frames = (max_output_frames - 1) * 4 + 1

            sampled_noise = torch.zeros(
                [1, 16, max_output_frames, 44, 80],
                device=self.device, dtype=self.weight_dtype)
            cond_concat = torch.zeros(
                [1, 20, max_output_frames, 44, 80],
                device=self.device, dtype=self.weight_dtype)
            visual_context = torch.zeros(
                [1, 1, 257, 1280],
                device=self.device, dtype=self.weight_dtype)
            mouse_cond = torch.zeros(
                [1, num_frames, 2],
                device=self.device, dtype=self.weight_dtype)
            keyboard_cond = torch.zeros(
                [1, num_frames, 4],
                device=self.device, dtype=self.weight_dtype)

            dist.broadcast(sampled_noise, src=0)
            dist.broadcast(cond_concat, src=0)
            dist.broadcast(visual_context, src=0)
            dist.broadcast(mouse_cond, src=0)
            dist.broadcast(keyboard_cond, src=0)

            conditional_dict = {
                "cond_concat": cond_concat,
                "visual_context": visual_context,
                "mouse_cond": mouse_cond,
                "keyboard_cond": keyboard_cond,
            }

            with torch.no_grad():
                self._inference_loop_worker(sampled_noise, conditional_dict)

    async def _inference_loop(self, noise: torch.Tensor, conditional_dict: dict, websocket):
        """Main inference loop for rank 0."""
        batch_size, num_channels, num_frames, height, width = noise.shape
        num_frame_per_block = self.num_frame_per_block
        num_blocks = num_frames // num_frame_per_block
        max_blocks = self.args.max_blocks or num_blocks

        output = torch.zeros_like(noise)
        vae_cache = [None] * len(ZERO_VAE_CACHE)

        # Initialize caches
        kv_cache = self._initialize_kv_cache(batch_size, noise.dtype, self.device)
        kv_cache_mouse = self._initialize_kv_cache_action(batch_size, noise.dtype, self.device, 880)
        kv_cache_keyboard = self._initialize_kv_cache_action(batch_size, noise.dtype, self.device, 1)
        crossattn_cache = self._initialize_crossattn_cache(batch_size, noise.dtype, self.device)

        current_start_frame = 0

        for block_index in range(min(max_blocks, num_blocks)):
            if self.should_stop:
                # Signal stop to workers
                continue_signal = torch.zeros(1, device=self.device, dtype=torch.long)
                dist.broadcast(continue_signal, src=0)
                break

            # Signal continue
            continue_signal = torch.ones(1, device=self.device, dtype=torch.long)
            dist.broadcast(continue_signal, src=0)

            block_start_time = time.time()

            # Request action
            await websocket.send_json({
                "type": "request_action",
                "block": block_index + 1,
                "mode": self.mode
            })

            # Wait for action
            try:
                action_data = await asyncio.wait_for(self.action_queue.get(), timeout=120.0)
                if action_data is None:
                    continue_signal = torch.zeros(1, device=self.device, dtype=torch.long)
                    dist.broadcast(continue_signal, src=0)
                    break
                action = {
                    "keyboard": action_data.get("keyboard", "q"),
                    "mouse": action_data.get("mouse", "u")
                }
            except asyncio.TimeoutError:
                action = {"keyboard": "q", "mouse": "u"}

            # Broadcast action to all ranks
            action = self._broadcast_action(action)

            # Generate block
            current_actions = self._parse_action_to_tensor(action, self.device)
            noisy_input = noise[:, :, current_start_frame:current_start_frame + num_frame_per_block]

            new_act, conditional_dict = cond_current(
                conditional_dict, current_start_frame, num_frame_per_block,
                replace=current_actions, mode=self.mode)

            # Denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = torch.ones(
                    [batch_size, num_frame_per_block],
                    device=self.device, dtype=torch.int64) * current_timestep

                denoised_pred = self.generator(
                    x=noisy_input,
                    t=timestep,
                    visual_context=new_act["visual_context"],
                    cond_concat=new_act["cond_concat"],
                    mouse_cond=new_act.get("mouse_cond"),
                    keyboard_cond=new_act.get("keyboard_cond"),
                    kv_cache=kv_cache,
                    kv_cache_mouse=kv_cache_mouse,
                    kv_cache_keyboard=kv_cache_keyboard,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )

                if index < len(self.denoising_step_list) - 1:
                    next_timestep = self.denoising_step_list[index + 1]
                    # Simple noise addition for next step
                    noisy_input = denoised_pred + torch.randn_like(denoised_pred) * (next_timestep / 1000.0)

            output[:, :, current_start_frame:current_start_frame + num_frame_per_block] = denoised_pred

            # Context pass
            context_timestep = torch.ones_like(timestep) * self.context_noise
            self.generator(
                x=denoised_pred,
                t=context_timestep,
                visual_context=new_act["visual_context"],
                cond_concat=new_act["cond_concat"],
                mouse_cond=new_act.get("mouse_cond"),
                keyboard_cond=new_act.get("keyboard_cond"),
                kv_cache=kv_cache,
                kv_cache_mouse=kv_cache_mouse,
                kv_cache_keyboard=kv_cache_keyboard,
                crossattn_cache=crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length
            )

            # VAE decode (only rank 0 sends frames)
            denoised_pred_t = denoised_pred.transpose(1, 2)
            video, vae_cache = self.vae_decoder(denoised_pred_t.half(), *vae_cache)

            video_np = rearrange(video, "B T C H W -> B T H W C")
            video_np = ((video_np.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)[0]

            # Send frames
            for frame_idx, frame in enumerate(video_np):
                await websocket.send_json({
                    "type": "frame",
                    "data": self._frame_to_base64(frame),
                    "block": block_index + 1,
                    "frame": frame_idx
                })
                await asyncio.sleep(0.04)

            current_start_frame += num_frame_per_block

            block_time = time.time() - block_start_time
            await websocket.send_json({
                "type": "block_complete",
                "block": block_index + 1,
                "time": f"{block_time:.2f}s",
                "gpus": WORLD_SIZE
            })

        # Signal completion
        continue_signal = torch.zeros(1, device=self.device, dtype=torch.long)
        dist.broadcast(continue_signal, src=0)

        await websocket.send_json({"type": "complete"})

    def _inference_loop_worker(self, noise: torch.Tensor, conditional_dict: dict):
        """Inference loop for worker ranks."""
        batch_size, num_channels, num_frames, height, width = noise.shape
        num_frame_per_block = self.num_frame_per_block
        num_blocks = num_frames // num_frame_per_block
        max_blocks = self.args.max_blocks or num_blocks

        # Initialize caches
        kv_cache = self._initialize_kv_cache(batch_size, noise.dtype, self.device)
        kv_cache_mouse = self._initialize_kv_cache_action(batch_size, noise.dtype, self.device, 880)
        kv_cache_keyboard = self._initialize_kv_cache_action(batch_size, noise.dtype, self.device, 1)
        crossattn_cache = self._initialize_crossattn_cache(batch_size, noise.dtype, self.device)

        current_start_frame = 0

        for block_index in range(min(max_blocks, num_blocks)):
            # Wait for continue signal
            continue_signal = torch.zeros(1, device=self.device, dtype=torch.long)
            dist.broadcast(continue_signal, src=0)

            if continue_signal.item() == 0:
                break

            # Receive action from rank 0
            action = self._broadcast_action(None)

            # Generate block (same as rank 0)
            current_actions = self._parse_action_to_tensor(action, self.device)
            noisy_input = noise[:, :, current_start_frame:current_start_frame + num_frame_per_block]

            new_act, conditional_dict = cond_current(
                conditional_dict, current_start_frame, num_frame_per_block,
                replace=current_actions, mode=self.mode)

            # Denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = torch.ones(
                    [batch_size, num_frame_per_block],
                    device=self.device, dtype=torch.int64) * current_timestep

                denoised_pred = self.generator(
                    x=noisy_input,
                    t=timestep,
                    visual_context=new_act["visual_context"],
                    cond_concat=new_act["cond_concat"],
                    mouse_cond=new_act.get("mouse_cond"),
                    keyboard_cond=new_act.get("keyboard_cond"),
                    kv_cache=kv_cache,
                    kv_cache_mouse=kv_cache_mouse,
                    kv_cache_keyboard=kv_cache_keyboard,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )

                if index < len(self.denoising_step_list) - 1:
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = denoised_pred + torch.randn_like(denoised_pred) * (next_timestep / 1000.0)

            # Context pass
            context_timestep = torch.ones_like(timestep) * self.context_noise
            self.generator(
                x=denoised_pred,
                t=context_timestep,
                visual_context=new_act["visual_context"],
                cond_concat=new_act["cond_concat"],
                mouse_cond=new_act.get("mouse_cond"),
                keyboard_cond=new_act.get("keyboard_cond"),
                kv_cache=kv_cache,
                kv_cache_mouse=kv_cache_mouse,
                kv_cache_keyboard=kv_cache_keyboard,
                crossattn_cache=crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length
            )

            current_start_frame += num_frame_per_block

    def stop(self):
        """Stop inference."""
        self.should_stop = True


# Global engine
inference_engine: Optional[TensorParallelInferenceEngine] = None


def run_rank0_server(args, engine: TensorParallelInferenceEngine):
    """Run the web server on rank 0."""
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, FileResponse

    app = FastAPI(title="Matrix-Game 2.0 Tensor Parallel Server")

    @app.get("/", response_class=HTMLResponse)
    async def get_index():
        return FileResponse("web_ui/index.html")

    @app.get("/app.html", response_class=HTMLResponse)
    async def get_app():
        return FileResponse("web_ui/app.html")

    @app.get("/assets/{filename}")
    async def get_asset(filename: str):
        return FileResponse(f"web_ui/assets/{filename}")

    @app.get("/api/status")
    async def get_status():
        return {
            "ready": True,
            "running": engine.is_running,
            "mode": engine.mode,
            "tensor_parallel": True,
            "world_size": WORLD_SIZE
        }

    @app.get("/api/images")
    async def list_images():
        images = []
        demo_dir = "demo_images"
        if os.path.exists(demo_dir):
            for subdir in os.listdir(demo_dir):
                subdir_path = os.path.join(demo_dir, subdir)
                if os.path.isdir(subdir_path):
                    for img in os.listdir(subdir_path):
                        if img.endswith(('.png', '.jpg', '.jpeg')):
                            images.append(f"{demo_dir}/{subdir}/{img}")
        return {"images": images}

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()

        try:
            while True:
                data = await websocket.receive_json()
                msg_type = data.get("type")

                if msg_type == "start":
                    img_input = data.get("image", "demo_images/universal/0000.png")
                    if not engine.is_running:
                        asyncio.create_task(
                            engine.run_inference_rank0(img_input, websocket))
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Engine already running"
                        })

                elif msg_type == "action":
                    if engine.is_running:
                        await engine.action_queue.put(data)

                elif msg_type == "stop":
                    engine.stop()
                    await engine.action_queue.put(None)
                    await websocket.send_json({"type": "stopped"})

        except WebSocketDisconnect:
            engine.stop()
        except Exception as e:
            print(f"WebSocket error: {e}")
            engine.stop()

    import uvicorn
    print(f"\nStarting tensor parallel server at http://{args.host}:{args.port}")
    print(f"Using {WORLD_SIZE} GPUs with tensor parallelism\n")
    uvicorn.run(app, host=args.host, port=args.port, ws_max_size=50 * 1024 * 1024)


def parse_args():
    parser = argparse.ArgumentParser(description="Tensor Parallel Web Server")
    parser.add_argument("--config_path", type=str,
                        default="configs/inference_yaml/inference_universal.yaml")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--pretrained_model_path", type=str, default="Matrix-Game-2.0")
    parser.add_argument("--max_num_output_frames", type=int, default=360)
    parser.add_argument("--max_blocks", type=int, default=999999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--fast", action="store_true",
                        help="Use 2 denoising steps instead of 3 (~33% speedup)")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Skip kernel warmup (kernels will compile lazily on first inference)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    print_rank0(f"World size: {WORLD_SIZE}, Rank: {RANK}, Local rank: {LOCAL_RANK}")
    print_rank0(f"Available GPUs: {torch.cuda.device_count()}")

    # Initialize engine on all ranks
    inference_engine = TensorParallelInferenceEngine(args)

    if RANK == 0:
        # Rank 0 runs the web server
        run_rank0_server(args, inference_engine)
    else:
        # Other ranks run as workers
        inference_engine.run_inference_worker()
