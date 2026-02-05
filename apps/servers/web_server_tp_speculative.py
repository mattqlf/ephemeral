"""
Tensor Parallel + Speculative Hybrid Web Server

This server combines tensor parallelism with speculative pipelining:
- Primary TP group (GPUs 0 to N/2-1): Generates current block with tensor parallelism
- Speculative TP group (GPUs N/2 to N-1): Pre-generates next block assuming same action

Example with 4 GPUs:
- Primary group: GPU 0, 1 (TP world_size=2)
- Speculative group: GPU 2, 3 (TP world_size=2)

Benefits:
- Tensor parallelism reduces per-block latency
- Speculation provides near-zero latency when action is held

Usage:
    # Launch with torchrun (4 GPUs: 2 for primary, 2 for speculative)
    uv run torchrun --nproc_per_node=4 --module apps.servers.web_server_tp_speculative \
        --checkpoint_path Matrix-Game-2.0/base_distilled_model/base_distill.safetensors \
        --port 8888 --fast

    # Or use the launcher script (recommended)
    uv run python -m apps.launchers.launch_tp_speculative --num_gpus 4 --port 8888 --fast

Requires: 4+ GPUs (even number), split evenly between primary and speculative
"""

import argparse
import asyncio
import base64
import io
import os
import time
import warnings
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from threading import Thread, Lock, Event
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

# Get distributed info before other imports
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))

# Calculate group assignments
TP_SIZE = WORLD_SIZE // 2
IS_PRIMARY = RANK < TP_SIZE
IS_SPECULATIVE = not IS_PRIMARY
GROUP_RANK = RANK if IS_PRIMARY else RANK - TP_SIZE
IS_GROUP_LEADER = GROUP_RANK == 0
IS_PRIMARY_LEADER = IS_PRIMARY and IS_GROUP_LEADER  # Rank 0
IS_SPECULATIVE_LEADER = IS_SPECULATIVE and IS_GROUP_LEADER  # Rank TP_SIZE

torch.cuda.set_device(LOCAL_RANK)

from wan.vae.wanx_vae import get_wanx_vae_wrapper
from demo_utils.vae_block3 import VAEDecoderWrapper
from demo_utils.constant import ZERO_VAE_CACHE
from utils.misc import set_seed
from utils.conditions import Bench_actions_universal
from pipeline.causal_inference import cond_current
from pipeline.scheduler import FlowMatchDiscreteScheduler


def print_rank0(*args, **kwargs):
    """Print only on rank 0."""
    if RANK == 0:
        print(*args, **kwargs)


def print_leader(*args, **kwargs):
    """Print on group leaders."""
    if IS_GROUP_LEADER:
        prefix = "[PRIMARY]" if IS_PRIMARY else "[SPECULATIVE]"
        print(f"{prefix}", *args, **kwargs)


class TPSpeculativeInferenceEngine:
    """
    Hybrid Tensor Parallel + Speculative inference engine.

    Uses two TP groups:
    - Primary group: Generates blocks based on user input
    - Speculative group: Pre-generates next block assuming same action

    Simplified design:
    - Each group maintains independent inference state
    - On speculation HIT: Use speculative frames (faster delivery)
    - On speculation MISS: Use primary frames
    - Both groups always generate to keep state consistent
    """

    def __init__(self, args):
        self.args = args
        self.weight_dtype = torch.bfloat16
        self.rank = RANK
        self.world_size = WORLD_SIZE
        self.local_rank = LOCAL_RANK
        self.device = torch.device(f"cuda:{LOCAL_RANK}")

        self.tp_size = TP_SIZE
        self.is_primary = IS_PRIMARY
        self.is_speculative = IS_SPECULATIVE
        self.group_rank = GROUP_RANK
        self.is_group_leader = IS_GROUP_LEADER
        self.is_primary_leader = IS_PRIMARY_LEADER
        self.is_speculative_leader = IS_SPECULATIVE_LEADER

        print_rank0(f"Initializing TP+Speculative Engine")
        print_rank0(f"  Total GPUs: {WORLD_SIZE}")
        print_rank0(f"  TP size per group: {TP_SIZE}")
        print_rank0(f"  Primary ranks: 0-{TP_SIZE-1}")
        print_rank0(f"  Speculative ranks: {TP_SIZE}-{WORLD_SIZE-1}")
        print(f"[Rank {RANK}] Device: {self.device}, "
              f"{'PRIMARY' if IS_PRIMARY else 'SPECULATIVE'}, "
              f"group_rank={GROUP_RANK}, leader={IS_GROUP_LEADER}")

        self._init_distributed()
        self._init_config()
        self._init_model()

        # Primary leader specific
        if self.is_primary_leader:
            self.frame_process = v2.Compose([
                v2.Resize(size=(352, 640), antialias=True),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            self.is_running = False
            self.should_stop = False
            self.current_action = {"keyboard": "q", "mouse": "u"}
            self.action_lock = Lock()

            # Speculation tracking
            self.speculation_hit = False
            self.speculative_frames = None

        # Scheduler (shared)
        self.scheduler = FlowMatchDiscreteScheduler()

    def _init_distributed(self):
        """Initialize distributed process groups."""
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                world_size=WORLD_SIZE,
                rank=RANK,
            )

        # Create sub-groups for tensor parallelism
        primary_ranks = list(range(TP_SIZE))
        speculative_ranks = list(range(TP_SIZE, WORLD_SIZE))

        self.primary_group = dist.new_group(primary_ranks)
        self.speculative_group = dist.new_group(speculative_ranks)

        # Set the appropriate TP group for this rank
        if self.is_primary:
            self.tp_group = self.primary_group
        else:
            self.tp_group = self.speculative_group

        # Initialize tensor parallel state with the appropriate group
        from wan.distributed.tensor_parallel import set_tensor_parallel_state
        set_tensor_parallel_state(TP_SIZE, self.group_rank, self.tp_group)

        print(f"[Rank {RANK}] Process groups initialized")

    def _init_config(self):
        """Load configuration."""
        self.config = OmegaConf.load(self.args.config_path)
        self.mode = self.config.get('mode', 'universal')

        if getattr(self.args, 'fast', False):
            print_rank0("[fast mode] Using 2 denoising steps")
            self.config.denoising_step_list = [1000, 500]

    def _init_model(self):
        """Initialize model with tensor parallelism."""
        print_leader("Loading model weights...")
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")

        # Load checkpoint
        main_checkpoint = None
        if self.args.checkpoint_path:
            print_leader(f"Loading checkpoint: {self.args.checkpoint_path}")
            main_checkpoint = load_file(self.args.checkpoint_path)

        # Create and convert model to TP
        from utils.wan_wrapper import WanDiffusionWrapper
        from wan.modules.causal_model_tp import TPCausalWanModel

        print_leader("Creating base model...")
        base_generator = WanDiffusionWrapper(
            **getattr(self.config, "model_kwargs", {}),
            is_causal=True
        )

        if main_checkpoint:
            base_generator.load_state_dict(main_checkpoint)

        base_generator = base_generator.to(device=self.device, dtype=self.weight_dtype)

        print_leader(f"Converting to tensor parallel (TP size={TP_SIZE})...")
        self.generator = TPCausalWanModel.from_pretrained_model(
            base_generator.model,
            world_size=TP_SIZE,
            rank=self.group_rank
        )
        self.generator = self.generator.to(device=self.device, dtype=self.weight_dtype)
        self.generator.eval()

        del base_generator
        torch.cuda.empty_cache()

        # Load VAE
        print_leader("Loading VAE...")
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

        print_leader("Compiling VAE decoder...")
        self.vae_decoder.compile(mode="max-autotune-no-cudagraphs")

        # Store config values
        self.denoising_step_list = self.config.denoising_step_list
        self.num_frame_per_block = self.config.get('num_frame_per_block', 3)
        self.frame_seq_length = 880
        self.context_noise = self.config.get('context_noise', 900)

        # Synchronize within each group then globally
        dist.barrier(self.tp_group)
        dist.barrier()

        print_leader("Model initialization complete!")

        if not getattr(self.args, 'no_warmup', False):
            self._warmup()
        else:
            print_leader("Skipping warmup (--no-warmup)")

    def _warmup(self):
        """Warmup the model."""
        print_leader("Warming up model...")

        device = self.device
        num_frame_per_block = self.num_frame_per_block
        max_output_frames = self.args.max_num_output_frames
        num_frames = (max_output_frames - 1) * 4 + 1

        # Dummy inputs
        test_image = torch.randn([1, 3, 1, 352, 640], device=device, dtype=self.weight_dtype)
        with torch.no_grad():
            visual_context_shape = self.vae.clip.encode_video(test_image).shape

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

        kv_cache = self._initialize_kv_cache(1, self.weight_dtype, device)
        kv_cache_mouse = self._initialize_kv_cache_action(1, self.weight_dtype, device, 880)
        kv_cache_keyboard = self._initialize_kv_cache_action(1, self.weight_dtype, device, 1)
        crossattn_cache = self._initialize_crossattn_cache(1, self.weight_dtype, device)

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
                    kv_cache=kv_cache,
                    kv_cache_mouse=kv_cache_mouse,
                    kv_cache_keyboard=kv_cache_keyboard,
                    crossattn_cache=crossattn_cache,
                    current_start=0
                )

            # VAE warmup on leaders
            if self.is_group_leader:
                vae_cache = [None] * len(ZERO_VAE_CACHE)
                dummy_latent = output.transpose(1, 2)
                _, vae_cache = self.vae_decoder(dummy_latent.half(), *vae_cache)

        dist.barrier(self.tp_group)
        dist.barrier()
        torch.cuda.empty_cache()
        print_leader("Warmup complete!")

    def _initialize_kv_cache(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> List[dict]:
        """Initialize KV cache for self-attention (sharded for TP)."""
        num_layers = self.generator.num_layers
        num_heads = self.generator.num_heads // TP_SIZE  # Sharded
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
        """Initialize KV cache for action modules (not sharded)."""
        num_layers = self.generator.num_layers
        num_heads = 16
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
        """Initialize cross-attention cache (sharded for TP)."""
        num_layers = self.generator.num_layers
        num_heads = self.generator.num_heads // TP_SIZE
        head_dim = self.generator.dim // self.generator.num_heads

        cache = []
        for _ in range(num_layers):
            cache.append({
                "k": torch.zeros([batch_size, 257, num_heads, head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 257, num_heads, head_dim], dtype=dtype, device=device),
                "is_init": False,
            })
        return cache

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

    def _broadcast_in_group(self, tensor: torch.Tensor, src_rank: int = 0):
        """Broadcast tensor within the TP group."""
        actual_src = src_rank if self.is_primary else src_rank + TP_SIZE
        dist.broadcast(tensor, src=actual_src, group=self.tp_group)

    def _broadcast_action_in_group(self, action: Optional[dict]) -> dict:
        """Broadcast action within the TP group."""
        if self.is_group_leader:
            keyboard_key = action.get("keyboard", "q") if action else "q"
            mouse_key = action.get("mouse", "u") if action else "u"
            key_map = {"w": 0, "a": 1, "s": 2, "d": 3, "q": 4}
            mouse_map = {"i": 0, "j": 1, "k": 2, "l": 3, "u": 4}
            action_tensor = torch.tensor(
                [key_map.get(keyboard_key.lower(), 4), mouse_map.get(mouse_key.lower(), 4)],
                dtype=torch.long, device=self.device
            )
        else:
            action_tensor = torch.zeros(2, dtype=torch.long, device=self.device)

        self._broadcast_in_group(action_tensor, src_rank=0)

        key_map_inv = {0: "w", 1: "a", 2: "s", 3: "d", 4: "q"}
        mouse_map_inv = {0: "i", 1: "j", 2: "k", 3: "l", 4: "u"}
        return {
            "keyboard": key_map_inv[action_tensor[0].item()],
            "mouse": mouse_map_inv[action_tensor[1].item()]
        }

    def _get_current_action(self) -> dict:
        """Get current action (primary leader only)."""
        with self.action_lock:
            return self.current_action.copy()

    def _set_current_action(self, action: dict):
        """Set current action (primary leader only)."""
        with self.action_lock:
            self.current_action = action.copy()

    def _resizecrop(self, image, th, tw):
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
        img = Image.fromarray(frame.astype(np.uint8))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _generate_block_tp(
        self,
        noise_slice: torch.Tensor,
        conditional_dict: dict,
        current_start_frame: int,
        action: dict,
        kv_cache: List[dict],
        kv_cache_mouse: List[dict],
        kv_cache_keyboard: List[dict],
        crossattn_cache: List[dict],
        vae_cache: List,
    ) -> tuple:
        """Generate a single block using tensor parallelism within the group."""
        device = self.device
        batch_size = noise_slice.shape[0]
        num_frame_per_block = self.num_frame_per_block

        # Broadcast action to all ranks in the TP group
        action = self._broadcast_action_in_group(action if self.is_group_leader else None)

        # Parse action to tensor
        current_actions = self._parse_action_to_tensor(action, device)

        # Update conditions with action
        new_act, conditional_dict = cond_current(
            conditional_dict, current_start_frame, num_frame_per_block,
            replace=current_actions, mode=self.mode)

        # Denoising loop
        noisy_input = noise_slice
        for index, current_timestep in enumerate(self.denoising_step_list):
            timestep = torch.ones(
                [batch_size, num_frame_per_block],
                device=device, dtype=torch.int64) * current_timestep

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
                noisy_input = self.scheduler.add_noise(
                    rearrange(denoised_pred, 'b c f h w -> (b f) c h w'),
                    torch.randn_like(rearrange(denoised_pred, 'b c f h w -> (b f) c h w')),
                    next_timestep * torch.ones(
                        [batch_size * num_frame_per_block], device=device, dtype=torch.long)
                )
                noisy_input = rearrange(noisy_input, '(b f) c h w -> b c f h w', b=batch_size)

        # Context pass to update KV cache
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

        # VAE decode (only on group leader)
        video_np = None
        if self.is_group_leader:
            denoised_pred_t = denoised_pred.transpose(1, 2)
            video, vae_cache = self.vae_decoder(denoised_pred_t.half(), *vae_cache)
            video_np = rearrange(video, "B T C H W -> B T H W C")
            video_np = ((video_np.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)[0]

        return denoised_pred, video_np, vae_cache, conditional_dict, action

    async def run_inference_primary_leader(self, img_input: str, websocket):
        """Run inference loop (primary leader - rank 0)."""
        assert self.is_primary_leader

        self.is_running = True
        self.should_stop = False

        with self.action_lock:
            self.current_action = {"keyboard": "q", "mouse": "u"}

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
                "message": f"Ready. [TP+SPECULATIVE: {WORLD_SIZE} GPUs, TP={TP_SIZE}]"
            })

            # Signal all ranks to start
            start_signal = torch.ones(1, device=self.device, dtype=torch.long)
            dist.broadcast(start_signal, src=0)

            # Broadcast initial tensors to primary group
            self._broadcast_in_group(sampled_noise, src_rank=0)
            self._broadcast_in_group(conditional_dict["cond_concat"], src_rank=0)
            self._broadcast_in_group(conditional_dict["visual_context"], src_rank=0)
            self._broadcast_in_group(conditional_dict["mouse_cond"], src_rank=0)
            self._broadcast_in_group(conditional_dict["keyboard_cond"], src_rank=0)

            # Send to speculative leader for speculative group
            dist.send(sampled_noise, dst=TP_SIZE)
            dist.send(conditional_dict["cond_concat"], dst=TP_SIZE)
            dist.send(conditional_dict["visual_context"], dst=TP_SIZE)
            dist.send(conditional_dict["mouse_cond"], dst=TP_SIZE)
            dist.send(conditional_dict["keyboard_cond"], dst=TP_SIZE)

            with torch.no_grad():
                await self._primary_inference_loop(
                    noise=sampled_noise,
                    conditional_dict=conditional_dict,
                    websocket=websocket
                )

        except Exception as e:
            import traceback
            traceback.print_exc()
            await websocket.send_json({"type": "error", "message": str(e)})

            stop_signal = torch.zeros(1, device=self.device, dtype=torch.long)
            dist.broadcast(stop_signal, src=0)
        finally:
            self.is_running = False

    def run_primary_worker(self):
        """Run as primary worker (ranks 1 to TP_SIZE-1)."""
        assert self.is_primary and not self.is_group_leader

        while True:
            # Wait for start signal
            start_signal = torch.zeros(1, device=self.device, dtype=torch.long)
            dist.broadcast(start_signal, src=0)

            if start_signal.item() == 0:
                continue

            # Receive initial tensors
            max_output_frames = self.args.max_num_output_frames
            num_frames = (max_output_frames - 1) * 4 + 1

            sampled_noise = torch.zeros([1, 16, max_output_frames, 44, 80],
                                         device=self.device, dtype=self.weight_dtype)
            cond_concat = torch.zeros([1, 20, max_output_frames, 44, 80],
                                       device=self.device, dtype=self.weight_dtype)
            visual_context = torch.zeros([1, 1, 257, 1280],
                                          device=self.device, dtype=self.weight_dtype)
            mouse_cond = torch.zeros([1, num_frames, 2],
                                      device=self.device, dtype=self.weight_dtype)
            keyboard_cond = torch.zeros([1, num_frames, 4],
                                         device=self.device, dtype=self.weight_dtype)

            self._broadcast_in_group(sampled_noise, src_rank=0)
            self._broadcast_in_group(cond_concat, src_rank=0)
            self._broadcast_in_group(visual_context, src_rank=0)
            self._broadcast_in_group(mouse_cond, src_rank=0)
            self._broadcast_in_group(keyboard_cond, src_rank=0)

            conditional_dict = {
                "cond_concat": cond_concat,
                "visual_context": visual_context,
                "mouse_cond": mouse_cond,
                "keyboard_cond": keyboard_cond,
            }

            with torch.no_grad():
                self._primary_worker_loop(sampled_noise, conditional_dict)

    def run_speculative_leader(self):
        """Run as speculative leader (rank TP_SIZE)."""
        assert self.is_speculative_leader

        while True:
            # Wait for start signal
            start_signal = torch.zeros(1, device=self.device, dtype=torch.long)
            dist.broadcast(start_signal, src=0)

            if start_signal.item() == 0:
                continue

            # Receive initial tensors from primary leader
            max_output_frames = self.args.max_num_output_frames
            num_frames = (max_output_frames - 1) * 4 + 1

            sampled_noise = torch.zeros([1, 16, max_output_frames, 44, 80],
                                         device=self.device, dtype=self.weight_dtype)
            cond_concat = torch.zeros([1, 20, max_output_frames, 44, 80],
                                       device=self.device, dtype=self.weight_dtype)
            visual_context = torch.zeros([1, 1, 257, 1280],
                                          device=self.device, dtype=self.weight_dtype)
            mouse_cond = torch.zeros([1, num_frames, 2],
                                      device=self.device, dtype=self.weight_dtype)
            keyboard_cond = torch.zeros([1, num_frames, 4],
                                         device=self.device, dtype=self.weight_dtype)

            dist.recv(sampled_noise, src=0)
            dist.recv(cond_concat, src=0)
            dist.recv(visual_context, src=0)
            dist.recv(mouse_cond, src=0)
            dist.recv(keyboard_cond, src=0)

            # Broadcast to speculative group workers
            self._broadcast_in_group(sampled_noise, src_rank=0)
            self._broadcast_in_group(cond_concat, src_rank=0)
            self._broadcast_in_group(visual_context, src_rank=0)
            self._broadcast_in_group(mouse_cond, src_rank=0)
            self._broadcast_in_group(keyboard_cond, src_rank=0)

            conditional_dict = {
                "cond_concat": cond_concat,
                "visual_context": visual_context,
                "mouse_cond": mouse_cond,
                "keyboard_cond": keyboard_cond,
            }

            with torch.no_grad():
                self._speculative_loop(sampled_noise, conditional_dict)

    def run_speculative_worker(self):
        """Run as speculative worker (ranks TP_SIZE+1 to WORLD_SIZE-1)."""
        assert self.is_speculative and not self.is_group_leader

        while True:
            # Wait for start signal
            start_signal = torch.zeros(1, device=self.device, dtype=torch.long)
            dist.broadcast(start_signal, src=0)

            if start_signal.item() == 0:
                continue

            # Receive tensors from speculative leader
            max_output_frames = self.args.max_num_output_frames
            num_frames = (max_output_frames - 1) * 4 + 1

            sampled_noise = torch.zeros([1, 16, max_output_frames, 44, 80],
                                         device=self.device, dtype=self.weight_dtype)
            cond_concat = torch.zeros([1, 20, max_output_frames, 44, 80],
                                       device=self.device, dtype=self.weight_dtype)
            visual_context = torch.zeros([1, 1, 257, 1280],
                                          device=self.device, dtype=self.weight_dtype)
            mouse_cond = torch.zeros([1, num_frames, 2],
                                      device=self.device, dtype=self.weight_dtype)
            keyboard_cond = torch.zeros([1, num_frames, 4],
                                         device=self.device, dtype=self.weight_dtype)

            self._broadcast_in_group(sampled_noise, src_rank=0)
            self._broadcast_in_group(cond_concat, src_rank=0)
            self._broadcast_in_group(visual_context, src_rank=0)
            self._broadcast_in_group(mouse_cond, src_rank=0)
            self._broadcast_in_group(keyboard_cond, src_rank=0)

            conditional_dict = {
                "cond_concat": cond_concat,
                "visual_context": visual_context,
                "mouse_cond": mouse_cond,
                "keyboard_cond": keyboard_cond,
            }

            with torch.no_grad():
                self._speculative_worker_loop(sampled_noise, conditional_dict)

    async def _primary_inference_loop(self, noise: torch.Tensor, conditional_dict: dict, websocket):
        """Main inference loop for primary leader with speculative coordination."""
        batch_size, num_channels, num_frames_total, height, width = noise.shape
        num_frame_per_block = self.num_frame_per_block
        num_blocks = num_frames_total // num_frame_per_block
        max_blocks = self.args.max_blocks or num_blocks

        vae_cache = [None] * len(ZERO_VAE_CACHE)

        # Initialize caches for primary
        kv_cache = self._initialize_kv_cache(batch_size, noise.dtype, self.device)
        kv_cache_mouse = self._initialize_kv_cache_action(batch_size, noise.dtype, self.device, 880)
        kv_cache_keyboard = self._initialize_kv_cache_action(batch_size, noise.dtype, self.device, 1)
        crossattn_cache = self._initialize_crossattn_cache(batch_size, noise.dtype, self.device)

        # Prepare attention masks
        self.generator.block_mask = self.generator._prepare_blockwise_causal_attn_mask(
            self.device, num_frames=num_frame_per_block, frame_seqlen=self.frame_seq_length,
            num_frame_per_block=num_frame_per_block, local_attn_size=self.generator.local_attn_size
        )
        self.generator.num_frame_per_block = num_frame_per_block

        current_start_frame = 0
        speculation_hits = 0
        speculation_misses = 0
        last_action = {"keyboard": "q", "mouse": "u"}

        await websocket.send_json({
            "type": "start_streaming",
            "mode": self.mode,
            "message": "Hold keys to control. [TP+SPECULATIVE]"
        })

        for block_index in range(min(max_blocks, num_blocks)):
            if self.should_stop:
                # Signal stop to all workers
                control = torch.tensor([0, 0, 0, 0], device=self.device, dtype=torch.long)
                self._broadcast_in_group(control, src_rank=0)
                dist.send(control[:2], dst=TP_SIZE)
                break

            block_start_time = time.time()

            # Get current action
            action = self._get_current_action()

            # Signal primary workers: [continue_flag, block_index, frame_start, action_code]
            action_code = self._encode_action_single(action)
            control = torch.tensor([1, block_index, current_start_frame, action_code],
                                    device=self.device, dtype=torch.long)
            self._broadcast_in_group(control, src_rank=0)

            # Send action to speculative leader
            spec_control = torch.tensor([block_index, action_code], device=self.device, dtype=torch.long)
            dist.send(spec_control, dst=TP_SIZE)

            # Generate block on primary group
            noise_slice = noise[:, :, current_start_frame:current_start_frame + num_frame_per_block]
            denoised_pred, video_np, vae_cache, conditional_dict, used_action = self._generate_block_tp(
                noise_slice, conditional_dict, current_start_frame, action,
                kv_cache, kv_cache_mouse, kv_cache_keyboard, crossattn_cache, vae_cache
            )

            # Receive speculation result from speculative leader
            spec_result = torch.zeros(2, device=self.device, dtype=torch.long)
            dist.recv(spec_result, src=TP_SIZE)
            spec_action_code = spec_result[1].item()
            spec_action = self._decode_action_single(spec_action_code)

            # Determine which frames to use
            use_speculation = False
            if block_index > 0 and spec_action == last_action and last_action == action:
                # Speculation hit - speculative generated same action we need
                use_speculation = True
                speculation_hits += 1
                print(f"[block {block_index + 1}] SPECULATION HIT! action={action}")

                # Receive speculative frames
                frame_shape = video_np.shape
                spec_frames = torch.zeros(frame_shape, device=self.device, dtype=torch.uint8)
                dist.recv(spec_frames, src=TP_SIZE)
                video_np = spec_frames.cpu().numpy()
            else:
                if block_index > 0:
                    print(f"[block {block_index + 1}] SPECULATION MISS - spec={spec_action}, need={action}")
                else:
                    print(f"[block {block_index + 1}] Generating (first block)")
                speculation_misses += 1

                # Receive and discard speculative frames
                frame_shape = video_np.shape
                spec_frames = torch.zeros(frame_shape, device=self.device, dtype=torch.uint8)
                dist.recv(spec_frames, src=TP_SIZE)

            # Send frames to client
            for frame_idx, frame in enumerate(video_np):
                await websocket.send_json({
                    "type": "frame",
                    "data": self._frame_to_base64(frame),
                    "block": block_index + 1,
                    "frame": frame_idx
                })
                await asyncio.sleep(0.033)

            last_action = action
            current_start_frame += num_frame_per_block

            block_time = time.time() - block_start_time
            total = speculation_hits + speculation_misses
            hit_rate = (speculation_hits / total * 100) if total > 0 else 0

            await websocket.send_json({
                "type": "block_complete",
                "block": block_index + 1,
                "time": f"{block_time:.2f}s",
                "speculation_hit": use_speculation,
                "hit_rate": f"{hit_rate:.1f}%"
            })

        # Signal end to all workers
        control = torch.tensor([0, 0, 0, 0], device=self.device, dtype=torch.long)
        self._broadcast_in_group(control, src_rank=0)
        dist.send(control[:2], dst=TP_SIZE)

        total = speculation_hits + speculation_misses
        if total > 0:
            print(f"\n[SPECULATION STATS] Hits: {speculation_hits}/{total} ({100*speculation_hits/total:.1f}%)")

        await websocket.send_json({"type": "complete"})

    def _encode_action_single(self, action: dict) -> int:
        """Encode action to single int."""
        key_map = {"w": 0, "a": 1, "s": 2, "d": 3, "q": 4}
        mouse_map = {"i": 0, "j": 1, "k": 2, "l": 3, "u": 4}
        k = key_map.get(action["keyboard"].lower(), 4)
        m = mouse_map.get(action["mouse"].lower(), 4)
        return k * 5 + m

    def _decode_action_single(self, code: int) -> dict:
        """Decode action from single int."""
        key_map_inv = {0: "w", 1: "a", 2: "s", 3: "d", 4: "q"}
        mouse_map_inv = {0: "i", 1: "j", 2: "k", 3: "l", 4: "u"}
        k = code // 5
        m = code % 5
        return {"keyboard": key_map_inv[k], "mouse": mouse_map_inv[m]}

    def _primary_worker_loop(self, noise: torch.Tensor, conditional_dict: dict):
        """Worker loop for primary workers."""
        batch_size = noise.shape[0]
        num_frame_per_block = self.num_frame_per_block

        kv_cache = self._initialize_kv_cache(batch_size, noise.dtype, self.device)
        kv_cache_mouse = self._initialize_kv_cache_action(batch_size, noise.dtype, self.device, 880)
        kv_cache_keyboard = self._initialize_kv_cache_action(batch_size, noise.dtype, self.device, 1)
        crossattn_cache = self._initialize_crossattn_cache(batch_size, noise.dtype, self.device)

        self.generator.block_mask = self.generator._prepare_blockwise_causal_attn_mask(
            self.device, num_frames=num_frame_per_block, frame_seqlen=self.frame_seq_length,
            num_frame_per_block=num_frame_per_block, local_attn_size=self.generator.local_attn_size
        )
        self.generator.num_frame_per_block = num_frame_per_block

        vae_cache = [None] * len(ZERO_VAE_CACHE)

        while True:
            # Wait for control signal
            control = torch.zeros(4, device=self.device, dtype=torch.long)
            self._broadcast_in_group(control, src_rank=0)

            if control[0].item() == 0:
                break

            block_index = control[1].item()
            current_start_frame = control[2].item()
            action_code = control[3].item()
            action = self._decode_action_single(action_code)

            noise_slice = noise[:, :, current_start_frame:current_start_frame + num_frame_per_block]
            self._generate_block_tp(
                noise_slice, conditional_dict, current_start_frame, action,
                kv_cache, kv_cache_mouse, kv_cache_keyboard, crossattn_cache, vae_cache
            )

    def _speculative_loop(self, noise: torch.Tensor, conditional_dict: dict):
        """Speculative leader loop - generates blocks ahead assuming same action."""
        batch_size = noise.shape[0]
        num_frame_per_block = self.num_frame_per_block
        num_frames_total = noise.shape[2]
        num_blocks = num_frames_total // num_frame_per_block

        kv_cache = self._initialize_kv_cache(batch_size, noise.dtype, self.device)
        kv_cache_mouse = self._initialize_kv_cache_action(batch_size, noise.dtype, self.device, 880)
        kv_cache_keyboard = self._initialize_kv_cache_action(batch_size, noise.dtype, self.device, 1)
        crossattn_cache = self._initialize_crossattn_cache(batch_size, noise.dtype, self.device)
        vae_cache = [None] * len(ZERO_VAE_CACHE)

        self.generator.block_mask = self.generator._prepare_blockwise_causal_attn_mask(
            self.device, num_frames=num_frame_per_block, frame_seqlen=self.frame_seq_length,
            num_frame_per_block=num_frame_per_block, local_attn_size=self.generator.local_attn_size
        )
        self.generator.num_frame_per_block = num_frame_per_block

        current_start_frame = 0
        last_action = {"keyboard": "q", "mouse": "u"}

        while True:
            # Receive control from primary leader
            spec_control = torch.zeros(2, device=self.device, dtype=torch.long)
            dist.recv(spec_control, src=0)

            block_index = spec_control[0].item()
            action_code = spec_control[1].item()

            if block_index < 0:
                # End signal
                control = torch.tensor([0, 0, 0, 0], device=self.device, dtype=torch.long)
                self._broadcast_in_group(control, src_rank=0)
                break

            action = self._decode_action_single(action_code)

            # Signal workers
            control = torch.tensor([1, block_index, current_start_frame, action_code],
                                    device=self.device, dtype=torch.long)
            self._broadcast_in_group(control, src_rank=0)

            # Generate block speculatively (using PREVIOUS action, since we're ahead)
            # First block uses provided action, subsequent blocks use last_action (speculation)
            spec_action = last_action if block_index > 0 else action

            noise_slice = noise[:, :, current_start_frame:current_start_frame + num_frame_per_block]
            denoised_pred, video_np, vae_cache, conditional_dict, used_action = self._generate_block_tp(
                noise_slice, conditional_dict, current_start_frame, spec_action,
                kv_cache, kv_cache_mouse, kv_cache_keyboard, crossattn_cache, vae_cache
            )

            # Send result back to primary
            result_action_code = self._encode_action_single(spec_action)
            spec_result = torch.tensor([block_index, result_action_code], device=self.device, dtype=torch.long)
            dist.send(spec_result, dst=0)

            # Send frames
            frames_tensor = torch.from_numpy(video_np).to(self.device, dtype=torch.uint8)
            dist.send(frames_tensor.contiguous(), dst=0)

            last_action = action
            current_start_frame += num_frame_per_block

    def _speculative_worker_loop(self, noise: torch.Tensor, conditional_dict: dict):
        """Speculative worker loop."""
        batch_size = noise.shape[0]
        num_frame_per_block = self.num_frame_per_block

        kv_cache = self._initialize_kv_cache(batch_size, noise.dtype, self.device)
        kv_cache_mouse = self._initialize_kv_cache_action(batch_size, noise.dtype, self.device, 880)
        kv_cache_keyboard = self._initialize_kv_cache_action(batch_size, noise.dtype, self.device, 1)
        crossattn_cache = self._initialize_crossattn_cache(batch_size, noise.dtype, self.device)
        vae_cache = [None] * len(ZERO_VAE_CACHE)

        self.generator.block_mask = self.generator._prepare_blockwise_causal_attn_mask(
            self.device, num_frames=num_frame_per_block, frame_seqlen=self.frame_seq_length,
            num_frame_per_block=num_frame_per_block, local_attn_size=self.generator.local_attn_size
        )
        self.generator.num_frame_per_block = num_frame_per_block

        while True:
            control = torch.zeros(4, device=self.device, dtype=torch.long)
            self._broadcast_in_group(control, src_rank=0)

            if control[0].item() == 0:
                break

            block_index = control[1].item()
            current_start_frame = control[2].item()
            action_code = control[3].item()
            action = self._decode_action_single(action_code)

            noise_slice = noise[:, :, current_start_frame:current_start_frame + num_frame_per_block]
            self._generate_block_tp(
                noise_slice, conditional_dict, current_start_frame, action,
                kv_cache, kv_cache_mouse, kv_cache_keyboard, crossattn_cache, vae_cache
            )

    def stop(self):
        self.should_stop = True


# FastAPI setup (only on primary leader)
if IS_PRIMARY_LEADER:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, FileResponse

    app = FastAPI(title="Matrix-Game 2.0 TP+Speculative Server")
    inference_engine: Optional[TPSpeculativeInferenceEngine] = None

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
            "ready": inference_engine is not None,
            "running": inference_engine.is_running if inference_engine else False,
            "mode": inference_engine.mode if inference_engine else None,
            "tp_speculative": True,
            "world_size": WORLD_SIZE,
            "tp_size": TP_SIZE
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
                    if inference_engine and not inference_engine.is_running:
                        asyncio.create_task(
                            inference_engine.run_inference_primary_leader(img_input, websocket))
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Engine not ready or already running"
                        })

                elif msg_type == "action":
                    if inference_engine and inference_engine.is_running:
                        inference_engine._set_current_action({
                            "keyboard": data.get("keyboard", "q"),
                            "mouse": data.get("mouse", "u")
                        })

                elif msg_type == "stop":
                    if inference_engine:
                        inference_engine.stop()
                        await websocket.send_json({"type": "stopped"})

        except WebSocketDisconnect:
            if inference_engine:
                inference_engine.stop()
        except Exception as e:
            print(f"WebSocket error: {e}")
            if inference_engine:
                inference_engine.stop()


def parse_args():
    parser = argparse.ArgumentParser(description="TP+Speculative Multi-GPU Web Server")
    parser.add_argument("--config_path", type=str,
                        default="configs/inference_yaml/inference_universal.yaml")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--pretrained_model_path", type=str, default="Matrix-Game-2.0")
    parser.add_argument("--max_num_output_frames", type=int, default=360)
    parser.add_argument("--max_blocks", type=int, default=999999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if WORLD_SIZE < 4 or WORLD_SIZE % 2 != 0:
        if RANK == 0:
            print(f"Error: TP+Speculative requires at least 4 GPUs (even number)")
            print(f"Got WORLD_SIZE={WORLD_SIZE}")
        return

    print_rank0(f"\n{'='*60}")
    print_rank0(f"TP+Speculative Server")
    print_rank0(f"{'='*60}")
    print_rank0(f"Total GPUs: {WORLD_SIZE}")
    print_rank0(f"TP size: {TP_SIZE}")
    print_rank0(f"Primary ranks: 0-{TP_SIZE-1}")
    print_rank0(f"Speculative ranks: {TP_SIZE}-{WORLD_SIZE-1}")
    print_rank0(f"{'='*60}\n")

    global inference_engine
    engine = TPSpeculativeInferenceEngine(args)

    if IS_PRIMARY_LEADER:
        inference_engine = engine
        import uvicorn
        print(f"\nStarting TP+Speculative server at http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port, ws_max_size=50 * 1024 * 1024)
    elif IS_PRIMARY and not IS_GROUP_LEADER:
        engine.run_primary_worker()
    elif IS_SPECULATIVE_LEADER:
        engine.run_speculative_leader()
    else:
        engine.run_speculative_worker()


if __name__ == "__main__":
    main()
