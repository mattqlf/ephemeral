"""
Speculative Pipelining Web Server for Multi-GPU Inference

This server runs two inference engines on separate GPUs:
- Primary engine (GPU 0): Generates current block based on user input
- Speculative engine (GPU 1): Pre-generates next block assuming same action

If the user's next action matches the speculation, the pre-generated block
is used immediately (near-zero latency). If not, it's discarded and regenerated.

Usage:
    uv run python web_server_speculative.py \
        --config_path configs/inference_yaml/inference_universal.yaml \
        --checkpoint_path Matrix-Game-2.0/base_distilled_model/base_distill.safetensors \
        --pretrained_model_path Matrix-Game-2.0 \
        --port 8888

Requires: 2 GPUs with sufficient VRAM (~24GB each)
"""

import argparse
import asyncio
import base64
import copy
import io
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from threading import Thread, Lock, Event
import queue

import numpy as np
import torch
import torch._dynamo

torch._dynamo.config.cache_size_limit = 64

from PIL import Image
from einops import rearrange
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torchvision.transforms import v2

from pipeline import CausalInferenceStreamingPipeline
from pipeline.causal_inference import cond_current
from wan.vae.wanx_vae import get_wanx_vae_wrapper
from demo_utils.vae_block3 import VAEDecoderWrapper
from demo_utils.constant import ZERO_VAE_CACHE
from utils.misc import set_seed
from utils.conditions import Bench_actions_universal
from utils.wan_wrapper import WanDiffusionWrapper

app = FastAPI(title="Matrix-Game 2.0 Speculative Server")


def clone_kv_cache(kv_cache: List[dict], target_device: torch.device) -> List[dict]:
    """Deep clone KV cache to target device."""
    if kv_cache is None:
        return None
    cloned = []
    for block_cache in kv_cache:
        cloned.append({
            "k": block_cache["k"].clone().to(target_device),
            "v": block_cache["v"].clone().to(target_device),
            "global_end_index": block_cache["global_end_index"].clone().to(target_device),
            "local_end_index": block_cache["local_end_index"].clone().to(target_device),
        })
    return cloned


def clone_crossattn_cache(cache: List[dict], target_device: torch.device) -> List[dict]:
    """Deep clone cross-attention cache to target device."""
    if cache is None:
        return None
    cloned = []
    for block_cache in cache:
        cloned.append({
            "k": block_cache["k"].clone().to(target_device),
            "v": block_cache["v"].clone().to(target_device),
            "is_init": block_cache["is_init"],
        })
    return cloned


def clone_vae_cache(vae_cache: List, target_device: torch.device) -> List:
    """Deep clone VAE decoder cache to target device."""
    if vae_cache is None:
        return None
    cloned = []
    for item in vae_cache:
        if item is None:
            cloned.append(None)
        elif isinstance(item, torch.Tensor):
            cloned.append(item.clone().to(target_device))
        else:
            cloned.append(item)
    return cloned


def copy_kv_cache_inplace(src: List[dict], dst: List[dict]):
    """Copy KV cache values in-place (same device)."""
    for src_block, dst_block in zip(src, dst):
        dst_block["k"].copy_(src_block["k"])
        dst_block["v"].copy_(src_block["v"])
        dst_block["global_end_index"].copy_(src_block["global_end_index"])
        dst_block["local_end_index"].copy_(src_block["local_end_index"])


def copy_crossattn_cache_inplace(src: List[dict], dst: List[dict]):
    """Copy cross-attention cache values in-place (same device)."""
    for src_block, dst_block in zip(src, dst):
        dst_block["k"].copy_(src_block["k"])
        dst_block["v"].copy_(src_block["v"])
        dst_block["is_init"] = src_block["is_init"]


def copy_vae_cache_inplace(src: List, dst: List):
    """Copy VAE cache values in-place (same device)."""
    for i, item in enumerate(src):
        if item is None:
            dst[i] = None
        elif isinstance(item, torch.Tensor):
            if dst[i] is None:
                dst[i] = item.clone()
            else:
                dst[i].copy_(item)
        else:
            dst[i] = item


@dataclass
class InferenceState:
    """Complete state needed to continue inference from a checkpoint."""
    block_index: int
    current_start_frame: int
    kv_cache1: List[dict]
    kv_cache_mouse: List[dict]
    kv_cache_keyboard: List[dict]
    crossattn_cache: List[dict]
    vae_cache: List
    conditional_dict: dict
    output: torch.Tensor
    noise: torch.Tensor
    device: torch.device


@dataclass
class SpeculativeResult:
    """Result from speculative generation."""
    block_index: int
    action: Dict[str, str]
    frames: np.ndarray
    denoised_pred: torch.Tensor
    vae_cache: List
    kv_cache1: List[dict]
    kv_cache_mouse: List[dict]
    kv_cache_keyboard: List[dict]
    crossattn_cache: List[dict]
    conditional_dict: dict
    valid: bool = True


class SpeculativeInferenceEngine:
    """
    Manages two inference engines for speculative pipelining.

    Primary (GPU 0): Generates blocks based on actual user input
    Speculative (GPU 1): Pre-generates next block assuming same action
    """

    def __init__(self, args):
        self.args = args
        self.weight_dtype = torch.bfloat16

        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus < 2:
            raise RuntimeError(
                f"Speculative mode requires 2 GPUs, but only {self.num_gpus} available. "
                "Use web_server.py for single-GPU inference."
            )

        self.device_primary = torch.device("cuda:0")
        self.device_speculative = torch.device("cuda:1")

        print(f"Primary engine: {self.device_primary}")
        print(f"Speculative engine: {self.device_speculative}")

        self._init_config()

        # Speculation state
        self.speculative_result: Optional[SpeculativeResult] = None
        self.speculation_lock = Lock()
        self.speculation_task_queue = queue.Queue()
        self.speculation_ready = Event()
        self.stop_speculation = Event()

        # Session state
        self.is_running = False
        self.should_stop = False

        # Continuous action tracking - always stores latest user input
        self.current_action = {"keyboard": "q", "mouse": "u"}
        self.action_lock = Lock()

        self._init_engines()

        self.frame_process = v2.Compose([
            v2.Resize(size=(352, 640), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _init_config(self):
        self.config = OmegaConf.load(self.args.config_path)
        self.mode = self.config.get('mode', 'universal')

        if getattr(self.args, 'fast', False):
            print("[fast mode] Using 2 denoising steps (was 3)")
            self.config.denoising_step_list = [1000, 500]

    def _init_engines(self):
        """Initialize both inference engines in parallel."""
        print("Initializing speculative inference engines...")
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")

        # Load shared weights once
        print("Loading shared model weights...")
        vae_path = os.path.join(self.args.pretrained_model_path, "Wan2.1_VAE.pth")
        vae_state_dict = torch.load(vae_path, map_location="cpu", mmap=True)
        decoder_state_dict = {k: v for k, v in vae_state_dict.items()
                              if 'decoder.' in k or 'conv2' in k}

        main_checkpoint = None
        if self.args.checkpoint_path:
            print(f"Loading checkpoint: {self.args.checkpoint_path}")
            main_checkpoint = load_file(self.args.checkpoint_path)

        # Initialize both engines
        print(f"Creating primary engine on {self.device_primary}...")
        self.primary = self._create_engine(
            self.device_primary, decoder_state_dict, main_checkpoint)

        print(f"Creating speculative engine on {self.device_speculative}...")
        self.speculative = self._create_engine(
            self.device_speculative, decoder_state_dict, main_checkpoint)

        # Start speculation worker thread
        self.speculation_thread = Thread(
            target=self._speculation_worker, daemon=True, name="speculation-worker")
        self.speculation_thread.start()

        print("Speculative engines ready!")
        if not getattr(self.args, 'no_warmup', False):
            self._warmup_engines()
        else:
            print("Skipping warmup (--no-warmup). Kernels will compile on first inference.")

    def _create_engine(self, device: torch.device, decoder_state_dict: dict,
                       main_checkpoint: Optional[dict]) -> dict:
        """Create a single inference engine on the specified device."""
        generator = WanDiffusionWrapper(
            **getattr(self.config, "model_kwargs", {}), is_causal=True)

        vae_decoder = VAEDecoderWrapper()
        vae_decoder.load_state_dict(copy.deepcopy(decoder_state_dict))
        vae_decoder.to(device, torch.float16)
        vae_decoder.requires_grad_(False)
        vae_decoder.eval()
        print(f"  Compiling VAE decoder on {device}...")
        vae_decoder.compile(mode="max-autotune-no-cudagraphs")

        pipeline = CausalInferenceStreamingPipeline(
            self.config, generator=generator, vae_decoder=vae_decoder)

        if main_checkpoint:
            pipeline.generator.load_state_dict(copy.deepcopy(main_checkpoint))

        pipeline = pipeline.to(device=device, dtype=self.weight_dtype)
        pipeline.vae_decoder.to(torch.float16)

        vae = get_wanx_vae_wrapper(self.args.pretrained_model_path, torch.float16)
        vae.requires_grad_(False)
        vae.eval()
        vae = vae.to(device, self.weight_dtype)

        return {
            "pipeline": pipeline,
            "vae": vae,
            "device": device,
        }

    def _warmup_engines(self):
        """Warmup both engines to compile kernels."""
        print("Warming up engines...")
        for name, engine in [("primary", self.primary), ("speculative", self.speculative)]:
            print(f"  Warming up {name} engine...")
            self._warmup_single_engine(engine)
        print("Warmup complete!")

    def _warmup_single_engine(self, engine: dict):
        """Warmup a single engine."""
        pipeline = engine["pipeline"]
        device = engine["device"]

        num_frame_per_block = pipeline.num_frame_per_block
        denoising_steps = pipeline.denoising_step_list

        # Create dummy inputs
        test_image = torch.randn([1, 3, 1, 352, 640], device=device, dtype=self.weight_dtype)
        with torch.no_grad():
            visual_context_shape = engine["vae"].clip.encode_video(test_image).shape

        max_output_frames = self.args.max_num_output_frames
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

        cond = cond_current(conditional_dict, 0, num_frame_per_block)
        noise_slice = noise[:, :, :num_frame_per_block]

        pipeline._initialize_kv_cache(1, self.weight_dtype, device)
        pipeline._initialize_kv_cache_mouse_and_keyboard(1, self.weight_dtype, device)
        pipeline._initialize_crossattn_cache(1, self.weight_dtype, device)

        with torch.no_grad():
            output = noise_slice
            for current_timestep in denoising_steps:
                timestep = torch.ones([1, num_frame_per_block], device=device, dtype=torch.int64) * current_timestep
                _, output = pipeline.generator(
                    noisy_image_or_video=output,
                    conditional_dict=cond,
                    timestep=timestep,
                    kv_cache=pipeline.kv_cache1,
                    kv_cache_mouse=pipeline.kv_cache_mouse,
                    kv_cache_keyboard=pipeline.kv_cache_keyboard,
                    crossattn_cache=pipeline.crossattn_cache,
                    current_start=0
                )

            # Context pass
            context_timestep = torch.ones([1, num_frame_per_block], device=device, dtype=torch.int64) * pipeline.args.context_noise
            pipeline.generator(
                noisy_image_or_video=output,
                conditional_dict=cond,
                timestep=context_timestep,
                kv_cache=pipeline.kv_cache1,
                kv_cache_mouse=pipeline.kv_cache_mouse,
                kv_cache_keyboard=pipeline.kv_cache_keyboard,
                crossattn_cache=pipeline.crossattn_cache,
                current_start=0
            )

            # VAE decoder warmup
            vae_cache = [None] * len(ZERO_VAE_CACHE)
            dummy_latent = output.transpose(1, 2)
            _, vae_cache = pipeline.vae_decoder(dummy_latent.half(), *vae_cache)
            _, _ = pipeline.vae_decoder(dummy_latent.half(), *vae_cache)

        torch.cuda.empty_cache()

    def _speculation_worker(self):
        """Background thread that generates speculative blocks on GPU 1."""
        print("[speculation] Worker thread started")

        while not self.stop_speculation.is_set():
            try:
                task = self.speculation_task_queue.get(timeout=0.5)
                if task is None:
                    break

                action, state = task
                print(f"[speculation] Generating block {state.block_index + 1} with action {action}")

                start_time = time.time()
                result = self._generate_block_on_speculative(action, state)
                elapsed = time.time() - start_time

                with self.speculation_lock:
                    self.speculative_result = result
                    self.speculation_ready.set()

                print(f"[speculation] Block {state.block_index + 1} ready in {elapsed:.2f}s")

            except queue.Empty:
                continue
            except Exception as e:
                import traceback
                print(f"[speculation] Worker error: {e}")
                traceback.print_exc()

        print("[speculation] Worker thread stopped")

    def _generate_block_on_speculative(self, action: Dict[str, str],
                                        state: InferenceState) -> SpeculativeResult:
        """Generate a single block on the speculative engine."""
        pipeline = self.speculative["pipeline"]
        device = self.device_speculative

        # Transfer state to speculative device
        pipeline.kv_cache1 = clone_kv_cache(state.kv_cache1, device)
        pipeline.kv_cache_mouse = clone_kv_cache(state.kv_cache_mouse, device)
        pipeline.kv_cache_keyboard = clone_kv_cache(state.kv_cache_keyboard, device)
        pipeline.crossattn_cache = clone_crossattn_cache(state.crossattn_cache, device)
        vae_cache = clone_vae_cache(state.vae_cache, device)

        # Clone conditional dict to speculative device
        conditional_dict = {
            "cond_concat": state.conditional_dict["cond_concat"].to(device),
            "visual_context": state.conditional_dict["visual_context"].to(device),
            "keyboard_cond": state.conditional_dict["keyboard_cond"].clone().to(device),
            "mouse_cond": state.conditional_dict["mouse_cond"].clone().to(device),
        }

        noise = state.noise.to(device)
        num_frame_per_block = pipeline.num_frame_per_block
        current_start_frame = state.current_start_frame
        batch_size = 1

        # Parse action to tensor format
        current_actions = self._parse_action_to_tensor(action, device)

        # Get noise slice for this block
        noisy_input = noise[:, :, current_start_frame:current_start_frame + num_frame_per_block]

        # Update conditions with the speculated action
        new_act, conditional_dict = cond_current(
            conditional_dict, current_start_frame, num_frame_per_block,
            replace=current_actions, mode=self.mode)

        # Denoising loop
        with torch.no_grad():
            for index, current_timestep in enumerate(pipeline.denoising_step_list):
                timestep = torch.ones(
                    [batch_size, num_frame_per_block],
                    device=device, dtype=torch.int64) * current_timestep

                if index < len(pipeline.denoising_step_list) - 1:
                    _, denoised_pred = pipeline.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=new_act,
                        timestep=timestep,
                        kv_cache=pipeline.kv_cache1,
                        kv_cache_mouse=pipeline.kv_cache_mouse,
                        kv_cache_keyboard=pipeline.kv_cache_keyboard,
                        crossattn_cache=pipeline.crossattn_cache,
                        current_start=current_start_frame * pipeline.frame_seq_length
                    )
                    next_timestep = pipeline.denoising_step_list[index + 1]
                    noisy_input = pipeline.scheduler.add_noise(
                        rearrange(denoised_pred, 'b c f h w -> (b f) c h w'),
                        torch.randn_like(rearrange(denoised_pred, 'b c f h w -> (b f) c h w')),
                        next_timestep * torch.ones(
                            [batch_size * num_frame_per_block], device=device, dtype=torch.long)
                    )
                    noisy_input = rearrange(noisy_input, '(b f) c h w -> b c f h w', b=batch_size)
                else:
                    _, denoised_pred = pipeline.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=new_act,
                        timestep=timestep,
                        kv_cache=pipeline.kv_cache1,
                        kv_cache_mouse=pipeline.kv_cache_mouse,
                        kv_cache_keyboard=pipeline.kv_cache_keyboard,
                        crossattn_cache=pipeline.crossattn_cache,
                        current_start=current_start_frame * pipeline.frame_seq_length
                    )

            # Update KV cache with context pass
            context_timestep = torch.ones_like(timestep) * pipeline.args.context_noise
            pipeline.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=new_act,
                timestep=context_timestep,
                kv_cache=pipeline.kv_cache1,
                kv_cache_mouse=pipeline.kv_cache_mouse,
                kv_cache_keyboard=pipeline.kv_cache_keyboard,
                crossattn_cache=pipeline.crossattn_cache,
                current_start=current_start_frame * pipeline.frame_seq_length,
            )

            # VAE decode
            denoised_pred_t = denoised_pred.transpose(1, 2)
            video, vae_cache = pipeline.vae_decoder(denoised_pred_t.half(), *vae_cache)

            video_np = rearrange(video, "B T C H W -> B T H W C")
            video_np = ((video_np.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)[0]

        return SpeculativeResult(
            block_index=state.block_index + 1,
            action=action,
            frames=video_np,
            denoised_pred=denoised_pred,
            vae_cache=vae_cache,
            kv_cache1=pipeline.kv_cache1,
            kv_cache_mouse=pipeline.kv_cache_mouse,
            kv_cache_keyboard=pipeline.kv_cache_keyboard,
            crossattn_cache=pipeline.crossattn_cache,
            conditional_dict=conditional_dict,
            valid=True
        )

    def _parse_action_to_tensor(self, action: Dict[str, str], device: torch.device) -> dict:
        """Parse action from dict format to tensor format."""
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

    async def run_inference(self, img_input: str, websocket: WebSocket):
        """Run speculative inference loop."""
        self.is_running = True
        self.should_stop = False

        # Reset action tracking and clear speculation state
        with self.action_lock:
            self.current_action = {"keyboard": "q", "mouse": "u"}

        while not self.speculation_task_queue.empty():
            try:
                self.speculation_task_queue.get_nowait()
            except:
                break

        with self.speculation_lock:
            self.speculative_result = None
        self.speculation_ready.clear()

        engine = self.primary
        pipeline = engine["pipeline"]
        vae = engine["vae"]
        device = engine["device"]

        try:
            # Load and process image
            if img_input.startswith('data:image'):
                print(f"Processing uploaded image...")
                header, b64_data = img_input.split(',', 1)
                img_bytes = base64.b64decode(b64_data)
                image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            else:
                print(f"Loading image from: {img_input}")
                from diffusers.utils import load_image
                image = load_image(img_input)

            image = self._resizecrop(image, 352, 640)
            image_tensor = self.frame_process(image)[None, :, None, :, :].to(
                dtype=self.weight_dtype, device=device)

            # Encode input image
            max_output_frames = self.args.max_num_output_frames
            padding_video = torch.zeros_like(image_tensor).repeat(1, 1, 4 * (max_output_frames - 1), 1, 1)
            img_cond = torch.concat([image_tensor, padding_video], dim=2)
            tiler_kwargs = {"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}
            img_cond = vae.encode(img_cond, device=device, **tiler_kwargs).to(device)

            mask_cond = torch.ones_like(img_cond)
            mask_cond[:, :, 1:] = 0
            cond_concat = torch.cat([mask_cond[:, :4], img_cond], dim=1)
            visual_context = vae.clip.encode_video(image_tensor)

            sampled_noise = torch.randn(
                [1, 16, max_output_frames, 44, 80],
                device=device, dtype=self.weight_dtype
            )

            num_frames = (max_output_frames - 1) * 4 + 1

            conditional_dict = {
                "cond_concat": cond_concat.to(device=device, dtype=self.weight_dtype),
                "visual_context": visual_context.to(device=device, dtype=self.weight_dtype)
            }

            cond_data = Bench_actions_universal(num_frames)
            mouse_condition = cond_data['mouse_condition'].unsqueeze(0).to(
                device=device, dtype=self.weight_dtype)
            conditional_dict['mouse_cond'] = mouse_condition

            keyboard_condition = cond_data['keyboard_condition'].unsqueeze(0).to(
                device=device, dtype=self.weight_dtype)
            conditional_dict['keyboard_cond'] = keyboard_condition

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
                "message": "Ready for input. Press keys to control. [SPECULATIVE MODE]"
            })

            with torch.no_grad():
                await self._speculative_inference_loop(
                    engine=engine,
                    noise=sampled_noise,
                    conditional_dict=conditional_dict,
                    websocket=websocket
                )

        except Exception as e:
            import traceback
            traceback.print_exc()
            await websocket.send_json({"type": "error", "message": str(e)})
        finally:
            self.is_running = False

    def _get_current_action(self) -> dict:
        """Get the current action from continuous tracking."""
        with self.action_lock:
            return self.current_action.copy()

    def _set_current_action(self, action: dict):
        """Update the current action from user input."""
        with self.action_lock:
            self.current_action = action.copy()

    async def _speculative_inference_loop(
        self,
        engine: dict,
        noise: torch.Tensor,
        conditional_dict: dict,
        websocket: WebSocket
    ):
        """
        Main inference loop with speculative pipelining.

        Key improvements over naive approach:
        1. Continuous action tracking - no blocking wait for user input
        2. Early speculation - queue next block immediately after context pass
        3. Speculation wait - wait for in-progress speculation up to small timeout
        4. Pipelined frame sending - async frame transmission
        """
        pipeline = engine["pipeline"]
        device = engine["device"]

        batch_size, num_channels, num_frames, height, width = noise.shape
        num_frame_per_block = pipeline.num_frame_per_block
        num_blocks = num_frames // num_frame_per_block
        max_blocks = self.args.max_blocks or num_blocks

        output = torch.zeros_like(noise)
        vae_cache = [None] * len(ZERO_VAE_CACHE)

        # Initialize KV caches
        pipeline._initialize_kv_cache(batch_size, noise.dtype, noise.device)
        pipeline._initialize_kv_cache_mouse_and_keyboard(batch_size, noise.dtype, noise.device)
        pipeline._initialize_crossattn_cache(batch_size, noise.dtype, noise.device)

        current_start_frame = 0
        speculation_hits = 0
        speculation_misses = 0

        # Notify client to start streaming actions
        await websocket.send_json({
            "type": "start_streaming",
            "mode": self.mode,
            "message": "Hold keys to control. Actions are streamed continuously."
        })

        for block_index in range(min(max_blocks, num_blocks)):
            if self.should_stop:
                break

            block_start_time = time.time()

            # Get current action (non-blocking - uses latest streamed input)
            action = self._get_current_action()

            # Check if speculation is ready
            use_speculation = False
            spec_result = None

            # For block > 0, wait for speculation from previous iteration
            if block_index > 0:
                # Wait for speculation to complete (queued at end of previous block)
                # Speculation takes ~1.5s, frame sending takes ~0.4s
                # So we may need to wait ~1.1s more after frames are sent
                if self.speculation_ready.wait(timeout=3.0):
                    # Speculation completed - check if it matches
                    with self.speculation_lock:
                        if (self.speculative_result is not None and
                            self.speculative_result.block_index == block_index + 1):
                            # Check if action still matches (user might have changed input)
                            current_action = self._get_current_action()
                            if self.speculative_result.action == current_action:
                                use_speculation = True
                                spec_result = self.speculative_result
                                action = current_action  # Use the matched action
                                self.speculative_result = None
                                speculation_hits += 1
                            else:
                                # Action changed - discard speculation
                                self.speculative_result = None
                else:
                    print(f"[block {block_index + 1}] Speculation timeout")

            if use_speculation:
                print(f"[block {block_index + 1}] SPECULATION HIT! action={action}")

                # Use speculative result
                video_np = spec_result.frames
                denoised_pred = spec_result.denoised_pred

                # Sync state from speculative back to primary
                copy_kv_cache_inplace(
                    clone_kv_cache(spec_result.kv_cache1, device), pipeline.kv_cache1)
                copy_kv_cache_inplace(
                    clone_kv_cache(spec_result.kv_cache_mouse, device), pipeline.kv_cache_mouse)
                copy_kv_cache_inplace(
                    clone_kv_cache(spec_result.kv_cache_keyboard, device), pipeline.kv_cache_keyboard)
                copy_crossattn_cache_inplace(
                    clone_crossattn_cache(spec_result.crossattn_cache, device), pipeline.crossattn_cache)
                vae_cache = clone_vae_cache(spec_result.vae_cache, device)

                # Update conditional dict
                conditional_dict["keyboard_cond"] = spec_result.conditional_dict["keyboard_cond"].to(device)
                conditional_dict["mouse_cond"] = spec_result.conditional_dict["mouse_cond"].to(device)

                output[:, :, current_start_frame:current_start_frame + num_frame_per_block] = \
                    denoised_pred.to(device)

            else:
                # No speculation available or action changed
                if block_index == 0:
                    print(f"[block {block_index + 1}] Generating (first block)")
                else:
                    print(f"[block {block_index + 1}] SPECULATION MISS - action changed or not ready")
                    with self.speculation_lock:
                        self.speculative_result = None
                speculation_misses += 1

                # Generate block on primary
                current_actions = self._parse_action_to_tensor(action, device)
                noisy_input = noise[:, :, current_start_frame:current_start_frame + num_frame_per_block]

                new_act, conditional_dict = cond_current(
                    conditional_dict, current_start_frame, num_frame_per_block,
                    replace=current_actions, mode=self.mode)

                for index, current_timestep in enumerate(pipeline.denoising_step_list):
                    timestep = torch.ones(
                        [batch_size, num_frame_per_block],
                        device=device, dtype=torch.int64) * current_timestep

                    if index < len(pipeline.denoising_step_list) - 1:
                        _, denoised_pred = pipeline.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=new_act,
                            timestep=timestep,
                            kv_cache=pipeline.kv_cache1,
                            kv_cache_mouse=pipeline.kv_cache_mouse,
                            kv_cache_keyboard=pipeline.kv_cache_keyboard,
                            crossattn_cache=pipeline.crossattn_cache,
                            current_start=current_start_frame * pipeline.frame_seq_length
                        )
                        next_timestep = pipeline.denoising_step_list[index + 1]
                        noisy_input = pipeline.scheduler.add_noise(
                            rearrange(denoised_pred, 'b c f h w -> (b f) c h w'),
                            torch.randn_like(rearrange(denoised_pred, 'b c f h w -> (b f) c h w')),
                            next_timestep * torch.ones(
                                [batch_size * num_frame_per_block], device=device, dtype=torch.long)
                        )
                        noisy_input = rearrange(noisy_input, '(b f) c h w -> b c f h w', b=batch_size)
                    else:
                        _, denoised_pred = pipeline.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=new_act,
                            timestep=timestep,
                            kv_cache=pipeline.kv_cache1,
                            kv_cache_mouse=pipeline.kv_cache_mouse,
                            kv_cache_keyboard=pipeline.kv_cache_keyboard,
                            crossattn_cache=pipeline.crossattn_cache,
                            current_start=current_start_frame * pipeline.frame_seq_length
                        )

                output[:, :, current_start_frame:current_start_frame + num_frame_per_block] = denoised_pred

                # Context pass to update KV cache
                context_timestep = torch.ones_like(timestep) * pipeline.args.context_noise
                pipeline.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=new_act,
                    timestep=context_timestep,
                    kv_cache=pipeline.kv_cache1,
                    kv_cache_mouse=pipeline.kv_cache_mouse,
                    kv_cache_keyboard=pipeline.kv_cache_keyboard,
                    crossattn_cache=pipeline.crossattn_cache,
                    current_start=current_start_frame * pipeline.frame_seq_length,
                )

                # VAE decode
                denoised_pred_t = denoised_pred.transpose(1, 2)
                video, vae_cache = pipeline.vae_decoder(denoised_pred_t.half(), *vae_cache)

                video_np = rearrange(video, "B T C H W -> B T H W C")
                video_np = ((video_np.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)[0]

            # Queue speculation IMMEDIATELY after we have state (before sending frames)
            # This maximizes time for speculation to complete
            if block_index + 1 < min(max_blocks, num_blocks):
                # Clear any stale speculation
                self.speculation_ready.clear()
                with self.speculation_lock:
                    self.speculative_result = None

                # Create state snapshot
                state = InferenceState(
                    block_index=block_index + 1,
                    current_start_frame=current_start_frame + num_frame_per_block,
                    kv_cache1=pipeline.kv_cache1,
                    kv_cache_mouse=pipeline.kv_cache_mouse,
                    kv_cache_keyboard=pipeline.kv_cache_keyboard,
                    crossattn_cache=pipeline.crossattn_cache,
                    vae_cache=vae_cache,
                    conditional_dict=conditional_dict,
                    output=output,
                    noise=noise,
                    device=device
                )

                # Queue speculation with current action (user likely continues same input)
                self.speculation_task_queue.put((action, state))

            # Send frames to client (this happens AFTER speculation is queued)
            for frame_idx, frame in enumerate(video_np):
                await websocket.send_json({
                    "type": "frame",
                    "data": self._frame_to_base64(frame),
                    "block": block_index + 1,
                    "frame": frame_idx
                })
                # Small delay between frames for smooth playback
                await asyncio.sleep(0.033)  # ~30fps frame rate

            current_start_frame += num_frame_per_block

            block_time = time.time() - block_start_time
            hit_rate = speculation_hits / (speculation_hits + speculation_misses) * 100 if (speculation_hits + speculation_misses) > 0 else 0

            await websocket.send_json({
                "type": "block_complete",
                "block": block_index + 1,
                "time": f"{block_time:.2f}s",
                "speculation_hit": use_speculation,
                "hit_rate": f"{hit_rate:.1f}%"
            })

        # Final stats
        total = speculation_hits + speculation_misses
        if total > 0:
            print(f"\n[SPECULATION STATS] Hits: {speculation_hits}/{total} ({100*speculation_hits/total:.1f}%)")

        await websocket.send_json({"type": "complete"})

    def stop(self):
        self.should_stop = True
        self.stop_speculation.set()


# Global engine
inference_engine: Optional[SpeculativeInferenceEngine] = None


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
        "speculative": True
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
                        inference_engine.run_inference(img_input, websocket))
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Engine not ready or already running"
                    })

            elif msg_type == "action":
                if inference_engine and inference_engine.is_running:
                    # Update continuous action tracking (non-blocking)
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
    parser = argparse.ArgumentParser(description="Speculative Multi-GPU Web Server")
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

    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory // 1024**3}GB)")

    print("\nInitializing speculative inference engine...")
    inference_engine = SpeculativeInferenceEngine(args)

    import uvicorn
    print(f"\nStarting speculative server at http://{args.host}:{args.port}")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        ws_max_size=50 * 1024 * 1024
    )
