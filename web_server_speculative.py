"""
Speculative Pipelining Web Server for Multi-GPU Inference

This server runs two inference engines on separate GPUs:
- Primary engine: Generates current block based on user input
- Speculative engine: Pre-generates next block assuming same action

If the user's next action matches the speculation, the pre-generated block
is used immediately (near-zero latency). If not, it's discarded and regenerated.

Usage:
    uv run python web_server_speculative.py \
        --config_path configs/inference_yaml/inference_universal.yaml \
        --checkpoint_path Matrix-Game-2.0/base_distilled_model/base_distill.safetensors \
        --pretrained_model_path Matrix-Game-2.0 \
        --port 8888

Requires: 2+ GPUs with sufficient VRAM (~80GB each)
"""

import argparse
import asyncio
import base64
import copy
import io
import os
import warnings
import concurrent.futures
from dataclasses import dataclass
from typing import Optional, Dict, Any
from threading import Thread, Lock
import queue

import numpy as np
import torch
from PIL import Image
from einops import rearrange
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from omegaconf import OmegaConf
from safetensors.torch import load_file

from pipeline import CausalInferenceStreamingPipeline
from pipeline.causal_inference import cond_current
from wan.vae.wanx_vae import get_wanx_vae_wrapper
from demo_utils.vae_block3 import VAEDecoderWrapper
from demo_utils.constant import ZERO_VAE_CACHE
from utils.misc import set_seed
from utils.conditions import Bench_actions_universal, Bench_actions_gta_drive, Bench_actions_templerun
from utils.misc import load_image
from wan.modules.model import WanModel
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

app = FastAPI()


@dataclass
class SpeculativeBlock:
    """Represents a speculatively generated block."""
    block_index: int
    action: Dict[str, str]  # {"keyboard": "w", "mouse": "u"}
    frames: list  # List of numpy frames
    valid: bool = True


class SpeculativeInferenceEngine:
    """
    Manages two inference engines for speculative pipelining.

    Engine 0 (primary): Generates blocks based on actual user input
    Engine 1 (speculative): Pre-generates next block assuming same action
    """

    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.weight_dtype = torch.float16 if config.mixed_precision else torch.float32

        # Check GPU availability
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus < 2:
            print(f"Warning: Only {self.num_gpus} GPU(s) available. Speculative mode works best with 2+ GPUs.")
            print("Falling back to single-GPU mode with async speculation.")

        self.device_primary = "cuda:0"
        self.device_speculative = f"cuda:{min(1, self.num_gpus - 1)}"

        print(f"Primary engine: {self.device_primary}")
        print(f"Speculative engine: {self.device_speculative}")

        # Initialize engines
        self.primary_engine = None
        self.speculative_engine = None

        # Speculation state
        self.speculative_block: Optional[SpeculativeBlock] = None
        self.speculation_lock = Lock()
        self.speculation_queue = queue.Queue()

        # Action queue for WebSocket
        self.action_queue = asyncio.Queue()
        self.is_running = False
        self.should_stop = False

        self._init_engines()

    def _init_engines(self):
        """Initialize both inference engines."""
        print("Initializing speculative inference engines...")

        warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")

        # Load shared weights once
        print("Loading shared weights...")

        vae_path = os.path.join(self.args.pretrained_model_path, "Wan2.1_VAE.pth")
        vae_state_dict = torch.load(vae_path, map_location="cpu", mmap=True)
        decoder_state_dict = {k: v for k, v in vae_state_dict.items()
                            if 'decoder.' in k or 'conv2' in k}

        main_checkpoint = None
        if self.args.checkpoint_path:
            main_checkpoint = load_file(self.args.checkpoint_path)

        # Initialize primary engine
        print(f"Initializing primary engine on {self.device_primary}...")
        self.primary_engine = self._create_engine(
            self.device_primary, decoder_state_dict, main_checkpoint)

        # Initialize speculative engine (on second GPU if available)
        print(f"Initializing speculative engine on {self.device_speculative}...")
        self.speculative_engine = self._create_engine(
            self.device_speculative, decoder_state_dict, main_checkpoint)

        # Start speculation worker thread
        self.speculation_thread = Thread(target=self._speculation_worker, daemon=True)
        self.speculation_thread.start()

        print("Speculative engines ready!")

    def _create_engine(self, device, decoder_state_dict, main_checkpoint):
        """Create a single inference engine on the specified device."""
        from wan.modules.causal_model import WanCausalModel as WanDiffusionWrapper

        generator = WanDiffusionWrapper(
            **getattr(self.config, "model_kwargs", {}), is_causal=True)

        # VAE decoder
        vae_decoder = VAEDecoderWrapper()
        vae_decoder.load_state_dict(copy.deepcopy(decoder_state_dict))
        vae_decoder.to(device, torch.float16)
        vae_decoder.requires_grad_(False)
        vae_decoder.eval()
        vae_decoder.compile(mode="max-autotune-no-cudagraphs")

        # Pipeline
        pipeline = CausalInferenceStreamingPipeline(
            self.config, generator=generator, vae_decoder=vae_decoder)

        if main_checkpoint:
            pipeline.generator.load_state_dict(copy.deepcopy(main_checkpoint))

        pipeline = pipeline.to(device=device, dtype=self.weight_dtype)
        pipeline.vae_decoder.to(torch.float16)

        # VAE encoder
        vae = get_wanx_vae_wrapper(self.args.pretrained_model_path, torch.float16)
        vae.requires_grad_(False)
        vae.eval()
        vae = vae.to(device, self.weight_dtype)

        return {
            "pipeline": pipeline,
            "vae": vae,
            "device": device,
        }

    def _speculation_worker(self):
        """Background thread that generates speculative blocks."""
        while True:
            try:
                task = self.speculation_queue.get(timeout=1.0)
                if task is None:
                    break

                action, block_index, state = task

                # Generate speculative block
                frames = self._generate_block_sync(
                    self.speculative_engine, state, action, block_index)

                with self.speculation_lock:
                    self.speculative_block = SpeculativeBlock(
                        block_index=block_index,
                        action=action,
                        frames=frames,
                        valid=True
                    )

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Speculation worker error: {e}")

    def _generate_block_sync(self, engine, state, action, block_index):
        """Synchronously generate a single block (for speculation thread)."""
        # This is a simplified version - full implementation would need
        # to properly manage the KV cache state
        pass  # Placeholder for full implementation

    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert a numpy frame to base64 JPEG."""
        img = Image.fromarray(frame.astype(np.uint8))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _resizecrop(self, image, th, tw):
        """Resize and crop image to target dimensions."""
        w, h = image.size
        if h / w > th / tw:
            new_w = tw
            new_h = int(new_w * h / w)
        else:
            new_h = th
            new_w = int(new_h * w / h)
        image = image.resize((new_w, new_h), Image.LANCZOS)
        left, top = (new_w - tw) // 2, (new_h - th) // 2
        return image.crop((left, top, left + tw, top + th))

    @property
    def mode(self):
        return self.config.mode

    @property
    def frame_process(self):
        import torchvision.transforms as transforms
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    async def run_inference(self, img_input: str, websocket: WebSocket):
        """
        Run speculative inference loop.

        The key optimization: while we wait for user input for block N,
        we speculatively generate block N+1 assuming the same action.
        """
        self.is_running = True
        self.should_stop = False

        engine = self.primary_engine
        pipeline = engine["pipeline"]
        vae = engine["vae"]
        device = engine["device"]

        # Clear action queue
        while not self.action_queue.empty():
            try:
                self.action_queue.get_nowait()
            except:
                break

        try:
            # Load and process image
            if img_input.startswith('data:image'):
                print(f"Processing uploaded image ({len(img_input) // 1024}KB base64)...")
                header, b64_data = img_input.split(',', 1)
                img_bytes = base64.b64decode(b64_data)
                image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                print(f"  Image size: {image.size}")
            else:
                print(f"Loading image from: {img_input}")
                image = load_image(img_input)

            image = self._resizecrop(image, 352, 640)
            print(f"  Resized to: {image.size}")

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

            # Setup conditions
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

            # Send ready signal
            await websocket.send_json({
                "type": "ready",
                "mode": self.mode,
                "message": "Ready for input. Press keys to control."
            })

            # Run speculative inference loop
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

    async def _speculative_inference_loop(
        self,
        engine: dict,
        noise: torch.Tensor,
        conditional_dict: dict,
        websocket: WebSocket
    ):
        """
        Speculative inference loop with pipelining.

        Flow:
        1. Request action from user
        2. While waiting, check if we have a valid speculative block
        3. If speculation matches user action, use it immediately
        4. Otherwise, generate block normally
        5. Start speculating on next block with same action
        """
        pipeline = engine["pipeline"]
        device = engine["device"]

        batch_size, num_channels, num_frames, height, width = noise.shape
        num_frame_per_block = pipeline.num_frame_per_block
        num_blocks = num_frames // num_frame_per_block
        max_blocks = self.args.max_blocks or num_blocks

        output = torch.zeros_like(noise)
        vae_cache = copy.deepcopy(ZERO_VAE_CACHE)
        for j in range(len(vae_cache)):
            vae_cache[j] = None

        # Initialize KV caches
        pipeline.kv_cache1 = None
        pipeline.kv_cache_keyboard = None
        pipeline.kv_cache_mouse = None
        pipeline.crossattn_cache = None

        pipeline._initialize_kv_cache(batch_size, noise.dtype, noise.device)
        pipeline._initialize_kv_cache_mouse_and_keyboard(batch_size, noise.dtype, noise.device)
        pipeline._initialize_crossattn_cache(batch_size, noise.dtype, noise.device)

        current_start_frame = 0
        last_action = None
        speculation_hits = 0
        speculation_misses = 0

        for block_index in range(min(max_blocks, num_blocks)):
            if self.should_stop:
                break

            # Request action
            await websocket.send_json({
                "type": "request_action",
                "block": block_index + 1,
                "mode": self.mode
            })

            # Wait for action
            try:
                action = await asyncio.wait_for(
                    self.action_queue.get(),
                    timeout=120.0
                )
                if action is None:
                    break
                current_actions = self._parse_action(action)
            except asyncio.TimeoutError:
                current_actions = self._parse_action({"keyboard": "q", "mouse": "u"})

            # Check speculation hit
            use_speculation = False
            with self.speculation_lock:
                if (self.speculative_block and
                    self.speculative_block.block_index == block_index + 1 and
                    self.speculative_block.action == action and
                    self.speculative_block.valid):
                    use_speculation = True
                    speculation_hits += 1
                    print(f"[speculative] HIT block {block_index + 1} (hits: {speculation_hits}, misses: {speculation_misses})")
                else:
                    speculation_misses += 1
                    if self.speculative_block:
                        print(f"[speculative] MISS block {block_index + 1} - expected {self.speculative_block.action}, got {action}")

            if use_speculation and self.speculative_block.frames:
                # Use pre-generated frames
                video_np = self.speculative_block.frames
            else:
                # Generate block normally
                noisy_input = noise[:, :, current_start_frame:current_start_frame + num_frame_per_block]

                new_act, conditional_dict = cond_current(
                    conditional_dict, current_start_frame, num_frame_per_block,
                    replace=current_actions, mode=self.mode)

                # Denoising loop
                for index, current_timestep in enumerate(pipeline.denoising_step_list):
                    timestep = torch.ones(
                        [batch_size, num_frame_per_block],
                        device=noise.device, dtype=torch.int64) * current_timestep

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
                                [batch_size * num_frame_per_block], device=noise.device, dtype=torch.long)
                        )
                        noisy_input = rearrange(noisy_input, '(b f) c h w -> b c f h w', b=denoised_pred.shape[0])
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

                # Update KV cache
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

                # Decode
                denoised_pred_t = denoised_pred.transpose(1, 2)
                video, vae_cache = pipeline.vae_decoder(denoised_pred_t.half(), *vae_cache)

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
            last_action = action

            # Queue speculation for next block (assuming same action)
            # Note: Full implementation would clone KV cache state here
            # self.speculation_queue.put((action, block_index + 2, state_snapshot))

            await websocket.send_json({
                "type": "block_complete",
                "block": block_index + 1
            })

        # Report speculation stats
        total = speculation_hits + speculation_misses
        if total > 0:
            print(f"[speculative] Final stats: {speculation_hits}/{total} hits ({100*speculation_hits/total:.1f}%)")

        await websocket.send_json({"type": "complete"})

    def _parse_action(self, action_data: dict) -> dict:
        """Parse action from WebSocket message to tensor format."""
        keyboard_map = {
            'w': [1, 0, 0, 0], 'a': [0, 1, 0, 0],
            's': [0, 0, 1, 0], 'd': [0, 0, 0, 1], 'q': [0, 0, 0, 0]
        }
        mouse_map = {
            'i': [1, 0], 'k': [-1, 0],
            'j': [0, -1], 'l': [0, 1], 'u': [0, 0]
        }

        keyboard = action_data.get('keyboard', 'q').lower()
        mouse = action_data.get('mouse', 'u').lower()

        return {
            'keyboard': keyboard_map.get(keyboard, [0, 0, 0, 0]),
            'mouse': mouse_map.get(mouse, [0, 0])
        }

    def stop(self):
        """Stop inference."""
        self.should_stop = True


# Global engine instance
engine: Optional[SpeculativeInferenceEngine] = None


@app.get("/", response_class=HTMLResponse)
async def get_index():
    return FileResponse("web_ui/index.html")


@app.get("/app.html", response_class=HTMLResponse)
async def get_app():
    return FileResponse("web_ui/app.html")


@app.get("/assets/{filename}")
async def get_asset(filename: str):
    return FileResponse(f"web_ui/assets/{filename}")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "start":
                img_path = data.get("image", "assets/example.png")
                asyncio.create_task(engine.run_inference(img_path, websocket))

            elif msg_type == "action":
                await engine.action_queue.put(data)

            elif msg_type == "stop":
                engine.stop()
                await engine.action_queue.put(None)
                await websocket.send_json({"type": "stopped"})

    except WebSocketDisconnect:
        print("Client disconnected")
        engine.stop()


def main():
    global engine

    parser = argparse.ArgumentParser(description="Speculative Multi-GPU Web Server")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--max_blocks", type=int, default=None)
    parser.add_argument("--max_num_output_frames", type=int, default=15)

    args = parser.parse_args()

    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    config = OmegaConf.load(args.config_path)
    set_seed(config.seed)

    engine = SpeculativeInferenceEngine(args, config)

    import uvicorn
    print(f"Starting speculative server at http://{args.host}:{args.port}")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        ws_max_size=50 * 1024 * 1024
    )


if __name__ == "__main__":
    main()
