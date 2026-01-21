"""
Web UI server for Matrix-Game 2.0 with WebSocket streaming.

Run with: uv run python web_server.py --checkpoint_path Matrix-Game-2.0/base_distilled_model/base_distill.safetensors
"""

import os
import io
import asyncio
import argparse
import base64
import copy
from queue import Queue, Empty
from typing import Optional

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn

from omegaconf import OmegaConf
from torchvision.transforms import v2
from diffusers.utils import load_image
from safetensors.torch import load_file
from einops import rearrange

from pipeline import CausalInferenceStreamingPipeline
from pipeline.causal_inference import cond_current
from wan.vae.wanx_vae import get_wanx_vae_wrapper
from demo_utils.vae_block3 import VAEDecoderWrapper
from demo_utils.constant import ZERO_VAE_CACHE
from utils.misc import set_seed
from utils.conditions import Bench_actions_universal, Bench_actions_gta_drive, Bench_actions_templerun
from utils.wan_wrapper import WanDiffusionWrapper


app = FastAPI(title="Matrix-Game 2.0 Web UI")

# Global state
inference_engine: Optional["InferenceEngine"] = None


class InferenceEngine:
    """Handles model loading and inference with WebSocket communication."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda")
        self.weight_dtype = torch.bfloat16

        self.action_queue = asyncio.Queue()
        self.is_running = False
        self.should_stop = False

        self._init_config()
        self._init_models()

        self.frame_process = v2.Compose([
            v2.Resize(size=(352, 640), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _init_config(self):
        self.config = OmegaConf.load(self.args.config_path)
        self.mode = self.config.get('mode', 'universal')

    def _init_models(self):
        print("Loading models...")

        # Initialize generator
        generator = WanDiffusionWrapper(
            **getattr(self.config, "model_kwargs", {}), is_causal=True)

        # Initialize VAE decoder
        current_vae_decoder = VAEDecoderWrapper()
        vae_state_dict = torch.load(
            os.path.join(self.args.pretrained_model_path, "Wan2.1_VAE.pth"),
            map_location="cpu"
        )
        decoder_state_dict = {}
        for key, value in vae_state_dict.items():
            if 'decoder.' in key or 'conv2' in key:
                decoder_state_dict[key] = value
        current_vae_decoder.load_state_dict(decoder_state_dict)
        current_vae_decoder.to(self.device, torch.float16)
        current_vae_decoder.requires_grad_(False)
        current_vae_decoder.eval()
        # Skip torch.compile for web server to avoid startup delay
        # current_vae_decoder.compile(mode="max-autotune-no-cudagraphs")

        # Initialize pipeline
        pipeline = CausalInferenceStreamingPipeline(
            self.config, generator=generator, vae_decoder=current_vae_decoder)

        if self.args.checkpoint_path:
            print(f"Loading checkpoint: {self.args.checkpoint_path}")
            state_dict = load_file(self.args.checkpoint_path)
            pipeline.generator.load_state_dict(state_dict)

        self.pipeline = pipeline.to(device=self.device, dtype=self.weight_dtype)
        self.pipeline.vae_decoder.to(torch.float16)

        # Initialize VAE encoder
        vae = get_wanx_vae_wrapper(self.args.pretrained_model_path, torch.float16)
        vae.requires_grad_(False)
        vae.eval()
        self.vae = vae.to(self.device, self.weight_dtype)

        print("Models loaded successfully!")

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
        """Convert a numpy frame to base64 JPEG."""
        img = Image.fromarray(frame.astype(np.uint8))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _parse_action(self, action_data: dict) -> dict:
        """Parse action from WebSocket message to tensor format."""
        mode = self.mode
        CAM_VALUE = 0.1

        keyboard_key = action_data.get("keyboard", "q").lower()
        mouse_key = action_data.get("mouse", "u").lower()

        if mode == "universal":
            CAMERA_VALUE_MAP = {
                "i": [CAM_VALUE, 0], "k": [-CAM_VALUE, 0],
                "j": [0, -CAM_VALUE], "l": [0, CAM_VALUE], "u": [0, 0]
            }
            KEYBOARD_IDX = {
                "w": [1, 0, 0, 0], "s": [0, 1, 0, 0],
                "a": [0, 0, 1, 0], "d": [0, 0, 0, 1], "q": [0, 0, 0, 0]
            }
            mouse_cond = torch.tensor(CAMERA_VALUE_MAP.get(mouse_key, [0, 0])).cuda()
            keyboard_cond = torch.tensor(KEYBOARD_IDX.get(keyboard_key, [0, 0, 0, 0])).cuda()

        elif mode == "gta_drive":
            CAMERA_VALUE_MAP = {
                "a": [0, -CAM_VALUE], "d": [0, CAM_VALUE], "q": [0, 0]
            }
            KEYBOARD_IDX = {"w": [1, 0], "s": [0, 1], "q": [0, 0]}
            # For GTA, steering (a/d) affects both mouse and movement
            mouse_cond = torch.tensor(CAMERA_VALUE_MAP.get(keyboard_key, [0, 0])).cuda()
            keyboard_cond = torch.tensor(KEYBOARD_IDX.get(keyboard_key, [0, 0])).cuda()

        else:  # templerun
            KEYBOARD_IDX = {
                "w": [0, 1, 0, 0, 0, 0, 0], "s": [0, 0, 1, 0, 0, 0, 0],
                "a": [0, 0, 0, 0, 0, 1, 0], "d": [0, 0, 0, 0, 0, 0, 1],
                "z": [0, 0, 0, 1, 0, 0, 0], "c": [0, 0, 0, 0, 1, 0, 0],
                "q": [1, 0, 0, 0, 0, 0, 0]
            }
            keyboard_cond = torch.tensor(KEYBOARD_IDX.get(keyboard_key, [1, 0, 0, 0, 0, 0, 0])).cuda()
            mouse_cond = None

        result = {"keyboard": keyboard_cond}
        if mouse_cond is not None:
            result["mouse"] = mouse_cond
        return result

    async def run_inference(self, img_path: str, websocket: WebSocket):
        """Run inference loop with WebSocket streaming."""
        self.is_running = True
        self.should_stop = False

        # Clear any pending actions
        while not self.action_queue.empty():
            try:
                self.action_queue.get_nowait()
            except:
                break

        try:
            # Load and process image
            image = load_image(img_path)
            image = self._resizecrop(image, 352, 640)

            image_tensor = self.frame_process(image)[None, :, None, :, :].to(
                dtype=self.weight_dtype, device=self.device)

            # Encode input image
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

            # Setup conditions
            conditional_dict = {
                "cond_concat": cond_concat.to(device=self.device, dtype=self.weight_dtype),
                "visual_context": visual_context.to(device=self.device, dtype=self.weight_dtype)
            }

            if self.mode == "universal":
                cond_data = Bench_actions_universal(num_frames)
                mouse_condition = cond_data['mouse_condition'].unsqueeze(0).to(
                    device=self.device, dtype=self.weight_dtype)
                conditional_dict['mouse_cond'] = mouse_condition
            elif self.mode == "gta_drive":
                cond_data = Bench_actions_gta_drive(num_frames)
                mouse_condition = cond_data['mouse_condition'].unsqueeze(0).to(
                    device=self.device, dtype=self.weight_dtype)
                conditional_dict['mouse_cond'] = mouse_condition
            else:
                cond_data = Bench_actions_templerun(num_frames)

            keyboard_condition = cond_data['keyboard_condition'].unsqueeze(0).to(
                device=self.device, dtype=self.weight_dtype)
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

            # Run inference loop
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
        finally:
            self.is_running = False

    async def _inference_loop(
        self,
        noise: torch.Tensor,
        conditional_dict: dict,
        websocket: WebSocket
    ):
        """Custom inference loop that streams frames via WebSocket."""
        batch_size, num_channels, num_frames, height, width = noise.shape
        num_frame_per_block = self.pipeline.num_frame_per_block

        num_blocks = num_frames // num_frame_per_block
        max_blocks = self.args.max_blocks or num_blocks

        output = torch.zeros_like(noise)
        videos = []
        vae_cache = copy.deepcopy(ZERO_VAE_CACHE)
        for j in range(len(vae_cache)):
            vae_cache[j] = None

        # Initialize KV caches
        self.pipeline.kv_cache1 = None
        self.pipeline.kv_cache_keyboard = None
        self.pipeline.kv_cache_mouse = None
        self.pipeline.crossattn_cache = None

        self.pipeline._initialize_kv_cache(batch_size, noise.dtype, noise.device)
        self.pipeline._initialize_kv_cache_mouse_and_keyboard(batch_size, noise.dtype, noise.device)
        self.pipeline._initialize_crossattn_cache(batch_size, noise.dtype, noise.device)

        current_start_frame = 0

        # Main generation loop
        for block_index in range(min(max_blocks, num_blocks)):
            if self.should_stop:
                break

            # Request action from client
            await websocket.send_json({
                "type": "request_action",
                "block": block_index + 1,
                "mode": self.mode
            })

            # Wait for action with timeout
            try:
                action = await asyncio.wait_for(
                    self.action_queue.get(),
                    timeout=120.0
                )
                if action is None:
                    break
                current_actions = self._parse_action(action)
            except asyncio.TimeoutError:
                # Use neutral action on timeout
                current_actions = self._parse_action({"keyboard": "q", "mouse": "u"})

            # Prepare input
            noisy_input = noise[:, :, current_start_frame:current_start_frame + num_frame_per_block]

            new_act, conditional_dict = cond_current(
                conditional_dict, current_start_frame, num_frame_per_block,
                replace=current_actions, mode=self.mode)

            # Denoising loop
            for index, current_timestep in enumerate(self.pipeline.denoising_step_list):
                timestep = torch.ones(
                    [batch_size, num_frame_per_block],
                    device=noise.device, dtype=torch.int64) * current_timestep

                if index < len(self.pipeline.denoising_step_list) - 1:
                    _, denoised_pred = self.pipeline.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=new_act,
                        timestep=timestep,
                        kv_cache=self.pipeline.kv_cache1,
                        kv_cache_mouse=self.pipeline.kv_cache_mouse,
                        kv_cache_keyboard=self.pipeline.kv_cache_keyboard,
                        crossattn_cache=self.pipeline.crossattn_cache,
                        current_start=current_start_frame * self.pipeline.frame_seq_length
                    )
                    next_timestep = self.pipeline.denoising_step_list[index + 1]
                    noisy_input = self.pipeline.scheduler.add_noise(
                        rearrange(denoised_pred, 'b c f h w -> (b f) c h w'),
                        torch.randn_like(rearrange(denoised_pred, 'b c f h w -> (b f) c h w')),
                        next_timestep * torch.ones(
                            [batch_size * num_frame_per_block], device=noise.device, dtype=torch.long)
                    )
                    noisy_input = rearrange(noisy_input, '(b f) c h w -> b c f h w', b=denoised_pred.shape[0])
                else:
                    _, denoised_pred = self.pipeline.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=new_act,
                        timestep=timestep,
                        kv_cache=self.pipeline.kv_cache1,
                        kv_cache_mouse=self.pipeline.kv_cache_mouse,
                        kv_cache_keyboard=self.pipeline.kv_cache_keyboard,
                        crossattn_cache=self.pipeline.crossattn_cache,
                        current_start=current_start_frame * self.pipeline.frame_seq_length
                    )

            # Record output
            output[:, :, current_start_frame:current_start_frame + num_frame_per_block] = denoised_pred

            # Update KV cache with clean context
            context_timestep = torch.ones_like(timestep) * self.pipeline.args.context_noise
            self.pipeline.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=new_act,
                timestep=context_timestep,
                kv_cache=self.pipeline.kv_cache1,
                kv_cache_mouse=self.pipeline.kv_cache_mouse,
                kv_cache_keyboard=self.pipeline.kv_cache_keyboard,
                crossattn_cache=self.pipeline.crossattn_cache,
                current_start=current_start_frame * self.pipeline.frame_seq_length,
            )

            # Decode and send frames
            denoised_pred_t = denoised_pred.transpose(1, 2)
            video, vae_cache = self.pipeline.vae_decoder(denoised_pred_t.half(), *vae_cache)
            videos.append(video)

            video_np = rearrange(video, "B T C H W -> B T H W C")
            video_np = ((video_np.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)[0]

            # Send each frame
            for frame_idx, frame in enumerate(video_np):
                await websocket.send_json({
                    "type": "frame",
                    "data": self._frame_to_base64(frame),
                    "block": block_index + 1,
                    "frame": frame_idx
                })
                await asyncio.sleep(0.04)  # ~25fps pacing

            current_start_frame += num_frame_per_block

            # Send block complete
            await websocket.send_json({
                "type": "block_complete",
                "block": block_index + 1
            })

        # Send completion
        await websocket.send_json({"type": "complete"})

    def stop(self):
        """Stop the inference loop."""
        self.should_stop = True


@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the main HTML page."""
    return FileResponse("web_ui/index.html")


@app.get("/app.html", response_class=HTMLResponse)
async def get_app():
    """Serve the app page."""
    return FileResponse("web_ui/app.html")


@app.get("/assets/{filename}")
async def get_asset(filename: str):
    """Serve static assets from web_ui/assets/."""
    return FileResponse(f"web_ui/assets/{filename}")


@app.get("/api/status")
async def get_status():
    """Get server status."""
    return {
        "ready": inference_engine is not None,
        "running": inference_engine.is_running if inference_engine else False,
        "mode": inference_engine.mode if inference_engine else None
    }


@app.get("/api/images")
async def list_images():
    """List available demo images."""
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
    """WebSocket endpoint for real-time communication."""
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "start":
                img_path = data.get("image", "demo_images/universal/0000.png")
                if inference_engine and not inference_engine.is_running:
                    asyncio.create_task(
                        inference_engine.run_inference(img_path, websocket))
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Engine not ready or already running"
                    })

            elif msg_type == "action":
                if inference_engine and inference_engine.is_running:
                    await inference_engine.action_queue.put(data)

            elif msg_type == "stop":
                if inference_engine:
                    inference_engine.stop()
                    await inference_engine.action_queue.put(None)
                    await websocket.send_json({"type": "stopped"})

    except WebSocketDisconnect:
        if inference_engine:
            inference_engine.stop()
    except Exception as e:
        print(f"WebSocket error: {e}")
        if inference_engine:
            inference_engine.stop()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str,
                        default="configs/inference_yaml/inference_universal.yaml")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--pretrained_model_path", type=str, default="Matrix-Game-2.0")
    parser.add_argument("--max_num_output_frames", type=int, default=360)
    parser.add_argument("--max_blocks", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    # Initialize inference engine
    print("Initializing inference engine...")
    inference_engine = InferenceEngine(args)

    # Create web_ui directory if needed
    os.makedirs("web_ui", exist_ok=True)

    # Start server
    print(f"Starting server at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
