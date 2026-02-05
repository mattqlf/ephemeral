"""
Web UI server for Matrix-Game 2.0 with WebSocket streaming.

Run with: uv run python -m apps.servers.web_server --checkpoint_path Matrix-Game-2.0/base_distilled_model/base_distill.safetensors
"""

import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyBsBvyfplPcR3g503sehLCbkU5nLkSY05A"

import io
import asyncio
import argparse
import base64
import copy
import logging
import time
from queue import Queue, Empty
from typing import Optional

import torch
import torch._dynamo

# Increase dynamo cache size to prevent VAE cache thrashing
torch._dynamo.config.cache_size_limit = 64
import numpy as np
from demo_utils.constant import ZERO_VAE_CACHE
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
from utils.misc import set_seed
from utils.conditions import Bench_actions_universal, Bench_actions_gta_drive, Bench_actions_templerun
from utils.wan_wrapper import WanDiffusionWrapper

from google import genai
from google.genai import types as genai_types

import anthropic
from suno_api.client import SunoClient

app = FastAPI(title="Matrix-Game 2.0 Web UI")
logger = logging.getLogger("ephemeral.web")

# Global state
inference_engine: Optional["InferenceEngine"] = None
suno_client: Optional[SunoClient] = None


def _websocket_client_label(websocket: WebSocket) -> str:
    client = getattr(websocket, "client", None)
    if client is None:
        return "unknown"
    return f"{client.host}:{client.port}"


def _summarize_action_payload(action_data: dict) -> str:
    keyboard = action_data.get("keyboard", "q")
    mouse = action_data.get("mouse", "u")
    return f"keyboard={keyboard} mouse={mouse}"


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

        # --fast flag: use 2 denoising steps instead of 3
        if getattr(self.args, 'fast', False):
            print("[fast mode] Using 2 denoising steps (was 3)")
            self.config.denoising_step_list = [1000, 500]

    def _init_models(self):
        import concurrent.futures
        import warnings

        print("Loading models (parallel)...")

        # Suppress FutureWarnings about weights_only
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")

        # Initialize generator structure (no weights yet)
        generator = WanDiffusionWrapper(
            **getattr(self.config, "model_kwargs", {}), is_causal=True)

        # Define loading functions for parallel execution
        def load_vae_decoder_weights():
            print("  Loading VAE decoder weights...")
            vae_path = os.path.join(self.args.pretrained_model_path, "Wan2.1_VAE.pth")
            vae_state_dict = torch.load(vae_path, map_location="cpu", mmap=True)
            decoder_state_dict = {k: v for k, v in vae_state_dict.items()
                                  if 'decoder.' in k or 'conv2' in k}
            print("  VAE decoder weights loaded")
            return decoder_state_dict

        def load_main_checkpoint():
            if self.args.checkpoint_path:
                print(f"  Loading main checkpoint...")
                state_dict = load_file(self.args.checkpoint_path)
                print("  Main checkpoint loaded")
                return state_dict
            return None

        def load_vae_encoder():
            print("  Loading VAE encoder...")
            vae = get_wanx_vae_wrapper(self.args.pretrained_model_path, torch.float16)
            vae.requires_grad_(False)
            vae.eval()
            print("  VAE encoder loaded")
            return vae

        # Load all weights in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_decoder = executor.submit(load_vae_decoder_weights)
            future_checkpoint = executor.submit(load_main_checkpoint)
            future_encoder = executor.submit(load_vae_encoder)

            decoder_state_dict = future_decoder.result()
            main_checkpoint = future_checkpoint.result()
            vae_encoder = future_encoder.result()

        print("Moving models to GPU...")

        # Initialize VAE decoder with loaded weights
        current_vae_decoder = VAEDecoderWrapper()
        current_vae_decoder.load_state_dict(decoder_state_dict)
        current_vae_decoder.to(self.device, torch.float16)
        current_vae_decoder.requires_grad_(False)
        current_vae_decoder.eval()
        if getattr(self.args, "no_compile", False):
            print("  Skipping VAE decoder compile (--no-compile). Using eager mode.")
        else:
            print("  Compiling VAE decoder (this takes ~30s on first run)...")
            current_vae_decoder.compile(mode="max-autotune-no-cudagraphs")

        # Initialize pipeline
        pipeline = CausalInferenceStreamingPipeline(
            self.config, generator=generator, vae_decoder=current_vae_decoder)

        if main_checkpoint:
            pipeline.generator.load_state_dict(main_checkpoint)

        self.pipeline = pipeline.to(device=self.device, dtype=self.weight_dtype)
        self.pipeline.vae_decoder.to(torch.float16)

        # Move VAE encoder to GPU
        self.vae = vae_encoder.to(self.device, self.weight_dtype)

        print("Models loaded successfully!")

        # Warmup primarily exists to trigger compile paths ahead of first inference.
        if getattr(self.args, "no_compile", False):
            print("Skipping warmup (--no-compile).")
        elif not getattr(self.args, 'no_warmup', False):
            self._warmup()
        else:
            print("Skipping warmup (--no-warmup). Kernels will compile on first inference.")

        print("\n  TIP: Keep this server running to avoid reloading models.\n")

    def _warmup(self):
        """Run complete warmup to trigger all torch.compile paths."""
        print("Warming up (compiling all kernels)...")

        num_frame_per_block = self.pipeline.num_frame_per_block
        denoising_steps = self.pipeline.denoising_step_list

        # Get actual visual_context shape from model
        print("  Detecting tensor shapes...")
        test_image = torch.randn([1, 3, 1, 352, 640], device=self.device, dtype=self.weight_dtype)
        with torch.no_grad():
            visual_context_shape = self.vae.clip.encode_video(test_image).shape
        print(f"    visual_context: {visual_context_shape}")

        # Create conditions for block 0 (simplest case - no KV cache complications)
        max_output_frames = self.args.max_num_output_frames
        num_frames = (max_output_frames - 1) * 4 + 1

        full_cond_concat = torch.randn(
            [1, 20, max_output_frames, 44, 80],
            device=self.device, dtype=self.weight_dtype
        )
        full_visual_context = torch.randn(
            visual_context_shape,
            device=self.device, dtype=self.weight_dtype
        )
        full_keyboard = torch.zeros(
            [1, num_frames, 4],
            device=self.device, dtype=self.weight_dtype
        )
        full_mouse = torch.zeros(
            [1, num_frames, 2],
            device=self.device, dtype=self.weight_dtype
        )
        sampled_noise = torch.randn(
            [1, 16, max_output_frames, 44, 80],
            device=self.device, dtype=self.weight_dtype
        )

        full_conditional_dict = {
            "cond_concat": full_cond_concat,
            "visual_context": full_visual_context,
            "keyboard_cond": full_keyboard,
            "mouse_cond": full_mouse,
        }

        # Slice for block 0
        cond = cond_current(full_conditional_dict, 0, num_frame_per_block)
        noise_slice = sampled_noise[:, :, :num_frame_per_block]

        # Initialize caches fresh for block 0
        self.pipeline._initialize_kv_cache(1, self.weight_dtype, self.device)
        self.pipeline._initialize_kv_cache_mouse_and_keyboard(1, self.weight_dtype, self.device)
        self.pipeline._initialize_crossattn_cache(1, self.weight_dtype, self.device)

        print(f"  Running {len(denoising_steps)} denoising steps + context pass...")

        with torch.no_grad():
            output = noise_slice

            # Run through all denoising timesteps to compile each one
            for step_idx, current_timestep in enumerate(denoising_steps):
                print(f"    Step {step_idx + 1}/{len(denoising_steps)} (timestep={current_timestep})...")
                timestep = torch.ones([1, num_frame_per_block], device=self.device, dtype=torch.int64) * current_timestep

                denoised_pred, _ = self.pipeline.generator(
                    noisy_image_or_video=output,
                    conditional_dict=cond,
                    timestep=timestep,
                    kv_cache=self.pipeline.kv_cache1,
                    kv_cache_mouse=self.pipeline.kv_cache_mouse,
                    kv_cache_keyboard=self.pipeline.kv_cache_keyboard,
                    crossattn_cache=self.pipeline.crossattn_cache,
                    current_start=0
                )
                output = denoised_pred

            # Context pass (timestep=0) to compile that path too
            print(f"    Context pass (timestep={self.pipeline.args.context_noise})...")
            context_timestep = torch.ones([1, num_frame_per_block], device=self.device, dtype=torch.int64) * self.pipeline.args.context_noise
            self.pipeline.generator(
                noisy_image_or_video=output,
                conditional_dict=cond,
                timestep=context_timestep,
                kv_cache=self.pipeline.kv_cache1,
                kv_cache_mouse=self.pipeline.kv_cache_mouse,
                kv_cache_keyboard=self.pipeline.kv_cache_keyboard,
                crossattn_cache=self.pipeline.crossattn_cache,
                current_start=0
            )

            # VAE decoder - first with None cache
            print("  VAE decoder (empty cache)...")
            vae_cache = [None] * len(ZERO_VAE_CACHE)
            dummy_latent = output.transpose(1, 2)
            _, vae_cache = self.pipeline.vae_decoder(dummy_latent.half(), *vae_cache)

            # VAE decoder - second with populated cache (different code path)
            print("  VAE decoder (populated cache)...")
            _, vae_cache = self.pipeline.vae_decoder(dummy_latent.half(), *vae_cache)

        # Clear CUDA cache
        torch.cuda.empty_cache()
        print("  Warmup complete!")

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

    async def _send_ws(self, websocket: WebSocket, payload: dict) -> bool:
        """Best-effort websocket send that treats disconnects as normal stop events."""
        msg_type = payload.get("type", "unknown")
        client_label = _websocket_client_label(websocket)
        try:
            await websocket.send_json(payload)
            return True
        except Exception as exc:
            self.should_stop = True
            logger.info(
                "[UI %s] WebSocket closed during send type=%s; stopping inference (%s)",
                client_label,
                msg_type,
                str(exc),
            )
            return False

    async def _generate_image_with_gemini(self, prompt: str, ref_image: Image.Image, websocket) -> Image.Image:
        """Generate a new image using Gemini with a text prompt and reference image."""
        client_label = _websocket_client_label(websocket)
        if not await self._send_ws(websocket, {
            "type": "generating_image",
            "message": f"Generating: {prompt[:60]}..."
        }):
            raise RuntimeError("Client disconnected during Gemini generation")

        logger.info("[UI %s] Gemini generation started (prompt_chars=%d)", client_label, len(prompt))

        def _call_gemini():
            client = genai.Client()
            contents = [prompt, ref_image]
            response = client.models.generate_content(
                model="gemini-3-pro-image-preview",
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE'],
                    image_config=genai_types.ImageConfig(
                        aspect_ratio="16:9",
                    ),
                ),
            )
            for part in response.parts:
                if part.text is not None:
                    logger.info("[UI %s] Gemini text response: %s", client_label, part.text[:200])
                elif part.inline_data is not None:
                    img_bytes = part.inline_data.data
                    return Image.open(io.BytesIO(img_bytes)).convert('RGB')
            return None

        loop = asyncio.get_event_loop()
        generated = await loop.run_in_executor(None, _call_gemini)

        if generated is None:
            raise RuntimeError("Gemini did not return an image")

        logger.info("[UI %s] Gemini generated image size=%s", client_label, generated.size)
        return generated

    async def _generate_music(self, prompt: str, image: Image.Image, websocket, captcha_token: str = ""):
        """Generate background music using Anthropic + Suno, running in the background."""
        client_label = _websocket_client_label(websocket)
        try:
            await websocket.send_json({
                "type": "music_status",
                "message": "Generating music prompt..."
            })

            # Step 1: Call Anthropic to create a Suno music prompt
            img_buffer = io.BytesIO()
            image.save(img_buffer, format="JPEG", quality=80)
            img_b64 = base64.b64encode(img_buffer.getvalue()).decode()

            def _call_anthropic():
                from suno_api.client import DEFAULT_ANTHROPIC_KEY
                client = anthropic.Anthropic(
                    api_key=os.environ.get("ANTHROPIC_API_KEY", DEFAULT_ANTHROPIC_KEY)
                )
                msg = client.messages.create(
                    model="claude-sonnet-4-5-20250929",
                    max_tokens=300,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": img_b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    "You are creating a prompt for Suno AI to generate background music for a video game world. "
                                    "Look at this game scene image"
                                    + (f" (the player described it as: \"{prompt}\")" if prompt else "")
                                    + ". Write a short Suno AI music generation prompt (2-3 sentences max) describing "
                                    "the perfect looping background music for this game world. Focus on mood, genre, "
                                    "instruments, tempo, and atmosphere. Make it instrumental. "
                                    "Output ONLY the Suno prompt, nothing else."
                                ),
                            },
                        ],
                    }],
                )
                return msg.content[0].text.strip()

            loop = asyncio.get_event_loop()
            music_prompt = await loop.run_in_executor(None, _call_anthropic)
            logger.info("[UI %s] Anthropic music prompt: %s", client_label, music_prompt)

            await websocket.send_json({
                "type": "music_status",
                "message": f"Composing: {music_prompt[:60]}..."
            })

            # Step 2: Generate song with Suno
            global suno_client
            if not suno_client:
                logger.warning("[UI %s] No Suno client configured, skipping music", client_label)
                return

            data = await suno_client.generate(
                description=music_prompt + " Instrumental, loopable, video game background music.",
                instrumental=True,
                captcha_token=captcha_token,
            )
            clips = data.get("clips", [])
            clip_ids = [c["id"] for c in clips]
            logger.info("[UI %s] Suno generation started, clip_ids=%s", client_label, clip_ids)

            await websocket.send_json({
                "type": "music_status",
                "message": "Waiting for music generation..."
            })

            # Step 3: Poll until songs are complete
            songs = await suno_client.wait_for_songs(clip_ids, timeout=300)
            audio_urls = [s.get("audio_url") for s in songs if s.get("audio_url")]

            if audio_urls:
                logger.info("[UI %s] Music ready: %s", client_label, audio_urls)
                await websocket.send_json({
                    "type": "music_ready",
                    "audio_urls": audio_urls,
                    "title": songs[0].get("title", ""),
                    "prompt": music_prompt,
                })
            else:
                logger.warning("[UI %s] Suno returned no audio URLs", client_label)

        except Exception as e:
            logger.exception("[UI %s] Music generation failed", client_label)
            try:
                await websocket.send_json({
                    "type": "music_status",
                    "message": f"Music generation failed: {str(e)[:100]}"
                })
            except Exception:
                pass

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

    async def run_inference(self, img_input: str, websocket: WebSocket, prompt: str = None, captcha_token: str = ""):
        """Run inference loop with WebSocket streaming."""
        client_label = _websocket_client_label(websocket)
        run_started_at = time.perf_counter()
        self.is_running = True
        self.should_stop = False

        # Clear any pending actions
        dropped_actions = 0
        while not self.action_queue.empty():
            try:
                self.action_queue.get_nowait()
                dropped_actions += 1
            except Exception:
                break
        if dropped_actions:
            logger.info("[UI %s] Cleared %d stale queued actions before start", client_label, dropped_actions)

        try:
            image_source = "upload" if img_input.startswith("data:image") else "path"
            logger.info(
                "[UI %s] Starting inference mode=%s image_source=%s prompt=%s",
                client_label,
                self.mode,
                image_source,
                "yes" if prompt else "no",
            )

            # Load and process image (supports file path or base64 data)
            if img_input.startswith('data:image'):
                # Handle base64 uploaded image
                logger.info("[UI %s] Processing uploaded image (~%d KB base64)", client_label, len(img_input) // 1024)
                header, b64_data = img_input.split(',', 1)
                img_bytes = base64.b64decode(b64_data)
                image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                logger.info("[UI %s] Uploaded image size=%s", client_label, image.size)
            else:
                # Handle file path
                logger.info("[UI %s] Loading image from path=%s", client_label, img_input)
                image = load_image(img_input)

            # If prompt provided, generate new image with Gemini
            if prompt:
                image = await self._generate_image_with_gemini(prompt, image, websocket)

            # Kick off background music generation (non-blocking)
            if captcha_token:
                asyncio.create_task(self._generate_music(prompt or "", image, websocket, captcha_token))

            image = self._resizecrop(image, 352, 640)
            logger.info("[UI %s] Input image normalized to %s", client_label, image.size)

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
            if not await self._send_ws(websocket, {
                "type": "frame",
                "data": self._frame_to_base64(init_frame),
                "block": 0
            }):
                return

            # Send ready signal
            if not await self._send_ws(websocket, {
                "type": "ready",
                "mode": self.mode,
                "message": "Ready for input. Press keys to control."
            }):
                return
            logger.info("[UI %s] Ready; waiting for interactive actions", client_label)

            # Run inference loop
            with torch.no_grad():
                await self._inference_loop(
                    noise=sampled_noise,
                    conditional_dict=conditional_dict,
                    websocket=websocket
                )

        except Exception as e:
            if self.should_stop:
                logger.info("[UI %s] Inference stopped due to disconnect/stop request", client_label)
            else:
                logger.exception("[UI %s] Inference failed", client_label)
                if not await self._send_ws(websocket, {"type": "error", "message": str(e)}):
                    logger.warning("[UI %s] Could not send error over websocket", client_label)
        finally:
            self.is_running = False
            elapsed = time.perf_counter() - run_started_at
            logger.info("[UI %s] Inference session ended after %.2fs", client_label, elapsed)

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
        total_blocks = min(max_blocks, num_blocks)
        client_label = _websocket_client_label(websocket)
        loop_started_at = time.perf_counter()
        logger.info(
            "[UI %s] Entering generation loop blocks=%d frames_per_block=%d denoise_steps=%s",
            client_label,
            total_blocks,
            num_frame_per_block,
            self.pipeline.denoising_step_list,
        )

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
        generated_blocks = 0

        # Main generation loop
        for block_index in range(total_blocks):
            if self.should_stop:
                logger.info("[UI %s] Stop flag set before block %d", client_label, block_index + 1)
                break
            block_started_at = time.perf_counter()
            block_num = block_index + 1

            # Request action from client
            if not await self._send_ws(websocket, {
                "type": "request_action",
                "block": block_num,
                "mode": self.mode
            }):
                return
            logger.info(
                "[UI %s] Block %d/%d requested action (queued_actions=%d)",
                client_label,
                block_num,
                total_blocks,
                self.action_queue.qsize(),
            )

            # Wait for action with timeout
            try:
                action_wait_started_at = time.perf_counter()
                action = await asyncio.wait_for(
                    self.action_queue.get(),
                    timeout=120.0
                )
                if action is None:
                    logger.info("[UI %s] Block %d received stop sentinel", client_label, block_num)
                    break
                current_actions = self._parse_action(action)
                action_wait_ms = (time.perf_counter() - action_wait_started_at) * 1000.0
                logger.info(
                    "[UI %s] Block %d action received in %.1fms (%s)",
                    client_label,
                    block_num,
                    action_wait_ms,
                    _summarize_action_payload(action),
                )
            except asyncio.TimeoutError:
                # Use neutral action on timeout
                current_actions = self._parse_action({"keyboard": "q", "mouse": "u"})
                logger.warning("[UI %s] Block %d action timeout; using neutral action", client_label, block_num)

            # Prepare input
            noisy_input = noise[:, :, current_start_frame:current_start_frame + num_frame_per_block]

            new_act, conditional_dict = cond_current(
                conditional_dict, current_start_frame, num_frame_per_block,
                replace=current_actions, mode=self.mode)

            # Denoising loop
            for index, current_timestep in enumerate(self.pipeline.denoising_step_list):
                logger.info(
                    "[UI %s] Block %d denoise step %d/%d (timestep=%d)",
                    client_label,
                    block_num,
                    index + 1,
                    len(self.pipeline.denoising_step_list),
                    current_timestep,
                )
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
                if not await self._send_ws(websocket, {
                    "type": "frame",
                    "data": self._frame_to_base64(frame),
                    "block": block_num,
                    "frame": frame_idx
                }):
                    return
                await asyncio.sleep(0.04)  # ~25fps pacing

            current_start_frame += num_frame_per_block
            generated_blocks += 1
            block_elapsed = time.perf_counter() - block_started_at
            block_fps = len(video_np) / block_elapsed if block_elapsed > 0 else 0.0
            logger.info(
                "[UI %s] Block %d complete frames=%d elapsed=%.2fs throughput=%.2f fps",
                client_label,
                block_num,
                len(video_np),
                block_elapsed,
                block_fps,
            )

            # Send block complete
            if not await self._send_ws(websocket, {
                "type": "block_complete",
                "block": block_num
            }):
                return

        # Send completion
        if not await self._send_ws(websocket, {"type": "complete"}):
            return
        total_elapsed = time.perf_counter() - loop_started_at
        logger.info(
            "[UI %s] Generation loop finished blocks=%d/%d elapsed=%.2fs",
            client_label,
            generated_blocks,
            total_blocks,
            total_elapsed,
        )

    def stop(self):
        """Stop the inference loop."""
        logger.info("Stop requested for active inference loop")
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
    client_label = _websocket_client_label(websocket)
    logger.info("[WS %s] Connected", client_label)

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            logger.info("[WS %s] Received message type=%s", client_label, msg_type)

            if msg_type == "start":
                img_input = data.get("image", "demo_images/universal/0000.png")
                prompt = data.get("prompt", None)
                captcha_token = data.get("captcha_token", "")
                logger.info(
                    "[WS %s] Start requested image_source=%s prompt=%s captcha=%s",
                    client_label,
                    "upload" if str(img_input).startswith("data:image") else "path",
                    "yes" if prompt else "no",
                    "yes" if captcha_token else "no",
                )
                if inference_engine and not inference_engine.is_running:
                    try:
                        asyncio.create_task(
                            inference_engine.run_inference(img_input, websocket, prompt=prompt, captcha_token=captcha_token))
                        logger.info("[WS %s] Inference task created", client_label)
                    except Exception as e:
                        logger.exception("[WS %s] Failed to start inference task", client_label)
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Failed to start: {str(e)}"
                        })
                else:
                    logger.warning("[WS %s] Start rejected: engine not ready or already running", client_label)
                    await websocket.send_json({
                        "type": "error",
                        "message": "Engine not ready or already running"
                    })

            elif msg_type == "action":
                if inference_engine and inference_engine.is_running:
                    await inference_engine.action_queue.put(data)
                    logger.info(
                        "[WS %s] Queued action (%s) queue_size=%d",
                        client_label,
                        _summarize_action_payload(data),
                        inference_engine.action_queue.qsize(),
                    )
                else:
                    logger.warning("[WS %s] Dropped action: inference is not running", client_label)

            elif msg_type == "stop":
                if inference_engine:
                    logger.info("[WS %s] Stop requested by client", client_label)
                    inference_engine.stop()
                    await inference_engine.action_queue.put(None)
                    await websocket.send_json({"type": "stopped"})
            else:
                logger.warning("[WS %s] Unknown message type=%s", client_label, msg_type)

    except WebSocketDisconnect:
        logger.info("[WS %s] Disconnected", client_label)
        if inference_engine:
            inference_engine.stop()
            await inference_engine.action_queue.put(None)
    except Exception as e:
        logger.exception("[WS %s] WebSocket error", client_label)
        if inference_engine:
            inference_engine.stop()
            await inference_engine.action_queue.put(None)


def parse_args():
    parser = argparse.ArgumentParser()
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
                        help="Use 2 denoising steps instead of 3 for faster inference (~33%% speedup, slight quality loss)")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile for the VAE decoder (lower startup overhead, potentially lower throughput)")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Skip kernel warmup (kernels will compile lazily on first inference)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    set_seed(args.seed)

    # Initialize inference engine
    logger.info("Initializing inference engine...")
    inference_engine = InferenceEngine(args)

    # Initialize Suno client (uses hardcoded default cookie if env var not set)
    suno_client = SunoClient()
    logger.info("Suno client initialized (music generation enabled)")

    # Create web_ui directory if needed
    os.makedirs("web_ui", exist_ok=True)

    # Start server
    logger.info("Starting server at http://%s:%d", args.host, args.port)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        ws_max_size=50 * 1024 * 1024  # 50MB max WebSocket message (for large image uploads)
    )
