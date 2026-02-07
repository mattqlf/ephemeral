"""
Web UI server for Matrix-Game 2.0 with WebSocket streaming.

Run with: uv run python -m apps.servers.web_server --checkpoint_path Matrix-Game-2.0/base_distilled_model/base_distill.safetensors
"""

import os
import sys
os.environ["GOOGLE_API_KEY"] = "AIzaSyBsBvyfplPcR3g503sehLCbkU5nLkSY05A"

# Must be set before model imports — flex_attention compile checks this at import time
if "--true-no-compile" in sys.argv:
    os.environ["MATRIX_NO_COMPILE"] = "1"

import io
import asyncio
import argparse
import base64
import copy
import logging
import random
import socket
import string
import time
from collections import Counter
from dataclasses import dataclass, field
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


@dataclass
class RoomState:
    room_id: str
    host_ws: WebSocket
    player_websockets: set = field(default_factory=set)
    player_names: dict = field(default_factory=dict)  # websocket -> name
    votes: list = field(default_factory=list)
    vote_event: asyncio.Event = field(default_factory=asyncio.Event)
    is_active: bool = True


room_state: Optional[RoomState] = None
server_port: int = 8080


def _get_lan_ip() -> str:
    """Get the LAN IP address of this machine."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _generate_room_id(length: int = 4) -> str:
    """Generate a short uppercase+digits room code."""
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


def _resolve_votes(votes: list) -> dict:
    """Pick the majority vote from a list of {keyboard, mouse} dicts. Random tiebreak."""
    if not votes:
        return {"keyboard": "q", "mouse": "u"}
    pairs = [(v.get("keyboard", "q"), v.get("mouse", "u")) for v in votes]
    counts = Counter(pairs)
    max_count = max(counts.values())
    winners = [pair for pair, count in counts.items() if count == max_count]
    winner = random.choice(winners)
    return {"keyboard": winner[0], "mouse": winner[1]}


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

        logger.info(
            "InferenceEngine init: checkpoint=%s pretrained=%s max_frames=%d fast=%s no_compile=%s no_warmup=%s true_no_compile=%s seed=%d",
            args.checkpoint_path,
            args.pretrained_model_path,
            args.max_num_output_frames,
            getattr(args, 'fast', False),
            getattr(args, 'no_compile', False),
            getattr(args, 'no_warmup', False),
            getattr(args, 'true_no_compile', False),
            args.seed,
        )

        self._init_config()
        self._init_models()

        self.frame_process = v2.Compose([
            v2.Resize(size=(352, 640), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _init_config(self):
        logger.info("Loading config from %s", self.args.config_path)
        self.config = OmegaConf.load(self.args.config_path)
        self.mode = self.config.get('mode', 'universal')
        logger.info("Config loaded: mode=%s denoising_steps=%s", self.mode, list(self.config.get('denoising_step_list', [])))

        # --fast flag: use 2 denoising steps instead of 3
        if getattr(self.args, 'fast', False):
            logger.info("[fast mode] Overriding to 2 denoising steps (was %s)", list(self.config.get('denoising_step_list', [])))
            self.config.denoising_step_list = [1000, 500]

    def _init_models(self):
        import concurrent.futures
        import warnings

        models_start = time.perf_counter()
        logger.info("Loading models (parallel)...")

        # Suppress FutureWarnings about weights_only
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")

        # Initialize generator structure (no weights yet)
        generator = WanDiffusionWrapper(
            **getattr(self.config, "model_kwargs", {}), is_causal=True)
        logger.info("Generator structure initialized")

        # Define loading functions for parallel execution
        def load_vae_decoder_weights():
            t0 = time.perf_counter()
            vae_path = os.path.join(self.args.pretrained_model_path, "Wan2.1_VAE.pth")
            logger.info("  Loading VAE decoder weights from %s", vae_path)
            vae_state_dict = torch.load(vae_path, map_location="cpu", mmap=True)
            decoder_state_dict = {k: v for k, v in vae_state_dict.items()
                                  if 'decoder.' in k or 'conv2' in k}
            logger.info("  VAE decoder weights loaded (%d keys, %.2fs)", len(decoder_state_dict), time.perf_counter() - t0)
            return decoder_state_dict

        def load_main_checkpoint():
            if self.args.checkpoint_path:
                t0 = time.perf_counter()
                logger.info("  Loading main checkpoint from %s", self.args.checkpoint_path)
                state_dict = load_file(self.args.checkpoint_path)
                logger.info("  Main checkpoint loaded (%d keys, %.2fs)", len(state_dict), time.perf_counter() - t0)
                return state_dict
            logger.info("  No checkpoint path specified, skipping")
            return None

        def load_vae_encoder():
            t0 = time.perf_counter()
            logger.info("  Loading VAE encoder from %s", self.args.pretrained_model_path)
            vae = get_wanx_vae_wrapper(self.args.pretrained_model_path, torch.float16)
            vae.requires_grad_(False)
            vae.eval()
            logger.info("  VAE encoder loaded (%.2fs)", time.perf_counter() - t0)
            return vae

        # Load all weights in parallel
        logger.info("Starting parallel weight loading (3 threads)...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_decoder = executor.submit(load_vae_decoder_weights)
            future_checkpoint = executor.submit(load_main_checkpoint)
            future_encoder = executor.submit(load_vae_encoder)

            decoder_state_dict = future_decoder.result()
            main_checkpoint = future_checkpoint.result()
            vae_encoder = future_encoder.result()

        parallel_elapsed = time.perf_counter() - models_start
        logger.info("Parallel weight loading complete (%.2fs)", parallel_elapsed)

        gpu_start = time.perf_counter()
        logger.info("Moving models to GPU (device=%s)...", self.device)

        # Initialize VAE decoder with loaded weights
        current_vae_decoder = VAEDecoderWrapper()
        current_vae_decoder.load_state_dict(decoder_state_dict)
        current_vae_decoder.to(self.device, torch.float16)
        current_vae_decoder.requires_grad_(False)
        current_vae_decoder.eval()
        if getattr(self.args, "no_compile", False):
            logger.info("  Skipping VAE decoder compile (--no-compile). Using eager mode.")
        else:
            compile_start = time.perf_counter()
            logger.info("  Compiling VAE decoder (this takes ~30s on first run)...")
            current_vae_decoder.compile(mode="max-autotune-no-cudagraphs")
            logger.info("  VAE decoder compiled (%.2fs)", time.perf_counter() - compile_start)

        # Initialize pipeline
        pipeline = CausalInferenceStreamingPipeline(
            self.config, generator=generator, vae_decoder=current_vae_decoder)

        if main_checkpoint:
            pipeline.generator.load_state_dict(main_checkpoint)
            logger.info("  Main checkpoint applied to generator")

        self.pipeline = pipeline.to(device=self.device, dtype=self.weight_dtype)
        self.pipeline.vae_decoder.to(torch.float16)

        # Move VAE encoder to GPU
        self.vae = vae_encoder.to(self.device, self.weight_dtype)

        gpu_elapsed = time.perf_counter() - gpu_start
        total_elapsed = time.perf_counter() - models_start
        logger.info("Models loaded and moved to GPU (gpu=%.2fs, total=%.2fs)", gpu_elapsed, total_elapsed)

        # Warmup primarily exists to trigger compile paths ahead of first inference.
        if getattr(self.args, "no_compile", False):
            logger.info("Skipping warmup (--no-compile).")
        elif not getattr(self.args, 'no_warmup', False):
            self._warmup()
        else:
            logger.info("Skipping warmup (--no-warmup). Kernels will compile lazily on first inference.")

        logger.info("TIP: Keep this server running to avoid reloading models.")

    def _warmup(self):
        """Run complete warmup to trigger all torch.compile paths."""
        warmup_start = time.perf_counter()
        logger.info("Warming up (compiling all kernels)...")

        num_frame_per_block = self.pipeline.num_frame_per_block
        denoising_steps = self.pipeline.denoising_step_list

        # Get actual visual_context shape from model
        logger.info("  Detecting tensor shapes...")
        test_image = torch.randn([1, 3, 1, 352, 640], device=self.device, dtype=self.weight_dtype)
        with torch.no_grad():
            visual_context_shape = self.vae.clip.encode_video(test_image).shape
        logger.info("    visual_context shape: %s", visual_context_shape)

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

        logger.info("  Running %d denoising steps + context pass...", len(denoising_steps))

        with torch.no_grad():
            output = noise_slice

            # Run through all denoising timesteps to compile each one
            for step_idx, current_timestep in enumerate(denoising_steps):
                step_start = time.perf_counter()
                logger.info("    Step %d/%d (timestep=%d)...", step_idx + 1, len(denoising_steps), current_timestep)
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
                logger.info("    Step %d/%d done (%.2fs)", step_idx + 1, len(denoising_steps), time.perf_counter() - step_start)
                output = denoised_pred

            # Context pass (timestep=0) to compile that path too
            ctx_start = time.perf_counter()
            logger.info("    Context pass (timestep=%s)...", self.pipeline.args.context_noise)
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
            logger.info("    Context pass done (%.2fs)", time.perf_counter() - ctx_start)

            # VAE decoder - first with None cache
            vae_start = time.perf_counter()
            logger.info("  VAE decoder (empty cache)...")
            vae_cache = [None] * len(ZERO_VAE_CACHE)
            dummy_latent = output.transpose(1, 2)
            _, vae_cache = self.pipeline.vae_decoder(dummy_latent.half(), *vae_cache)
            logger.info("  VAE decoder (empty cache) done (%.2fs)", time.perf_counter() - vae_start)

            # VAE decoder - second with populated cache (different code path)
            vae2_start = time.perf_counter()
            logger.info("  VAE decoder (populated cache)...")
            _, vae_cache = self.pipeline.vae_decoder(dummy_latent.half(), *vae_cache)
            logger.info("  VAE decoder (populated cache) done (%.2fs)", time.perf_counter() - vae2_start)

        # Clear CUDA cache
        torch.cuda.empty_cache()
        warmup_elapsed = time.perf_counter() - warmup_start
        logger.info("  Warmup complete! (total %.2fs)", warmup_elapsed)

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

    async def _generate_image_with_gemini(self, prompt: str, ref_image: Image.Image | None, websocket) -> Image.Image:
        """Generate a new image using Gemini with a text prompt and optional reference image."""
        client_label = _websocket_client_label(websocket)
        gemini_start = time.perf_counter()
        if not await self._send_ws(websocket, {
            "type": "generating_image",
            "message": f"Generating: {prompt[:60]}..."
        }):
            raise RuntimeError("Client disconnected during Gemini generation")

        logger.info("[Gemini %s] Generation started (prompt_chars=%d, has_ref_image=%s)", client_label, len(prompt), ref_image is not None)
        logger.info("[Gemini %s] Prompt: %s", client_label, prompt[:200])

        def _call_gemini():
            client = genai.Client()
            if ref_image is not None:
                contents = [prompt, ref_image]
            else:
                contents = [f"Generate an image: {prompt}"]
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
                    logger.info("[Gemini %s] Text response: %s", client_label, part.text[:200])
                elif part.inline_data is not None:
                    img_bytes = part.inline_data.data
                    return Image.open(io.BytesIO(img_bytes)).convert('RGB')
            return None

        loop = asyncio.get_event_loop()
        generated = await loop.run_in_executor(None, _call_gemini)
        gemini_elapsed = time.perf_counter() - gemini_start

        if generated is None:
            logger.error("[Gemini %s] No image returned after %.2fs", client_label, gemini_elapsed)
            raise RuntimeError("Gemini did not return an image")

        logger.info("[Gemini %s] Generated image size=%s (%.2fs)", client_label, generated.size, gemini_elapsed)
        return generated

    async def _generate_music(self, prompt: str, image: Image.Image | None, websocket, captcha_token: str = ""):
        """Generate background music using Anthropic + Suno, running in the background.

        Handles all input combinations:
        - image only: Anthropic describes the scene visually
        - text only: Anthropic uses the text description
        - image + text: Anthropic uses both
        """
        client_label = _websocket_client_label(websocket)
        music_start = time.perf_counter()
        logger.info("[Music %s] === Starting background music generation ===", client_label)
        logger.info("[Music %s] Prompt: %s", client_label, prompt[:100] if prompt else "(none)")
        logger.info("[Music %s] Captcha token: %d chars", client_label, len(captcha_token))
        logger.info("[Music %s] Image: %s", client_label, f"{image.size}" if image else "none")
        try:
            await websocket.send_json({
                "type": "music_status",
                "message": "Generating music prompt..."
            })

            # Step 1: Call Anthropic to create a Suno music prompt
            # Entire function runs in thread executor so it doesn't depend on event loop
            # (event loop gets blocked by synchronous GPU ops like VAE encoding)
            _image = image  # capture for closure

            def _encode_and_call_anthropic():
                from suno_api.client import DEFAULT_ANTHROPIC_KEY

                img_b64 = None
                if _image:
                    img_buffer = io.BytesIO()
                    _image.save(img_buffer, format="JPEG", quality=80)
                    img_b64 = base64.b64encode(img_buffer.getvalue()).decode()

                client = anthropic.Anthropic(
                    api_key=os.environ.get("ANTHROPIC_API_KEY", DEFAULT_ANTHROPIC_KEY)
                )

                # Build message content based on available inputs
                content = []
                if img_b64:
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_b64,
                        },
                    })

                suffix = (
                    "Write a short Suno AI music generation prompt (2-3 sentences max) describing "
                    "the perfect looping background music for this game world. Focus on mood, genre, "
                    "instruments, tempo, and atmosphere. Make it instrumental. "
                    "Output ONLY the Suno prompt, nothing else."
                )

                if _image and prompt:
                    text_part = (
                        "You are creating a prompt for Suno AI to generate background music for a video game world. "
                        f"Look at this game scene image (the player described it as: \"{prompt}\"). " + suffix
                    )
                elif _image:
                    text_part = (
                        "You are creating a prompt for Suno AI to generate background music for a video game world. "
                        "Look at this game scene image. " + suffix
                    )
                else:
                    text_part = (
                        "You are creating a prompt for Suno AI to generate background music for a video game world. "
                        f"The player described the world as: \"{prompt}\". " + suffix
                    )

                content.append({"type": "text", "text": text_part})

                msg = client.messages.create(
                    model="claude-sonnet-4-5-20250929",
                    max_tokens=300,
                    messages=[{"role": "user", "content": content}],
                )
                return msg.content[0].text.strip()

            logger.info("[Music %s] Calling Anthropic for music prompt...", client_label)
            loop = asyncio.get_event_loop()
            music_prompt = await loop.run_in_executor(None, _encode_and_call_anthropic)
            anthropic_elapsed = time.perf_counter() - music_start
            logger.info("[Music %s] Anthropic responded in %.2fs: %s", client_label, anthropic_elapsed, music_prompt)

            await websocket.send_json({
                "type": "music_status",
                "message": f"Composing: {music_prompt[:60]}..."
            })

            # Step 2: Generate song with Suno
            global suno_client
            if not suno_client:
                logger.warning("[Music %s] No Suno client configured, skipping music", client_label)
                return

            suno_start = time.perf_counter()
            logger.info("[Music %s] Sending generation request to Suno...", client_label)
            data = await suno_client.generate(
                description=music_prompt + " Instrumental, loopable, video game background music.",
                instrumental=True,
                captcha_token=captcha_token,
            )
            clips = data.get("clips", [])
            clip_ids = [c["id"] for c in clips]
            suno_submit_elapsed = time.perf_counter() - suno_start
            logger.info(
                "[Music %s] Suno accepted generation in %.2fs: clip_ids=%s",
                client_label, suno_submit_elapsed, clip_ids,
            )

            await websocket.send_json({
                "type": "music_status",
                "message": "Waiting for music generation..."
            })

            # Step 3: Poll until first song is complete (Suno generates 2, play whichever finishes first)
            logger.info("[Music %s] Polling for first song completion (timeout=300s)...", client_label)
            poll_start = time.perf_counter()
            deadline = time.perf_counter() + 300
            poll_count = 0
            first_ready = None
            while time.perf_counter() < deadline:
                songs = await suno_client.get_songs(clip_ids)
                poll_count += 1
                for s in songs:
                    if s.get("audio_url"):
                        first_ready = s
                        break
                if first_ready:
                    break
                statuses = [s.get("status", "?") for s in songs]
                logger.info("[Music %s] Poll #%d: statuses=%s", client_label, poll_count, statuses)
                await asyncio.sleep(5)

            poll_elapsed = time.perf_counter() - poll_start
            if first_ready:
                total_elapsed = time.perf_counter() - music_start
                logger.info(
                    "[Music %s] First song ready in %.2fs (poll=%.2fs, %d polls): %s",
                    client_label, total_elapsed, poll_elapsed, poll_count, first_ready.get("audio_url"),
                )
                await websocket.send_json({
                    "type": "music_ready",
                    "audio_urls": [first_ready["audio_url"]],
                    "title": first_ready.get("title", ""),
                    "prompt": music_prompt,
                })
            else:
                total_elapsed = time.perf_counter() - music_start
                song_statuses = [s.get("status", "?") for s in songs]
                logger.warning(
                    "[Music %s] No songs ready after %.2fs (%d polls, statuses=%s)",
                    client_label, total_elapsed, poll_count, song_statuses,
                )

        except Exception as e:
            total_elapsed = time.perf_counter() - music_start
            logger.exception("[Music %s] Music generation failed after %.2fs", client_label, total_elapsed)
            try:
                await websocket.send_json({
                    "type": "music_status",
                    "message": f"Music generation failed: {str(e)[:100]}"
                })
            except Exception:
                pass

    async def _generate_caption(self, frame: np.ndarray, block_num: int, websocket):
        """Send a frame to Claude and get a caption + visibility-optimized color.

        Runs entirely in a thread executor so the event loop (and inference) are never blocked.
        The frontend simulates the typewriter effect locally.
        """
        client_label = _websocket_client_label(websocket)
        caption_start = time.perf_counter()
        logger.info("[Caption %s] Starting caption for block %d", client_label, block_num)
        try:
            img = Image.fromarray(frame.astype(np.uint8))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=70)
            img_b64 = base64.b64encode(buf.getvalue()).decode()
            logger.info("[Caption %s] Block %d image encoded (%d bytes)", client_label, block_num, len(buf.getvalue()))

            def _call_claude():
                from suno_api.client import DEFAULT_ANTHROPIC_KEY
                client = anthropic.Anthropic(
                    api_key=os.environ.get("ANTHROPIC_API_KEY", DEFAULT_ANTHROPIC_KEY)
                )
                msg = client.messages.create(
                    model="claude-sonnet-4-5-20250929",
                    max_tokens=100,
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
                                    "You are a narrator for a real-time interactive world exploration game. "
                                    "Respond with EXACTLY two lines, nothing else:\n"
                                    "Line 1: A hex color code (e.g. #FF6B35) that will be maximally visible "
                                    "as caption text overlaid on this image. Pick a color that contrasts strongly "
                                    "with the dominant colors in the lower portion of the image where the caption appears.\n"
                                    "Line 2: A single short, atmospheric sentence (max 15 words) describing what you see. "
                                    "Be poetic and immersive. No quotes, no labels, just the narration."
                                ),
                            },
                        ],
                    }],
                )
                return msg.content[0].text.strip()

            logger.info("[Caption %s] Block %d calling Anthropic...", client_label, block_num)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, _call_claude)
            api_elapsed = time.perf_counter() - caption_start
            logger.info("[Caption %s] Block %d Anthropic responded (%.2fs): %s", client_label, block_num, api_elapsed, result)

            # Parse color and text from response
            lines = result.strip().split('\n', 1)
            color = "#e0e0e0"  # default fallback
            caption_text = result
            if len(lines) == 2 and lines[0].startswith('#') and len(lines[0]) == 7:
                color = lines[0].strip()
                caption_text = lines[1].strip()
            logger.info("[Caption %s] Block %d parsed: color=%s text=%s", client_label, block_num, color, caption_text)

            await self._send_ws(websocket, {
                "type": "caption",
                "block": block_num,
                "text": caption_text,
                "color": color,
            })

            caption_elapsed = time.perf_counter() - caption_start
            logger.info("[Caption %s] Block %d complete (%.2fs)", client_label, block_num, caption_elapsed)

        except Exception as e:
            caption_elapsed = time.perf_counter() - caption_start
            logger.warning("[Caption %s] Block %d failed after %.2fs: %s", client_label, block_num, caption_elapsed, e)

    async def _get_next_action(self) -> Optional[dict]:
        """Get next action: from vote buffer (room mode) or action_queue (solo mode)."""
        global room_state
        if room_state is None or not room_state.is_active:
            return await asyncio.wait_for(self.action_queue.get(), timeout=120.0)

        # Room mode: if votes are buffered, resolve immediately
        rs = room_state
        if rs.votes:
            result = _resolve_votes(rs.votes)
            vote_count = len(rs.votes)
            rs.votes.clear()
            rs.vote_event.clear()
            logger.info("[Room] Resolved %d votes -> keyboard=%s mouse=%s", vote_count, result["keyboard"], result["mouse"])
            return result

        # No votes yet — wait for first vote or room closure
        rs.vote_event.clear()
        try:
            await asyncio.wait_for(rs.vote_event.wait(), timeout=120.0)
        except asyncio.TimeoutError:
            return {"keyboard": "q", "mouse": "u"}

        # Room may have closed while waiting
        if room_state is None or not room_state.is_active:
            return await asyncio.wait_for(self.action_queue.get(), timeout=120.0)

        if rs.votes:
            result = _resolve_votes(rs.votes)
            vote_count = len(rs.votes)
            rs.votes.clear()
            rs.vote_event.clear()
            logger.info("[Room] Resolved %d votes (after wait) -> keyboard=%s mouse=%s", vote_count, result["keyboard"], result["mouse"])
            return result

        return {"keyboard": "q", "mouse": "u"}

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

    async def start_music(self, captcha_token: str, websocket: WebSocket):
        """Start music generation with the captcha token (called when token arrives via WS)."""
        client_label = _websocket_client_label(websocket)
        if not captcha_token:
            logger.warning("[Music %s] Empty captcha token, skipping music", client_label)
            return
        prompt = getattr(self, '_music_prompt', '')
        image = getattr(self, '_music_image', None)
        logger.info("[Music %s] Captcha token received (%d chars), starting music generation", client_label, len(captcha_token))
        asyncio.create_task(self._generate_music(prompt, image, websocket, captcha_token))

    async def run_inference(self, img_input: str | None, websocket: WebSocket, prompt: str = None):
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
            image_source = "upload" if (img_input and img_input.startswith("data:image")) else ("path" if img_input else "none")
            logger.info(
                "[UI %s] Starting inference mode=%s image_source=%s prompt=%s",
                client_label,
                self.mode,
                image_source,
                "yes" if prompt else "no",
            )

            # Load image if provided (supports file path or base64 data)
            image = None
            if img_input:
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

            # Store for music generation immediately (captcha token may arrive any time)
            # Music gen runs in parallel with Gemini — doesn't need the Gemini-generated image
            self._music_prompt = prompt or ""
            self._music_image = image  # uploaded image or None; good enough for Anthropic music prompt

            # Generate image with Gemini if prompt provided
            if prompt:
                image = await self._generate_image_with_gemini(prompt, image, websocket)

            image = self._resizecrop(image, 352, 640)
            logger.info("[UI %s] Input image normalized to %s", client_label, image.size)

            # Send image preview to client (when Gemini generated the image)
            if prompt:
                buf = io.BytesIO()
                image.save(buf, format='JPEG', quality=85)
                await self._send_ws(websocket, {
                    "type": "image_preview",
                    "data": base64.b64encode(buf.getvalue()).decode()
                })

            image_tensor = self.frame_process(image)[None, :, None, :, :].to(
                dtype=self.weight_dtype, device=self.device)

            # Encode input image
            max_output_frames = self.args.max_num_output_frames
            logger.info("[UI %s] VAE encoding input image (max_output_frames=%d)...", client_label, max_output_frames)
            vae_encode_start = time.perf_counter()
            padding_video = torch.zeros_like(image_tensor).repeat(1, 1, 4 * (max_output_frames - 1), 1, 1)
            img_cond = torch.concat([image_tensor, padding_video], dim=2)
            tiler_kwargs = {"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}
            img_cond = self.vae.encode(img_cond, device=self.device, **tiler_kwargs).to(self.device)

            mask_cond = torch.ones_like(img_cond)
            mask_cond[:, :, 1:] = 0
            cond_concat = torch.cat([mask_cond[:, :4], img_cond], dim=1)
            visual_context = self.vae.clip.encode_video(image_tensor)
            vae_encode_elapsed = time.perf_counter() - vae_encode_start
            logger.info(
                "[UI %s] VAE encode complete (%.2fs) cond_concat=%s visual_context=%s",
                client_label, vae_encode_elapsed, list(cond_concat.shape), list(visual_context.shape),
            )

            sampled_noise = torch.randn(
                [1, 16, max_output_frames, 44, 80],
                device=self.device, dtype=self.weight_dtype
            )

            num_frames = (max_output_frames - 1) * 4 + 1

            # Setup conditions
            logger.info("[UI %s] Setting up conditions (mode=%s, num_frames=%d)...", client_label, self.mode, num_frames)
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
            logger.info("[UI %s] Conditions ready (keys=%s)", client_label, list(conditional_dict.keys()))

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
        logger.info("[UI %s] Initializing KV caches (batch=%d, dtype=%s, device=%s)...", client_label, batch_size, noise.dtype, noise.device)
        kv_start = time.perf_counter()
        self.pipeline.kv_cache1 = None
        self.pipeline.kv_cache_keyboard = None
        self.pipeline.kv_cache_mouse = None
        self.pipeline.crossattn_cache = None

        self.pipeline._initialize_kv_cache(batch_size, noise.dtype, noise.device)
        self.pipeline._initialize_kv_cache_mouse_and_keyboard(batch_size, noise.dtype, noise.device)
        self.pipeline._initialize_crossattn_cache(batch_size, noise.dtype, noise.device)
        logger.info("[UI %s] KV caches initialized (%.2fs)", client_label, time.perf_counter() - kv_start)

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
                action = await self._get_next_action()
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
            ctx_pass_start = time.perf_counter()
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
            logger.info("[UI %s] Block %d context pass (%.2fs)", client_label, block_num, time.perf_counter() - ctx_pass_start)

            # Decode and send frames
            vae_decode_start = time.perf_counter()
            denoised_pred_t = denoised_pred.transpose(1, 2)
            video, vae_cache = self.pipeline.vae_decoder(denoised_pred_t.half(), *vae_cache)
            videos.append(video)
            logger.info("[UI %s] Block %d VAE decode (%.2fs)", client_label, block_num, time.perf_counter() - vae_decode_start)

            video_np = rearrange(video, "B T C H W -> B T H W C")
            video_np = ((video_np.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)[0]

            # Send each frame
            send_start = time.perf_counter()
            for frame_idx, frame in enumerate(video_np):
                if not await self._send_ws(websocket, {
                    "type": "frame",
                    "data": self._frame_to_base64(frame),
                    "block": block_num,
                    "frame": frame_idx
                }):
                    return
                await asyncio.sleep(0.04)  # ~25fps pacing
            logger.info("[UI %s] Block %d sent %d frames (%.2fs)", client_label, block_num, len(video_np), time.perf_counter() - send_start)

            # Fire off caption every 3 blocks (non-blocking)
            if block_num % 3 == 1:
                logger.info("[UI %s] Spawning caption task for block %d", client_label, block_num)
                asyncio.create_task(self._generate_caption(video_np[-1], block_num, websocket))

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


async def _broadcast_player_count():
    """Send updated player count to host and all players."""
    global room_state
    rs = room_state
    if rs is None:
        return
    count = len(rs.player_websockets)
    names = list(rs.player_names.values())
    msg = {"type": "room_update", "player_count": count, "names": names}
    try:
        await rs.host_ws.send_json(msg)
    except Exception:
        pass
    for pw in list(rs.player_websockets):
        try:
            await pw.send_json(msg)
        except Exception:
            pass


async def _send_action_feed(name: str, keyboard: str, mouse: str):
    """Send an action feed entry to the host."""
    global room_state
    rs = room_state
    if rs is None:
        return
    try:
        await rs.host_ws.send_json({
            "type": "action_feed",
            "name": name,
            "keyboard": keyboard,
            "mouse": mouse,
        })
    except Exception:
        pass


async def _close_room():
    """Close the active room: notify all players and host, reset state."""
    global room_state
    rs = room_state
    if rs is None:
        return
    rs.is_active = False
    rs.vote_event.set()  # wake any waiting _get_next_action
    close_msg = {"type": "room_closed"}
    for pw in list(rs.player_websockets):
        try:
            await pw.send_json(close_msg)
            await pw.close()
        except Exception:
            pass
    try:
        await rs.host_ws.send_json(close_msg)
    except Exception:
        pass
    logger.info("[Room] Room %s closed (%d players were connected)", rs.room_id, len(rs.player_websockets))
    room_state = None


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


@app.get("/room/{room_id}", response_class=HTMLResponse)
async def get_room(room_id: str):
    """Serve the mobile remote control page if room exists."""
    global room_state
    if room_state is None or room_state.room_id != room_id or not room_state.is_active:
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse("Room not found", status_code=404)
    return FileResponse("web_ui/remote.html")


@app.websocket("/ws/room/{room_id}")
async def room_websocket(websocket: WebSocket, room_id: str):
    """WebSocket endpoint for mobile remote control players."""
    global room_state
    await websocket.accept()
    if room_state is None or room_state.room_id != room_id or not room_state.is_active:
        await websocket.close(code=4004, reason="Room not found")
        return
    rs = room_state
    rs.player_websockets.add(websocket)
    player_name = "Player"

    try:
        # Wait for first message which should be a join with name
        first = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
        if first.get("type") == "join":
            player_name = (first.get("name") or "Player")[:16]

        rs.player_names[websocket] = player_name
        count = len(rs.player_websockets)
        logger.info("[Room] %s joined room %s (player_count=%d)", player_name, room_id, count)

        await websocket.send_json({"type": "room_joined", "player_count": count})
        await _broadcast_player_count()

        while True:
            data = await websocket.receive_json()
            if data.get("type") == "vote":
                if rs.is_active:
                    kb = data.get("keyboard", "q")
                    ms = data.get("mouse", "u")
                    rs.votes.append({"keyboard": kb, "mouse": ms})
                    rs.vote_event.set()
                    await _send_action_feed(player_name, kb, ms)
                    logger.info("[Room] Vote from %s: keyboard=%s mouse=%s (buffered=%d)",
                                player_name, kb, ms, len(rs.votes))
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        rs.player_websockets.discard(websocket)
        rs.player_names.pop(websocket, None)
        logger.info("[Room] %s left room %s (player_count=%d)", player_name, room_id, len(rs.player_websockets))
        if rs.is_active:
            await _broadcast_player_count()


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
                img_input = data.get("image", None)
                prompt = data.get("prompt", None)
                logger.info(
                    "[WS %s] Start requested image_source=%s prompt=%s",
                    client_label,
                    "upload" if (img_input and str(img_input).startswith("data:image")) else ("path" if img_input else "none"),
                    "yes" if prompt else "no",
                )
                if inference_engine and not inference_engine.is_running:
                    try:
                        asyncio.create_task(
                            inference_engine.run_inference(img_input, websocket, prompt=prompt))
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

            elif msg_type == "captcha_token":
                token = data.get("token", "")
                logger.info("[WS %s] Captcha token received (%d chars)", client_label, len(token))
                if inference_engine and inference_engine.is_running:
                    await inference_engine.start_music(token, websocket)
                else:
                    logger.warning("[WS %s] Captcha token received but inference not running", client_label)

            elif msg_type == "action":
                if inference_engine and inference_engine.is_running:
                    if room_state and room_state.is_active:
                        kb = data.get("keyboard", "q")
                        ms = data.get("mouse", "u")
                        room_state.votes.append({"keyboard": kb, "mouse": ms})
                        room_state.vote_event.set()
                        await _send_action_feed("Host", kb, ms)
                        logger.info(
                            "[WS %s] Host vote (%s) buffered=%d",
                            client_label,
                            _summarize_action_payload(data),
                            len(room_state.votes),
                        )
                    else:
                        await inference_engine.action_queue.put(data)
                        logger.info(
                            "[WS %s] Queued action (%s) queue_size=%d",
                            client_label,
                            _summarize_action_payload(data),
                            inference_engine.action_queue.qsize(),
                        )
                else:
                    logger.warning("[WS %s] Dropped action: inference is not running", client_label)

            elif msg_type == "open_room":
                if room_state is not None:
                    await websocket.send_json({"type": "error", "message": "Room already open"})
                else:
                    rid = _generate_room_id()
                    room_state = RoomState(room_id=rid, host_ws=websocket)
                    lan_ip = _get_lan_ip()
                    url = f"http://{lan_ip}:{server_port}/room/{rid}"
                    logger.info("[WS %s] Room opened: %s -> %s", client_label, rid, url)
                    await websocket.send_json({
                        "type": "room_opened",
                        "room_id": rid,
                        "url": url,
                        "player_count": 0,
                    })

            elif msg_type == "close_room":
                if room_state and room_state.host_ws == websocket:
                    logger.info("[WS %s] Close room requested", client_label)
                    await _close_room()

            elif msg_type == "stop":
                if inference_engine:
                    logger.info("[WS %s] Stop requested by client", client_label)
                    if room_state and room_state.host_ws == websocket:
                        await _close_room()
                    inference_engine.stop()
                    await inference_engine.action_queue.put(None)
                    await websocket.send_json({"type": "stopped"})
            else:
                logger.warning("[WS %s] Unknown message type=%s", client_label, msg_type)

    except WebSocketDisconnect:
        logger.info("[WS %s] Disconnected", client_label)
        if room_state and room_state.host_ws == websocket:
            await _close_room()
        if inference_engine:
            inference_engine.stop()
            await inference_engine.action_queue.put(None)
    except Exception as e:
        logger.exception("[WS %s] WebSocket error", client_label)
        if room_state and room_state.host_ws == websocket:
            await _close_room()
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
    parser.add_argument("--true-no-compile", action="store_true",
                        help="Disable ALL torch.compile including flex_attention. Implies --no-compile and --no-warmup. "
                             "Near-instant startup but significantly slower inference (~2-4x)")
    args = parser.parse_args()

    # --true-no-compile implies --no-compile and --no-warmup
    if args.true_no_compile:
        args.no_compile = True
        args.no_warmup = True

    return args


if __name__ == "__main__":
    args = parse_args()
    server_port = args.port
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
