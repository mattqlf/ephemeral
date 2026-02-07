"""
Web UI server for Matrix-Game 2.0 with WebSocket streaming.

Run with: uv run python -m apps.servers.web_server --checkpoint_path Matrix-Game-2.0/base_distilled_model/base_distill.safetensors
"""

import os
import sys
import time as _time

_t0 = _time.perf_counter()
print("[startup] Loading modules...", flush=True)

os.environ["GOOGLE_API_KEY"] = "AIzaSyBsBvyfplPcR3g503sehLCbkU5nLkSY05A"

# Persist Triton/inductor kernel caches so subsequent launches skip recompilation.
# On RunPod, ~/.triton/ is ephemeral — /workspace/ survives restarts.
if os.path.isdir("/workspace") and "TRITON_CACHE_DIR" not in os.environ:
    os.environ["TRITON_CACHE_DIR"] = "/workspace/.cache/triton"
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")

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
from typing import Optional

def _elapsed():
    return f"{_time.perf_counter() - _t0:.1f}s"

print(f"[startup] Importing torch...", flush=True)
import torch
import torch._dynamo
print(f"[startup] torch imported ({_elapsed()})", flush=True)

# Increase dynamo cache size to prevent VAE cache thrashing
torch._dynamo.config.cache_size_limit = 64
import numpy as np

print(f"[startup] Importing dependencies...", flush=True)
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
print(f"[startup] Dependencies imported ({_elapsed()})", flush=True)

print(f"[startup] Importing model pipeline (flex_attention compile happens here)...", flush=True)
from pipeline import CausalInferenceStreamingPipeline
from pipeline.causal_inference import cond_current
print(f"[startup] Pipeline imported ({_elapsed()})", flush=True)

print(f"[startup] Importing VAE and utilities...", flush=True)
from wan.vae.wanx_vae import get_wanx_vae_wrapper
from demo_utils.vae_block3 import VAEDecoderWrapper
from utils.misc import set_seed
from utils.conditions import Bench_actions_universal, Bench_actions_gta_drive, Bench_actions_templerun
from utils.wan_wrapper import WanDiffusionWrapper
print(f"[startup] VAE and utilities imported ({_elapsed()})", flush=True)

from google import genai
from google.genai import types as genai_types
import anthropic
from suno_api.client import SunoClient

print(f"[startup] All modules loaded ({_elapsed()})", flush=True)

app = FastAPI(title="Matrix-Game 2.0 Web UI")
logger = logging.getLogger("ephemeral.web")

# Global state
inference_engine: Optional["InferenceEngine"] = None
suno_client: Optional[SunoClient] = None


@dataclass
class WorldRunner:
    world_id: int
    device: torch.device
    pipeline: "CausalInferenceStreamingPipeline"
    action_queue: asyncio.Queue
    # Per-run state (set during run_inference):
    conditional_dict: dict = None
    vae_cache: list = None
    noise: torch.Tensor = None


@dataclass
class RoomState:
    room_id: str
    host_ws: WebSocket
    player_websockets: set = field(default_factory=set)
    player_names: dict = field(default_factory=dict)  # websocket -> name
    votes: dict = field(default_factory=dict)          # world_id -> list[dict]
    vote_event: asyncio.Event = field(default_factory=asyncio.Event)
    player_worlds: dict = field(default_factory=dict)  # websocket -> world_id
    is_active: bool = True


room_state: Optional[RoomState] = None
server_port: int = 8080


# ── Modal per-player multiplayer ──

@dataclass
class ModalPlayerState:
    player_id: str
    name: str
    websocket: WebSocket
    worker: object  # Modal class instance handle
    status: str  # "queued" | "initializing" | "active" | "disconnected"
    action_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    inference_task: Optional[asyncio.Task] = None


@dataclass
class ModalRoomState:
    room_id: str
    host_ws: WebSocket
    seed_image_b64: str  # base64 of the seed image (no data: prefix)
    mode: str
    players: dict = field(default_factory=dict)  # player_id -> ModalPlayerState
    queue: list = field(default_factory=list)  # ordered player_ids waiting
    active_count: int = 0
    max_active: int = 8
    is_active: bool = True
    music_urls: list = field(default_factory=list)


modal_room_state: Optional[ModalRoomState] = None


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
        self.weight_dtype = torch.bfloat16

        # Per-world GPU devices
        num_gpus = torch.cuda.device_count()
        if args.num_worlds > 1 and args.num_worlds > num_gpus:
            raise RuntimeError(
                f"--num-worlds={args.num_worlds} requires {args.num_worlds} GPUs, "
                f"only {num_gpus} available"
            )
        self.devices = [torch.device(f"cuda:{i}") for i in range(args.num_worlds)]
        self.device = self.devices[0]  # primary device for shared ops (VAE encode, image processing)

        self.is_running = False
        self.should_stop = False
        self.num_worlds = args.num_worlds
        self.worlds: list[WorldRunner] = []
        self._ws_lock: asyncio.Lock = None  # lazy init, for thread-safe WebSocket sends

        logger.info(
            "InferenceEngine init: checkpoint=%s pretrained=%s max_frames=%d resolution=%dx%d fast=%s no_compile=%s seed=%d num_worlds=%d devices=%s",
            args.checkpoint_path,
            args.pretrained_model_path,
            args.max_num_output_frames,
            args.res_width,
            args.res_height,
            getattr(args, 'fast', False),
            getattr(args, 'no_compile', False),
            args.seed,
            args.num_worlds,
            [str(d) for d in self.devices],
        )

        self._init_config()
        self._init_models()

        self.frame_process = v2.Compose([
            v2.Resize(size=(self.res_height, self.res_width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _init_config(self):
        logger.info("Loading config from %s", self.args.config_path)
        self.config = OmegaConf.load(self.args.config_path)
        self.mode = self.config.get('mode', 'universal')
        logger.info("Config loaded: mode=%s denoising_steps=%s", self.mode, list(self.config.get('denoising_step_list', [])))

        # Store resolution for use throughout the engine
        self.res_width = self.args.res_width
        self.res_height = self.args.res_height
        self.latent_w = self.args.latent_w
        self.latent_h = self.args.latent_h
        logger.info("Resolution: %dx%d (latent %dx%d)", self.res_width, self.res_height, self.latent_w, self.latent_h)

        # Override image_or_video_shape with resolved latent dimensions
        shape = list(self.config.image_or_video_shape)
        shape[3] = self.latent_h
        shape[4] = self.latent_w
        self.config.image_or_video_shape = shape
        logger.info("image_or_video_shape: %s", shape)

        # --num-frames: override frames per block
        if self.args.num_frames is not None:
            logger.info("Overriding num_frame_per_block: %d -> %d", self.config.num_frame_per_block, self.args.num_frames)
            self.config.num_frame_per_block = self.args.num_frames

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

        # Create an engine on each world's GPU
        gpu_start = time.perf_counter()
        self.worlds = []
        for i, device in enumerate(self.devices):
            logger.info("Creating world %d engine on %s...", i, device)
            pipeline = self._create_world_engine(device, decoder_state_dict, main_checkpoint)
            world = WorldRunner(
                world_id=i, device=device, pipeline=pipeline,
                action_queue=asyncio.Queue())
            self.worlds.append(world)
            logger.info("World %d engine ready on %s", i, device)

        # Keep backward compat references (world 0 = primary)
        self.pipeline = self.worlds[0].pipeline

        # VAE encoder only on primary device (used for input image encoding)
        self.vae = vae_encoder.to(self.device, self.weight_dtype)

        gpu_elapsed = time.perf_counter() - gpu_start
        total_elapsed = time.perf_counter() - models_start
        logger.info("All %d world engine(s) created (gpu=%.2fs, total=%.2fs)", len(self.worlds), gpu_elapsed, total_elapsed)

        # Warmup primarily exists to trigger compile paths ahead of first inference.
        if getattr(self.args, "no_compile", False):
            logger.info("Skipping warmup (--no-compile).")
        elif not getattr(self.args, 'no_warmup', False):
            self._warmup()
        else:
            logger.info("Skipping warmup (--no-warmup). Kernels will compile lazily on first inference.")

        logger.info("TIP: Keep this server running to avoid reloading models.")

    def _create_world_engine(self, device: torch.device, decoder_state_dict: dict,
                             main_checkpoint: dict | None) -> CausalInferenceStreamingPipeline:
        """Create a single inference pipeline on the specified device.

        Follows the pattern from web_server_speculative._create_engine:
        fresh model instances with deepcopied weights per GPU.
        """
        generator = WanDiffusionWrapper(
            **getattr(self.config, "model_kwargs", {}), is_causal=True)

        vae_decoder = VAEDecoderWrapper()
        vae_decoder.load_state_dict(copy.deepcopy(decoder_state_dict))
        vae_decoder.to(device, torch.float16)
        vae_decoder.requires_grad_(False)
        vae_decoder.eval()

        if getattr(self.args, "no_compile", False):
            logger.info("  Skipping VAE decoder compile on %s (--no-compile)", device)
        else:
            compile_start = time.perf_counter()
            logger.info("  Compiling VAE decoder on %s...", device)
            vae_decoder.compile(mode="max-autotune-no-cudagraphs")
            logger.info("  VAE decoder compiled on %s (%.2fs)", device, time.perf_counter() - compile_start)

        pipeline = CausalInferenceStreamingPipeline(
            self.config, generator=generator, vae_decoder=vae_decoder)

        if main_checkpoint:
            pipeline.generator.load_state_dict(copy.deepcopy(main_checkpoint))
            logger.info("  Main checkpoint applied to generator on %s", device)

        pipeline = pipeline.to(device=device, dtype=self.weight_dtype)
        pipeline.vae_decoder.to(torch.float16)

        return pipeline

    def _warmup(self):
        """Run complete warmup on each world's pipeline to trigger all torch.compile paths."""
        warmup_start = time.perf_counter()
        logger.info("Warming up %d world(s) (compiling all kernels)...", len(self.worlds))
        for world in self.worlds:
            self._warmup_single(world)
        warmup_elapsed = time.perf_counter() - warmup_start
        logger.info("  Warmup complete! (total %.2fs)", warmup_elapsed)

    def _warmup_single(self, world: "WorldRunner"):
        """Run warmup on a single world's pipeline."""
        pipeline = world.pipeline
        device = world.device
        world_start = time.perf_counter()
        logger.info("  Warming up world %d on %s...", world.world_id, device)

        num_frame_per_block = pipeline.num_frame_per_block
        denoising_steps = pipeline.denoising_step_list

        # Get actual visual_context shape from model (use primary VAE encoder)
        logger.info("  Detecting tensor shapes...")
        test_image = torch.randn([1, 3, 1, self.res_height, self.res_width], device=self.device, dtype=self.weight_dtype)
        with torch.no_grad():
            visual_context_shape = self.vae.clip.encode_video(test_image).shape
        logger.info("    visual_context shape: %s", visual_context_shape)

        # Create conditions for block 0
        max_output_frames = self.args.max_num_output_frames
        num_frames = (max_output_frames - 1) * 4 + 1

        full_cond_concat = torch.randn(
            [1, 20, max_output_frames, self.latent_h, self.latent_w],
            device=device, dtype=self.weight_dtype
        )
        full_visual_context = torch.randn(
            visual_context_shape,
            device=device, dtype=self.weight_dtype
        )
        full_keyboard = torch.zeros(
            [1, num_frames, 4],
            device=device, dtype=self.weight_dtype
        )
        full_mouse = torch.zeros(
            [1, num_frames, 2],
            device=device, dtype=self.weight_dtype
        )
        sampled_noise = torch.randn(
            [1, 16, max_output_frames, self.latent_h, self.latent_w],
            device=device, dtype=self.weight_dtype
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
        pipeline._initialize_kv_cache(1, self.weight_dtype, device)
        pipeline._initialize_kv_cache_mouse_and_keyboard(1, self.weight_dtype, device)
        pipeline._initialize_crossattn_cache(1, self.weight_dtype, device)

        logger.info("  Running %d denoising steps + context pass...", len(denoising_steps))

        with torch.no_grad():
            output = noise_slice

            for step_idx, current_timestep in enumerate(denoising_steps):
                step_start = time.perf_counter()
                logger.info("    Step %d/%d (timestep=%d)...", step_idx + 1, len(denoising_steps), current_timestep)
                timestep = torch.ones([1, num_frame_per_block], device=device, dtype=torch.int64) * current_timestep

                denoised_pred, _ = pipeline.generator(
                    noisy_image_or_video=output,
                    conditional_dict=cond,
                    timestep=timestep,
                    kv_cache=pipeline.kv_cache1,
                    kv_cache_mouse=pipeline.kv_cache_mouse,
                    kv_cache_keyboard=pipeline.kv_cache_keyboard,
                    crossattn_cache=pipeline.crossattn_cache,
                    current_start=0
                )
                logger.info("    Step %d/%d done (%.2fs)", step_idx + 1, len(denoising_steps), time.perf_counter() - step_start)
                output = denoised_pred

            # Context pass
            ctx_start = time.perf_counter()
            logger.info("    Context pass (timestep=%s)...", pipeline.args.context_noise)
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
            logger.info("    Context pass done (%.2fs)", time.perf_counter() - ctx_start)

            # VAE decoder - first with None cache
            vae_start = time.perf_counter()
            logger.info("  VAE decoder (empty cache)...")
            vae_cache = [None] * len(ZERO_VAE_CACHE)
            dummy_latent = output.transpose(1, 2)
            _, vae_cache = pipeline.vae_decoder(dummy_latent.half(), *vae_cache)
            logger.info("  VAE decoder (empty cache) done (%.2fs)", time.perf_counter() - vae_start)

            # VAE decoder - second with populated cache
            vae2_start = time.perf_counter()
            logger.info("  VAE decoder (populated cache)...")
            _, vae_cache = pipeline.vae_decoder(dummy_latent.half(), *vae_cache)
            logger.info("  VAE decoder (populated cache) done (%.2fs)", time.perf_counter() - vae2_start)

        torch.cuda.empty_cache()
        logger.info("  World %d warmup done (%.2fs)", world.world_id, time.perf_counter() - world_start)

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

    @staticmethod
    def _random_action():
        """Return a random action for uncontrolled worlds."""
        return {
            "keyboard": random.choice(["w", "a", "s", "d", "q"]),
            "mouse": random.choice(["i", "j", "k", "l", "u"]),
        }

    async def _send_ws(self, websocket: WebSocket, payload: dict) -> bool:
        """Best-effort websocket send that treats disconnects as normal stop events.

        Thread-safe via asyncio.Lock — multiple concurrent world tasks may send
        frames simultaneously.
        """
        if self._ws_lock is None:
            self._ws_lock = asyncio.Lock()
        msg_type = payload.get("type", "unknown")
        client_label = _websocket_client_label(websocket)
        try:
            async with self._ws_lock:
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
                model=self.args.suno_model,
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

    async def _generate_caption(self, frame: np.ndarray, block_num: int, websocket, world_id: int = 0):
        """Send a frame to Claude and get a caption + visibility-optimized color.

        Runs entirely in a thread executor so the event loop (and inference) are never blocked.
        The frontend simulates the typewriter effect locally.
        """
        client_label = _websocket_client_label(websocket)
        caption_start = time.perf_counter()
        logger.info("[Caption %s] Starting caption for block %d world %d", client_label, block_num, world_id)
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
                                    "Line 1: Either #000000 or #FFFFFF — pick whichever contrasts better "
                                    "with the lower portion of the image where the caption appears.\n"
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
            color = "#FFFFFF"  # default fallback
            caption_text = result
            if len(lines) == 2 and lines[0].strip() in ("#000000", "#FFFFFF"):
                color = lines[0].strip()
                caption_text = lines[1].strip()
            logger.info("[Caption %s] Block %d parsed: color=%s text=%s", client_label, block_num, color, caption_text)

            await self._send_ws(websocket, {
                "type": "caption",
                "block": block_num,
                "text": caption_text,
                "color": color,
                "world_id": world_id,
            })

            caption_elapsed = time.perf_counter() - caption_start
            logger.info("[Caption %s] Block %d complete (%.2fs)", client_label, block_num, caption_elapsed)

        except Exception as e:
            caption_elapsed = time.perf_counter() - caption_start
            logger.warning("[Caption %s] Block %d failed after %.2fs: %s", client_label, block_num, caption_elapsed, e)

    async def _get_next_action(self, world_id: int = 0) -> Optional[dict]:
        """Get next action for a world.

        All inputs are votes. In room mode, buffered votes are resolved by
        majority; if none buffered, block until the first vote arrives.
        In solo mode, world 0 reads from its action_queue; other worlds use random.

        Handles solo<->room transitions via a sentinel pushed into action_queue
        when a room opens (see open_room handler).
        """
        global room_state

        while True:
            rs = room_state

            # ── Solo mode ──
            if rs is None or not rs.is_active:
                if world_id == 0:
                    action = await asyncio.wait_for(
                        self.worlds[0].action_queue.get(), timeout=120.0)
                    if action and action.get("_room_transition"):
                        continue  # Room just opened — re-enter loop in room mode
                    return action
                return self._random_action()

            # ── Room mode: check buffered votes ──
            world_votes = rs.votes.get(world_id, [])
            if world_votes:
                result = _resolve_votes(world_votes)
                vote_count = len(world_votes)
                rs.votes[world_id] = []
                logger.info("[Room] World %d resolved %d votes -> keyboard=%s mouse=%s",
                            world_id, vote_count, result["keyboard"], result["mouse"])
                return result

            # No votes buffered — non-host worlds with no controller get random
            if world_id > 0:
                has_controller = any(wid == world_id for wid in rs.player_worlds.values())
                if not has_controller:
                    return self._random_action()

            # Wait for a vote (any world) or room closure
            rs.vote_event.clear()
            try:
                await asyncio.wait_for(rs.vote_event.wait(), timeout=120.0)
            except asyncio.TimeoutError:
                return {"keyboard": "q", "mouse": "u"}

            # Room may have closed while waiting — loop back to re-check
            if room_state is None or not room_state.is_active:
                continue

            # Re-check votes after waking
            world_votes = rs.votes.get(world_id, [])
            if world_votes:
                result = _resolve_votes(world_votes)
                vote_count = len(world_votes)
                rs.votes[world_id] = []
                logger.info("[Room] World %d resolved %d votes (after wait) -> keyboard=%s mouse=%s",
                            world_id, vote_count, result["keyboard"], result["mouse"])
                return result

            # Woke up from vote for a different world — loop back
            continue

    def _parse_action(self, action_data: dict, device: torch.device = None) -> dict:
        """Parse action from WebSocket message to tensor format."""
        device = device or self.device
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
            mouse_cond = torch.tensor(CAMERA_VALUE_MAP.get(mouse_key, [0, 0]), device=device)
            keyboard_cond = torch.tensor(KEYBOARD_IDX.get(keyboard_key, [0, 0, 0, 0]), device=device)

        elif mode == "gta_drive":
            CAMERA_VALUE_MAP = {
                "a": [0, -CAM_VALUE], "d": [0, CAM_VALUE], "q": [0, 0]
            }
            KEYBOARD_IDX = {"w": [1, 0], "s": [0, 1], "q": [0, 0]}
            mouse_cond = torch.tensor(CAMERA_VALUE_MAP.get(keyboard_key, [0, 0]), device=device)
            keyboard_cond = torch.tensor(KEYBOARD_IDX.get(keyboard_key, [0, 0]), device=device)

        else:  # templerun
            KEYBOARD_IDX = {
                "w": [0, 1, 0, 0, 0, 0, 0], "s": [0, 0, 1, 0, 0, 0, 0],
                "a": [0, 0, 0, 0, 0, 1, 0], "d": [0, 0, 0, 0, 0, 0, 1],
                "z": [0, 0, 0, 1, 0, 0, 0], "c": [0, 0, 0, 0, 1, 0, 0],
                "q": [1, 0, 0, 0, 0, 0, 0]
            }
            keyboard_cond = torch.tensor(KEYBOARD_IDX.get(keyboard_key, [1, 0, 0, 0, 0, 0, 0]), device=device)
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

        # Clear any pending actions from all world queues
        for world in self.worlds:
            dropped = 0
            while not world.action_queue.empty():
                try:
                    world.action_queue.get_nowait()
                    dropped += 1
                except Exception:
                    break
            if dropped:
                logger.info("[UI %s] Cleared %d stale queued actions from world %d", client_label, dropped, world.world_id)

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
                    logger.info("[UI %s] Processing uploaded image (~%d KB base64)", client_label, len(img_input) // 1024)
                    header, b64_data = img_input.split(',', 1)
                    img_bytes = base64.b64decode(b64_data)
                    image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                    logger.info("[UI %s] Uploaded image size=%s", client_label, image.size)
                else:
                    logger.info("[UI %s] Loading image from path=%s", client_label, img_input)
                    image = load_image(img_input)

            # Store for music generation immediately (captcha token may arrive any time)
            self._music_prompt = prompt or ""
            self._music_image = image

            # Generate image with Gemini if prompt provided
            if prompt:
                image = await self._generate_image_with_gemini(prompt, image, websocket)

            image = self._resizecrop(image, self.res_height, self.res_width)
            logger.info("[UI %s] Input image normalized to %s", client_label, image.size)

            # Send image preview to client (when Gemini generated the image)
            if prompt:
                buf = io.BytesIO()
                image.save(buf, format='JPEG', quality=85)
                await self._send_ws(websocket, {
                    "type": "image_preview",
                    "data": base64.b64encode(buf.getvalue()).decode()
                })

            # Encode on primary device
            image_tensor = self.frame_process(image)[None, :, None, :, :].to(
                dtype=self.weight_dtype, device=self.device)

            max_output_frames = self.args.max_num_output_frames
            logger.info("[UI %s] VAE encoding input image (max_output_frames=%d)...", client_label, max_output_frames)
            vae_encode_start = time.perf_counter()
            padding_video = torch.zeros_like(image_tensor).repeat(1, 1, 4 * (max_output_frames - 1), 1, 1)
            img_cond = torch.concat([image_tensor, padding_video], dim=2)
            tiler_kwargs = {"tiled": True, "tile_size": [self.latent_h, self.latent_w], "tile_stride": [(self.latent_h + 1) // 2, (self.latent_w + 1) // 2]}
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
                [1, 16, max_output_frames, self.latent_h, self.latent_w],
                device=self.device, dtype=self.weight_dtype
            )

            num_frames = (max_output_frames - 1) * 4 + 1

            # Setup conditions (on primary device)
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

            # Copy tensors to each world's device
            for world in self.worlds:
                dev = world.device
                world.conditional_dict = {
                    "cond_concat": conditional_dict["cond_concat"].to(dev),
                    "visual_context": conditional_dict["visual_context"].to(dev),
                    "keyboard_cond": conditional_dict["keyboard_cond"].clone().to(dev),
                }
                if "mouse_cond" in conditional_dict:
                    world.conditional_dict["mouse_cond"] = conditional_dict["mouse_cond"].clone().to(dev)
                world.noise = sampled_noise.clone().to(dev)
                world.vae_cache = [None] * len(ZERO_VAE_CACHE)

            # Send ready signal first so frontend can set up the multi-world grid
            if not await self._send_ws(websocket, {
                "type": "ready",
                "mode": self.mode,
                "resolution": f"{self.res_width}x{self.res_height}",
                "num_worlds": self.num_worlds,
                "message": "Ready for input. Press keys to control."
            }):
                return

            # Send initial frame for each world (all start with the same image)
            init_frame = np.array(image.resize((self.res_width, self.res_height)))
            init_b64 = self._frame_to_base64(init_frame)
            for w_id in range(self.num_worlds):
                if not await self._send_ws(websocket, {
                    "type": "frame",
                    "data": init_b64,
                    "block": 0,
                    "world_id": w_id,
                }):
                    return
            logger.info("[UI %s] Ready; launching %d independent world loop(s)", client_label, self.num_worlds)

            # Launch N concurrent world tasks
            with torch.no_grad():
                tasks = [
                    asyncio.create_task(self._world_inference_loop(world, websocket))
                    for world in self.worlds
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error("[UI %s] World %d loop failed:", client_label, i, exc_info=result)

            # Send completion
            await self._send_ws(websocket, {"type": "complete"})

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

    async def _world_inference_loop(self, world: "WorldRunner", websocket: WebSocket):
        """Independent inference loop for a single world on its own GPU.

        Each world runs this concurrently. GPU work is dispatched via
        run_in_executor so worlds on different GPUs execute in true parallel.
        """
        pipeline = world.pipeline
        device = world.device
        w_id = world.world_id
        noise = world.noise

        batch_size = noise.shape[0]
        num_frames = noise.shape[2]
        num_frame_per_block = pipeline.num_frame_per_block

        num_blocks = num_frames // num_frame_per_block
        max_blocks = self.args.max_blocks or num_blocks
        total_blocks = min(max_blocks, num_blocks)
        client_label = _websocket_client_label(websocket)
        loop_started_at = time.perf_counter()

        logger.info(
            "[UI %s] World %d starting loop on %s (blocks=%d, denoise_steps=%s)",
            client_label, w_id, device, total_blocks, pipeline.denoising_step_list,
        )

        # Initialize KV caches on this world's device
        pipeline.kv_cache1 = None
        pipeline.kv_cache_keyboard = None
        pipeline.kv_cache_mouse = None
        pipeline.crossattn_cache = None
        pipeline._initialize_kv_cache(batch_size, noise.dtype, device)
        pipeline._initialize_kv_cache_mouse_and_keyboard(batch_size, noise.dtype, device)
        pipeline._initialize_crossattn_cache(batch_size, noise.dtype, device)

        current_start_frame = 0
        generated_blocks = 0

        for block_index in range(total_blocks):
            if self.should_stop:
                logger.info("[UI %s] World %d stop flag set before block %d", client_label, w_id, block_index + 1)
                break

            block_started_at = time.perf_counter()
            block_num = block_index + 1

            # Send request_action with world_id
            await self._send_ws(websocket, {
                "type": "request_action",
                "block": block_num,
                "mode": self.mode,
                "world_id": w_id,
            })

            # Wait for action from this world's queue
            try:
                action = await self._get_next_action(world_id=w_id)
                if action is None:
                    logger.info("[UI %s] World %d block %d received stop sentinel", client_label, w_id, block_num)
                    break
            except asyncio.TimeoutError:
                action = {"keyboard": "q", "mouse": "u"}
                logger.warning("[UI %s] World %d block %d action timeout; using neutral", client_label, w_id, block_num)

            # Run GPU work in thread executor (allows other worlds to proceed)
            loop = asyncio.get_event_loop()
            video_np, world.vae_cache, world.conditional_dict = await loop.run_in_executor(
                None, self._run_block_sync, world, current_start_frame, action)

            # Send frames
            for frame_idx, frame in enumerate(video_np):
                if not await self._send_ws(websocket, {
                    "type": "frame",
                    "data": self._frame_to_base64(frame),
                    "block": block_num,
                    "frame": frame_idx,
                    "world_id": w_id,
                }):
                    return
                await asyncio.sleep(0.01)

            current_start_frame += num_frame_per_block
            generated_blocks += 1

            # Caption every 3 blocks
            if block_num % 3 == 1:
                asyncio.create_task(self._generate_caption(video_np[-1], block_num, websocket, world_id=w_id))

            block_elapsed = time.perf_counter() - block_started_at
            logger.info("[UI %s] World %d block %d done (%.2fs)", client_label, w_id, block_num, block_elapsed)

            # Send block complete per world
            await self._send_ws(websocket, {
                "type": "block_complete",
                "block": block_num,
                "world_id": w_id,
            })

        total_elapsed = time.perf_counter() - loop_started_at
        logger.info(
            "[UI %s] World %d loop finished blocks=%d/%d elapsed=%.2fs",
            client_label, w_id, generated_blocks, total_blocks, total_elapsed,
        )

    def _run_block_sync(self, world: "WorldRunner", current_start_frame: int,
                        action: dict) -> tuple:
        """Synchronous GPU work for one block of one world.

        Called from thread pool via run_in_executor. Returns (video_np, vae_cache, conditional_dict).
        Note: torch.no_grad() is thread-local, so we must set it here (the caller's
        context manager doesn't carry over to executor threads).
        """
        with torch.no_grad():
            pipeline = world.pipeline
            device = world.device
            noise = world.noise
            conditional_dict = world.conditional_dict
            vae_cache = world.vae_cache

            num_frame_per_block = pipeline.num_frame_per_block
            batch_size = noise.shape[0]

            current_actions = self._parse_action(action, device=device)
            noisy_input = noise[:, :, current_start_frame:current_start_frame + num_frame_per_block]

            new_act, conditional_dict = cond_current(
                conditional_dict, current_start_frame, num_frame_per_block,
                replace=current_actions, mode=self.mode)

            # Denoising loop
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

            # Context pass
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

            return video_np, vae_cache, conditional_dict

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


# ── Modal Room helpers ──

def _get_modal_worker_class():
    """Lazily import and return the Modal worker class."""
    from modal_worker.worker import MatrixGameWorker
    return MatrixGameWorker


async def _modal_broadcast_player_list():
    """Send updated player list to Modal room host."""
    global modal_room_state
    mrs = modal_room_state
    if mrs is None:
        return
    players_info = []
    for pid, ps in mrs.players.items():
        players_info.append({
            "player_id": pid,
            "name": ps.name,
            "status": ps.status,
        })
    queued_info = [{"player_id": pid, "name": mrs.players[pid].name}
                   for pid in mrs.queue if pid in mrs.players]
    msg = {
        "type": "modal_room_update",
        "active_count": mrs.active_count,
        "max_active": mrs.max_active,
        "players": players_info,
        "queue": queued_info,
    }
    try:
        await mrs.host_ws.send_json(msg)
    except Exception:
        pass


async def _modal_activate_next_queued():
    """Activate the next queued player if there's capacity."""
    global modal_room_state
    mrs = modal_room_state
    if mrs is None or not mrs.is_active:
        return
    while mrs.queue and mrs.active_count < mrs.max_active:
        next_pid = mrs.queue.pop(0)
        ps = mrs.players.get(next_pid)
        if ps is None or ps.status == "disconnected":
            continue
        # Spawn worker for this queued player
        asyncio.create_task(_modal_spawn_player(mrs, ps))
        break


async def _modal_spawn_player(mrs: "ModalRoomState", ps: "ModalPlayerState"):
    """Spawn a Modal worker and start inference loop for a player."""
    ps.status = "initializing"
    mrs.active_count += 1
    await _modal_broadcast_player_list()

    try:
        await ps.websocket.send_json({
            "type": "initializing",
            "message": "Loading your world...",
        })
    except Exception:
        ps.status = "disconnected"
        mrs.active_count -= 1
        await _modal_activate_next_queued()
        return

    try:
        WorkerCls = _get_modal_worker_class()
        worker = WorkerCls()
        ps.worker = worker

        # Init session on Modal (runs on remote GPU)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: worker.init_session.remote(
                image_b64=mrs.seed_image_b64, mode=mrs.mode
            )
        )

        ps.status = "active"
        await _modal_broadcast_player_list()

        # Send initial frame to player
        await ps.websocket.send_json({
            "type": "ready",
            "mode": result["mode"],
            "resolution": result["resolution"],
        })
        await ps.websocket.send_json({
            "type": "frame",
            "data": result["initial_frame_b64"],
            "block": 0,
        })

        # Send initial frame to host grid
        try:
            await mrs.host_ws.send_json({
                "type": "modal_player_frame",
                "player_id": ps.player_id,
                "player_name": ps.name,
                "data": result["initial_frame_b64"],
                "block": 0,
            })
        except Exception:
            pass

        # Send music if available
        if mrs.music_urls:
            try:
                await ps.websocket.send_json({
                    "type": "music_ready",
                    "audio_urls": mrs.music_urls,
                })
            except Exception:
                pass

        # Start inference loop
        ps.inference_task = asyncio.create_task(
            _modal_player_inference_loop(mrs, ps.player_id)
        )

    except Exception as e:
        logger.exception("[ModalRoom] Failed to spawn worker for %s", ps.name)
        ps.status = "disconnected"
        mrs.active_count -= 1
        try:
            await ps.websocket.send_json({
                "type": "error",
                "message": f"Failed to start your world: {str(e)[:100]}",
            })
        except Exception:
            pass
        await _modal_activate_next_queued()


async def _modal_player_inference_loop(mrs: "ModalRoomState", player_id: str):
    """Run inference blocks for one player until disconnect or room close."""
    ps = mrs.players.get(player_id)
    if ps is None:
        return

    worker = ps.worker
    max_blocks = 999999  # effectively unlimited
    block_num = 0
    loop = asyncio.get_event_loop()

    try:
        for _ in range(max_blocks):
            if not mrs.is_active or ps.status == "disconnected":
                break

            block_num += 1

            # Request action from player
            try:
                await ps.websocket.send_json({
                    "type": "request_action",
                    "block": block_num,
                })
            except Exception:
                break

            # Wait for action (timeout → neutral)
            try:
                action = await asyncio.wait_for(ps.action_queue.get(), timeout=120.0)
                if action is None:  # sentinel for disconnect
                    break
            except asyncio.TimeoutError:
                action = {"keyboard": "q", "mouse": "u"}

            # Run block on Modal worker
            try:
                result = await loop.run_in_executor(
                    None, lambda a=action: worker.run_block.remote(action=a)
                )
            except Exception as e:
                logger.warning("[ModalRoom] run_block failed for %s: %s", ps.name, e)
                break

            # Send frames to player
            for frame_idx, frame_b64 in enumerate(result["frames_b64"]):
                try:
                    await ps.websocket.send_json({
                        "type": "frame",
                        "data": frame_b64,
                        "block": block_num,
                        "frame": frame_idx,
                    })
                except Exception:
                    break
                await asyncio.sleep(0.01)

            # Send last frame to host grid
            try:
                await mrs.host_ws.send_json({
                    "type": "modal_player_frame",
                    "player_id": ps.player_id,
                    "player_name": ps.name,
                    "data": result["last_frame_b64"],
                    "block": block_num,
                })
            except Exception:
                pass

            # Caption every 3 blocks
            if block_num % 3 == 1:
                asyncio.create_task(
                    _modal_generate_caption(mrs, ps, result["last_frame_b64"], block_num)
                )

    except asyncio.CancelledError:
        logger.info("[ModalRoom] Inference loop cancelled for %s", ps.name)
    except Exception as e:
        logger.exception("[ModalRoom] Inference loop error for %s", ps.name)
    finally:
        # Cleanup — end session on Modal worker
        try:
            await loop.run_in_executor(None, lambda: worker.end_session.remote())
        except Exception:
            pass
        ps.status = "disconnected"
        # mrs may have been set to None if room was closed
        if mrs.is_active:
            mrs.active_count = max(0, mrs.active_count - 1)
            logger.info("[ModalRoom] Player %s loop ended (block=%d)", ps.name, block_num)
            await _modal_broadcast_player_list()
            await _modal_activate_next_queued()
        else:
            logger.info("[ModalRoom] Player %s loop ended (room closed, block=%d)", ps.name, block_num)


async def _modal_generate_caption(mrs: "ModalRoomState", ps: "ModalPlayerState",
                                   frame_b64: str, block_num: int):
    """Generate caption for a Modal player's frame via Claude API."""
    try:
        from suno_api.client import DEFAULT_ANTHROPIC_KEY

        def _call_claude():
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
                                "data": frame_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "You are a narrator for a real-time interactive world exploration game. "
                                "Respond with EXACTLY two lines, nothing else:\n"
                                "Line 1: A hex color code (e.g. #FF6B35) that will be maximally visible "
                                "as caption text overlaid on this image.\n"
                                "Line 2: A single short, atmospheric sentence (max 15 words). "
                                "Be poetic and immersive. No quotes, no labels, just the narration."
                            ),
                        },
                    ],
                }],
            )
            return msg.content[0].text.strip()

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _call_claude)

        lines = result.strip().split("\n", 1)
        color = "#e0e0e0"
        caption_text = result
        if len(lines) == 2 and lines[0].startswith("#") and len(lines[0]) == 7:
            color = lines[0].strip()
            caption_text = lines[1].strip()

        # Send to player
        try:
            await ps.websocket.send_json({
                "type": "caption",
                "block": block_num,
                "text": caption_text,
                "color": color,
            })
        except Exception:
            pass

        # Send to host
        try:
            await mrs.host_ws.send_json({
                "type": "modal_player_caption",
                "player_id": ps.player_id,
                "player_name": ps.name,
                "block": block_num,
                "text": caption_text,
                "color": color,
            })
        except Exception:
            pass

    except Exception as e:
        logger.warning("[ModalRoom] Caption failed for %s block %d: %s", ps.name, block_num, e)


async def _close_modal_room():
    """Close the Modal room: end all workers, notify all players."""
    global modal_room_state
    mrs = modal_room_state
    if mrs is None:
        return
    mrs.is_active = False

    # Stop all players
    for pid, ps in list(mrs.players.items()):
        # Send sentinel to action queue
        try:
            ps.action_queue.put_nowait(None)
        except Exception:
            pass
        # Cancel inference task
        if ps.inference_task and not ps.inference_task.done():
            ps.inference_task.cancel()
        # Notify player
        try:
            await ps.websocket.send_json({"type": "room_closed"})
            await ps.websocket.close()
        except Exception:
            pass

    # Notify host
    try:
        await mrs.host_ws.send_json({"type": "modal_room_closed"})
    except Exception:
        pass

    logger.info("[ModalRoom] Room %s closed (%d players)", mrs.room_id, len(mrs.players))
    modal_room_state = None


# ── Modal Room endpoints ──

@app.get("/modal-room/{room_id}", response_class=HTMLResponse)
async def get_modal_room(room_id: str):
    """Serve the mobile page for a Modal room player."""
    global modal_room_state
    if modal_room_state is None or modal_room_state.room_id != room_id or not modal_room_state.is_active:
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse("Room not found", status_code=404)
    return FileResponse("web_ui/remote.html")


@app.websocket("/ws/modal-room/{room_id}")
async def modal_room_websocket(websocket: WebSocket, room_id: str):
    """WebSocket endpoint for Modal room players (per-player GPU worlds)."""
    global modal_room_state
    await websocket.accept()

    mrs = modal_room_state
    if mrs is None or mrs.room_id != room_id or not mrs.is_active:
        await websocket.close(code=4004, reason="Room not found")
        return

    # Wait for join message
    player_name = "Player"
    try:
        first = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
        if first.get("type") == "join":
            player_name = (first.get("name") or "Player")[:16]
    except Exception:
        await websocket.close(code=4000, reason="Join timeout")
        return

    # Create player state
    player_id = f"p_{_generate_room_id(6)}"
    ps = ModalPlayerState(
        player_id=player_id,
        name=player_name,
        websocket=websocket,
        worker=None,
        status="queued",
    )
    mrs.players[player_id] = ps

    logger.info("[ModalRoom] %s (%s) joined room %s", player_name, player_id, room_id)

    # Check capacity
    if mrs.active_count < mrs.max_active:
        # Spawn immediately
        asyncio.create_task(_modal_spawn_player(mrs, ps))
    else:
        # Add to queue
        mrs.queue.append(player_id)
        position = len(mrs.queue)
        await websocket.send_json({
            "type": "queued",
            "position": position,
            "message": f"You are #{position} in queue. Please wait...",
        })
        logger.info("[ModalRoom] %s queued at position %d", player_name, position)

    await _modal_broadcast_player_list()

    # Main message loop
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "action":
                # Player sending an action for their world
                await ps.action_queue.put({
                    "keyboard": data.get("keyboard", "q"),
                    "mouse": data.get("mouse", "u"),
                })
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        ps.status = "disconnected"
        # Send sentinel to wake up inference loop
        try:
            ps.action_queue.put_nowait(None)
        except Exception:
            pass
        # Remove from queue if still queued
        if player_id in mrs.queue:
            mrs.queue.remove(player_id)
        logger.info("[ModalRoom] %s (%s) disconnected", player_name, player_id)
        await _modal_broadcast_player_list()


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

        # World selection for multi-world mode
        num_worlds = inference_engine.num_worlds if inference_engine else 1
        player_world_id = 0
        if num_worlds > 1:
            await websocket.send_json({"type": "select_world", "num_worlds": num_worlds})
            try:
                sel_msg = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
                if sel_msg.get("type") == "select_world":
                    chosen = sel_msg.get("world_id", 0)
                    if 0 <= chosen < num_worlds:
                        player_world_id = chosen
            except asyncio.TimeoutError:
                pass
            rs.player_worlds[websocket] = player_world_id
            logger.info("[Room] %s selected world %d", player_name, player_world_id)

        while True:
            data = await websocket.receive_json()
            if data.get("type") == "vote":
                if rs.is_active:
                    kb = data.get("keyboard", "q")
                    ms = data.get("mouse", "u")
                    w_id = rs.player_worlds.get(websocket, 0)
                    rs.votes.setdefault(w_id, []).append({"keyboard": kb, "mouse": ms})
                    rs.vote_event.set()
                    await _send_action_feed(player_name, kb, ms)
                    logger.info("[Room] Vote from %s (world %d): keyboard=%s mouse=%s (buffered=%d)",
                                player_name, w_id, kb, ms, len(rs.votes.get(w_id, [])))
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        rs.player_websockets.discard(websocket)
        rs.player_names.pop(websocket, None)
        rs.player_worlds.pop(websocket, None)
        logger.info("[Room] %s left room %s (player_count=%d)", player_name, room_id, len(rs.player_websockets))
        if rs.is_active:
            await _broadcast_player_count()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication."""
    global room_state
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
                        # Room mode: host action is a vote for world 0
                        kb = data.get("keyboard", "q")
                        ms = data.get("mouse", "u")
                        room_state.votes.setdefault(0, []).append({"keyboard": kb, "mouse": ms})
                        room_state.vote_event.set()
                        await _send_action_feed("Host", kb, ms)
                    else:
                        # Solo mode: direct to world 0's action_queue
                        await inference_engine.worlds[0].action_queue.put(data)
                        logger.info(
                            "[WS %s] Queued action (%s) queue_size=%d",
                            client_label,
                            _summarize_action_payload(data),
                            inference_engine.worlds[0].action_queue.qsize(),
                        )
                else:
                    logger.warning("[WS %s] Dropped action: inference is not running", client_label)

            elif msg_type == "open_room":
                if room_state is not None:
                    await websocket.send_json({"type": "error", "message": "Room already open"})
                else:
                    rid = _generate_room_id()
                    room_state = RoomState(room_id=rid, host_ws=websocket)
                    # Unblock _get_next_action if it's waiting on action_queue in solo mode
                    if inference_engine and inference_engine.is_running:
                        await inference_engine.worlds[0].action_queue.put({"_room_transition": True})
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

            elif msg_type == "open_modal_room":
                if modal_room_state is not None:
                    await websocket.send_json({"type": "error", "message": "Modal room already open"})
                else:
                    # Requires seed image (base64, no data: prefix)
                    seed_image_b64 = data.get("seed_image_b64", "")
                    mode = data.get("mode", "universal")

                    if not seed_image_b64:
                        # If data: prefix provided, strip it
                        raw = data.get("image", "")
                        if raw and "," in raw:
                            seed_image_b64 = raw.split(",", 1)[1]
                        elif raw and os.path.isfile(raw):
                            # Server-side image path — load and encode
                            import base64 as b64mod
                            with open(raw, "rb") as f:
                                seed_image_b64 = b64mod.b64encode(f.read()).decode()
                            logger.info("[WS %s] Loaded seed image from path: %s (%d KB)",
                                        client_label, raw, len(seed_image_b64) // 1024)
                        elif raw:
                            seed_image_b64 = raw

                    if not seed_image_b64:
                        await websocket.send_json({"type": "error", "message": "Seed image required for Modal room"})
                    else:
                        rid = _generate_room_id()
                        modal_room_state = ModalRoomState(
                            room_id=rid,
                            host_ws=websocket,
                            seed_image_b64=seed_image_b64,
                            mode=mode,
                        )
                        lan_ip = _get_lan_ip()
                        url = f"http://{lan_ip}:{server_port}/modal-room/{rid}"
                        logger.info("[WS %s] Modal room opened: %s -> %s", client_label, rid, url)
                        await websocket.send_json({
                            "type": "modal_room_opened",
                            "room_id": rid,
                            "url": url,
                        })

            elif msg_type == "close_modal_room":
                if modal_room_state and modal_room_state.host_ws == websocket:
                    logger.info("[WS %s] Close modal room requested", client_label)
                    await _close_modal_room()

            elif msg_type == "stop":
                if inference_engine:
                    logger.info("[WS %s] Stop requested by client", client_label)
                    if room_state and room_state.host_ws == websocket:
                        await _close_room()
                    inference_engine.stop()
                    for w in inference_engine.worlds:
                        await w.action_queue.put(None)
                    await websocket.send_json({"type": "stopped"})

            else:
                logger.warning("[WS %s] Unknown message type=%s", client_label, msg_type)

    except WebSocketDisconnect:
        logger.info("[WS %s] Disconnected", client_label)
        if room_state and room_state.host_ws == websocket:
            await _close_room()
        if modal_room_state and modal_room_state.host_ws == websocket:
            await _close_modal_room()
        if inference_engine:
            inference_engine.stop()
            for w in inference_engine.worlds:
                await w.action_queue.put(None)
    except Exception as e:
        logger.exception("[WS %s] WebSocket error", client_label)
        if room_state and room_state.host_ws == websocket:
            await _close_room()
        if modal_room_state and modal_room_state.host_ws == websocket:
            await _close_modal_room()
        if inference_engine:
            inference_engine.stop()
            for w in inference_engine.worlds:
                await w.action_queue.put(None)


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
    parser.add_argument("--resolution", type=str, default="640x352",
                        help="Output resolution WxH. Supported: 640x352 (default), 832x480, 1280x720, 480x832, 720x1280")
    parser.add_argument("--num-frames", type=int, default=None,
                        help="Frames per generation block (default: from config, typically 3). Higher = more temporal context per block")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile for the VAE decoder (lower startup overhead, potentially lower throughput)")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Skip kernel warmup (kernels will compile lazily on first inference)")
    parser.add_argument("--true-no-compile", action="store_true",
                        help="Disable ALL torch.compile including flex_attention. Implies --no-compile and --no-warmup. "
                             "Near-instant startup but significantly slower inference (~2-4x)")
    parser.add_argument("--num-worlds", type=int, default=1,
                        help="Number of parallel worlds. Each world runs independently on its own GPU.")
    parser.add_argument("--suno-model", type=str, default="chirp-auk-turbo",
                        choices=["chirp-auk-turbo", "chirp-crow"],
                        help="Suno music model: chirp-auk-turbo (v4.5, default) or chirp-crow (v5)")
    args = parser.parse_args()

    # Parse resolution
    try:
        w, h = args.resolution.split('x')
        args.res_width, args.res_height = int(w), int(h)
        if args.res_width % 8 != 0 or args.res_height % 8 != 0:
            parser.error(f"Resolution {args.resolution} must be divisible by 8 (VAE stride)")
        args.latent_w, args.latent_h = args.res_width // 8, args.res_height // 8
    except ValueError:
        parser.error(f"Invalid resolution format: {args.resolution}. Use WxH (e.g. 832x480)")

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
