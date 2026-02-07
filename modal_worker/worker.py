"""
Modal stateful worker for Ephemeral.

Each instance holds a full model + KV caches on a single GPU,
serving one player's world.

Usage:
    modal run modal_worker/worker.py  # standalone test
"""

import sys
import os
import io
import base64
import copy
import time

import modal

# Import app and volume from app.py
from modal_worker.app import app, weights_volume, image

WEIGHTS_PATH = "/weights"
CONFIG_PATH = "/app/configs/inference_yaml/inference_universal.yaml"
RESOLUTION = (640, 352)  # (width, height)


@app.cls(
    gpu="H200",
    timeout=1800,
    volumes={WEIGHTS_PATH: weights_volume},
    scaledown_window=1800,  # 30 min — pre-init pool workers must stay alive
)
@modal.concurrent(max_inputs=1)
class EphemeralWorker:
    """One-player-per-GPU stateful inference worker.

    Each unique session_id gets its own dedicated container pool,
    ensuring all .remote() calls for a session hit the same container.
    Without this, Modal routes calls to ANY container in a shared pool,
    which breaks stateful KV-cache inference across blocks.
    """

    session_id: str = modal.parameter()

    @modal.enter()
    def load_model(self):
        """Load model weights once when the container starts."""
        sys.path.insert(0, "/app")
        os.chdir("/app")  # Ensure relative paths in configs resolve correctly

        import torch
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = 64

        import numpy as np
        from omegaconf import OmegaConf
        from torchvision.transforms import v2
        from safetensors.torch import load_file
        from PIL import Image

        from demo_utils.constant import ZERO_VAE_CACHE
        from demo_utils.vae_block3 import VAEDecoderWrapper
        from pipeline import CausalInferenceStreamingPipeline
        from wan.vae.wanx_vae import get_wanx_vae_wrapper
        from utils.wan_wrapper import WanDiffusionWrapper

        self.device = torch.device("cuda:0")
        self.weight_dtype = torch.bfloat16

        res_w, res_h = RESOLUTION
        self.res_width = res_w
        self.res_height = res_h
        self.latent_w = res_w // 8
        self.latent_h = res_h // 8

        # Load config
        config = OmegaConf.load(CONFIG_PATH)
        self.mode = config.get("mode", "universal")

        # Override latent dims in config shape
        shape = list(config.image_or_video_shape)
        shape[3] = self.latent_h
        shape[4] = self.latent_w
        config.image_or_video_shape = shape

        self.max_output_frames = 121  # 30 blocks × 4 frames/block + 1
        self.max_blocks = 999999

        # Parallel load weights
        import concurrent.futures

        def _load_vae_decoder_weights():
            vae_path = os.path.join(WEIGHTS_PATH, "Wan2.1_VAE.pth")
            vae_sd = torch.load(vae_path, map_location="cpu", mmap=True)
            return {k: v for k, v in vae_sd.items() if "decoder." in k or "conv2" in k}

        def _load_main_checkpoint():
            ckpt_path = os.path.join(WEIGHTS_PATH, "base_distill.safetensors")
            return load_file(ckpt_path)

        def _load_vae_encoder():
            vae = get_wanx_vae_wrapper(WEIGHTS_PATH, torch.float16)
            vae.requires_grad_(False)
            vae.eval()
            return vae

        t0 = time.perf_counter()
        print("[worker] Loading weights (VAE decoder, main checkpoint, VAE encoder) in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            f_dec = executor.submit(_load_vae_decoder_weights)
            f_ckpt = executor.submit(_load_main_checkpoint)
            f_enc = executor.submit(_load_vae_encoder)
            decoder_sd = f_dec.result()
            main_ckpt = f_ckpt.result()
            vae_encoder = f_enc.result()
        print(f"[worker] Weights loaded in {time.perf_counter() - t0:.1f}s")

        # Build pipeline
        t1 = time.perf_counter()
        print("[worker] Building pipeline (WanDiffusionWrapper + VAEDecoder + CausalInferencePipeline)...")
        generator = WanDiffusionWrapper(
            **getattr(config, "model_kwargs", {}), is_causal=True
        )
        vae_decoder = VAEDecoderWrapper()
        vae_decoder.load_state_dict(decoder_sd)
        vae_decoder.to(self.device, torch.float16)
        vae_decoder.requires_grad_(False)
        vae_decoder.eval()
        pipeline = CausalInferenceStreamingPipeline(
            config, generator=generator, vae_decoder=vae_decoder
        )
        pipeline.generator.load_state_dict(main_ckpt)
        pipeline = pipeline.to(device=self.device, dtype=self.weight_dtype)
        pipeline.vae_decoder.to(torch.float16)

        self.pipeline = pipeline
        self.config = config
        self.vae = vae_encoder.to(self.device, self.weight_dtype)
        print(f"[worker] Pipeline built and moved to GPU in {time.perf_counter() - t1:.1f}s")
        self.ZERO_VAE_CACHE = ZERO_VAE_CACHE

        self.frame_process = v2.Compose([
            v2.Resize(size=(self.res_height, self.res_width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # Per-session state (set in init_session)
        self.conditional_dict = None
        self.vae_cache = None
        self.noise = None
        self.current_start_frame = 0
        self.block_index = 0

        print(f"[worker] Model loaded on {self.device}, mode={self.mode}")

    @modal.method()
    def init_session(self, image_b64: str, mode: str = "universal") -> dict:
        """
        Initialize a new game session with a seed image.

        Args:
            image_b64: Base64-encoded JPEG/PNG image (no data: prefix).
            mode: Game mode (universal, gta_drive, templerun).

        Returns:
            {"initial_frame_b64": str, "mode": str, "resolution": str}
        """
        import torch
        import numpy as np
        from PIL import Image
        from einops import rearrange
        from utils.conditions import (
            Bench_actions_universal, Bench_actions_gta_drive, Bench_actions_templerun,
        )

        t0 = time.perf_counter()
        self.mode = mode
        print(f"[worker] init_session starting (mode={mode}, image={len(image_b64)//1024}KB)")

        # Decode image
        t_step = time.perf_counter()
        img_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image = self._resizecrop(image, self.res_height, self.res_width)
        print(f"[worker]   image decoded + resized: {time.perf_counter() - t_step:.2f}s")

        # VAE encode
        t_step = time.perf_counter()
        image_tensor = self.frame_process(image)[None, :, None, :, :].to(
            dtype=self.weight_dtype, device=self.device
        )

        max_out = self.max_output_frames
        with torch.no_grad():
            padding = torch.zeros_like(image_tensor).repeat(1, 1, 4 * (max_out - 1), 1, 1)
            img_cond = torch.concat([image_tensor, padding], dim=2)
            tiler_kwargs = {
                "tiled": True,
                "tile_size": [self.latent_h, self.latent_w],
                "tile_stride": [(self.latent_h + 1) // 2, (self.latent_w + 1) // 2],
            }
            img_cond = self.vae.encode(img_cond, device=self.device, **tiler_kwargs).to(self.device)
        print(f"[worker]   VAE encode: {time.perf_counter() - t_step:.2f}s")

        t_step = time.perf_counter()
        with torch.no_grad():
            mask_cond = torch.ones_like(img_cond)
            mask_cond[:, :, 1:] = 0
            cond_concat = torch.cat([mask_cond[:, :4], img_cond], dim=1)
            visual_context = self.vae.clip.encode_video(image_tensor)

            noise = torch.randn(
                [1, 16, max_out, self.latent_h, self.latent_w],
                device=self.device, dtype=self.weight_dtype,
            )
        print(f"[worker]   CLIP encode + noise: {time.perf_counter() - t_step:.2f}s")

        num_frames = (max_out - 1) * 4 + 1

        # Build conditions
        t_step = time.perf_counter()
        conditional_dict = {
            "cond_concat": cond_concat.to(device=self.device, dtype=self.weight_dtype),
            "visual_context": visual_context.to(device=self.device, dtype=self.weight_dtype),
        }

        if mode == "universal":
            cond_data = Bench_actions_universal(num_frames)
            conditional_dict["mouse_cond"] = (
                cond_data["mouse_condition"].unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)
            )
        elif mode == "gta_drive":
            cond_data = Bench_actions_gta_drive(num_frames)
            conditional_dict["mouse_cond"] = (
                cond_data["mouse_condition"].unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)
            )
        else:
            cond_data = Bench_actions_templerun(num_frames)

        conditional_dict["keyboard_cond"] = (
            cond_data["keyboard_condition"].unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)
        )
        print(f"[worker]   conditions built: {time.perf_counter() - t_step:.2f}s")

        # Initialize KV caches
        t_step = time.perf_counter()
        pipeline = self.pipeline
        pipeline._initialize_kv_cache(1, self.weight_dtype, self.device)
        pipeline._initialize_kv_cache_mouse_and_keyboard(1, self.weight_dtype, self.device)
        pipeline._initialize_crossattn_cache(1, self.weight_dtype, self.device)
        print(f"[worker]   KV caches initialized: {time.perf_counter() - t_step:.2f}s")

        # Store session state
        self.conditional_dict = conditional_dict
        self.noise = noise
        self.vae_cache = [None] * len(self.ZERO_VAE_CACHE)
        self.current_start_frame = 0
        self.block_index = 0

        # Build initial frame b64
        init_frame = np.array(image.resize((self.res_width, self.res_height)))
        init_b64 = self._frame_to_base64(init_frame)

        elapsed = time.perf_counter() - t0
        print(f"[worker] init_session complete: {elapsed:.1f}s total (mode={mode}, res={self.res_width}x{self.res_height})")

        return {
            "initial_frame_b64": init_b64,
            "mode": mode,
            "resolution": f"{self.res_width}x{self.res_height}",
        }

    @modal.method()
    def run_block(self, action: dict) -> dict:
        """
        Run one inference block (denoise + context pass + VAE decode).

        Args:
            action: {"keyboard": "w", "mouse": "u"} — same format as web_server.

        Returns:
            {"frames_b64": [str, ...], "block_index": int, "last_frame_b64": str}
        """
        import torch
        import numpy as np
        from einops import rearrange
        from pipeline.causal_inference import cond_current

        if self.noise is None:
            raise RuntimeError("run_block called before init_session")

        pipeline = self.pipeline
        device = self.device
        noise = self.noise
        conditional_dict = self.conditional_dict
        vae_cache = self.vae_cache
        current_start = self.current_start_frame
        num_frame_per_block = pipeline.num_frame_per_block

        # Bounds check: ensure we have enough frames left in noise/cond tensors
        if current_start + num_frame_per_block > noise.shape[2]:
            max_blocks = noise.shape[2] // num_frame_per_block
            print(f"[worker] session limit reached: block {self.block_index + 1} exceeds max {max_blocks}")
            return {"session_limit": True, "max_blocks": max_blocks}

        t0 = time.perf_counter()

        # Parse action to tensors
        current_actions = self._parse_action(action, device)
        noisy_input = noise[:, :, current_start:current_start + num_frame_per_block]

        with torch.no_grad():
            new_act, conditional_dict = cond_current(
                conditional_dict, current_start, num_frame_per_block,
                replace=current_actions, mode=self.mode,
            )

            # Denoising loop
            for index, current_timestep in enumerate(pipeline.denoising_step_list):
                timestep = torch.ones(
                    [1, num_frame_per_block], device=device, dtype=torch.int64
                ) * current_timestep

                if index < len(pipeline.denoising_step_list) - 1:
                    _, denoised_pred = pipeline.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=new_act,
                        timestep=timestep,
                        kv_cache=pipeline.kv_cache1,
                        kv_cache_mouse=pipeline.kv_cache_mouse,
                        kv_cache_keyboard=pipeline.kv_cache_keyboard,
                        crossattn_cache=pipeline.crossattn_cache,
                        current_start=current_start * pipeline.frame_seq_length,
                    )
                    next_timestep = pipeline.denoising_step_list[index + 1]
                    noisy_input = pipeline.scheduler.add_noise(
                        rearrange(denoised_pred, "b c f h w -> (b f) c h w"),
                        torch.randn_like(rearrange(denoised_pred, "b c f h w -> (b f) c h w")),
                        next_timestep * torch.ones(
                            [1 * num_frame_per_block], device=device, dtype=torch.long
                        ),
                    )
                    noisy_input = rearrange(noisy_input, "(b f) c h w -> b c f h w", b=1)
                else:
                    _, denoised_pred = pipeline.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=new_act,
                        timestep=timestep,
                        kv_cache=pipeline.kv_cache1,
                        kv_cache_mouse=pipeline.kv_cache_mouse,
                        kv_cache_keyboard=pipeline.kv_cache_keyboard,
                        crossattn_cache=pipeline.crossattn_cache,
                        current_start=current_start * pipeline.frame_seq_length,
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
                current_start=current_start * pipeline.frame_seq_length,
            )

            # VAE decode
            denoised_pred_t = denoised_pred.transpose(1, 2)
            video, vae_cache = pipeline.vae_decoder(denoised_pred_t.half(), *vae_cache)

            video_np = rearrange(video, "B T C H W -> B T H W C")
            video_np = ((video_np.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)[0]

        # Update session state
        self.conditional_dict = conditional_dict
        self.vae_cache = vae_cache
        self.current_start_frame = current_start + num_frame_per_block
        self.block_index += 1

        # Encode frames
        frames_b64 = [self._frame_to_base64(frame) for frame in video_np]
        last_frame_b64 = frames_b64[-1]

        elapsed = time.perf_counter() - t0
        max_blocks = noise.shape[2] // num_frame_per_block
        print(f"[worker] block {self.block_index}/{max_blocks} done in {elapsed:.1f}s "
              f"(start_frame={current_start}, action={action})")

        return {
            "frames_b64": frames_b64,
            "block_index": self.block_index,
            "last_frame_b64": last_frame_b64,
        }

    @modal.method()
    def end_session(self):
        """Clear session state and free GPU memory."""
        import torch

        self.conditional_dict = None
        self.vae_cache = None
        self.noise = None
        self.current_start_frame = 0
        self.block_index = 0

        # Clear pipeline caches
        self.pipeline.kv_cache1 = None
        self.pipeline.kv_cache_mouse = None
        self.pipeline.kv_cache_keyboard = None
        self.pipeline.crossattn_cache = None
        torch.cuda.empty_cache()
        print("[worker] Session ended, GPU memory freed")

    @modal.method()
    def health_check(self) -> dict:
        """Simple health check to verify the worker is alive."""
        import torch
        return {
            "status": "ok",
            "device": str(self.device),
            "mode": self.mode,
            "block_index": self.block_index,
            "gpu_memory_allocated_mb": torch.cuda.memory_allocated() // (1024 * 1024),
        }

    # ── Helpers ──

    def _resizecrop(self, image, th, tw):
        from PIL import Image as PILImage
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

    def _frame_to_base64(self, frame) -> str:
        import numpy as np
        from PIL import Image as PILImage
        img = PILImage.fromarray(frame.astype(np.uint8))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _parse_action(self, action_data: dict, device) -> dict:
        import torch

        CAM_VALUE = 0.1
        keyboard_key = action_data.get("keyboard", "q").lower()
        mouse_key = action_data.get("mouse", "u").lower()

        if self.mode == "universal":
            CAMERA_VALUE_MAP = {
                "i": [CAM_VALUE, 0], "k": [-CAM_VALUE, 0],
                "j": [0, -CAM_VALUE], "l": [0, CAM_VALUE], "u": [0, 0],
            }
            KEYBOARD_IDX = {
                "w": [1, 0, 0, 0], "s": [0, 1, 0, 0],
                "a": [0, 0, 1, 0], "d": [0, 0, 0, 1], "q": [0, 0, 0, 0],
            }
            mouse_cond = torch.tensor(CAMERA_VALUE_MAP.get(mouse_key, [0, 0]), device=device)
            keyboard_cond = torch.tensor(KEYBOARD_IDX.get(keyboard_key, [0, 0, 0, 0]), device=device)
        elif self.mode == "gta_drive":
            CAMERA_VALUE_MAP = {
                "a": [0, -CAM_VALUE], "d": [0, CAM_VALUE], "q": [0, 0],
            }
            KEYBOARD_IDX = {"w": [1, 0], "s": [0, 1], "q": [0, 0]}
            mouse_cond = torch.tensor(CAMERA_VALUE_MAP.get(keyboard_key, [0, 0]), device=device)
            keyboard_cond = torch.tensor(KEYBOARD_IDX.get(keyboard_key, [0, 0]), device=device)
        else:  # templerun
            KEYBOARD_IDX = {
                "w": [0, 1, 0, 0, 0, 0, 0], "s": [0, 0, 1, 0, 0, 0, 0],
                "a": [0, 0, 0, 0, 0, 1, 0], "d": [0, 0, 0, 0, 0, 0, 1],
                "z": [0, 0, 0, 1, 0, 0, 0], "c": [0, 0, 0, 0, 1, 0, 0],
                "q": [1, 0, 0, 0, 0, 0, 0],
            }
            keyboard_cond = torch.tensor(
                KEYBOARD_IDX.get(keyboard_key, [1, 0, 0, 0, 0, 0, 0]), device=device
            )
            mouse_cond = None

        result = {"keyboard": keyboard_cond}
        if mouse_cond is not None:
            result["mouse"] = mouse_cond
        return result


# ── Standalone test entrypoint ──

@app.local_entrypoint()
def main():
    """Test the worker: init_session + 3x run_block + end_session."""
    import base64

    # Create a simple test image (solid blue 640x352)
    from PIL import Image as PILImage

    img = PILImage.new("RGB", (640, 352), color=(50, 100, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    import uuid
    worker = EphemeralWorker(session_id=str(uuid.uuid4()))

    print("=== init_session ===")
    result = worker.init_session.remote(image_b64=img_b64, mode="universal")
    print(f"  mode={result['mode']}, resolution={result['resolution']}")
    print(f"  initial_frame_b64: {len(result['initial_frame_b64'])} chars")

    for i in range(3):
        actions = [
            {"keyboard": "w", "mouse": "u"},
            {"keyboard": "d", "mouse": "j"},
            {"keyboard": "q", "mouse": "l"},
        ]
        print(f"\n=== run_block {i + 1} (action: {actions[i]}) ===")
        result = worker.run_block.remote(action=actions[i])
        print(f"  block_index={result['block_index']}, frames={len(result['frames_b64'])}")

        # Save last frame as JPEG for verification
        frame_bytes = base64.b64decode(result["last_frame_b64"])
        fname = f"/tmp/ephemeral_test_block_{i + 1}.jpg"
        with open(fname, "wb") as f:
            f.write(frame_bytes)
        print(f"  Saved: {fname}")

    print("\n=== end_session ===")
    worker.end_session.remote()
    print("Done!")
