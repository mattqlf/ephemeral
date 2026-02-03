"""
Web UI server for LingBot-World with WebSocket chunk-based streaming.

This server wraps LingBot-World's WanI2V model for interactive camera-controlled
world generation. It generates video in 17-frame chunks with user-directed camera
movement between chunks.

Launch via: python launch_lingbot.py --ckpt_dir <path> --port 8889

Architecture:
- Uses FSDP + Ulysses sequence parallelism across 4 GPUs
- Only rank 0 runs the FastAPI server and WebSocket handling
- All ranks participate in model inference via distributed collectives
- Chunk-based: generate 17 frames, stream them, wait for next camera direction
"""

import asyncio
import argparse
import base64
import gc
import io
import logging
import math
import os
import sys
import types
from contextlib import contextmanager
from functools import partial
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF
from PIL import Image
from einops import rearrange
from tqdm import tqdm

from lingbot.wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from lingbot.wan.distributed.util import init_distributed_group, get_world_size
from lingbot.wan.distributed.fsdp import shard_model
from lingbot.wan.distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from lingbot.wan.modules.model import WanModel
from lingbot.wan.modules.t5 import T5EncoderModel
from lingbot.wan.modules.vae2_1 import Wan2_1_VAE
from lingbot.wan.utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from lingbot.wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from lingbot.wan.utils.cam_utils import (
    compute_relative_poses,
    interpolate_camera_poses,
    get_plucker_embeddings,
    get_Ks_transformed,
)

# Only import web server deps on rank 0
rank = int(os.getenv("RANK", 0))
if rank == 0:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, FileResponse
    import uvicorn


# Camera direction -> incremental pose transform
# These define small camera movements for interactive control
DIRECTION_TRANSFORMS = {
    "forward":  {"translation": np.array([0.0, 0.0, 0.3]),  "rotation_axis": None,    "rotation_angle": 0.0},
    "backward": {"translation": np.array([0.0, 0.0, -0.3]), "rotation_axis": None,    "rotation_angle": 0.0},
    "left":     {"translation": np.array([0.0, 0.0, 0.0]),  "rotation_axis": "y",     "rotation_angle": -8.0},
    "right":    {"translation": np.array([0.0, 0.0, 0.0]),  "rotation_axis": "y",     "rotation_angle": 8.0},
    "up":       {"translation": np.array([0.0, 0.0, 0.0]),  "rotation_axis": "x",     "rotation_angle": -5.0},
    "down":     {"translation": np.array([0.0, 0.0, 0.0]),  "rotation_axis": "x",     "rotation_angle": 5.0},
    "none":     {"translation": np.array([0.0, 0.0, 0.0]),  "rotation_axis": None,    "rotation_angle": 0.0},
}

# Default camera intrinsics (480p)
DEFAULT_INTRINSICS = np.array([600.0, 600.0, 416.0, 240.0])  # fx, fy, cx, cy


def rotation_matrix(axis: str, angle_deg: float) -> np.ndarray:
    """Create a 3x3 rotation matrix from axis and angle in degrees."""
    angle = np.radians(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return np.eye(3)


def build_chunk_poses(last_pose: np.ndarray, direction: str, num_frames: int = 17) -> np.ndarray:
    """
    Build camera poses for a chunk based on the current pose and a direction command.

    Returns an array of shape [num_frames, 4, 4] with interpolated poses from
    last_pose to the target pose defined by direction.
    """
    transform = DIRECTION_TRANSFORMS.get(direction, DIRECTION_TRANSFORMS["none"])

    # Compute target pose
    target_pose = last_pose.copy()
    # Apply rotation
    if transform["rotation_axis"] is not None:
        R_delta = rotation_matrix(transform["rotation_axis"], transform["rotation_angle"])
        target_pose[:3, :3] = last_pose[:3, :3] @ R_delta
    # Apply translation (in camera frame)
    target_pose[:3, 3] = last_pose[:3, 3] + last_pose[:3, :3] @ transform["translation"]

    # Interpolate between last_pose and target_pose
    src_poses = np.stack([last_pose, target_pose])
    poses = interpolate_camera_poses(
        src_indices=np.array([0, num_frames - 1]),
        src_rot_mat=src_poses[:, :3, :3],
        src_trans_vec=src_poses[:, :3, 3],
        tgt_indices=np.linspace(0, num_frames - 1, num_frames),
    )
    return poses.numpy()


def load_preset_path(preset_dir: str, max_frames: int = None) -> tuple:
    """Load camera poses and intrinsics from a preset directory."""
    poses = np.load(os.path.join(preset_dir, "poses.npy"))
    intrinsics = np.load(os.path.join(preset_dir, "intrinsics.npy"))
    prompt_file = os.path.join(preset_dir, "prompt.txt")
    prompt = ""
    if os.path.exists(prompt_file):
        with open(prompt_file, "r") as f:
            prompt = f.read().strip()
    if max_frames is not None:
        len_poses = ((len(poses) - 1) // 4) * 4 + 1
        poses = poses[:min(max_frames, len_poses)]
        intrinsics = intrinsics[:min(max_frames, len_poses)]
    return poses, intrinsics, prompt


class LingBotEngine:
    """Wraps LingBot-World WanI2V for chunk-based interactive generation."""

    def __init__(self, args):
        self.args = args
        self.rank = int(os.getenv("RANK", 0))
        self.world_size = int(os.getenv("WORLD_SIZE", 1))
        self.local_rank = int(os.getenv("LOCAL_RANK", 0))
        self.device = torch.device(f"cuda:{self.local_rank}")

        self.is_running = False
        self.should_stop = False
        self.action_queue = asyncio.Queue() if self.rank == 0 else None

        # Broadcast queue for distributing commands from rank 0 to all ranks
        self._command_event = None
        self._current_command = None

        self._init_model()

    def _init_model(self):
        """Initialize WanI2V model with distributed setup."""
        cfg = WAN_CONFIGS["i2v-A14B"]
        self.config = cfg

        use_sp = self.args.ulysses_size > 1
        if use_sp:
            init_distributed_group()
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        shard_fn = partial(shard_model, device_id=self.local_rank)

        if self.rank == 0:
            logging.info("Loading T5 text encoder...")
        self.text_encoder = T5EncoderModel(
            text_len=cfg.text_len,
            dtype=cfg.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(self.args.ckpt_dir, cfg.t5_checkpoint),
            tokenizer_path=os.path.join(self.args.ckpt_dir, cfg.t5_tokenizer),
            shard_fn=shard_fn if self.args.t5_fsdp else None,
        )

        if self.rank == 0:
            logging.info("Loading VAE...")
        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(self.args.ckpt_dir, cfg.vae_checkpoint),
            device=self.device,
        )

        if self.rank == 0:
            logging.info("Loading low-noise DiT model...")
        self.low_noise_model = WanModel.from_pretrained(
            self.args.ckpt_dir, subfolder=cfg.low_noise_checkpoint, torch_dtype=torch.bfloat16)
        self.low_noise_model = self._configure_model(
            self.low_noise_model, use_sp, self.args.dit_fsdp, shard_fn)

        if self.rank == 0:
            logging.info("Loading high-noise DiT model...")
        self.high_noise_model = WanModel.from_pretrained(
            self.args.ckpt_dir, subfolder=cfg.high_noise_checkpoint, torch_dtype=torch.bfloat16)
        self.high_noise_model = self._configure_model(
            self.high_noise_model, use_sp, self.args.dit_fsdp, shard_fn)

        self.num_train_timesteps = cfg.num_train_timesteps
        self.boundary = cfg.boundary
        self.param_dtype = cfg.param_dtype
        self.vae_stride = cfg.vae_stride
        self.patch_size = cfg.patch_size

        if self.rank == 0:
            logging.info("LingBot-World model loaded successfully!")

    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn):
        model.eval().requires_grad_(False)
        if use_sp:
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(sp_attn_forward, block.self_attn)
            model.forward = types.MethodType(sp_dit_forward, model)
        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            model = shard_fn(model)
        else:
            model.to(self.device)
        return model

    def _prepare_model_for_timestep(self, t, boundary, offload_model=False):
        if t.item() >= boundary:
            required, offload = 'high_noise_model', 'low_noise_model'
        else:
            required, offload = 'low_noise_model', 'high_noise_model'
        if offload_model:
            if next(getattr(self, offload).parameters()).device.type == 'cuda':
                getattr(self, offload).to('cpu')
            if next(getattr(self, required).parameters()).device.type == 'cpu':
                getattr(self, required).to(self.device)
        return getattr(self, required)

    def _frame_to_base64(self, frame: np.ndarray) -> str:
        img = Image.fromarray(frame.astype(np.uint8))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _prepare_camera_condition(self, c2ws: np.ndarray, intrinsics: np.ndarray,
                                  h: int, w: int, lat_h: int, lat_w: int, lat_f: int):
        """Prepare Plücker embedding camera condition from poses and intrinsics."""
        Ks = torch.from_numpy(intrinsics).float()
        if Ks.ndim == 1:
            Ks = Ks.unsqueeze(0).repeat(len(c2ws), 1)

        # Transform intrinsics for the actual output resolution
        Ks = get_Ks_transformed(Ks,
                                height_org=480,
                                width_org=832,
                                height_resize=h,
                                width_resize=w,
                                height_final=h,
                                width_final=w)
        Ks = Ks[0]

        c2ws_tensor = torch.from_numpy(c2ws).float() if isinstance(c2ws, np.ndarray) else c2ws.float()

        # Interpolate to latent temporal resolution
        len_c2ws = len(c2ws_tensor)
        c2ws_infer = interpolate_camera_poses(
            src_indices=np.linspace(0, len_c2ws - 1, len_c2ws),
            src_rot_mat=c2ws_tensor[:, :3, :3].numpy(),
            src_trans_vec=c2ws_tensor[:, :3, 3].numpy(),
            tgt_indices=np.linspace(0, len_c2ws - 1, int((len_c2ws - 1) // 4) + 1),
        )
        c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
        Ks = Ks.repeat(len(c2ws_infer), 1)

        c2ws_infer = c2ws_infer.to(self.device)
        Ks = Ks.to(self.device)

        c2ws_plucker_emb = get_plucker_embeddings(c2ws_infer, Ks, h, w)
        c2ws_plucker_emb = rearrange(
            c2ws_plucker_emb,
            'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
            c1=int(h // lat_h),
            c2=int(w // lat_w),
        )
        c2ws_plucker_emb = c2ws_plucker_emb[None, ...]
        c2ws_plucker_emb = rearrange(
            c2ws_plucker_emb,
            'b (f h w) c -> b c f h w',
            f=lat_f, h=lat_h, w=lat_w,
        ).to(self.param_dtype)

        return {"c2ws_plucker_emb": c2ws_plucker_emb.chunk(1, dim=0)}

    async def generate_chunk(self, prompt: str, img: Image.Image,
                             c2ws: np.ndarray, intrinsics: np.ndarray,
                             sampling_steps: int = 25, shift: float = 10.0,
                             guide_scale: tuple = (5.0, 5.0),
                             max_area: int = 480 * 832,
                             websocket=None, chunk_index: int = 0):
        """
        Generate a single chunk of video frames.

        This runs the full diffusion process for the given poses and returns
        the generated video tensor and decoded frames.
        """
        cfg = self.config
        frame_num = len(c2ws)
        # Ensure frame_num is 4n+1
        frame_num = ((frame_num - 1) // 4) * 4 + 1
        c2ws = c2ws[:frame_num]

        img_tensor = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h_img, w_img = img_tensor.shape[1:]
        aspect_ratio = h_img / w_img
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]
        lat_f = (F - 1) // self.vae_stride[0] + 1
        max_seq_len = lat_f * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(self.args.seed)
        noise = torch.randn(16, lat_f, lat_h, lat_w,
                            dtype=torch.float32, generator=seed_g, device=self.device)

        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        n_prompt = cfg.sample_neg_prompt

        # Text encoding
        if not self.args.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        # Camera conditioning
        dit_cond_dict = self._prepare_camera_condition(
            c2ws, intrinsics, h, w, lat_h, lat_w, lat_f)

        # Encode image
        y = self.vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(
                    img_tensor[None].cpu(), size=(h, w), mode='bicubic').transpose(0, 1),
                torch.zeros(3, F - 1, h, w)
            ], dim=1).to(self.device)
        ])[0]
        y = torch.concat([msk, y])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_low = getattr(self.low_noise_model, 'no_sync', noop_no_sync)
        no_sync_high = getattr(self.high_noise_model, 'no_sync', noop_no_sync)

        with (
            torch.amp.autocast('cuda', dtype=self.param_dtype),
            torch.no_grad(),
            no_sync_low(),
            no_sync_high(),
        ):
            boundary = self.boundary * self.num_train_timesteps

            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                sampling_steps, device=self.device, shift=shift)
            timesteps = sample_scheduler.timesteps

            latent = noise

            arg_c = {
                'context': [context[0]],
                'seq_len': max_seq_len,
                'y': [y],
                'dit_cond_dict': dit_cond_dict,
            }
            arg_null = {
                'context': context_null,
                'seq_len': max_seq_len,
                'y': [y],
                'dit_cond_dict': dit_cond_dict,
            }

            torch.cuda.empty_cache()

            for step_idx, t in enumerate(tqdm(timesteps, disable=(self.rank != 0))):
                latent_model_input = [latent.to(self.device)]
                timestep = torch.stack([t]).to(self.device)

                model = self._prepare_model_for_timestep(t, boundary, offload_model=False)
                sample_guide_scale = guide_scale[1] if t.item() >= boundary else guide_scale[0]

                noise_pred_cond = model(latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = model(latent_model_input, t=timestep, **arg_null)[0]
                noise_pred = noise_pred_uncond + sample_guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0), t, latent.unsqueeze(0),
                    return_dict=False, generator=seed_g)[0]
                latent = temp_x0.squeeze(0)

                del latent_model_input, timestep

                # Send progress update
                if self.rank == 0 and websocket:
                    try:
                        await websocket.send_json({
                            "type": "generating",
                            "chunk": chunk_index,
                            "step": step_idx + 1,
                            "total_steps": len(timesteps),
                        })
                    except Exception:
                        pass

            # Decode video
            x0 = [latent]
            if self.rank == 0:
                videos = self.vae.decode(x0)
                video = videos[0]  # [C, F, H, W] in range [-1, 1]
            else:
                video = None

        del noise, latent, x0, sample_scheduler
        gc.collect()
        torch.cuda.empty_cache()

        if dist.is_initialized():
            dist.barrier()

        return video

    async def run_interactive(self, img: Image.Image, prompt: str,
                              num_chunks: int, sampling_steps: int,
                              size: str, camera_preset: str, websocket):
        """
        Run interactive chunk-based generation loop.

        For each chunk:
        1. Wait for camera direction from user (or use preset path)
        2. Generate 17 frames with that camera motion
        3. Stream frames to client
        4. Use last frame as input for next chunk
        """
        self.is_running = True
        self.should_stop = False

        size_parts = size.split("*")
        max_area = int(size_parts[0]) * int(size_parts[1])
        guide_scale = self.config.sample_guide_scale

        # Load preset if specified
        preset_poses = None
        preset_intrinsics = None
        if camera_preset and camera_preset != "interactive":
            preset_dir = f"configs/lingbot/camera_presets/{camera_preset}"
            if os.path.exists(preset_dir):
                preset_poses, preset_intrinsics, preset_prompt = load_preset_path(preset_dir)
                if not prompt and preset_prompt:
                    prompt = preset_prompt
                # Calculate how many chunks this preset supports
                total_preset_frames = len(preset_poses)
                num_chunks = min(num_chunks, (total_preset_frames - 1) // 16)

        if not prompt:
            prompt = "A cinematic first-person exploration through a detailed environment with smooth camera movement."

        # Current camera state
        current_pose = np.eye(4)
        chunk_frames = 17  # 4n+1 where n=4
        current_img = img

        try:
            # Send initial frame
            if self.rank == 0:
                init_frame = np.array(img.resize((832, 480)))
                await websocket.send_json({
                    "type": "frame",
                    "data": self._frame_to_base64(init_frame),
                    "chunk": 0,
                    "frame": 0,
                })

            for chunk_idx in range(num_chunks):
                if self.should_stop:
                    break

                # Get camera poses for this chunk
                if preset_poses is not None:
                    # Use preset path
                    start_frame = chunk_idx * 16
                    end_frame = start_frame + chunk_frames
                    if end_frame > len(preset_poses):
                        break
                    chunk_c2ws = preset_poses[start_frame:end_frame]
                    chunk_intrinsics = preset_intrinsics[start_frame:end_frame]

                    if self.rank == 0:
                        await websocket.send_json({
                            "type": "generating",
                            "chunk": chunk_idx + 1,
                            "step": 0,
                            "total_steps": sampling_steps,
                        })
                else:
                    # Interactive mode: request direction from user
                    if self.rank == 0:
                        await websocket.send_json({
                            "type": "request_action",
                            "chunk": chunk_idx + 1,
                        })

                        # Wait for direction
                        try:
                            action = await asyncio.wait_for(
                                self.action_queue.get(), timeout=300.0)
                            if action is None:
                                break
                            direction = action.get("direction", "none")
                        except asyncio.TimeoutError:
                            direction = "none"

                    # Broadcast direction from rank 0 to all ranks
                    if dist.is_initialized():
                        direction_list = [direction] if self.rank == 0 else [None]
                        dist.broadcast_object_list(direction_list, src=0)
                        direction = direction_list[0]

                    chunk_c2ws = build_chunk_poses(current_pose, direction, chunk_frames)
                    chunk_intrinsics = np.tile(DEFAULT_INTRINSICS, (chunk_frames, 1))

                # Generate this chunk
                video = await self.generate_chunk(
                    prompt=prompt,
                    img=current_img,
                    c2ws=chunk_c2ws,
                    intrinsics=chunk_intrinsics,
                    sampling_steps=sampling_steps,
                    shift=self.args.sample_shift,
                    guide_scale=guide_scale,
                    max_area=max_area,
                    websocket=websocket,
                    chunk_index=chunk_idx + 1,
                )

                if self.rank == 0 and video is not None:
                    # video is [C, F, H, W] in range [-1, 1]
                    video_np = rearrange(video, 'c f h w -> f h w c')
                    video_np = ((video_np.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)

                    # Stream frames
                    for frame_idx, frame in enumerate(video_np):
                        if self.should_stop:
                            break
                        await websocket.send_json({
                            "type": "frame",
                            "data": self._frame_to_base64(frame),
                            "chunk": chunk_idx + 1,
                            "frame": frame_idx,
                        })
                        await asyncio.sleep(0.06)  # ~16fps pacing

                    # Use last frame as input for next chunk
                    last_frame = video_np[-1]
                    current_img = Image.fromarray(last_frame)

                    # Update current pose
                    current_pose = chunk_c2ws[-1]

                    await websocket.send_json({
                        "type": "chunk_complete",
                        "chunk": chunk_idx + 1,
                    })

                # Sync all ranks
                if dist.is_initialized():
                    dist.barrier()

            if self.rank == 0:
                await websocket.send_json({"type": "complete"})

        except Exception as e:
            import traceback
            traceback.print_exc()
            if self.rank == 0 and websocket:
                await websocket.send_json({"type": "error", "message": str(e)})
        finally:
            self.is_running = False

    def stop(self):
        self.should_stop = True


# ---- FastAPI Application (rank 0 only) ----

engine: Optional[LingBotEngine] = None

if rank == 0:
    app = FastAPI(title="LingBot-World Web UI")

    @app.get("/", response_class=HTMLResponse)
    async def get_index():
        return FileResponse("web_ui/lingbot.html")

    @app.get("/lingbot.html", response_class=HTMLResponse)
    async def get_lingbot():
        return FileResponse("web_ui/lingbot.html")

    @app.get("/assets/{filename}")
    async def get_asset(filename: str):
        return FileResponse(f"web_ui/assets/{filename}")

    @app.get("/api/status")
    async def get_status():
        return {
            "ready": engine is not None,
            "running": engine.is_running if engine else False,
            "model": "lingbot-world",
        }

    @app.get("/api/presets")
    async def list_presets():
        """List available camera preset paths."""
        presets = []
        preset_dir = "configs/lingbot/camera_presets"
        if os.path.exists(preset_dir):
            for name in sorted(os.listdir(preset_dir)):
                path = os.path.join(preset_dir, name)
                if os.path.isdir(path) and os.path.exists(os.path.join(path, "poses.npy")):
                    prompt = ""
                    prompt_file = os.path.join(path, "prompt.txt")
                    if os.path.exists(prompt_file):
                        with open(prompt_file, "r") as f:
                            prompt = f.read().strip()[:100]
                    poses = np.load(os.path.join(path, "poses.npy"))
                    presets.append({
                        "name": name,
                        "frames": len(poses),
                        "prompt_preview": prompt,
                    })
        return {"presets": presets}

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()

        try:
            while True:
                data = await websocket.receive_json()
                msg_type = data.get("type")

                if msg_type == "start":
                    if engine and not engine.is_running:
                        # Parse image
                        img_data = data.get("image", "")
                        prompt = data.get("prompt", "")
                        camera_preset = data.get("camera_preset", "interactive")
                        sampling_steps = int(data.get("steps", 25))
                        size = data.get("size", "480*832")
                        num_chunks = int(data.get("num_chunks", 10))

                        if img_data.startswith("data:image"):
                            header, b64 = img_data.split(",", 1)
                            img_bytes = base64.b64decode(b64)
                            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        elif os.path.exists(img_data):
                            img = Image.open(img_data).convert("RGB")
                        else:
                            # Use preset image if available
                            preset_dir = f"configs/lingbot/camera_presets/{camera_preset}"
                            img_path = os.path.join(preset_dir, "image.jpg")
                            if os.path.exists(img_path):
                                img = Image.open(img_path).convert("RGB")
                            else:
                                await websocket.send_json({
                                    "type": "error",
                                    "message": "No image provided and no preset image found"
                                })
                                continue

                        asyncio.create_task(
                            engine.run_interactive(
                                img=img,
                                prompt=prompt,
                                num_chunks=num_chunks,
                                sampling_steps=sampling_steps,
                                size=size,
                                camera_preset=camera_preset,
                                websocket=websocket,
                            )
                        )
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Engine not ready or already running"
                        })

                elif msg_type == "action":
                    if engine and engine.is_running:
                        await engine.action_queue.put(data)

                elif msg_type == "stop":
                    if engine:
                        engine.stop()
                        await engine.action_queue.put(None)
                        await websocket.send_json({"type": "stopped"})

        except WebSocketDisconnect:
            if engine:
                engine.stop()
        except Exception as e:
            logging.error(f"WebSocket error: {e}")
            if engine:
                engine.stop()


def parse_args():
    parser = argparse.ArgumentParser(description="LingBot-World Web Server")
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Path to LingBot-World checkpoint directory")
    parser.add_argument("--size", type=str, default="480*832",
                        choices=list(SUPPORTED_SIZES.get("i2v-A14B", [])),
                        help="Output resolution (default: 480*832)")
    parser.add_argument("--sample_shift", type=float, default=10.0,
                        help="Sampling shift factor (default: 10.0)")
    parser.add_argument("--ulysses_size", type=int, default=1,
                        help="Ulysses sequence parallel size (default: 1, set to world_size for SP)")
    parser.add_argument("--t5_fsdp", action="store_true", default=False,
                        help="Enable FSDP for T5")
    parser.add_argument("--t5_cpu", action="store_true", default=False,
                        help="Place T5 on CPU")
    parser.add_argument("--dit_fsdp", action="store_true", default=False,
                        help="Enable FSDP for DiT models")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8889,
                        help="Server port (default: 8889)")
    return parser.parse_args()


def main():
    global engine

    args = parse_args()

    this_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))

    # Setup logging
    if this_rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)

    # Initialize distributed
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=this_rank,
            world_size=world_size)

    # Auto-configure ulysses_size to match world_size
    if args.ulysses_size == 1 and world_size > 1:
        args.ulysses_size = world_size
        if this_rank == 0:
            logging.info(f"Auto-setting ulysses_size={world_size}")

    # Initialize engine
    if this_rank == 0:
        logging.info(f"Initializing LingBot-World engine (rank {this_rank}/{world_size})...")

    engine = LingBotEngine(args)

    # Only rank 0 runs the web server
    if this_rank == 0:
        logging.info(f"Starting LingBot-World server at http://{args.host}:{args.port}")
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            ws_max_size=50 * 1024 * 1024,
        )
    else:
        # Other ranks wait for commands via distributed collectives
        # In this architecture, they participate in inference through
        # FSDP/SP collective operations triggered during generate_chunk
        logging.info(f"Rank {this_rank} waiting for distributed inference calls...")
        while True:
            try:
                if dist.is_initialized():
                    dist.barrier()
            except Exception:
                break


if __name__ == "__main__":
    main()
