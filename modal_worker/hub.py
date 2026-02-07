"""
Modal ASGI hub for Ephemeral.

Serves the web UI, handles WebSocket orchestration for host + players,
and dispatches GPU inference to EphemeralWorker containers via .remote().

Deploy:
    modal deploy modal_worker/hub.py
"""

import asyncio
import base64
import io
import logging
import os
import random
import string
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import modal
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse

from modal_worker.app import app, hub_image

# Import worker module so EphemeralWorker gets registered with the same app.
# This ensures `modal deploy modal_worker/hub.py` deploys both hub + GPU workers.
import modal_worker.worker  # noqa: F401

logger = logging.getLogger("hub")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


# ── Dataclasses ──


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
    # Pre-initialized workers: list of (worker, init_result) ready for instant assignment
    ready_workers: list = field(default_factory=list)
    pool_target: int = 8  # match max_active — all slots pre-initialized


@dataclass
class SoloPlayState:
    """State for the host's solo play session (single Modal worker)."""
    websocket: WebSocket
    worker: object
    seed_image_b64: str = ""
    action_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    inference_task: Optional[asyncio.Task] = None
    should_stop: bool = False
    mode: str = "universal"


# ── Global state (single container, max_containers=1) ──

modal_room_state: Optional[ModalRoomState] = None
solo_play_state: Optional[SoloPlayState] = None
suno_client = None  # Lazy-init SunoClient

# Pre-initialized worker pool (built during solo play, transferred to room on open)
_preinit_pool: list = []          # list of (worker, init_result)
_preinit_seed_b64: str = ""       # seed image the pool was built for
_preinit_task: Optional[asyncio.Task] = None  # background pre-init task


# ── Helpers ──


def _generate_room_id(length: int = 4) -> str:
    """Generate a short uppercase+digits room code."""
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


def _get_modal_worker_class():
    """Lazily import and return the Modal worker class."""
    from modal_worker.worker import EphemeralWorker
    return EphemeralWorker


def _get_suno_client():
    """Lazy-init and return the global Suno client."""
    global suno_client
    if suno_client is None:
        from suno_api.client import SunoClient
        cookie = os.environ.get("SUNO_COOKIE", "")
        if cookie:
            suno_client = SunoClient(cookie=cookie)
            logger.info("[Hub] SunoClient initialized (%d char cookie)", len(cookie))
        else:
            logger.warning("[Hub] No SUNO_COOKIE set, music generation disabled")
    return suno_client


def _resolve_seed_image(data: dict) -> str:
    """Extract seed image base64 from a WS message. Returns empty string if none."""
    seed_image_b64 = data.get("seed_image_b64", "")
    if not seed_image_b64:
        raw = data.get("image", "")
        if raw and "," in raw:
            # data:image/...;base64,XXX → strip prefix
            seed_image_b64 = raw.split(",", 1)[1]
        elif raw and os.path.isfile(f"/app/{raw}"):
            # Server-side demo image path → load and encode
            with open(f"/app/{raw}", "rb") as f:
                seed_image_b64 = base64.b64encode(f.read()).decode()
            logger.info("[Hub] Loaded seed image from path: %s", raw)
        elif raw:
            seed_image_b64 = raw
    return seed_image_b64


# ── Gemini image generation ──


async def _generate_image_with_gemini(prompt: str, ref_image_b64: str | None,
                                       websocket: WebSocket) -> str:
    """Generate an image via Gemini. Returns base64 JPEG (no prefix).

    Args:
        prompt: Text description of the world to generate.
        ref_image_b64: Optional reference image as base64 (no data: prefix).
        websocket: Host WebSocket to send status updates.

    Returns:
        Base64-encoded JPEG image string.
    """
    from PIL import Image

    await websocket.send_json({
        "type": "generating_image",
        "message": f"Generating: {prompt[:60]}...",
    })

    def _call_gemini():
        from google import genai
        from google.genai import types as genai_types

        client = genai.Client()  # uses GOOGLE_API_KEY env var

        contents = []
        if ref_image_b64:
            ref_bytes = base64.b64decode(ref_image_b64)
            ref_img = Image.open(io.BytesIO(ref_bytes)).convert("RGB")
            contents = [prompt, ref_img]
        else:
            contents = [f"Generate an image: {prompt}"]

        response = client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=contents,
            config=genai_types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=genai_types.ImageConfig(
                    aspect_ratio="16:9",
                ),
            ),
        )

        for part in response.parts:
            if part.text is not None:
                logger.info("[Gemini] Text response: %s", part.text[:200])
            elif part.inline_data is not None:
                img_bytes = part.inline_data.data
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                return base64.b64encode(buf.getvalue()).decode()
        return None

    loop = asyncio.get_event_loop()
    t0 = time.perf_counter()
    result_b64 = await loop.run_in_executor(None, _call_gemini)
    elapsed = time.perf_counter() - t0

    if result_b64 is None:
        logger.error("[Gemini] No image returned after %.1fs", elapsed)
        raise RuntimeError("Gemini did not return an image")

    logger.info("[Gemini] Image generated in %.1fs (%d KB)", elapsed, len(result_b64) // 1024)
    return result_b64


# ── Solo play caption ──


async def _solo_generate_caption(websocket: WebSocket, frame_b64: str, block_num: int):
    """Generate caption for the host's solo play frame via Claude API."""
    try:
        import anthropic

        def _call_claude():
            client = anthropic.Anthropic()
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

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _call_claude)

        lines = result.strip().split("\n", 1)
        color = "#FFFFFF"
        caption_text = result
        if len(lines) == 2 and lines[0].strip() in ("#000000", "#FFFFFF"):
            color = lines[0].strip()
            caption_text = lines[1].strip()

        try:
            await websocket.send_json({
                "type": "caption",
                "block": block_num,
                "text": caption_text,
                "color": color,
                "world_id": 0,
            })
        except Exception:
            pass

    except Exception as e:
        logger.warning("[Hub] Solo caption failed block %d: %s", block_num, e)


# ── Music generation (Anthropic → Suno) ──


async def _generate_music(prompt: str, image_b64: str | None, websocket: WebSocket,
                          captcha_token: str):
    """Generate background music using Anthropic + Suno, running in the background."""
    try:
        import anthropic

        await websocket.send_json({
            "type": "music_status",
            "message": "Generating music prompt...",
        })

        # Step 1: Anthropic generates a Suno prompt
        def _encode_and_call_anthropic():
            client = anthropic.Anthropic()
            content = []
            if image_b64:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_b64,
                    },
                })

            suffix = (
                "Write a short Suno AI music generation prompt (2-3 sentences max) describing "
                "the perfect looping background music for this game world. Focus on mood, genre, "
                "instruments, tempo, and atmosphere. Make it instrumental. "
                "Output ONLY the Suno prompt, nothing else."
            )

            if image_b64 and prompt:
                text_part = (
                    "You are creating a prompt for Suno AI to generate background music for a video game world. "
                    f'Look at this game scene image (the player described it as: "{prompt}"). ' + suffix
                )
            elif image_b64:
                text_part = (
                    "You are creating a prompt for Suno AI to generate background music for a video game world. "
                    "Look at this game scene image. " + suffix
                )
            else:
                text_part = (
                    "You are creating a prompt for Suno AI to generate background music for a video game world. "
                    f'The player described the world as: "{prompt}". ' + suffix
                )

            content.append({"type": "text", "text": text_part})
            msg = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=300,
                messages=[{"role": "user", "content": content}],
            )
            return msg.content[0].text.strip()

        loop = asyncio.get_event_loop()
        music_prompt = await loop.run_in_executor(None, _encode_and_call_anthropic)
        logger.info("[Music] Anthropic prompt: %s", music_prompt)

        await websocket.send_json({
            "type": "music_status",
            "message": f"Composing: {music_prompt[:60]}...",
        })

        # Step 2: Suno generation
        client = _get_suno_client()
        if not client:
            logger.warning("[Music] No Suno client, skipping music")
            return

        data = await client.generate(
            description=music_prompt + " Instrumental, loopable, video game background music.",
            instrumental=True,
            captcha_token=captcha_token,
        )
        clips = data.get("clips", [])
        clip_ids = [c["id"] for c in clips]
        logger.info("[Music] Suno generation started: clip_ids=%s", clip_ids)

        await websocket.send_json({
            "type": "music_status",
            "message": "Waiting for music generation...",
        })

        # Step 3: Poll until first song ready
        deadline = time.perf_counter() + 300
        first_ready = None
        while time.perf_counter() < deadline:
            songs = await client.get_songs(clip_ids)
            for s in songs:
                if s.get("audio_url"):
                    first_ready = s
                    break
            if first_ready:
                break
            await asyncio.sleep(5)

        if first_ready:
            audio_urls = [first_ready["audio_url"]]
            logger.info("[Music] First song ready: %s", first_ready["audio_url"])
            await websocket.send_json({
                "type": "music_ready",
                "audio_urls": audio_urls,
                "title": first_ready.get("title", ""),
                "prompt": music_prompt,
            })
            # Store for Modal room players who join later
            if modal_room_state and modal_room_state.is_active:
                modal_room_state.music_urls = audio_urls
        else:
            logger.warning("[Music] No songs ready after timeout")

    except Exception as e:
        logger.exception("[Music] Music generation failed")
        try:
            await websocket.send_json({
                "type": "music_status",
                "message": f"Music generation failed: {str(e)[:100]}",
            })
        except Exception:
            pass


# ── Solo play inference loop ──


async def _solo_inference_loop(sps: SoloPlayState):
    """Run inference blocks on a Modal worker for solo play."""
    websocket = sps.websocket
    worker = sps.worker
    block_num = 0
    loop = asyncio.get_event_loop()

    try:
        while True:
            if sps.should_stop:
                break

            block_num += 1

            # Request action
            try:
                await websocket.send_json({
                    "type": "request_action",
                    "block": block_num,
                    "mode": sps.mode,
                    "world_id": 0,
                })
            except Exception:
                break

            # Wait for action (no timeout — wait until player acts or disconnects)
            action = await sps.action_queue.get()
            if action is None:
                break

            # Run block on Modal worker
            try:
                result = await loop.run_in_executor(
                    None, lambda a=action: worker.run_block.remote(action=a)
                )
            except Exception as e:
                if "init_session" in str(e) and sps.seed_image_b64:
                    # Container was recycled — re-initialize and retry
                    logger.warning("[Hub] Solo container recycled, re-initializing session...")
                    try:
                        await loop.run_in_executor(
                            None, lambda: worker.init_session.remote(
                                image_b64=sps.seed_image_b64, mode=sps.mode
                            )
                        )
                        result = await loop.run_in_executor(
                            None, lambda a=action: worker.run_block.remote(action=a)
                        )
                    except Exception as e2:
                        logger.warning("[Hub] Solo re-init also failed: %s", e2)
                        break
                else:
                    logger.warning("[Hub] Solo run_block failed: %s", e)
                    break

            # Check for session limit
            if result.get("session_limit"):
                logger.info("[Hub] Solo session limit reached at block %d (max %s)",
                            block_num, result.get("max_blocks"))
                try:
                    await websocket.send_json({
                        "type": "session_ended",
                        "reason": f"Session limit reached ({result.get('max_blocks')} blocks)",
                    })
                except Exception:
                    pass
                break

            # Send frames
            for frame_idx, frame_b64 in enumerate(result["frames_b64"]):
                try:
                    await websocket.send_json({
                        "type": "frame",
                        "data": frame_b64,
                        "block": block_num,
                        "frame": frame_idx,
                        "world_id": 0,
                    })
                except Exception:
                    break
                await asyncio.sleep(0.01)

            # Caption every 3 blocks
            if block_num % 3 == 1:
                asyncio.create_task(
                    _solo_generate_caption(websocket, result["last_frame_b64"], block_num)
                )

            # Block complete
            try:
                await websocket.send_json({
                    "type": "block_complete",
                    "block": block_num,
                    "world_id": 0,
                })
            except Exception:
                break

    except asyncio.CancelledError:
        logger.info("[Hub] Solo inference loop cancelled")
    except Exception:
        logger.exception("[Hub] Solo inference loop error")
    finally:
        try:
            await loop.run_in_executor(None, lambda: worker.end_session.remote())
        except Exception:
            pass
        try:
            await websocket.send_json({"type": "complete"})
        except Exception:
            pass
        logger.info("[Hub] Solo inference loop ended (block=%d)", block_num)


async def _start_solo_play(websocket: WebSocket, data: dict):
    """Handle the 'start' message: Gemini image gen → spawn worker → inference loop."""
    global solo_play_state

    if solo_play_state is not None:
        await websocket.send_json({"type": "error", "message": "Already running"})
        return

    img_input = data.get("image")
    prompt = data.get("prompt")
    mode = "universal"

    try:
        # Resolve seed image
        seed_image_b64 = ""
        if img_input:
            if isinstance(img_input, str) and "," in img_input:
                seed_image_b64 = img_input.split(",", 1)[1]
            elif isinstance(img_input, str) and os.path.isfile(f"/app/{img_input}"):
                with open(f"/app/{img_input}", "rb") as f:
                    seed_image_b64 = base64.b64encode(f.read()).decode()
            elif img_input:
                seed_image_b64 = img_input

        ref_image_for_gemini = seed_image_b64 or None

        # Generate image with Gemini if prompt provided
        if prompt:
            seed_image_b64 = await _generate_image_with_gemini(
                prompt, ref_image_for_gemini, websocket
            )
            # Send preview
            await websocket.send_json({
                "type": "image_preview",
                "data": seed_image_b64,
            })

        if not seed_image_b64:
            await websocket.send_json({"type": "error", "message": "No image provided"})
            return

        # Start pre-initializing 8 room workers in background
        # They'll be ready by the time the user clicks "Open Room"
        global _preinit_task
        await _cleanup_preinit_pool()  # clear any stale pool
        _preinit_task = asyncio.create_task(
            _background_preinit_pool(seed_image_b64, mode, count=2)
        )

        # Spawn Modal worker
        WorkerCls = _get_modal_worker_class()
        worker = WorkerCls(session_id=str(uuid.uuid4()))

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: worker.init_session.remote(image_b64=seed_image_b64, mode=mode)
        )

        # Send ready
        await websocket.send_json({
            "type": "ready",
            "mode": result["mode"],
            "resolution": result["resolution"],
            "num_worlds": 1,
            "message": "Ready for input. Press keys to control.",
        })

        # Send initial frame
        await websocket.send_json({
            "type": "frame",
            "data": result["initial_frame_b64"],
            "block": 0,
            "world_id": 0,
        })

        # Create solo play state
        sps = SoloPlayState(
            websocket=websocket,
            worker=worker,
            seed_image_b64=seed_image_b64,
            mode=mode,
        )
        sps._music_prompt = prompt or ""
        sps._music_image_b64 = seed_image_b64
        solo_play_state = sps

        # Start inference loop
        sps.inference_task = asyncio.create_task(_solo_inference_loop(sps))

    except Exception as e:
        logger.exception("[Hub] Failed to start solo play")
        await websocket.send_json({"type": "error", "message": str(e)[:200]})


async def _stop_solo_play():
    """Stop the solo play session."""
    global solo_play_state
    sps = solo_play_state
    if sps is None:
        return
    sps.should_stop = True
    try:
        sps.action_queue.put_nowait(None)
    except Exception:
        pass
    if sps.inference_task and not sps.inference_task.done():
        sps.inference_task.cancel()
    solo_play_state = None


# ── Modal Room orchestration ──


async def _modal_broadcast_player_list():
    """Send updated player list to the host."""
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
    queued_info = [
        {"player_id": pid, "name": mrs.players[pid].name}
        for pid in mrs.queue if pid in mrs.players
    ]
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
    mrs = modal_room_state
    if mrs is None or not mrs.is_active:
        return
    while mrs.queue and mrs.active_count < mrs.max_active:
        next_pid = mrs.queue.pop(0)
        ps = mrs.players.get(next_pid)
        if ps is None or ps.status == "disconnected":
            continue
        asyncio.create_task(_modal_spawn_player(mrs, ps))
        break


async def _modal_spawn_player(mrs: ModalRoomState, ps: ModalPlayerState):
    """Assign a pre-initialized worker to a player, or spawn on-demand."""
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
        # Try to grab a pre-initialized worker: room pool first, then global pool
        if mrs.ready_workers:
            worker, result = mrs.ready_workers.pop(0)
            ps.worker = worker
            logger.info("[Hub] Assigned pre-init worker to %s from room pool (instant)", ps.name)
        elif _preinit_pool:
            worker, result = _preinit_pool.pop(0)
            ps.worker = worker
            logger.info("[Hub] Assigned pre-init worker to %s from global pool (instant)", ps.name)
        else:
            # Fallback: spawn on-demand (slower)
            logger.info("[Hub] No pre-init workers, spawning on-demand for %s", ps.name)
            WorkerCls = _get_modal_worker_class()
            worker = WorkerCls(session_id=str(uuid.uuid4()))
            ps.worker = worker
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: worker.init_session.remote(
                    image_b64=mrs.seed_image_b64, mode=mrs.mode
                )
            )

        ps.status = "active"
        await _modal_broadcast_player_list()

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

        if mrs.music_urls:
            try:
                await ps.websocket.send_json({
                    "type": "music_ready",
                    "audio_urls": mrs.music_urls,
                })
            except Exception:
                pass

        ps.inference_task = asyncio.create_task(
            _modal_player_inference_loop(mrs, ps.player_id)
        )

    except Exception as e:
        logger.exception("[Hub] Failed to spawn worker for %s", ps.name)
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


async def _modal_player_inference_loop(mrs: ModalRoomState, player_id: str):
    """Run inference blocks for one player until disconnect or room close."""
    ps = mrs.players.get(player_id)
    if ps is None:
        return

    worker = ps.worker
    block_num = 0
    loop = asyncio.get_event_loop()

    end_reason = "unknown"
    try:
        while True:
            if not mrs.is_active:
                end_reason = "room_closed"
                break
            if ps.status == "disconnected":
                end_reason = "player_disconnected"
                break

            block_num += 1

            try:
                await ps.websocket.send_json({
                    "type": "request_action",
                    "block": block_num,
                })
            except Exception as e:
                end_reason = f"send_request_action_failed: {e}"
                break

            action = await ps.action_queue.get()
            if action is None:
                end_reason = "action_queue_none (player disconnected)"
                break

            try:
                result = await loop.run_in_executor(
                    None, lambda a=action: worker.run_block.remote(action=a)
                )
            except Exception as e:
                if "init_session" in str(e) and mrs.seed_image_b64:
                    # Container was recycled — re-initialize and retry
                    logger.warning("[Hub] Container recycled for %s, re-initializing...", ps.name)
                    try:
                        await loop.run_in_executor(
                            None, lambda: worker.init_session.remote(
                                image_b64=mrs.seed_image_b64, mode=mrs.mode
                            )
                        )
                        result = await loop.run_in_executor(
                            None, lambda a=action: worker.run_block.remote(action=a)
                        )
                    except Exception as e2:
                        end_reason = f"re-init_failed: {e2}"
                        logger.warning("[Hub] Re-init also failed for %s: %s", ps.name, e2)
                        break
                else:
                    end_reason = f"run_block_failed: {e}"
                    logger.warning("[Hub] run_block failed for %s: %s", ps.name, e)
                    break

            # Check for session limit
            if result.get("session_limit"):
                end_reason = f"session_limit ({result.get('max_blocks')} blocks)"
                logger.info("[Hub] Session limit for %s at block %d", ps.name, block_num)
                break

            for frame_idx, frame_b64 in enumerate(result["frames_b64"]):
                try:
                    await ps.websocket.send_json({
                        "type": "frame",
                        "data": frame_b64,
                        "block": block_num,
                        "frame": frame_idx,
                    })
                except Exception as e:
                    end_reason = f"send_frame_failed: {e}"
                    break
                await asyncio.sleep(0.01)
            else:
                # Only continue if the frame loop didn't break
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

                if block_num % 3 == 1:
                    asyncio.create_task(
                        _modal_generate_caption(mrs, ps, result["last_frame_b64"], block_num)
                    )
                continue
            # frame loop broke — exit outer loop too
            break

    except asyncio.CancelledError:
        end_reason = "cancelled"
        logger.info("[Hub] Inference loop cancelled for %s", ps.name)
    except Exception:
        end_reason = "unhandled_exception"
        logger.exception("[Hub] Inference loop error for %s", ps.name)
    finally:
        logger.info("[Hub] Player %s loop ended: reason=%s, blocks=%d", ps.name, end_reason, block_num)
        try:
            await loop.run_in_executor(None, lambda: worker.end_session.remote())
        except Exception:
            pass
        # Notify the player client so the UI knows the session ended
        try:
            await ps.websocket.send_json({
                "type": "session_ended",
                "reason": end_reason,
                "blocks": block_num,
            })
        except Exception:
            pass
        ps.status = "disconnected"
        if mrs.is_active:
            mrs.active_count = max(0, mrs.active_count - 1)
            await _modal_broadcast_player_list()
            await _modal_activate_next_queued()
        else:
            logger.info("[Hub] (room was already closed)")


async def _modal_generate_caption(
    mrs: ModalRoomState, ps: ModalPlayerState, frame_b64: str, block_num: int
):
    """Generate caption for a player's frame via Claude API."""
    try:
        import anthropic

        def _call_claude():
            client = anthropic.Anthropic()
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

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _call_claude)

        lines = result.strip().split("\n", 1)
        color = "#FFFFFF"
        caption_text = result
        if len(lines) == 2 and lines[0].strip() in ("#000000", "#FFFFFF"):
            color = lines[0].strip()
            caption_text = lines[1].strip()

        try:
            await ps.websocket.send_json({
                "type": "caption",
                "block": block_num,
                "text": caption_text,
                "color": color,
            })
        except Exception:
            pass

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
        logger.warning("[Hub] Caption failed for %s block %d: %s", ps.name, block_num, e)


async def _close_modal_room():
    """Close the room: end all workers, notify all players."""
    global modal_room_state
    mrs = modal_room_state
    if mrs is None:
        return
    mrs.is_active = False

    for pid, ps in list(mrs.players.items()):
        try:
            ps.action_queue.put_nowait(None)
        except Exception:
            pass
        if ps.inference_task and not ps.inference_task.done():
            ps.inference_task.cancel()
        try:
            await ps.websocket.send_json({"type": "room_closed"})
            await ps.websocket.close()
        except Exception:
            pass

    try:
        await mrs.host_ws.send_json({"type": "modal_room_closed"})
    except Exception:
        pass

    # Clean up unused pre-initialized workers
    loop = asyncio.get_event_loop()
    for worker, _ in mrs.ready_workers:
        try:
            await loop.run_in_executor(None, lambda w=worker: w.end_session.remote())
        except Exception:
            pass
    mrs.ready_workers.clear()

    logger.info("[Hub] Room %s closed (%d players)", mrs.room_id, len(mrs.players))
    modal_room_state = None


async def _background_preinit_pool(seed_image_b64: str, mode: str, count: int = 8):
    """Pre-initialize a global pool of workers in the background.

    Called during solo play so workers are ready before the room even opens.
    """
    global _preinit_pool, _preinit_seed_b64
    _preinit_seed_b64 = seed_image_b64
    _preinit_pool.clear()

    logger.info("[Hub] Background pre-init: spawning %d workers...", count)
    loop = asyncio.get_event_loop()
    WorkerCls = _get_modal_worker_class()

    async def _init_one(i):
        try:
            t0 = time.perf_counter()
            w = WorkerCls(session_id=str(uuid.uuid4()))
            result = await loop.run_in_executor(
                None, lambda: w.init_session.remote(
                    image_b64=seed_image_b64, mode=mode
                )
            )
            elapsed = time.perf_counter() - t0
            _preinit_pool.append((w, result))
            logger.info("[Hub] Background pre-init worker %d ready (%.1fs)", i, elapsed)
        except Exception as e:
            logger.warning("[Hub] Background pre-init worker %d failed: %s", i, e)

    tasks = [asyncio.create_task(_init_one(i)) for i in range(count)]
    await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("[Hub] Background pre-init complete: %d workers ready", len(_preinit_pool))


def _transfer_pool_to_room(mrs: ModalRoomState):
    """Move pre-initialized workers from global pool to room state if seed matches."""
    global _preinit_pool, _preinit_seed_b64
    if _preinit_seed_b64 == mrs.seed_image_b64 and _preinit_pool:
        transferred = len(_preinit_pool)
        mrs.ready_workers.extend(_preinit_pool)
        _preinit_pool = []
        _preinit_seed_b64 = ""
        logger.info("[Hub] Transferred %d pre-init workers to room %s", transferred, mrs.room_id)
        return transferred
    return 0


async def _cleanup_preinit_pool():
    """End sessions on any leftover global pool workers."""
    global _preinit_pool, _preinit_seed_b64, _preinit_task
    if not _preinit_pool:
        return
    logger.info("[Hub] Cleaning up %d unused pre-init workers", len(_preinit_pool))
    loop = asyncio.get_event_loop()
    for w, _ in _preinit_pool:
        try:
            await loop.run_in_executor(None, lambda w=w: w.end_session.remote())
        except Exception:
            pass
    _preinit_pool.clear()
    _preinit_seed_b64 = ""
    if _preinit_task and not _preinit_task.done():
        _preinit_task.cancel()
    _preinit_task = None


async def _preinit_workers(mrs: ModalRoomState):
    """Pre-initialize workers for a room (used when no global pool exists)."""
    needed = mrs.pool_target - len(mrs.ready_workers)
    if needed <= 0:
        return
    logger.info("[Hub] Pre-initializing %d more workers for room %s...", needed, mrs.room_id)
    loop = asyncio.get_event_loop()
    WorkerCls = _get_modal_worker_class()

    async def _init_one(i):
        try:
            t0 = time.perf_counter()
            w = WorkerCls(session_id=str(uuid.uuid4()))
            result = await loop.run_in_executor(
                None, lambda: w.init_session.remote(
                    image_b64=mrs.seed_image_b64, mode=mrs.mode
                )
            )
            elapsed = time.perf_counter() - t0
            if mrs.is_active:
                mrs.ready_workers.append((w, result))
                logger.info("[Hub] Pre-init worker %d ready (%.1fs)", i, elapsed)
            else:
                await loop.run_in_executor(None, lambda: w.end_session.remote())
        except Exception as e:
            logger.warning("[Hub] Pre-init worker %d failed: %s", i, e)

    tasks = [asyncio.create_task(_init_one(i)) for i in range(needed)]
    await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("[Hub] Pre-init complete: %d workers ready", len(mrs.ready_workers))


async def _refill_worker_pool(mrs: ModalRoomState):
    """Top up the ready_workers pool after one is consumed."""
    if not mrs.is_active:
        return
    needed = mrs.pool_target - len(mrs.ready_workers)
    if needed <= 0:
        return
    logger.info("[Hub] Refilling pool: spawning %d more workers", needed)
    loop = asyncio.get_event_loop()
    WorkerCls = _get_modal_worker_class()

    for i in range(needed):
        try:
            w = WorkerCls(session_id=str(uuid.uuid4()))
            result = await loop.run_in_executor(
                None, lambda: w.init_session.remote(
                    image_b64=mrs.seed_image_b64, mode=mrs.mode
                )
            )
            if mrs.is_active:
                mrs.ready_workers.append((w, result))
                logger.info("[Hub] Pool refill: worker ready")
            else:
                await loop.run_in_executor(None, lambda: w.end_session.remote())
                break
        except Exception as e:
            logger.warning("[Hub] Pool refill failed: %s", e)


# ── ASGI entry point ──


@app.function(
    image=hub_image,
    secrets=[
        modal.Secret.from_name("anthropic-api-key"),
        modal.Secret.from_name("google-api-key"),
        modal.Secret.from_name("suno-cookie"),
    ],
    max_containers=1,
    scaledown_window=600,
    timeout=86400,  # 24 hours — WebSocket connections must stay alive
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def web():
    web_app = FastAPI(title="Ephemeral")

    # ── Static file routes ──

    @web_app.get("/", response_class=HTMLResponse)
    async def get_index():
        return FileResponse("/app/web_ui/index.html")

    @web_app.get("/app.html", response_class=HTMLResponse)
    async def get_app():
        return FileResponse("/app/web_ui/app.html")

    @web_app.get("/assets/{filename}")
    async def get_asset(filename: str):
        return FileResponse(f"/app/web_ui/assets/{filename}")

    @web_app.get("/modal-room/{room_id}", response_class=HTMLResponse)
    async def get_modal_room(room_id: str):
        if modal_room_state is None or modal_room_state.room_id != room_id or not modal_room_state.is_active:
            return PlainTextResponse("Room not found", status_code=404)
        return FileResponse("/app/web_ui/remote.html")

    # ── API routes ──

    @web_app.get("/api/status")
    async def get_status():
        return {
            "ready": True,
            "running": solo_play_state is not None or modal_room_state is not None,
            "mode": "universal",
        }

    @web_app.get("/api/images")
    async def list_images():
        images = []
        demo_dir = "/app/demo_images"
        if os.path.exists(demo_dir):
            for subdir in sorted(os.listdir(demo_dir)):
                subdir_path = os.path.join(demo_dir, subdir)
                if os.path.isdir(subdir_path):
                    for img in sorted(os.listdir(subdir_path)):
                        if img.endswith((".png", ".jpg", ".jpeg")):
                            images.append(f"demo_images/{subdir}/{img}")
        return {"images": images}

    @web_app.get("/demo_images/{subdir}/{filename}")
    async def get_demo_image(subdir: str, filename: str):
        return FileResponse(f"/app/demo_images/{subdir}/{filename}")

    # ── Host WebSocket ──

    @web_app.websocket("/ws")
    async def host_websocket(websocket: WebSocket):
        global modal_room_state, solo_play_state
        await websocket.accept()
        logger.info("[Hub] Host connected")
        _pending_music_prompt = ""

        try:
            while True:
                data = await websocket.receive_json()
                msg_type = data.get("type")

                if msg_type == "start":
                    # Store prompt early — captcha_token may arrive before solo_play_state is set
                    _pending_music_prompt = data.get("prompt", "")
                    asyncio.create_task(_start_solo_play(websocket, data))

                elif msg_type == "action":
                    # Solo mode: push to solo action queue
                    if solo_play_state and solo_play_state.websocket == websocket:
                        await solo_play_state.action_queue.put({
                            "keyboard": data.get("keyboard", "q"),
                            "mouse": data.get("mouse", "u"),
                        })

                elif msg_type == "captcha_token":
                    token = data.get("token", "")
                    if token:
                        # Use prompt/image from solo_play_state if ready, else fall back to pending
                        image_b64 = None
                        prompt_text = _pending_music_prompt
                        if solo_play_state:
                            image_b64 = getattr(solo_play_state, "_music_image_b64", None)
                            prompt_text = getattr(solo_play_state, "_music_prompt", "") or prompt_text
                        asyncio.create_task(
                            _generate_music(prompt_text, image_b64, websocket, token)
                        )

                elif msg_type == "open_modal_room":
                    if modal_room_state is not None:
                        await websocket.send_json({"type": "error", "message": "Room already open"})
                        continue

                    seed_image_b64 = _resolve_seed_image(data)
                    prompt = data.get("prompt", "")
                    mode = data.get("mode", "universal")

                    # Generate image with Gemini if prompt provided
                    if prompt:
                        try:
                            ref_image = seed_image_b64 or None
                            seed_image_b64 = await _generate_image_with_gemini(
                                prompt, ref_image, websocket
                            )
                            await websocket.send_json({
                                "type": "image_preview",
                                "data": seed_image_b64,
                            })
                        except Exception as e:
                            logger.exception("[Hub] Gemini failed for room")
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Image generation failed: {str(e)[:100]}",
                            })
                            continue

                    if not seed_image_b64:
                        await websocket.send_json({"type": "error", "message": "Provide an image or prompt"})
                        continue

                    rid = _generate_room_id()
                    modal_room_state = ModalRoomState(
                        room_id=rid,
                        host_ws=websocket,
                        seed_image_b64=seed_image_b64,
                        mode=mode,
                    )

                    host = websocket.headers.get("host", "localhost")
                    scheme = "https" if "modal" in host else "http"
                    url = f"{scheme}://{host}/modal-room/{rid}"

                    logger.info("[Hub] Room opened: %s -> %s", rid, url)
                    await websocket.send_json({
                        "type": "modal_room_opened",
                        "room_id": rid,
                        "url": url,
                    })

                    # Transfer whatever workers are ready from the background pool.
                    # Do NOT spawn a second batch — the background task from solo play
                    # will continue filling _preinit_pool, and spawn_player will grab from it.
                    transferred = _transfer_pool_to_room(modal_room_state)
                    logger.info("[Hub] Transferred %d pre-init workers to room (background pool still running)", transferred)

                elif msg_type == "close_modal_room":
                    if modal_room_state and modal_room_state.host_ws == websocket:
                        await _close_modal_room()

                elif msg_type == "stop":
                    await _stop_solo_play()
                    if modal_room_state and modal_room_state.host_ws == websocket:
                        await _close_modal_room()
                    await websocket.send_json({"type": "stopped"})

        except WebSocketDisconnect:
            logger.info("[Hub] Host disconnected")
            await _stop_solo_play()
            await _cleanup_preinit_pool()
            if modal_room_state and modal_room_state.host_ws == websocket:
                await _close_modal_room()
        except Exception:
            logger.exception("[Hub] Host WebSocket error")
            await _stop_solo_play()
            await _cleanup_preinit_pool()
            if modal_room_state and modal_room_state.host_ws == websocket:
                await _close_modal_room()

    # ── Player WebSocket ──

    @web_app.websocket("/ws/modal-room/{room_id}")
    async def player_websocket(websocket: WebSocket, room_id: str):
        global modal_room_state
        await websocket.accept()

        mrs = modal_room_state
        if mrs is None or mrs.room_id != room_id or not mrs.is_active:
            await websocket.close(code=4004, reason="Room not found")
            return

        player_name = "Player"
        try:
            first = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
            if first.get("type") == "join":
                player_name = (first.get("name") or "Player")[:16]
        except Exception:
            await websocket.close(code=4000, reason="Join timeout")
            return

        player_id = f"p_{_generate_room_id(6)}"
        ps = ModalPlayerState(
            player_id=player_id,
            name=player_name,
            websocket=websocket,
            worker=None,
            status="queued",
        )
        mrs.players[player_id] = ps

        logger.info("[Hub] %s (%s) joined room %s", player_name, player_id, room_id)

        if mrs.active_count < mrs.max_active:
            asyncio.create_task(_modal_spawn_player(mrs, ps))
        else:
            mrs.queue.append(player_id)
            position = len(mrs.queue)
            await websocket.send_json({
                "type": "queued",
                "position": position,
                "message": f"You are #{position} in queue. Please wait...",
            })
            logger.info("[Hub] %s queued at position %d", player_name, position)

        await _modal_broadcast_player_list()

        try:
            while True:
                data = await websocket.receive_json()
                msg_type = data.get("type")

                if msg_type == "action":
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
            try:
                ps.action_queue.put_nowait(None)
            except Exception:
                pass
            if player_id in mrs.queue:
                mrs.queue.remove(player_id)
            logger.info("[Hub] %s (%s) disconnected", player_name, player_id)
            await _modal_broadcast_player_list()

    return web_app
