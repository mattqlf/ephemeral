"""
Mock LingBot-World server for local UI testing without GPUs.

Simulates the full WebSocket protocol with fake frame generation
(solid color frames with text overlay) so you can test the entire
frontend flow: start, direction input, progress updates, frame
streaming, chunk completion.

Usage:
    uv run python -m apps.servers.web_server_lingbot_mock
    # Then open http://localhost:8889
"""

import asyncio
import base64
import io
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn


app = FastAPI(title="LingBot-World Mock Server")

# State
is_running = False
should_stop = False
action_queue: asyncio.Queue = asyncio.Queue()


def make_fake_frame(width: int, height: int, chunk: int, frame: int,
                    step: int = 0, direction: str = "") -> str:
    """Generate a colored test frame with info text overlay."""
    # Shift hue based on chunk/frame for visual feedback
    r = int(40 + 15 * (chunk % 8))
    g = int(20 + 10 * ((chunk * 3 + frame) % 12))
    b = int(80 + 20 * ((chunk * 7 + frame * 3) % 10))

    img = Image.new("RGB", (width, height), (r, g, b))
    draw = ImageDraw.Draw(img)

    # Draw grid lines for spatial reference
    for x in range(0, width, 80):
        draw.line([(x, 0), (x, height)], fill=(r + 30, g + 30, b + 30), width=1)
    for y in range(0, height, 80):
        draw.line([(0, y), (width, y)], fill=(r + 30, g + 30, b + 30), width=1)

    # Text info
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 18)
        font_sm = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 14)
    except Exception:
        font = ImageFont.load_default()
        font_sm = font

    draw.text((20, 20), "LINGBOT-WORLD [MOCK]", fill=(200, 200, 200), font=font)
    draw.text((20, 50), f"Chunk: {chunk}  Frame: {frame}", fill=(180, 180, 180), font=font_sm)
    if direction:
        draw.text((20, 75), f"Direction: {direction}", fill=(123, 97, 255), font=font_sm)
    if step > 0:
        draw.text((20, 100), f"Step: {step}", fill=(150, 150, 150), font=font_sm)

    # Moving element to show frame progression
    cx = int(100 + frame * 40) % (width - 40)
    cy = height // 2 + int(30 * np.sin(frame * 0.5 + chunk))
    draw.ellipse([cx, cy, cx + 30, cy + 30], fill=(123, 97, 255))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


async def run_mock_generation(websocket: WebSocket, data: dict):
    global is_running, should_stop

    is_running = True
    should_stop = False

    # Clear queue
    while not action_queue.empty():
        try:
            action_queue.get_nowait()
        except Exception:
            break

    num_chunks = int(data.get("num_chunks", 5))
    steps = int(data.get("steps", 25))
    size = data.get("size", "480*832")
    camera_preset = data.get("camera_preset", "interactive")
    prompt = data.get("prompt", "mock scene")

    parts = size.split("*")
    h, w = int(parts[0]), int(parts[1])

    # Send initial frame
    await websocket.send_json({
        "type": "frame",
        "data": make_fake_frame(w, h, 0, 0, direction="start"),
        "chunk": 0,
        "frame": 0,
    })

    try:
        for chunk_idx in range(num_chunks):
            if should_stop:
                break

            direction = "none"

            if camera_preset == "interactive":
                # Ask for direction
                await websocket.send_json({
                    "type": "request_action",
                    "chunk": chunk_idx + 1,
                })

                # Wait for user input
                try:
                    action = await asyncio.wait_for(action_queue.get(), timeout=300.0)
                    if action is None:
                        break
                    direction = action.get("direction", "none")
                except asyncio.TimeoutError:
                    direction = "none"
            else:
                direction = ["forward", "left", "right", "forward"][chunk_idx % 4]

            # Simulate denoising steps
            mock_steps = min(steps, 8)  # Fewer steps for faster mock
            for step in range(mock_steps):
                if should_stop:
                    break
                await websocket.send_json({
                    "type": "generating",
                    "chunk": chunk_idx + 1,
                    "step": step + 1,
                    "total_steps": mock_steps,
                })
                await asyncio.sleep(0.15)  # Simulate compute time

            if should_stop:
                break

            # Stream 17 mock frames
            for frame_idx in range(17):
                if should_stop:
                    break
                await websocket.send_json({
                    "type": "frame",
                    "data": make_fake_frame(w, h, chunk_idx + 1, frame_idx,
                                            step=mock_steps, direction=direction),
                    "chunk": chunk_idx + 1,
                    "frame": frame_idx,
                })
                await asyncio.sleep(0.06)

            if not should_stop:
                await websocket.send_json({
                    "type": "chunk_complete",
                    "chunk": chunk_idx + 1,
                })

        await websocket.send_json({"type": "complete"})

    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        is_running = False


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
    return {"ready": True, "running": is_running, "model": "lingbot-world-mock"}


@app.get("/api/presets")
async def list_presets():
    presets = []
    preset_dir = "configs/lingbot/camera_presets"
    if os.path.exists(preset_dir):
        for name in sorted(os.listdir(preset_dir)):
            path = os.path.join(preset_dir, name)
            if os.path.isdir(path) and os.path.exists(os.path.join(path, "poses.npy")):
                poses = np.load(os.path.join(path, "poses.npy"))
                prompt = ""
                prompt_file = os.path.join(path, "prompt.txt")
                if os.path.exists(prompt_file):
                    with open(prompt_file) as f:
                        prompt = f.read().strip()[:100]
                presets.append({
                    "name": name,
                    "frames": len(poses),
                    "prompt_preview": prompt,
                })
    return {"presets": presets}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global should_stop
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "start":
                if not is_running:
                    asyncio.create_task(run_mock_generation(websocket, data))
                else:
                    await websocket.send_json({
                        "type": "error", "message": "Already running"})

            elif msg_type == "action":
                if is_running:
                    await action_queue.put(data)

            elif msg_type == "stop":
                should_stop = True
                await action_queue.put(None)
                await websocket.send_json({"type": "stopped"})

    except WebSocketDisconnect:
        should_stop = True
    except Exception as e:
        print(f"WebSocket error: {e}")
        should_stop = True


if __name__ == "__main__":
    print("=" * 50)
    print("LingBot-World MOCK Server")
    print("=" * 50)
    print("No GPU required - generates colored test frames")
    print("Open http://localhost:8889 to test the UI")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8889, ws_max_size=50 * 1024 * 1024)
