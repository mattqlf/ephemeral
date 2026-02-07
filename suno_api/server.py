"""FastAPI server wrapping the Suno client."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel

from .client import SunoClient

COOKIE = os.environ.get("SUNO_COOKIE", "")
JWT = os.environ.get("SUNO_JWT", "")
SESSION_ID = os.environ.get("SUNO_SESSION_ID", "")
DOWNLOAD_DIR = Path(os.environ.get("SUNO_DOWNLOAD_DIR", "./downloads"))

client: SunoClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client
    if not COOKIE and not JWT:
        raise RuntimeError(
            "Set SUNO_COOKIE (browser cookie) or SUNO_JWT (token from Clerk response)."
        )
    client = SunoClient(cookie=COOKIE, jwt=JWT, session_id=SESSION_ID)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    yield
    await client.close()


app = FastAPI(title="Suno API", lifespan=lifespan)


# ── schemas ───────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str = ""
    description: str = ""
    tags: str = ""
    title: str = ""
    model: str = "chirp-auk-turbo"
    instrumental: bool = False
    captcha_token: str = ""
    wait: bool = False


class LyricsRequest(BaseModel):
    prompt: str
    wait: bool = True


# ── routes ────────────────────────────────────────────────────────────

@app.get("/credits")
async def credits():
    return await client.get_credits()


@app.post("/generate")
async def generate(req: GenerateRequest):
    """Generate a song. Returns 2 clip variations.

    Requires `captcha_token` (hCaptcha token from browser).
    Set `wait=true` to block until songs are fully generated (~90s).
    """
    data = await client.generate(
        prompt=req.prompt,
        description=req.description,
        tags=req.tags,
        title=req.title,
        model=req.model,
        instrumental=req.instrumental,
        captcha_token=req.captcha_token,
    )

    clips = data.get("clips", [])
    ids = [c["id"] for c in clips]

    if not req.wait:
        return {"clip_ids": ids, "clips": clips}

    songs = await client.wait_for_songs(ids)
    return {"clips": songs}


@app.post("/lyrics")
async def lyrics(req: LyricsRequest):
    """Generate lyrics from a description. No captcha required."""
    data = await client.generate_lyrics(req.prompt)
    lyrics_id = data["id"]
    if not req.wait:
        return {"lyrics_id": lyrics_id}
    return await client.wait_for_lyrics(lyrics_id)


@app.get("/songs")
async def songs(ids: str = Query(..., description="Comma-separated clip IDs")):
    return await client.get_songs(ids.split(","))


@app.get("/songs/{song_id}")
async def song(song_id: str):
    s = await client.get_song(song_id)
    if not s:
        raise HTTPException(404, "Song not found")
    return s


@app.get("/songs/{song_id}/audio")
async def audio(song_id: str):
    """Stream the MP3 audio for a completed song."""
    s = await client.get_song(song_id)
    if not s:
        raise HTTPException(404, "Song not found")
    url = s.get("audio_url")
    if not url:
        raise HTTPException(409, "Song not ready yet")
    data = await client.download(url)
    return Response(content=data, media_type="audio/mpeg")


@app.post("/songs/{song_id}/download")
async def download(song_id: str):
    """Download a song to the local SUNO_DOWNLOAD_DIR."""
    s = await client.get_song(song_id)
    if not s:
        raise HTTPException(404, "Song not found")
    url = s.get("audio_url")
    if not url:
        raise HTTPException(409, "Song not ready yet")

    data = await client.download(url)
    title = s.get("title", song_id).replace("/", "_")
    filename = f"{title}_{song_id[:8]}.mp3"
    path = DOWNLOAD_DIR / filename
    path.write_bytes(data)
    return {"path": str(path), "size_bytes": len(data)}
