"""Suno API client — handles auth, generation, polling, and download."""

import asyncio
import logging
import time
import uuid
import httpx

logger = logging.getLogger("suno.client")


CLERK_BASE = "https://clerk.suno.com"
STUDIO_BASE = "https://studio-api.prod.suno.com"
CLERK_JS_VERSION = "5.117.0"

DEFAULT_COOKIE = (
    "__client=eyJhbGciOiJSUzI1NiIsImtpZCI6InN1bm8tYXBpLXJzMjU2LWtleS0xIiwidHlwIjoiSldUIn0."
    "eyJzdW5vLmNvbS9jbGFpbXMvY2xpZW50X2lkIjoiY2xpZW50XzRaZ0x3NTZBMlVic1FVdXFMcWZaWlgiLCJzdW5vLmNvbS9jbGFpbXMvdG9rZW5fdHlwZSI6InJlZnJlc2giLCJpc3MiOiJodHRwczovL2F1dGguc3Vuby5jb20iLCJleHAiOjE4MDE4Njc2NzR9."
    "jWgIdLrdcnnWVhPrV_oGYBOabpbwCQLJ0A6zszos8ks9MHtOjti5qERz2I8R-6vAqzqtxA0Ukvjs-Dk9wFBgRu-lm5vbo8nLroyimqwn3ixHH2yCqk5C6V_-rax53dO0bM-Hhm6e2JxJ2ix6sMbfhQg-8jRo7fzzQL5gUUoDhn7dQcpUrbMnFxLfS9hMeXknmAuqYARVXKhOC9aBOoUkwIN9FTmL33XDqOi2AKjdAoL_l0rfTUw0TIEVXtTeGVVd8oZDXMP1LQ_YqldFzFEEMyBMNM6sqk384KH4u9o2_0tCiovvdXzces5EaHwYwUXuBkDe_Iwj7Odtqy52j7iLcQ;"
    "__client_uat=1768569564"
)
DEFAULT_ANTHROPIC_KEY = "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"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Origin": "https://suno.com",
    "Referer": "https://suno.com/",
}


class SunoClient:
    def __init__(self, cookie: str = "", jwt: str = "", session_id: str = ""):
        self._cookie = cookie or DEFAULT_COOKIE
        self._sid: str | None = session_id or None
        self._jwt: str | None = jwt or None
        self._jwt_ts: float = time.time() if jwt else 0
        self._http = httpx.AsyncClient(headers=HEADERS, timeout=60)
        logger.info(
            "SunoClient initialized (cookie=%d chars, jwt=%s, session_id=%s)",
            len(self._cookie),
            "provided" if jwt else "none",
            session_id or "none",
        )

    # ── auth ──────────────────────────────────────────────────────────

    async def _get_session_id(self) -> str:
        if self._sid:
            logger.debug("Using cached session_id=%s", self._sid)
            return self._sid
        logger.info("Fetching session ID from Clerk...")
        r = await self._http.get(
            f"{CLERK_BASE}/v1/client",
            params={"_clerk_js_version": CLERK_JS_VERSION},
            headers={"Cookie": self._cookie},
        )
        r.raise_for_status()
        self._sid = r.json()["response"]["last_active_session_id"]
        logger.info("Got session_id=%s", self._sid)
        return self._sid

    async def _refresh_token(self) -> str:
        sid = await self._get_session_id()
        logger.info("Refreshing JWT token (session=%s)...", sid)
        r = await self._http.post(
            f"{CLERK_BASE}/v1/client/sessions/{sid}/tokens",
            params={"_clerk_js_version": CLERK_JS_VERSION},
            headers={"Cookie": self._cookie},
            content="",
        )
        r.raise_for_status()
        self._jwt = r.json()["jwt"]
        self._jwt_ts = time.time()
        logger.info("JWT refreshed (token=%s...)", self._jwt[:40] if self._jwt else "?")
        return self._jwt

    async def _token(self) -> str:
        """Return a valid JWT, refreshing if older than 60 s."""
        age = time.time() - self._jwt_ts
        if not self._jwt or age > 60:
            logger.info("Token stale (age=%.1fs), refreshing...", age)
            await self._refresh_token()
        return self._jwt

    async def _auth_headers(self) -> dict:
        return {"Authorization": f"Bearer {await self._token()}"}

    # ── studio API helpers ────────────────────────────────────────────

    async def _studio_get(self, path: str, **kwargs) -> dict | list:
        logger.info("GET %s%s", STUDIO_BASE, path)
        r = await self._http.get(
            f"{STUDIO_BASE}{path}",
            headers=await self._auth_headers(),
            **kwargs,
        )
        logger.info("GET %s -> %d", path, r.status_code)
        if r.status_code == 401:
            logger.warning("GET %s returned 401, refreshing token and retrying...", path)
            await self._refresh_token()
            r = await self._http.get(
                f"{STUDIO_BASE}{path}",
                headers=await self._auth_headers(),
                **kwargs,
            )
            logger.info("GET %s retry -> %d", path, r.status_code)
        r.raise_for_status()
        return r.json()

    async def _studio_post(self, path: str, json: dict) -> dict:
        logger.info("POST %s%s (payload_keys=%s)", STUDIO_BASE, path, list(json.keys()))
        r = await self._http.post(
            f"{STUDIO_BASE}{path}",
            headers=await self._auth_headers(),
            json=json,
        )
        logger.info("POST %s -> %d", path, r.status_code)
        if r.status_code == 401:
            logger.warning("POST %s returned 401, refreshing token and retrying...", path)
            await self._refresh_token()
            r = await self._http.post(
                f"{STUDIO_BASE}{path}",
                headers=await self._auth_headers(),
                json=json,
            )
            logger.info("POST %s retry -> %d", path, r.status_code)
        r.raise_for_status()
        return r.json()

    # ── public methods ────────────────────────────────────────────────

    async def get_credits(self) -> dict:
        logger.info("Fetching credit balance...")
        data = await self._studio_get("/api/billing/info/")
        monthly_limit = data.get("monthly_limit", "?")
        monthly_usage = data.get("monthly_usage", "?")
        logger.info("Credits: usage=%s / limit=%s", monthly_usage, monthly_limit)
        return data

    async def generate(
        self,
        *,
        prompt: str = "",
        description: str = "",
        tags: str = "",
        title: str = "",
        model: str = "chirp-auk-turbo",
        instrumental: bool = False,
        captcha_token: str = "",
    ) -> dict:
        """Start a generation. Returns response with clips list (usually 2).

        Use `description` for AI-written lyrics, or `prompt` for custom lyrics.
        `captcha_token` is required — an hCaptcha token from the browser.
        """
        payload = {
            "token": captcha_token,
            "generation_type": "TEXT",
            "mv": model,
            "prompt": prompt,
            "make_instrumental": instrumental,
            "user_uploaded_images_b64": None,
            "metadata": {
                "web_client_pathname": "/create",
                "is_max_mode": False,
                "create_mode": "simple" if description else "custom",
                "create_session_token": str(uuid.uuid4()),
                "disable_volume_normalization": False,
                "can_control_sliders": [],
                "lyrics_model": "default",
            },
            "override_fields": [],
            "cover_clip_id": None,
            "cover_start_s": None,
            "cover_end_s": None,
            "persona_id": None,
            "artist_clip_id": None,
            "artist_start_s": None,
            "artist_end_s": None,
            "continue_clip_id": None,
            "continued_aligned_prompt": None,
            "continue_at": None,
            "transaction_uuid": str(uuid.uuid4()),
        }

        if description:
            payload["gpt_description_prompt"] = description
        else:
            payload["tags"] = tags
            payload["title"] = title

        logger.info(
            "Starting generation: model=%s instrumental=%s captcha=%d chars description=%s",
            model,
            instrumental,
            len(captcha_token),
            (description or prompt)[:80],
        )
        data = await self._studio_post("/api/generate/v2-web/", payload)
        clips = data.get("clips", [])
        clip_ids = [c.get("id", "?") for c in clips]
        logger.info("Generation started: %d clips, ids=%s", len(clips), clip_ids)
        return data

    async def generate_lyrics(self, prompt: str) -> dict:
        logger.info("Generating lyrics (prompt=%s)", prompt[:80])
        data = await self._studio_post("/api/generate/lyrics/", {"prompt": prompt})
        logger.info("Lyrics generation started: id=%s", data.get("id", "?"))
        return data

    async def get_lyrics(self, lyrics_id: str) -> dict:
        return await self._studio_get(f"/api/generate/lyrics/{lyrics_id}")

    async def wait_for_lyrics(
        self, lyrics_id: str, timeout: float = 60, poll_interval: float = 2
    ) -> dict:
        logger.info("Waiting for lyrics id=%s (timeout=%.0fs, poll=%.0fs)", lyrics_id, timeout, poll_interval)
        deadline = time.time() + timeout
        poll_count = 0
        while time.time() < deadline:
            data = await self.get_lyrics(lyrics_id)
            poll_count += 1
            status = data.get("status", "unknown")
            logger.info("Lyrics poll #%d: status=%s", poll_count, status)
            if status == "complete":
                logger.info("Lyrics complete after %d polls", poll_count)
                return data
            await asyncio.sleep(poll_interval)
        logger.warning("Lyrics timed out after %.0fs (%d polls)", timeout, poll_count)
        return await self.get_lyrics(lyrics_id)

    async def get_songs(self, ids: list[str]) -> list[dict]:
        logger.debug("Fetching songs: ids=%s", ids)
        return await self._studio_get("/api/feed/", params={"ids": ",".join(ids)})

    async def get_song(self, song_id: str) -> dict:
        songs = await self.get_songs([song_id])
        return songs[0] if songs else {}

    async def wait_for_songs(
        self,
        ids: list[str],
        timeout: float = 300,
        poll_interval: float = 5,
    ) -> list[dict]:
        """Poll until all songs are complete or timeout."""
        logger.info("Waiting for %d songs: ids=%s (timeout=%.0fs, poll=%.0fs)", len(ids), ids, timeout, poll_interval)
        deadline = time.time() + timeout
        poll_count = 0
        while time.time() < deadline:
            songs = await self.get_songs(ids)
            poll_count += 1
            statuses = [s.get("status", "unknown") for s in songs]
            logger.info("Song poll #%d: statuses=%s", poll_count, statuses)
            if all(s == "complete" for s in statuses):
                audio_urls = [s.get("audio_url", "none") for s in songs]
                logger.info(
                    "All %d songs complete after %d polls! audio_urls=%s",
                    len(songs), poll_count, audio_urls,
                )
                return songs
            await asyncio.sleep(poll_interval)
        logger.warning("Songs timed out after %.0fs (%d polls)", timeout, poll_count)
        return await self.get_songs(ids)

    async def download(self, audio_url: str) -> bytes:
        logger.info("Downloading audio: %s", audio_url)
        r = await self._http.get(audio_url)
        r.raise_for_status()
        logger.info("Downloaded %d bytes from %s", len(r.content), audio_url)
        return r.content

    async def close(self):
        await self._http.aclose()
