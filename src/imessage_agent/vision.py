"""Captions for incoming photos, via a local vision model in Ollama.

The image bytes live in RAM and in a 0600 file inside a 0700 temp dir only for the
duration of one `sips` conversion (HEIC -> JPEG, downscaled); the caption that comes
back is what enters the chat context. Nothing is logged but counts and error classes.
"""

from __future__ import annotations

import asyncio
import base64
import os
import shutil
import tempfile
from pathlib import Path

import httpx

CAPTION_PROMPT = (
    "Describe this photo in one or two casual sentences for a friend who can't see it: what's "
    "in it, who or what is doing what, any readable text, the vibe. No preamble, no lists."
)


class OllamaVision:
    name = "ollama-vision"

    def __init__(self, base_url: str, model: str, max_width: int = 1024, keep_alive: str = "2m", timeout: float = 120.0):
        self.model = model
        self.max_width = max_width
        self.keep_alive = keep_alive
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout)

    async def _to_jpeg(self, data: bytes) -> bytes:
        """Normalise with macOS `sips`: HEIC/PNG/anything -> JPEG, downscaled. Local only."""
        tmp = Path(tempfile.mkdtemp(prefix="imsg-photo-"))
        os.chmod(tmp, 0o700)
        try:
            src, dst = tmp / "in", tmp / "out.jpg"
            src.write_bytes(data)
            os.chmod(src, 0o600)
            proc = await asyncio.create_subprocess_exec(
                "sips", "-s", "format", "jpeg", "-s", "formatOptions", "80",
                "--resampleHeightWidthMax", str(self.max_width), str(src), "--out", str(dst),
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=60)
            if proc.returncode == 0 and dst.exists():
                return dst.read_bytes()
            return data  # let the model try the original
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    async def caption(self, data: bytes) -> str:
        jpeg = await self._to_jpeg(data)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": CAPTION_PROMPT, "images": [base64.b64encode(jpeg).decode()]}],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0.2, "num_predict": 120},
        }
        r = await self._client.post("/api/chat", json=payload)
        r.raise_for_status()
        text = ((r.json().get("message") or {}).get("content") or "").strip()
        return " ".join(text.split())[:300]

    async def has_model(self) -> bool | None:
        try:
            r = await self._client.get("/api/tags", timeout=5.0)
            r.raise_for_status()
            names = {m.get("name") for m in r.json().get("models", [])}
        except (httpx.HTTPError, ValueError):
            return None
        want = self.model if ":" in self.model else f"{self.model}:latest"
        return want in names or self.model in names

    async def aclose(self) -> None:
        await self._client.aclose()
