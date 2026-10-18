"""Text generation backends.

Default: Ollama on localhost (POST /api/chat, stream=false).
Optional: Anthropic API, only when CLOUD_LLM_ENABLED=true. Text only, never images.
"""

from __future__ import annotations

import re
from typing import Protocol

import httpx

from .context import ContextMessage

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def build_messages(persona: str, bot_name: str, history: list[ContextMessage]) -> list[dict[str, str]]:
    """Convert rolling context into chat messages.

    Human messages are prefixed with the sender's name so the model can address people.
    Consecutive human messages are merged into one user turn (models want alternation).
    """
    msgs: list[dict[str, str]] = []
    buf: list[str] = []

    def flush():
        if buf:
            msgs.append({"role": "user", "content": "\n".join(buf)})
            buf.clear()

    for m in history:
        if m.from_bot:
            flush()
            msgs.append({"role": "assistant", "content": m.text})
        else:
            buf.append(f"{m.sender}: {m.text}")
    flush()
    if not msgs or msgs[-1]["role"] != "user":
        msgs.append({"role": "user", "content": f"(someone mentioned {bot_name})"})
    return msgs


def clean_reply(text: str, bot_name: str) -> str:
    text = _THINK_RE.sub("", text).strip()
    # Some models echo a "Name:" prefix the way the history is formatted.
    prefix = f"{bot_name}:"
    if text.lower().startswith(prefix.lower()):
        text = text[len(prefix):].strip()
    return text.strip().strip('"').strip()


class TextBackend(Protocol):
    name: str

    async def reply(self, system: str, messages: list[dict[str, str]]) -> str: ...
    async def healthy(self) -> bool: ...
    async def aclose(self) -> None: ...


class OllamaBackend:
    name = "ollama"

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 300,
        timeout: float = 120.0,
        keep_alive: str = "30m",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.keep_alive = keep_alive
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def reply(self, system: str, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }
        r = await self._client.post("/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
        return (data.get("message") or {}).get("content", "") or ""

    async def healthy(self) -> bool:
        try:
            r = await self._client.get("/api/tags", timeout=5.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def has_model(self) -> bool | None:
        try:
            r = await self._client.get("/api/tags", timeout=5.0)
            r.raise_for_status()
            names = {m.get("name") for m in r.json().get("models", [])}
        except (httpx.HTTPError, ValueError):
            return None
        want = self.model if ":" in self.model else f"{self.model}:latest"
        return want in names or self.model in names

    async def unload(self) -> None:
        """Ask Ollama to drop the model from memory (frees RAM for image generation)."""
        try:
            await self._client.post("/api/generate", json={"model": self.model, "keep_alive": 0}, timeout=30.0)
        except httpx.HTTPError:
            pass

    async def aclose(self) -> None:
        await self._client.aclose()


class AnthropicBackend:
    """Cloud text mode. Only constructed when CLOUD_LLM_ENABLED=true."""

    name = "anthropic"

    def __init__(self, api_key: str, model: str, max_tokens: int = 400, timeout: float = 60.0):
        import anthropic  # optional dependency: uv sync --extra cloud

        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout, max_retries=1)

    async def reply(self, system: str, messages: list[dict[str, str]]) -> str:
        resp = await self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
            output_config={"effort": "low"},
        )
        if resp.stop_reason == "refusal":
            return "can't help with that one"
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

    async def healthy(self) -> bool:
        return True  # no probe: we never call out unless a user actually triggers the bot

    async def aclose(self) -> None:
        await self._client.close()
