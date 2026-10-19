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

# --- cloud fact lookups -------------------------------------------------------
NEEDS_FACTS_SYSTEM = (
    "You are a strict classifier for a group-chat bot. Decide whether answering the LAST "
    "message requires information the bot cannot reliably know from memory: anything current "
    "or time-sensitive (news, scores, schedules, prices, weather, hours, who won or what happened "
    "recently) or a precise fact that should be looked up (stats, dates, numbers, specs, addresses). "
    "Opinions, jokes, roasts, advice, chit-chat and general knowledge do NOT. "
    "Answer with exactly one word: yes or no."
)

CLOUD_FACTS_ADDENDUM = (
    "\n\nFor this reply you have a web search tool. Use it when the question needs current or "
    "precise facts (scores, schedules, news, prices, weather, stats); one or two quick searches "
    "are enough. You already told the chat you were going to look it up, so come straight back "
    "with what you found, in the same short texting style described above and fully in character: "
    "facts first, then the bit, no markdown, no links unless someone asks for one, and name the "
    "source in a few words only if it matters."
)

# Sent before a cloud lookup so the chat knows the pause is on purpose.
LOOKUP_NOTICES = [
    "hold on, gotta look that up",
    "one sec, lemme check that",
    "gimme a minute, i'll find out",
    "not off the top of my head, checking",
    "hang on, looking it up",
]


REACTION_ADDENDUM = (
    "\n\nRight now you are NOT replying. You are only deciding whether the LAST message deserves a "
    "quick reaction, the way a friend fires off 'lmaooo', '😂😂', 'wait what', '💀' or 'W'. The default "
    "is NONE: plans, logistics, questions, money, 'ok', 'see you', and anything ordinary get NONE. React "
    "only when the message is genuinely funny, wild, a big win or a big L, or begging to be roasted. "
    "Answer with exactly the word NONE, or the reaction itself: at most 6 words or a few emoji, nothing "
    "else, no quotes, no explanation."
)

MEMORY_SYSTEM = (
    "You keep private notes for {bot_name} about one group chat so {bot_name} can talk to these people "
    "like it knows them. You get the current notes and the most recent messages. Output the UPDATED "
    "notes as plain markdown and nothing else, under {max_chars} characters, with exactly these sections: "
    "'## People' (one bullet per person: how they text, personality tells, what they care about, running "
    "jokes about them) and '## Group' (dynamics, inside jokes, recurring topics, who ribs whom). Keep what "
    "is still true, merge new tells, drop stale or trivial stuff, prefer durable traits over one-off events. "
    "Never include phone numbers, emails, addresses, or anything that looks like a secret, and never copy "
    "long quotes. If nothing new was learned, output the current notes unchanged."
)


def memory_messages(current_notes: str, history: list[ContextMessage]) -> list[dict[str, str]]:
    lines = [f"{m.sender}: {m.text}" for m in history]
    content = (
        "Current notes:\n" + (current_notes.strip() or "(none yet)") +
        "\n\nRecent messages:\n" + "\n".join(lines) +
        "\n\nOutput the updated notes now."
    )
    return [{"role": "user", "content": content}]


def parse_reaction(text: str) -> str | None:
    t = _THINK_RE.sub("", text or "").strip().strip('"').strip("'").strip()
    if not t or t.upper().startswith("NONE") or "\n" in t or len(t) > 40:
        return None
    return t


def facts_router_messages(history: list[ContextMessage]) -> list[dict[str, str]]:
    """The recent chat as one user turn, for the yes/no router."""
    lines = [f"{m.sender}: {m.text}" for m in history[-8:]]
    return [{"role": "user", "content": "\n".join(lines) + "\n\nDoes the LAST message need outside information? yes or no."}]


def parse_yes_no(text: str) -> bool:
    words = _THINK_RE.sub("", text or "").strip().lower().split()
    return bool(words) and words[0].strip(".,!") == "yes"


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
        num_ctx: int = 8192,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.keep_alive = keep_alive
        self.num_ctx = num_ctx
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def reply(self, system: str, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens, "num_ctx": self.num_ctx},
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

    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int = 400,
        timeout: float = 60.0,
        web_search: bool = False,
        max_searches: int = 3,
    ):
        import anthropic  # optional dependency: uv sync --extra cloud

        self.model = model
        self.max_tokens = max_tokens
        self.web_search = web_search
        self.max_searches = max_searches
        self._client = anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout, max_retries=1)

    async def reply(self, system: str, messages: list[dict[str, str]]) -> str:
        kwargs: dict = {}
        if self.web_search and self.max_searches > 0:
            # Server-side tool: the search runs on Anthropic's side, results come back in
            # the same response. No extra round trips from this process.
            kwargs["tools"] = [{"type": "web_search_20260209", "name": "web_search", "max_uses": self.max_searches}]
        resp = await self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
            output_config={"effort": "low"},
            **kwargs,
        )
        if resp.stop_reason == "refusal":
            return "can't help with that one"
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

    async def healthy(self) -> bool:
        return True  # no probe: we never call out unless a user actually triggers the bot

    async def aclose(self) -> None:
        await self._client.close()
