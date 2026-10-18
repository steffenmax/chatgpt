"""Core agent: turns a BlueBubbles webhook event into (maybe) a reply."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any

from .bluebubbles import BlueBubblesClient
from .config import Settings
from .context import ContextStore
from .imagegen import ImageJob, ImageQueue
from .llm import TextBackend, build_messages, clean_reply
from .logsafe import exc_name, log_event
from .triggers import detect, help_text

NEW_MESSAGE = "new-message"


class SenderResolver:
    """Maps a handle address to a display name. Static map first, then BlueBubbles contacts."""

    def __init__(self, bb: BlueBubblesClient | None, static: dict[str, str], use_contacts: bool = True):
        self._bb = bb
        self._static = dict(static)
        self._use_contacts = use_contacts and bb is not None
        self._cache: dict[str, str] = {}

    async def name_for(self, address: str | None) -> str:
        if not address:
            return "someone"
        if address in self._static:
            return self._static[address]
        if address in self._cache:
            return self._cache[address]
        name = address
        if self._use_contacts:
            try:
                contacts = await self._bb.query_contacts([address])  # type: ignore[union-attr]
                name = _pick_contact_name(contacts) or address
            except Exception as exc:  # noqa: BLE001
                log_event("contacts.lookup_failed", error=exc_name(exc))
        self._cache[address] = name
        return name


def _pick_contact_name(contacts: Any) -> str | None:
    if not isinstance(contacts, list):
        return None
    for c in contacts:
        if not isinstance(c, dict):
            continue
        for key in ("displayName", "nickname"):
            v = c.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        first = (c.get("firstName") or "").strip()
        last = (c.get("lastName") or "").strip()
        if first or last:
            return (first + " " + last).strip()
    return None


class Agent:
    def __init__(
        self,
        settings: Settings,
        bb: BlueBubblesClient,
        llm: TextBackend,
        context: ContextStore,
        images: ImageQueue | None,
        resolver: SenderResolver | None = None,
        llm_gate: asyncio.Semaphore | None = None,
    ):
        self.s = settings
        self.bb = bb
        self.llm = llm
        self.context = context
        self.images = images
        self.resolver = resolver or SenderResolver(bb, settings.sender_names)
        self._allowed = set(settings.allowed_chat_guids)
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._unlisted_logged: dict[str, float] = {}
        # One LLM call at a time; the image queue holds the same gate while generating.
        self._llm_gate = llm_gate or asyncio.Semaphore(1)
        self._tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------------ intake
    def schedule(self, payload: dict[str, Any]) -> None:
        """Fire-and-forget so the webhook returns immediately."""
        t = asyncio.create_task(self.process(payload))
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    async def process(self, payload: dict[str, Any]) -> str:
        """Returns a short status string (used by tests and logs). Never contains text."""
        if not isinstance(payload, dict) or payload.get("type") != NEW_MESSAGE:
            return "ignored:event"
        data = payload.get("data")
        if not isinstance(data, dict):
            return "ignored:malformed"
        if data.get("isFromMe"):
            return "ignored:from_me"
        guid = data.get("guid")
        if isinstance(guid, str) and guid:
            if guid in self._seen:
                return "ignored:duplicate"
            self._seen[guid] = None
            while len(self._seen) > 1000:
                self._seen.popitem(last=False)
        if data.get("associatedMessageType") or data.get("itemType"):
            return "ignored:reaction_or_system"

        chats = data.get("chats") or []
        chat_guid = chats[0].get("guid") if chats and isinstance(chats[0], dict) else None
        if not chat_guid:
            log_event("message.no_chat_guid", message_guid=guid)
            return "ignored:no_chat"

        text = data.get("text") if isinstance(data.get("text"), str) else ""
        trigger = detect(text, self.s.bot_name)

        if chat_guid not in self._allowed:
            if trigger.kind != "none":
                self._log_unlisted(chat_guid)
            return "ignored:not_allowlisted"

        if not text:
            return "ignored:empty"

        address = (data.get("handle") or {}).get("address") if isinstance(data.get("handle"), dict) else None
        sender = await self.resolver.name_for(address)

        if trigger.kind in ("none", "mention"):
            self.context.add(chat_guid, sender, text)
        if trigger.kind == "none":
            return "stored"

        log_event("trigger", chat_guid=chat_guid, kind=trigger.kind)
        try:
            if trigger.kind == "help":
                await self._say(chat_guid, help_text(self.s.bot_name, self.s.image_enabled and self.images is not None))
            elif trigger.kind == "forget":
                self.context.forget(chat_guid)
                log_event("context.forgot", chat_guid=chat_guid)
                await self._say(chat_guid, "done, wiped everything i had from this chat", remember=False)
            elif trigger.kind == "image":
                await self._handle_image(chat_guid, trigger.arg)
            elif trigger.kind == "mention":
                await self._handle_mention(chat_guid)
        except Exception as exc:  # noqa: BLE001
            log_event("handler.error", chat_guid=chat_guid, kind=trigger.kind, error=exc_name(exc))
            return f"error:{exc_name(exc)}"
        return f"handled:{trigger.kind}"

    # ---------------------------------------------------------------- handlers
    async def _handle_mention(self, chat_guid: str) -> None:
        history = self.context.get(chat_guid)
        messages = build_messages(self.s.persona, self.s.bot_name, history)
        started = time.monotonic()
        async with self._llm_gate:
            raw = await self.llm.reply(self.s.persona, messages)
        reply = clean_reply(raw, self.s.bot_name)
        log_event(
            "llm.reply",
            chat_guid=chat_guid,
            backend=self.llm.name,
            duration_ms=int((time.monotonic() - started) * 1000),
            chars=len(reply),
        )
        if not reply:
            reply = "hm, blanked. say that again?"
        await self._say(chat_guid, reply)

    async def _handle_image(self, chat_guid: str, prompt: str) -> None:
        if not self.s.image_enabled or self.images is None:
            await self._say(chat_guid, "image generation is turned off on my end", remember=False)
            return
        prompt = prompt.strip()
        if not prompt:
            await self._say(chat_guid, "give me something to draw: /image a corgi surfing", remember=False)
            return
        if len(prompt) > self.s.image_prompt_max_chars:
            prompt = prompt[: self.s.image_prompt_max_chars]
        job = ImageJob(chat_guid=chat_guid, prompt=prompt, job_id=uuid.uuid4().hex[:8])
        ahead = self.images.qsize()
        note = "on it, give me a minute" if ahead == 0 else f"on it, {ahead} ahead of you in the queue"
        await self._say(chat_guid, note, remember=False)
        await self.images.submit(job)

    async def send_image(self, job: ImageJob, png: Path) -> None:
        await self.bb.send_attachment(job.chat_guid, png, name=f"{self.s.bot_name.lower()}-{job.job_id}.png")
        log_event("image.sent", chat_guid=job.chat_guid, job_id=job.job_id)

    async def image_failed(self, job: ImageJob, reason: str) -> None:
        await self._say(job.chat_guid, "that image didn't come out, sorry. try again in a bit?", remember=False)

    async def _say(self, chat_guid: str, text: str, remember: bool = True) -> None:
        await self.bb.send_text(chat_guid, text)
        log_event("message.sent", chat_guid=chat_guid, chars=len(text))
        if remember:
            self.context.add(chat_guid, self.s.bot_name, text, from_bot=True)

    # ----------------------------------------------------------------- helpers
    def _log_unlisted(self, chat_guid: str) -> None:
        now = time.monotonic()
        last = self._unlisted_logged.get(chat_guid, 0.0)
        if now - last > 60:
            self._unlisted_logged[chat_guid] = now
            log_event("chat.not_allowlisted", chat_guid=chat_guid, allowed=False)

    async def drain(self) -> None:
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
