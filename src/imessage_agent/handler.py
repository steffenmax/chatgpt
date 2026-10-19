"""Core agent: turns a BlueBubbles webhook event into (maybe) a reply."""

from __future__ import annotations

import asyncio
import random
import time
import uuid
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any

from .bluebubbles import BlueBubblesClient
from .config import Settings
from .context import ContextStore
from .imagegen import ImageJob, ImageQueue
from .llm import (
    CLOUD_FACTS_ADDENDUM,
    LOOKUP_NOTICES,
    NEEDS_FACTS_SYSTEM,
    REACTION_ADDENDUM,
    TextBackend,
    build_messages,
    clean_reply,
    facts_router_messages,
    parse_reaction,
    parse_yes_no,
)
from .logsafe import exc_name, log_event
from .memory import MemoryStore
from .triggers import Trigger, detect, help_text

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
        cloud: TextBackend | None = None,
        memory: MemoryStore | None = None,
    ):
        self.s = settings
        self.bb = bb
        self.llm = llm
        # Optional second backend for questions that need outside information.
        self.cloud = cloud
        # Optional on-disk notes about each chat, maintained by the local model.
        self.memory = memory
        # GUIDs of the bot's own messages (from isFromMe webhooks) so an inline reply
        # to one of them counts as talking to the bot.
        self._bot_guids: OrderedDict[str, None] = OrderedDict()
        self._last_reaction: dict[str, float] = {}
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
        guid = data.get("guid")
        if data.get("isFromMe"):
            if isinstance(guid, str) and guid:
                self._bot_guids[guid] = None
                while len(self._bot_guids) > 500:
                    self._bot_guids.popitem(last=False)
            return "ignored:from_me"
        if isinstance(guid, str) and guid and guid in self._seen:
            log_event("message.ignored", status="duplicate")
            return "ignored:duplicate"
        if data.get("associatedMessageType") or data.get("itemType"):
            return "ignored:reaction_or_system"

        chats = data.get("chats") or []
        chat_guid = chats[0].get("guid") if chats and isinstance(chats[0], dict) else None
        if not chat_guid:
            log_event("message.no_chat_guid", message_guid=guid)
            return "ignored:no_chat"

        text = data.get("text") if isinstance(data.get("text"), str) else ""
        # BlueBubbles delivers the same message more than once, and the first copy can
        # arrive before the text is populated. Only count a delivery as "seen" once it
        # carried text, so a later copy with the text still gets handled.
        if isinstance(guid, str) and guid and text:
            self._seen[guid] = None
            while len(self._seen) > 1000:
                self._seen.popitem(last=False)
        trigger = detect(text, self.s.bot_name)
        reply_to = data.get("threadOriginatorGuid") or data.get("replyToGuid")
        is_reply_to_bot = trigger.kind == "none" and isinstance(reply_to, str) and reply_to in self._bot_guids
        if is_reply_to_bot and text:
            trigger = Trigger("mention", text)

        if chat_guid not in self._allowed:
            if trigger.kind != "none":
                self._log_unlisted(chat_guid)
            return "ignored:not_allowlisted"

        if not text:
            log_event("message.ignored", status="empty", chat_guid=chat_guid)
            return "ignored:empty"

        address = (data.get("handle") or {}).get("address") if isinstance(data.get("handle"), dict) else None
        sender = await self.resolver.name_for(address)

        if trigger.kind in ("none", "mention"):
            self.context.add(chat_guid, sender, text)
        elif trigger.kind == "ask":
            self.context.add(chat_guid, sender, trigger.arg or text)
        if trigger.kind in ("none", "mention", "ask"):
            self._maybe_update_memory(chat_guid)
        if trigger.kind == "none":
            if not self._maybe_quiet_question(chat_guid, text):
                self._maybe_react(chat_guid)
            return "stored"

        if is_reply_to_bot:
            log_event("trigger", chat_guid=chat_guid, kind=trigger.kind, reason="reply")
        else:
            log_event("trigger", chat_guid=chat_guid, kind=trigger.kind)
        try:
            if trigger.kind == "help":
                await self._say(
                    chat_guid,
                    help_text(
                        self.s.bot_name,
                        self.s.image_enabled and self.images is not None,
                        cloud_facts=self.cloud is not None,
                    ),
                )
            elif trigger.kind == "forget":
                self.context.forget(chat_guid)
                log_event("context.forgot", chat_guid=chat_guid)
                await self._say(chat_guid, "done, wiped everything i had from this chat", remember=False)
            elif trigger.kind == "image":
                await self._handle_image(chat_guid, trigger.arg)
            elif trigger.kind == "mention":
                await self._handle_mention(chat_guid)
            elif trigger.kind == "ask":
                await self._handle_mention(chat_guid, force_cloud=True)
        except Exception as exc:  # noqa: BLE001
            log_event("handler.error", chat_guid=chat_guid, kind=trigger.kind, error=exc_name(exc))
            return f"error:{exc_name(exc)}"
        return f"handled:{trigger.kind}"

    # ---------------------------------------------------------------- handlers
    async def _handle_mention(self, chat_guid: str, force_cloud: bool = False) -> None:
        history = self.context.get(chat_guid)
        messages = build_messages(self.s.persona, self.s.bot_name, history)
        backend = self.llm
        system = self._system_for(chat_guid)
        if self.cloud is not None:
            reason = "command" if force_cloud else "router"
            if force_cloud or await self._needs_facts(history):
                backend = self.cloud
                system = system + CLOUD_FACTS_ADDENDUM
                log_event("llm.escalate", chat_guid=chat_guid, reason=reason, backend=backend.name)
                # Say so first: the lookup takes 10-20 s and silence reads as ignoring them.
                await self._say(chat_guid, random.choice(LOOKUP_NOTICES), remember=False)
        started = time.monotonic()
        async with self._llm_gate:
            try:
                raw = await backend.reply(system, messages)
            except Exception as exc:  # noqa: BLE001
                if backend is self.llm:
                    raise
                # Cloud unavailable (no credits, network, rate limit): answer locally instead.
                log_event("llm.cloud_failed", chat_guid=chat_guid, backend=backend.name, error=exc_name(exc))
                backend, system = self.llm, self._system_for(chat_guid)
                raw = await backend.reply(system, messages)
        reply = clean_reply(raw, self.s.bot_name)
        log_event(
            "llm.reply",
            chat_guid=chat_guid,
            backend=backend.name,
            duration_ms=int((time.monotonic() - started) * 1000),
            chars=len(reply),
        )
        if not reply:
            reply = "hm, blanked. say that again?"
        await self._say(chat_guid, reply)

    async def _needs_facts(self, history) -> bool:
        """Ask the local model whether the last message needs outside information."""
        try:
            async with self._llm_gate:
                raw = await self.llm.reply(NEEDS_FACTS_SYSTEM, facts_router_messages(history))
        except Exception as exc:  # noqa: BLE001
            log_event("router.error", error=exc_name(exc))
            return False
        return parse_yes_no(raw)

    def _system_for(self, chat_guid: str) -> str:
        """Persona plus whatever notes the bot keeps about this chat."""
        system = self.s.persona
        notes = self.memory.load(chat_guid).strip() if self.memory else ""
        if notes:
            system += (
                "\n\nNotes you keep about this chat and the people in it, from earlier conversations "
                "(use them naturally, never read them out):\n" + notes
            )
        return system

    def _spawn(self, coro) -> None:
        t = asyncio.create_task(coro)
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    def _maybe_quiet_question(self, chat_guid: str, text: str) -> bool:
        """A question nobody answers for a while is treated as addressed to the bot."""
        if self.s.quiet_question_seconds <= 0 or not text.rstrip().endswith("?"):
            return False
        history = self.context.get(chat_guid)
        if not history:
            return False
        marker = history[-1]  # the message object just stored
        self._spawn(self._quiet_question(chat_guid, marker))
        return True

    def _maybe_react(self, chat_guid: str) -> None:
        """Sometimes fire a short text reaction, like a friend would."""
        if self.s.reaction_chance <= 0 or random.random() >= self.s.reaction_chance:
            return
        now = time.monotonic()
        if now - self._last_reaction.get(chat_guid, -1e9) < self.s.reaction_cooldown_seconds:
            return
        self._last_reaction[chat_guid] = now
        self._spawn(self._react(chat_guid))

    async def _react(self, chat_guid: str) -> None:
        history = self.context.get(chat_guid)
        if not history or history[-1].from_bot:
            return
        try:
            async with self._llm_gate:
                raw = await self.llm.reply(self._system_for(chat_guid) + REACTION_ADDENDUM,
                                           build_messages(self.s.persona, self.s.bot_name, history))
            reaction = parse_reaction(raw)
            if not reaction:
                return
            await self._say(chat_guid, reaction)
            log_event("reaction.sent", chat_guid=chat_guid, chars=len(reaction))
        except Exception as exc:  # noqa: BLE001
            log_event("handler.error", chat_guid=chat_guid, kind="reaction", error=exc_name(exc))

    def _maybe_update_memory(self, chat_guid: str) -> None:
        if self.memory is not None and self.memory.tick(chat_guid):
            self._spawn(self._update_memory(chat_guid))

    async def _update_memory(self, chat_guid: str) -> None:
        history = self.context.get(chat_guid)
        if not history:
            return
        try:
            async with self._llm_gate:
                await self.memory.update(chat_guid, self.s.bot_name, history, self.llm)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            log_event("memory.update_failed", chat_guid=chat_guid, error=exc_name(exc))

    async def _quiet_question(self, chat_guid: str, marker) -> None:
        await asyncio.sleep(self.s.quiet_question_seconds)
        history = self.context.get(chat_guid)
        if not history or history[-1] is not marker:
            return  # someone (or the bot) already replied, or the context was wiped
        log_event("trigger", chat_guid=chat_guid, kind="quiet_question")
        try:
            await self._handle_mention(chat_guid)
        except Exception as exc:  # noqa: BLE001
            log_event("handler.error", chat_guid=chat_guid, kind="quiet_question", error=exc_name(exc))

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
        if reason == "shutdown":
            text = "got restarted mid-picture, send that /image again"
        elif reason == "rejected":
            text = "not drawing that one, try a different prompt"
        else:
            text = "that image didn't come out, sorry. try again in a bit?"
        await self._say(job.chat_guid, text, remember=False)

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
