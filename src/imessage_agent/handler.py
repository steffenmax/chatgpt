"""Core agent: turns a BlueBubbles webhook event into (maybe) a reply."""

from __future__ import annotations

import asyncio
import random
import re
import time
import uuid
from dataclasses import replace
from datetime import datetime, timedelta
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any

from .bluebubbles import BlueBubblesClient
from .config import Settings
from .context import ContextStore
from .imagegen import ImageJob, ImageQueue
from .llm import (
    CLOUD_FACTS_ADDENDUM,
    INTENT_SCHEMA,
    INTENT_SYSTEM,
    LOOKUP_NOTICES,
    AFTERWORD_SUFFIX,
    BATCH_NUDGE,
    CONSISTENCY_RULE,
    REFUSAL_ROASTS,
    SELF_ONLY_RULE,
    afterword_ok,
    VOICE_SYSTEM_SUFFIX,
    Intent,
    TextBackend,
    build_messages,
    clean_reply,
    fix_speaker,
    intent_messages,
    is_refusal,
    looks_like_image_request,
    parse_intent,
    parse_reaction,
    reaction_addendum,
    speaker_names,
    stop_sequences,
    strip_unseen_urls,
    trim_reply,
)
from .llm import OllamaBackend
from .linkreader import LinkReader, find_urls, strip_cards
from .logsafe import exc_name, log_event
from .memory import MemoryStore
from .reminders import MAX_DAYS_AHEAD, Reminder, ReminderStore, describe, parse
from .task import TaskRunner
from .triggers import DEFAULT_MUTE_SECONDS, Trigger, detect, help_text, parse_duration

NEW_MESSAGE = "new-message"


def _norm_addr(a: str | None) -> str:
    """Compare iMessage handles loosely: '+1 (973) 555-0191' == '+19735550191'."""
    a = (a or "").strip().lower()
    return a if "@" in a else re.sub(r"[^\d+]", "", a)
_ECHOED_INSTRUCTION = re.compile(r"^\s*\w+\s*:|plain version|confirm that|let(ting)? the chat know|\bthey should\b")
_REFUSAL = re.compile(r"\b(can'?t|cannot|can not|won'?t|unable|not able|yourself)\b")


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
        reminders: ReminderStore | None = None,
        vision=None,
        tasks: TaskRunner | None = None,
        brief=None,
        links: LinkReader | None = None,
    ):
        self.s = settings
        self.bb = bb
        self.llm = llm
        # Optional second backend for questions that need outside information.
        self.cloud = cloud
        # Optional on-disk notes about each chat, maintained by the local model.
        self.memory = memory
        self.reminders = reminders
        # Optional local vision model: photos people send become short captions in context.
        self.vision = vision
        # Optional /task agent (Claude Agent SDK); one job at a time, owner chats only.
        self.tasks = tasks
        # Optional daily culture brief (CultureBrief), added to the prompt while fresh.
        self.brief = brief
        # Optional link reader: links people share get a short card of what's behind them.
        self.links = links
        self._link_reads: dict[str, set[asyncio.Task]] = {}
        self._task_job: asyncio.Task | None = None
        self._task_chat: str | None = None
        # A question the task agent asked: the next plain message in that chat answers it.
        self._task_question: tuple[str, asyncio.Future, str | None] | None = None
        # GUIDs of the bot's own messages (from isFromMe webhooks) so an inline reply
        # to one of them counts as talking to the bot.
        self._bot_guids: OrderedDict[str, None] = OrderedDict()
        self._last_reaction: dict[str, float] = {}
        self._last_reaction_text: dict[str, str] = {}
        self._last_roast: dict[str, str] = {}
        self._participants: dict[str, tuple[float, list[str]]] = {}
        # One reply at a time per chat, and mentions waiting to be answered together.
        self._chat_locks: dict[str, asyncio.Lock] = {}
        self._pending: dict[str, dict[str, Any]] = {}
        # "<name> chill" / "/mute 30m": chat -> monotonic time the bot may talk again.
        self._muted_until: dict[str, float] = {}
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
        if self.vision is not None and chat_guid in self._allowed:
            photos = [a for a in (data.get("attachments") or [])
                      if isinstance(a, dict) and str(a.get("mimeType") or "").startswith("image/") and a.get("guid")]
            if photos:
                captions = await self._caption_photos(chat_guid, photos[:2])
                text = (text + " " + captions).strip() if captions else text
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

        if self._task_question is not None and self._task_question[0] == chat_guid and trigger.kind in ("none", "mention"):
            chat_q, fut, asker = self._task_question
            from_addr = (data.get("handle") or {}).get("address") if isinstance(data.get("handle"), dict) else None
            if not fut.done() and (asker is None or _norm_addr(from_addr) == asker):
                self._task_question = None
                fut.set_result(text)
                log_event("task.answered", chat_guid=chat_guid, chars=len(text))
                return "handled:task_answer"

        address = (data.get("handle") or {}).get("address") if isinstance(data.get("handle"), dict) else None
        sender = await self.resolver.name_for(address)

        stored, has_link = None, False
        if trigger.kind in ("none", "mention"):
            stored = self.context.add(chat_guid, sender, text)
        elif trigger.kind == "ask":
            stored = self.context.add(chat_guid, sender, trigger.arg or text)
        if stored is not None and self.links is not None:
            urls = find_urls(stored.text)
            if urls:
                has_link = True
                self._read_links(chat_guid, stored, urls)
        if trigger.kind in ("none", "mention", "ask"):
            self._maybe_update_memory(chat_guid)
            if chat_guid in self._pending:  # the chat is still going: let it settle before replying
                self._pending[chat_guid]["last"] = time.monotonic()
        if trigger.kind == "none":
            if self._muted(chat_guid):
                return "stored"
            if has_link and self.s.link_quiet_seconds > 0:
                self._spawn(self._quiet_link(chat_guid, stored))
                return "stored"
            if not self._maybe_quiet_question(chat_guid, text):
                self._maybe_react(chat_guid)
            return "stored"

        if trigger.kind in ("mention", "ask") and self._muted(chat_guid):
            log_event("trigger", chat_guid=chat_guid, kind=trigger.kind, status="muted")
            return "muted"
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
                        reminders=self.reminders is not None,
                        task=self.tasks is not None and chat_guid in self.s.task_chat_guids,
                    ),
                )
            elif trigger.kind == "forget":
                self.context.forget(chat_guid)
                log_event("context.forgot", chat_guid=chat_guid)
                await self._say_in_voice(chat_guid, "done, i wiped everything i had from this chat", "done, wiped everything i had from this chat")
            elif trigger.kind == "mute":
                await self._handle_mute(chat_guid, trigger.arg)
            elif trigger.kind == "unmute":
                await self._handle_unmute(chat_guid)
            elif trigger.kind == "image":
                await self._handle_image(chat_guid, trigger.arg)
            elif trigger.kind in ("mention", "ask"):
                if self.s.reply_batch_seconds > 0:
                    self._queue_mention(chat_guid, stored, sender, force_cloud=trigger.kind == "ask")
                    return f"queued:{trigger.kind}"
                async with self._chat_lock(chat_guid):
                    await self._handle_mention(chat_guid, force_cloud=trigger.kind == "ask", marker=stored)
            elif trigger.kind == "remind":
                await self._handle_remind(chat_guid, sender, trigger.arg)
            elif trigger.kind == "reminders":
                await self._handle_reminders(chat_guid)
            elif trigger.kind == "task":
                await self._handle_task(chat_guid, sender, trigger.arg, address)
        except Exception as exc:  # noqa: BLE001
            log_event("handler.error", chat_guid=chat_guid, kind=trigger.kind, error=exc_name(exc))
            return f"error:{exc_name(exc)}"
        return f"handled:{trigger.kind}"

    # ---------------------------------------------------------------- handlers
    def _muted(self, chat_guid: str) -> bool:
        until = self._muted_until.get(chat_guid)
        if until is None:
            return False
        if time.monotonic() >= until:
            self._muted_until.pop(chat_guid, None)
            log_event("mute.ended", chat_guid=chat_guid)
            return False
        return True

    async def _handle_mute(self, chat_guid: str, arg: str) -> None:
        seconds = parse_duration(arg) or DEFAULT_MUTE_SECONDS
        self._muted_until[chat_guid] = time.monotonic() + seconds
        self._pending.pop(chat_guid, None)  # drop pings waiting for a reply
        log_event("mute.started", chat_guid=chat_guid, duration_ms=seconds * 1000)
        mins = round(seconds / 60)
        span = f"{mins} min" if mins < 60 else f"{mins // 60} hr" + (f" {mins % 60} min" if mins % 60 else "")
        await self._say(chat_guid, random.choice(
            (f"aight, going quiet for {span}", f"fine. {span} of silence, enjoy it", f"say less. back in {span}")))

    async def _handle_unmute(self, chat_guid: str) -> None:
        was = self._muted_until.pop(chat_guid, None) is not None
        log_event("mute.ended", chat_guid=chat_guid, reason="unmute")
        await self._say(chat_guid, random.choice(("i'm back", "missed me", "back. what'd i miss")) if was else "i wasn't gone")

    def _chat_lock(self, chat_guid: str) -> asyncio.Lock:
        return self._chat_locks.setdefault(chat_guid, asyncio.Lock())

    def _queue_mention(self, chat_guid: str, marker, sender: str, force_cloud: bool = False) -> None:
        """Answer mentions that arrive close together in one reply (REPLY_BATCH_SECONDS)."""
        now = time.monotonic()
        st = self._pending.get(chat_guid)
        if st is None:
            st = self._pending[chat_guid] = {"first": now, "last": now, "marker": marker, "who": [], "n": 0, "cloud": False}
            self._spawn(self._answer_batch(chat_guid))
        st["last"], st["marker"] = now, marker
        st["n"] += 1
        st["cloud"] = st["cloud"] or force_cloud
        if sender not in st["who"]:
            st["who"].append(sender)

    async def _answer_batch(self, chat_guid: str) -> None:
        async with self._chat_lock(chat_guid):  # wait out a reply still being written
            while True:
                st = self._pending.get(chat_guid)
                if st is None:
                    return  # muted meanwhile
                now = time.monotonic()
                quiet = st["last"] + self.s.reply_batch_seconds - now
                cap = st["first"] + max(self.s.reply_batch_max_seconds, self.s.reply_batch_seconds) - now
                wait = min(quiet, cap)
                if wait <= 0:
                    break
                await asyncio.sleep(wait)
            st = self._pending.pop(chat_guid)
            if self._muted(chat_guid):
                return
            if st["n"] > 1:
                log_event("mention.batched", chat_guid=chat_guid, count=st["n"])
            try:
                await self._handle_mention(chat_guid, force_cloud=st["cloud"], marker=st["marker"],
                                           pinged_by=st["who"] if st["n"] > 1 else None)
            except Exception as exc:  # noqa: BLE001
                log_event("handler.error", chat_guid=chat_guid, kind="mention", error=exc_name(exc))

    async def _handle_mention(self, chat_guid: str, force_cloud: bool = False, marker=None,
                              pinged_by: list[str] | None = None) -> None:
        await self._links_ready(chat_guid)
        history = self.context.get(chat_guid)
        # The router looks at the (last) message that pinged the bot, not whatever came after it.
        upto = history
        for i in range(len(history) - 1, -1, -1):
            if history[i] is marker:
                upto = history[: i + 1]
                break
        who = upto[-1].sender if upto else "someone"
        intent = Intent("lookup") if force_cloud else await self._intent(upto)
        if intent.action == "image" and self.images is not None:
            log_event("intent", chat_guid=chat_guid, kind="image")
            await self._handle_image(chat_guid, intent.prompt)
            return
        if intent.action == "remind" and self.reminders is not None:
            log_event("intent", chat_guid=chat_guid, kind="remind")
            await self._handle_remind(chat_guid, who, f"{intent.when} {intent.what}")
            return
        if intent.action == "forget":
            log_event("intent", chat_guid=chat_guid, kind="forget")
            self.context.forget(chat_guid)
            await self._say_in_voice(chat_guid, "done, i wiped everything i had from this chat", "done, wiped everything i had from this chat")
            return
        nudge = BATCH_NUDGE.replace("{who}", ", ".join(pinged_by)) if pinged_by else ""
        await self._reply(chat_guid, history, use_cloud=(intent.action == "lookup"),
                          reason="command" if force_cloud else "intent", nudge=nudge)

    async def _reply(self, chat_guid: str, history, use_cloud: bool, reason: str, nudge: str = "") -> None:
        messages = build_messages(self.s.persona, self.s.bot_name, history)
        backend = self.llm
        system = self._system_for(chat_guid) + nudge
        if use_cloud and self.cloud is not None:
            backend = self.cloud
            system = system + CLOUD_FACTS_ADDENDUM
            log_event("llm.escalate", chat_guid=chat_guid, reason=reason, backend=backend.name)
            # Say so first: the lookup takes 10-20 s and silence reads as ignoring them.
            await self._say(chat_guid, random.choice(LOOKUP_NOTICES), remember=False)
        names = speaker_names(history, self.s.bot_name)
        stops = stop_sequences(names)

        async def ask(b, sys_prompt):
            if isinstance(b, OllamaBackend):
                return await b.reply(sys_prompt + CONSISTENCY_RULE + SELF_ONLY_RULE, messages, stop=stops)
            return await b.reply(sys_prompt + CONSISTENCY_RULE + SELF_ONLY_RULE, messages)

        started = time.monotonic()
        async with self._llm_gate:
            try:
                raw = await ask(backend, system)
            except Exception as exc:  # noqa: BLE001
                if backend is self.llm:
                    raise
                # Cloud unavailable (no credits, network, rate limit): answer locally instead.
                log_event("llm.cloud_failed", chat_guid=chat_guid, backend=backend.name, error=exc_name(exc))
                backend, system = self.llm, self._system_for(chat_guid)
                raw = await ask(backend, system)
            reply, impersonated = fix_speaker(clean_reply(raw, self.s.bot_name), names, self.s.bot_name)
            if impersonated:
                # It wrote a line as one of the members: ask once more, then strip the label.
                log_event("reply.impersonation", chat_guid=chat_guid, attempt=1)
                raw = await ask(backend, system + f"\n\nYou are {self.s.bot_name}. Reply as {self.s.bot_name} only.")
                reply, impersonated = fix_speaker(clean_reply(raw, self.s.bot_name), names, self.s.bot_name)
                if impersonated:
                    log_event("reply.impersonation", chat_guid=chat_guid, attempt=2)
                    reply = re.sub(r"^\s*\w+\s*:\s*", "", reply.split("\n")[0]).strip()
        if backend is self.llm:
            seen = "\n".join(m.text for m in history)
            stripped = strip_unseen_urls(reply, seen)
            if stripped != reply:
                log_event("reply.links_removed", chat_guid=chat_guid)
                reply = stripped or "can't pull up links from here, look it up yourself"
        log_event(
            "llm.reply",
            chat_guid=chat_guid,
            backend=backend.name,
            duration_ms=int((time.monotonic() - started) * 1000),
            chars=len(reply),
        )
        if is_refusal(reply):
            # The model stepped out of character to decline: answer the way he would instead.
            log_event("reply.refusal_swapped", chat_guid=chat_guid, backend=backend.name)
            last = self._last_roast.get(chat_guid)
            reply = random.choice([r for r in REFUSAL_ROASTS if r != last])
            self._last_roast[chat_guid] = reply
        if backend is self.llm and self.s.reply_max_chars:
            trimmed = trim_reply(reply, self.s.reply_max_chars)
            if trimmed != reply:
                log_event("reply.trimmed", chat_guid=chat_guid, chars=len(reply))
                reply = trimmed
        if not reply:
            reply = "hm, blanked. say that again?"
        await self._say(chat_guid, reply)

    async def _intent(self, history) -> Intent:
        """Ask the local model which tool (if any) the last message calls for."""
        if not history:
            return Intent()
        system = INTENT_SYSTEM.replace("{bot_name}", self.s.bot_name)
        msgs = intent_messages(_plain(history))
        try:
            async with self._llm_gate:
                structured = getattr(self.llm, "structured", None)
                raw = await structured(system, msgs, INTENT_SCHEMA) if structured else await self.llm.reply(system, msgs)
        except Exception as exc:  # noqa: BLE001
            log_event("router.error", error=exc_name(exc))
            return Intent()
        intent = parse_intent(raw)
        if intent.action == "lookup" and self.cloud is None:
            return Intent()
        if intent.action == "image" and not looks_like_image_request(strip_cards(history[-1].text)):
            log_event("intent.vetoed", kind="image")
            return Intent()
        return intent

    async def _say_in_voice(
        self, chat_guid: str, fact: str, fallback: str, remember: bool = False, keep: tuple[str, ...] = ()
    ) -> None:
        """Deliver a fixed fact in the bot's own voice; fall back to the plain text if the model flakes.

        The model's phrasing is used only if it still contains every string in `keep`
        (times, numbers) and doesn't turn a done deal into a refusal.
        """
        text = fallback
        if not self.s.voice_acks:
            await self._say(chat_guid, text, remember=remember)
            return
        try:
            async with self._llm_gate:
                raw = await self.llm.reply(self._system_for(chat_guid) + VOICE_SYSTEM_SUFFIX + fact, [{"role": "user", "content": "(say it now)"}])
            cand = clean_reply(raw, self.s.bot_name)
            low = cand.lower()
            if (
                cand
                and len(cand) <= max(240, len(fallback) * 3)
                and "\n" not in cand
                and all(k.lower() in low for k in keep)
                and not (_REFUSAL.search(low) and not _REFUSAL.search(fact.lower()))
                and not _ECHOED_INSTRUCTION.search(low)
                and self.s.bot_name.lower() not in low
            ):
                text = cand
            else:
                log_event("voice.rejected", chat_guid=chat_guid)
        except Exception as exc:  # noqa: BLE001
            log_event("voice.failed", chat_guid=chat_guid, error=exc_name(exc))
        await self._say(chat_guid, text, remember=remember)

    async def _handle_remind(self, chat_guid: str, who: str, arg: str) -> None:
        if self.reminders is None:
            await self._say(chat_guid, "reminders are off on my end", remember=False)
            return
        m = re.match(r"^\s*cancel\s+#?(\d+)\s*$", arg, re.IGNORECASE)
        if m:
            ok = self.reminders.cancel(chat_guid, int(m.group(1)))
            await self._say(chat_guid, "dropped it" if ok else "no reminder with that number here", remember=False)
            return
        now = datetime.now().astimezone()
        due, what = parse(arg, now)
        if due is None or not what.strip():
            await self._say(chat_guid, "say it like: /remind in 20m call the dentist, or /remind tomorrow 9am venmo joe", remember=False)
            return
        if due - now > timedelta(days=MAX_DAYS_AHEAD):
            await self._say(chat_guid, f"that's too far out, i only do {MAX_DAYS_AHEAD} days", remember=False)
            return
        r = self.reminders.add(chat_guid, who, what.strip(), due.timestamp())
        if r is None:
            await self._say(chat_guid, "this chat has too many reminders pending, cancel some first", remember=False)
            return
        plain = f"got it {who}, i'll remind you {describe(due, now)} (#{r.id})"
        when = describe(due, now)
        hour = re.search(r"\d+", when)
        await self._say_in_voice(
            chat_guid,
            f"got it {who}, i'll remind you here {when}, that's #{r.id}",
            plain,
            keep=(hour.group(0) if hour else "", f"{r.id}"),
        )

    async def _handle_reminders(self, chat_guid: str) -> None:
        if self.reminders is None:
            await self._say(chat_guid, "reminders are off on my end", remember=False)
            return
        items = self.reminders.pending(chat_guid)
        if not items:
            await self._say(chat_guid, "nothing pending here", remember=False)
            return
        now = datetime.now().astimezone()
        lines = [f"#{r.id} {describe(datetime.fromtimestamp(r.due).astimezone(), now)}: {r.text} ({r.who})" for r in items[:15]]
        await self._say(chat_guid, "\n".join(lines), remember=False)

    async def _handle_task(self, chat_guid: str, who: str, request: str, address: str | None = None) -> None:
        if self.tasks is None:
            await self._say(chat_guid, "tasks are off on my end", remember=False)
            return
        if chat_guid not in self.s.task_chat_guids:
            log_event("task.refused", chat_guid=chat_guid, reason="not_task_chat")
            await self._say(chat_guid, "tasks only work in the owner's dm for now", remember=False)
            return
        owners = {_norm_addr(a) for a in self.s.task_owner_addresses}
        if owners and _norm_addr(address) not in owners:
            log_event("task.refused", chat_guid=chat_guid, reason="not_owner")
            await self._say(chat_guid, "only the boss can send me on tasks for now", remember=False)
            return
        requester = _norm_addr(address) if owners else None
        if not request.strip():
            await self._say(chat_guid, "give me the job: /task find 3 steakhouses near msg with a table saturday 8pm", remember=False)
            return
        if self._task_job is not None and not self._task_job.done():
            await self._say(chat_guid, "already working on one, hold on", remember=False)
            return
        await self._say_in_voice(chat_guid, "on it, gimme a few minutes to dig into that properly", "on it, gimme a few minutes")
        history = self.context.get(chat_guid)
        n = self.s.task_context_messages
        # the /task line itself is not stored in context, so the last n are all prior chat
        ctx = [f"{'you' if m.from_bot else m.sender}: {m.text}" for m in history[-n:]] if n else []
        self._task_chat = chat_guid
        self._task_job = asyncio.create_task(self._run_task(chat_guid, who, request.strip(), ctx, requester))

    async def _run_task(self, chat_guid: str, who: str, request: str, ctx: list[str], requester: str | None = None) -> None:
        log_event("task.started", chat_guid=chat_guid, chars=len(request))
        try:
            ask = self._task_asker(chat_guid, requester) if self.s.task_max_questions > 0 else None
            res = await self.tasks.run(request, who, ctx, ask=ask)  # type: ignore[union-attr]
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log_event("task.error", chat_guid=chat_guid, error=exc_name(exc))
            await self._say_in_voice(chat_guid, "that job broke on my end, try again later", "that one broke on my end, try again later")
            return
        log_event(
            "task.finished", chat_guid=chat_guid, status="ok" if res.ok else res.reason[:40],
            duration_ms=res.duration_ms, turns=res.turns, cost_cents=round(res.cost_usd * 100), files=len(res.files),
        )
        if not res.ok:
            why = {"timeout": "it took too long", "error_max_turns": "it took too many steps",
                   "error_max_budget_usd": "it hit the spending cap"}.get(res.reason, "something went wrong")
            plain = f"couldn't finish that one, {why}"
            await self._say_in_voice(chat_guid, plain, plain)
            if res.text:
                await self._deliver(chat_guid, "here's what i got before it fell apart:\n" + res.text[:700])
            return
        text = res.text if len(res.text) <= 1500 else res.text[:1500].rsplit(" ", 1)[0] + " …"
        await self._deliver(chat_guid, text)
        self.context.add(chat_guid, self.s.bot_name, f"[posted task results for {who}: {text[:600]}]", from_bot=True)
        for f in res.files[:3]:
            try:
                await self.bb.send_attachment(chat_guid, f, name=f.name)
            except Exception as exc:  # noqa: BLE001
                log_event("task.attach_failed", chat_guid=chat_guid, error=exc_name(exc))
        await self._afterword(chat_guid, who, request, text)

    async def _deliver(self, chat_guid: str, text: str) -> None:
        """Task results go out verbatim: the facts, with no assistant framing (the agent is told
        to write it that way). The bot's own voice comes before (the ack) and after (_afterword)."""
        await self._say(chat_guid, text, remember=False)

    async def _afterword(self, chat_guid: str, who: str, request: str, results: str) -> None:
        """One short closing text in the bot's voice reacting to the results. Skipped if it
        restates numbers or links (it could garble them), runs long, or reads off."""
        if not self.s.voice_acks:
            return
        history = self.context.get(chat_guid)
        names = speaker_names(history, self.s.bot_name)
        system = (self._system_for(chat_guid) + AFTERWORD_SUFFIX
                  .replace("{who}", who).replace("{request}", request[:200].replace('"', "'")))
        for attempt in (1, 2):
            try:
                async with self._llm_gate:
                    raw = await self.llm.reply(system, [{"role": "user", "content": results}])
            except Exception as exc:  # noqa: BLE001
                log_event("voice.failed", chat_guid=chat_guid, error=exc_name(exc))
                return
            text, impersonated = fix_speaker(clean_reply(raw, self.s.bot_name), names, self.s.bot_name)
            low = text.lower()
            if (not impersonated and afterword_ok(results, text) and not _ECHOED_INSTRUCTION.search(low)
                    and not (_REFUSAL.search(low) and not _REFUSAL.search(results.lower()))):
                await self._say(chat_guid, text)
                log_event("task.afterword", chat_guid=chat_guid, chars=len(text), attempt=attempt)
                return
        log_event("task.afterword", chat_guid=chat_guid, status="skipped")

    def _task_asker(self, chat_guid: str, requester: str | None = None):
        async def ask(question: str) -> str | None:
            fut = asyncio.get_running_loop().create_future()
            self._task_question = (chat_guid, fut, requester)
            await self._say_in_voice(chat_guid, f"quick q: {question}", f"quick q: {question}",
                                     keep=tuple(set(re.findall(r"\d+", question))))
            log_event("task.asked", chat_guid=chat_guid, chars=len(question))
            try:
                return await fut
            finally:
                if self._task_question is not None and self._task_question[1] is fut:
                    self._task_question = None
        return ask

    async def stop_task(self) -> None:
        """On shutdown: cancel a running task and say so."""
        job, chat = self._task_job, self._task_chat
        if job is None or job.done():
            return
        job.cancel()
        try:
            await job
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
        log_event("task.abandoned", chat_guid=chat or "?")
        try:
            await self._say(chat, "got restarted mid-task, send that /task again", remember=False)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001
            log_event("task.notify_failed", error=exc_name(exc))

    async def fire_reminder(self, r: Reminder) -> None:
        await self._say(r.chat_guid, f"⏰ {r.who}: {r.text}")

    def _system_for(self, chat_guid: str) -> str:
        """Persona plus whatever notes the bot keeps about this chat."""
        system = self.s.persona
        notes = self.memory.load(chat_guid).strip() if self.memory else ""
        if notes:
            system += (
                "\n\nNotes you keep about this chat and the people in it, from earlier conversations "
                "(use them naturally, never read them out):\n" + notes
            )
        if self.links is not None:
            system += (
                "\n\nWhen someone shares a link, a [link: ...] note after their message shows what it is: the "
                "post or page, and for X posts the top replies. React to what's actually in it like you "
                "opened it yourself, and riff on the replies if they're funny. Don't invent anything that "
                "isn't in the note, don't read the note out, and if it says you couldn't open it, say so."
            )
        brief = self.brief.load() if self.brief else ""
        if brief:
            system += (
                "\n\nWhat people are talking and joking about online right now. Drop a reference when it "
                "genuinely fits the conversation, like someone who's always online would; never force "
                "one in, never explain the joke, never list these:\n" + brief
            )
        return system

    def _read_links(self, chat_guid: str, msg, urls: list[str]) -> None:
        async def read():
            cards = [c for c in await asyncio.gather(*(self.links.read(u) for u in urls)) if c]  # type: ignore[union-attr]
            if cards:
                msg.text = msg.text + " " + " ".join(cards)

        t = asyncio.create_task(read())
        pending = self._link_reads.setdefault(chat_guid, set())
        pending.add(t)
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)
        t.add_done_callback(pending.discard)

    async def _links_ready(self, chat_guid: str, timeout: float = 30.0) -> None:
        """Wait for links still being read in this chat, so a reply can see them."""
        pending = list(self._link_reads.get(chat_guid) or ())
        if pending:
            await asyncio.wait(pending, timeout=timeout)

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
        if not history or history[-1].from_bot or self._muted(chat_guid):
            return
        try:
            async with self._llm_gate:
                last = self._last_reaction_text.get(chat_guid)
                raw = await self.llm.reply(self._system_for(chat_guid) + reaction_addendum(last),
                                           build_messages(self.s.persona, self.s.bot_name, history))
            reaction = parse_reaction(raw)
            if reaction and last and reaction.lower().rstrip("o!. ") == last.lower().rstrip("o!. "):
                reaction = None  # same as last time
            if not reaction:
                log_event("reaction.considered", chat_guid=chat_guid, status="none")
                return
            self._last_reaction_text[chat_guid] = reaction
            await self._say(chat_guid, reaction)
            log_event("reaction.sent", chat_guid=chat_guid, chars=len(reaction))
        except Exception as exc:  # noqa: BLE001
            log_event("handler.error", chat_guid=chat_guid, kind="reaction", error=exc_name(exc))

    def _maybe_update_memory(self, chat_guid: str) -> None:
        if self.memory is not None and self.memory.tick(chat_guid):
            self._spawn(self._update_memory(chat_guid))

    async def _chat_members(self, chat_guid: str) -> list[str]:
        """Display names of everyone in the chat (cached for an hour); [] if BlueBubbles can't say."""
        hit = self._participants.get(chat_guid)
        if hit and time.monotonic() - hit[0] < 3600:
            return hit[1]
        names: list[str] = []
        try:
            chat = await self.bb.get_chat(chat_guid)
            for p in chat.get("participants") or []:
                addr = p.get("address") if isinstance(p, dict) else None
                if addr:
                    name = await self.resolver.name_for(addr)
                    if name and name != addr and name not in names:  # unknown numbers stay out
                        names.append(name)
        except Exception as exc:  # noqa: BLE001
            log_event("chat.members_failed", chat_guid=chat_guid, error=exc_name(exc))
            return hit[1] if hit else []
        self._participants[chat_guid] = (time.monotonic(), names)
        return names

    async def _update_memory(self, chat_guid: str) -> None:
        history = self.context.get(chat_guid)
        if not history:
            return
        try:
            members = await self._chat_members(chat_guid)
            async with self._llm_gate:
                await self.memory.update(chat_guid, self.s.bot_name, _plain(history), self.llm, members)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            log_event("memory.update_failed", chat_guid=chat_guid, error=exc_name(exc))

    async def _quiet_question(self, chat_guid: str, marker) -> None:
        await asyncio.sleep(self.s.quiet_question_seconds)
        history = self.context.get(chat_guid)
        if not history or history[-1] is not marker or self._muted(chat_guid):
            return  # someone (or the bot) already replied, the context was wiped, or he's muted
        log_event("trigger", chat_guid=chat_guid, kind="quiet_question")
        if self.s.reply_batch_seconds > 0:
            self._queue_mention(chat_guid, marker, marker.sender)
            return
        try:
            async with self._chat_lock(chat_guid):
                await self._handle_mention(chat_guid, marker=marker)
        except Exception as exc:  # noqa: BLE001
            log_event("handler.error", chat_guid=chat_guid, kind="quiet_question", error=exc_name(exc))

    async def _quiet_link(self, chat_guid: str, marker) -> None:
        """A link nobody responds to gets the bot's own take after LINK_QUIET_SECONDS."""
        await asyncio.sleep(self.s.link_quiet_seconds)
        await self._links_ready(chat_guid)
        history = self.context.get(chat_guid)
        if not history or history[-1] is not marker or self._muted(chat_guid):
            return  # someone (or the bot) already said something, or he's muted
        if "[link: " not in marker.text or "[link: couldn't open it]" in marker.text:
            return  # nothing to talk about
        log_event("trigger", chat_guid=chat_guid, kind="quiet_link")
        try:
            async with self._chat_lock(chat_guid):
                history = self.context.get(chat_guid)
                if not history or history[-1] is not marker:
                    return  # someone spoke while a reply was being written
                await self._reply(chat_guid, history, use_cloud=False, reason="quiet_link", nudge=(
                    f"\n\n{marker.sender} shared that link a while ago and nobody has said anything. Give "
                    "your own quick take on it, one or two short lines, like you just opened it: the funniest "
                    "or wildest part, a roast, or a reply that caught your eye. Don't ask if anyone saw it."))
        except Exception as exc:  # noqa: BLE001
            log_event("handler.error", chat_guid=chat_guid, kind="quiet_link", error=exc_name(exc))

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
        await self._say_in_voice(
            chat_guid,
            "on it, the picture will take me a few minutes"
            + (f", {ahead} ahead of yours" if ahead else ""),
            note,
            keep=(str(ahead),) if ahead else (),
        )
        await self.images.submit(job)

    async def send_image(self, job: ImageJob, png: Path) -> None:
        await self.bb.send_attachment(job.chat_guid, png, name=f"{self.s.bot_name.lower()}-{job.job_id}.png")
        log_event("image.sent", chat_guid=job.chat_guid, job_id=job.job_id)
        # So the bot knows what it just posted when the chat reacts to it.
        self.context.add(job.chat_guid, self.s.bot_name, f"[sent a picture: {job.prompt}]", from_bot=True)

    async def _caption_photos(self, chat_guid: str, photos: list[dict]) -> str:
        parts = []
        for a in photos:
            data = None
            for attempt in range(3):  # the attachment may still be transferring when the webhook fires
                try:
                    data = await self.bb.download_attachment(str(a["guid"]))
                    break
                except Exception as exc:  # noqa: BLE001
                    if attempt == 2:
                        log_event("photo.download_failed", chat_guid=chat_guid, error=exc_name(exc), attempt=attempt + 1)
                    else:
                        await asyncio.sleep(2)
            if data is None:
                continue
            try:
                async with self._llm_gate:
                    cap = await self.vision.caption(data)
            except Exception as exc:  # noqa: BLE001
                log_event("photo.caption_failed", chat_guid=chat_guid, error=exc_name(exc))
                continue
            if cap:
                parts.append(f"[photo: {cap}]")
                log_event("photo.captioned", chat_guid=chat_guid, chars=len(cap))
        return " ".join(parts)

    async def image_failed(self, job: ImageJob, reason: str) -> None:
        if reason == "shutdown":
            fact, text = "i got restarted mid-picture, ask me for it again", "got restarted mid-picture, send that /image again"
        elif reason == "rejected":
            fact, text = "not drawing that one, try a different idea", "not drawing that one, try a different prompt"
        else:
            fact, text = "that picture didn't come out, try me again in a bit", "that image didn't come out, sorry. try again in a bit?"
        await self._say_in_voice(job.chat_guid, fact, text)

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


def _plain(history):
    """History without link cards, for the tool router and the memory notes."""
    return [replace(m, text=strip_cards(m.text)) if "[link: " in m.text else m for m in history]
