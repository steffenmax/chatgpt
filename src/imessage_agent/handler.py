"""Core agent: turns a BlueBubbles webhook event into (maybe) a reply."""

from __future__ import annotations

import asyncio
import random
import shutil
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
    FACT_FINDER_SYSTEM,
    FACTS_STRICT,
    LOOKED_UP,
    INTENT_SCHEMA,
    INTENT_SYSTEM,
    LOOKUP_NOTICES,
    ADDRESSEE_SCHEMA,
    BUDDY_MOOD_SCHEMA,
    BUDDY_MOOD_SYSTEM,
    ADDRESSEE_SYSTEM,
    AFTERWORD_SUFFIX,
    CONSISTENCY_RULE,
    REFUSAL_ROASTS,
    SELF_ONLY_RULE,
    UNANSWERED_GIF_SYSTEM,
    afterword_ok,
    VOICE_SYSTEM_SUFFIX,
    Intent,
    addressee_messages,
    parse_addressee,
    TextBackend,
    build_messages,
    clean_reply,
    facts_kept,
    fix_speaker,
    intent_messages,
    is_refusal,
    looks_like_gif_request,
    looks_like_image_request,
    looks_like_tweet_request,
    mentions_bots,
    parse_mood,
    parse_intent,
    parse_reaction,
    reaction_addendum,
    speaker_names,
    stop_sequences,
    strip_fake_notes,
    strip_unseen_urls,
    target_nudge,
    trim_reply,
)
from .llm import OllamaBackend
from .linkreader import LinkReader, find_urls, strip_cards
from .logsafe import exc_name, log_event
from .alerts import Alert
from .bookie import Bookie, odds_line, parse_bet
from .media import GifSearch, TweetFinder, clean_query, describe_tweet, gif_from_reaction
from .likeness import Likenesses, blocked
from .voice import SOUND_TAGS, VoiceSynth, new_workdir, repeat_last, strip_tags, wants_voice
from .recap import RECAP_SYSTEM, gather, wants_recap, window_from
from .sports import (ESPN, LEAGUES, Game, TeamIndex, fmt_odds, late_game, league_hint, live_odds, past_halfway,
                     too_late_to_bet)
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
# The designated person dissed the bot: a random (non-humiliating, within the likeness limits) scene.
BUDDY_SCENES = (
    "as a confused medieval knight who lost his horse", "riding a tiny tricycle in a business suit",
    "as a sad mall santa on break", "as a 1980s infomercial host selling a useless gadget",
    "as a minor league baseball mascot", "stuck inside a claw machine", "as a substitute teacher on his first day",
    "as a pirate captain who's afraid of water", "as a contestant on a cooking show burning everything",
    "as a renaissance painting subject looking unimpressed", "lifting a single tiny dumbbell with full effort",
    "as a weather reporter in a hurricane", "as a lifeguard at a kiddie pool", "as a dad at a barbecue grilling one hot dog",
)
BUDDY_THIS_U = ("this u?", "this u, {name}?", "bold words from this guy", "said the guy in this picture",
                "talk all that and look at this", "{name} rn")
BUDDY_COMEBACKS = ("big talk from {name}", "say that again {name}, i dare you", "noted. you're on my list {name}")
LIKENESS_NO = ("not with {name}'s actual face, pick something else", "nah, that one's off limits with a real guy's face",
               "i'll draw {name} doing a lot of dumb stuff, not that")
RECAP_ACKS = ("one sec, scrolling up", "ugh fine, reading", "hold on, getting the tape", "pulling the receipts",
              "give me a sec to catch up myself")
BET_QUIPS = (
    "bold.", "the house thanks you", "i'll frame the receipt", "brave. stupid, but brave", "noted. in pen",
    "your funeral", "love the confidence", "this is how it starts", "i've seen worse. not many", "say less",
)
BAILOUT_YES = ("fine. {n} bucks, don't make me regret it", "here's {n}. this is going on your permanent record",
               "{n} bucks from the ricky relief fund. pathetic")
BAILOUT_NO = ("you're not broke enough", "nice try, you've got money or bets riding", "one bailout a week, degenerate")

# What the bot says when it takes an image request (picked at random, never twice in a row).
IMAGE_ACKS = (
    "mmm.", "say less", "lemme cook", "the kitchen is open", "firing up the oven", "oh you sick puppy. one sec",
    "you're nasty for this one", "hold my beer", "art takes time, sit tight", "warming up the paintbrush",
    "this one's going in the louvre", "ok ok ok", "bet.", "don't rush genius", "chef's kiss incoming",
    "oh i got you", "say no more", "brb, being an artist", "you're not ready for this", "this might get me banned. on it",
    "consider it done", "grabbing the crayons", "god forgive me. starting now", "oh this is gonna be good",
    "cookin.", "unholy. i love it. one sec", "my therapist would hate this. on it", "you've been waiting to ask that huh",
    "mixing the paints", "witness greatness in a minute", "lock in. drawing", "the vision is coming to me",
    "absolutely deranged request. starting", "give me a sec to sin", "picasso mode activated",
)
IMAGE_QUEUE_ACKS = (
    "kitchen's backed up, {n} ahead of you", "{n} orders ahead of yours, patience", "wait your turn, {n} in line",
    "cookin {n} others first, you're next-ish", "{n} ahead of you. art has a line now", "take a number, {n} before you",
)

# A message without the name, right after the bot spoke, is judged only within this window.
FOLLOWUP_WINDOW_SECONDS = 30

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
        gifs: GifSearch | None = None,
        tweets: TweetFinder | None = None,
        espn: ESPN | None = None,
        bookie: Bookie | None = None,
        sports_fans: dict[str, list[str]] | None = None,
        likenesses: Likenesses | None = None,
        voice: VoiceSynth | None = None,
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
        # Optional GIF search (GIPHY) and funny-tweet search (X); see media.py.
        self.gifs = gifs
        self.tweets = tweets
        # NFL: ESPN client, the fake-money book, and who in the chat roots for whom.
        self.espn = espn
        self.bookie = bookie
        self.sports_fans = sports_fans or {}
        self.sports_poller = None  # alerts.SportsAlerts, set by main when NFL features are on
        # Opted-in people whose face image requests can use (trained add-ons).
        self.likenesses = likenesses
        # Optional local text-to-speech for voice notes.
        self.voice = voice
        self._link_reads: dict[str, set[asyncio.Task]] = {}
        self._task_job: asyncio.Task | None = None
        self._task_chat: str | None = None
        # A question the task agent asked: the next plain message in that chat answers it.
        self._task_question: tuple[str, asyncio.Future, str | None] | None = None
        # GUIDs of the bot's own messages (from isFromMe webhooks) -> when they arrived, so an
        # inline reply to one counts as talking to the bot, and a message right after one
        # gets a judgment call.
        self._bot_guids: OrderedDict[str, float] = OrderedDict()
        self._not_bot_guids: OrderedDict[str, None] = OrderedDict()
        self._last_reaction: dict[str, float] = {}
        self._last_reaction_text: dict[str, str] = {}
        self._last_roast: dict[str, str] = {}
        self._last_image_ack: dict[str, str] = {}
        self._buddy_last: dict[tuple[str, str], float] = {}  # (chat, "down"/"jab") -> when it last fired
        # message GUID -> {"chat", "who": addresses that laughed, "fired": milestones already said}
        self._laughs: OrderedDict[str, dict[str, Any]] = OrderedDict()
        # chat -> (last unanswered message, when it gets a GIF); one timer task per chat
        self._unanswered: dict[str, tuple[Any, float]] = {}
        self._unanswered_tasks: dict[str, asyncio.Task] = {}
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
                self._bot_guids[guid] = time.monotonic()
                while len(self._bot_guids) > 500:
                    self._bot_guids.popitem(last=False)
            return "ignored:from_me"
        if isinstance(guid, str) and guid and guid in self._seen:
            log_event("message.ignored", status="duplicate")
            return "ignored:duplicate"
        if data.get("associatedMessageType"):
            return self._on_tapback(data)
        if data.get("itemType"):
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
        # threadOriginatorGuid is a real inline reply. replyToGuid is only the previous message
        # in the chat, so a message right after the bot's is a maybe: judged below.
        thread = data.get("threadOriginatorGuid")
        is_reply_to_bot = trigger.kind == "none" and isinstance(thread, str) and await self._is_bot_message(thread)
        prev = data.get("replyToGuid")
        follows_bot = (trigger.kind == "none" and not is_reply_to_bot and isinstance(prev, str)
                       and time.monotonic() - self._bot_guids.get(prev, -1e9) < FOLLOWUP_WINDOW_SECONDS)
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
        how = "reply" if is_reply_to_bot else ""
        if trigger.kind == "none" and text and not self._muted(chat_guid):
            if ";-;" in chat_guid:  # a one-on-one chat with the bot: everything is for it
                trigger, how = Trigger("mention", text), "dm"
            elif follows_bot and await self._said_to_bot(chat_guid):
                trigger, how = Trigger("mention", text), "followup"
        if trigger.kind == "none" and stored is not None:
            self._arm_unanswered(chat_guid, stored)
        if (stored is not None and self.s.buddy_address and _norm_addr(address) == _norm_addr(self.s.buddy_address)
                and ";-;" not in chat_guid and not self._muted(chat_guid)):
            self._spawn(self._buddy_check(chat_guid, stored))
        if trigger.kind == "none":
            if self._muted(chat_guid):
                return "stored"
            if has_link and self.s.link_quiet_seconds > 0:
                self._spawn(self._quiet_link(chat_guid, stored))
                return "stored"
            if not self._maybe_quiet_question(chat_guid, text):
                self._maybe_react(chat_guid)
            return "stored"

        # "ricky 50 on the jets" is a bet, "ricky catch me up" a recap, not chat
        if trigger.kind == "mention" and self._bookie_here(chat_guid) and parse_bet(text):
            trigger = Trigger("bet", text)
        elif trigger.kind == "mention" and wants_recap(text):
            trigger = Trigger("recap", text)
        elif trigger.kind == "mention" and self.voice is not None and wants_voice(text):
            trigger = Trigger("voice", text)
        if trigger.kind in ("mention", "ask") and self._muted(chat_guid):
            log_event("trigger", chat_guid=chat_guid, kind=trigger.kind, status="muted")
            return "muted"
        if how:
            log_event("trigger", chat_guid=chat_guid, kind=trigger.kind, reason=how)
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
                        gifs=self.gifs is not None,
                        tweets=self.tweets is not None,
                        bookie=self._bookie_here(chat_guid),
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
            elif trigger.kind in ("optout", "optin"):
                await self._handle_opt(chat_guid, address, trigger.kind == "optin")
            elif trigger.kind == "voice":
                await self._handle_voice(chat_guid, trigger.arg, stored)
            elif trigger.kind == "recap":
                await self._handle_recap(chat_guid, sender, address, trigger.arg)
            elif trigger.kind in ("bet", "odds", "bets", "bank", "leaderboard", "bailout"):
                await self._handle_book(chat_guid, trigger.kind, trigger.arg, sender, address)
            elif trigger.kind == "gif":
                await self._handle_gif(chat_guid, trigger.arg)
            elif trigger.kind == "tweet":
                await self._handle_tweet(chat_guid, trigger.arg)
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
    async def _is_bot_message(self, guid: str) -> bool:
        """Is this message the bot's own? Asks BlueBubbles for ones sent before the last restart."""
        if guid in self._bot_guids:
            return True
        if guid in self._not_bot_guids:
            return False
        try:
            mine = bool((await self.bb.get_message(guid)).get("isFromMe"))
        except Exception as exc:  # noqa: BLE001
            log_event("reply.lookup_failed", error=exc_name(exc))
            return False
        if mine:
            self._bot_guids[guid] = 0.0  # old: counts for inline replies, not for the follow-up window
        else:
            self._not_bot_guids[guid] = None
            while len(self._not_bot_guids) > 500:
                self._not_bot_guids.popitem(last=False)
        return mine

    def _on_tapback(self, data: dict[str, Any]) -> str:
        """Count haha tapbacks per message; say something at HAHA_ACK_AT and HAHA_PRAISE_AT."""
        kind = str(data.get("associatedMessageType") or "").lower()
        if kind not in ("laugh", "-laugh", "2003", "3003") or not (self.s.haha_ack_at or self.s.haha_praise_at):
            return "ignored:reaction_or_system"
        chats = data.get("chats") or []
        chat_guid = chats[0].get("guid") if chats and isinstance(chats[0], dict) else None
        target = re.sub(r"^\w+:\d+/", "", str(data.get("associatedMessageGuid") or ""))
        who = (data.get("handle") or {}).get("address") if isinstance(data.get("handle"), dict) else None
        if not chat_guid or chat_guid not in self._allowed or not target or not who:
            return "ignored:reaction_or_system"
        st = self._laughs.get(target)
        if st is None:
            st = self._laughs[target] = {"chat": chat_guid, "who": set(), "fired": set()}
            while len(self._laughs) > 500:
                self._laughs.popitem(last=False)
        if kind in ("-laugh", "3003"):
            st["who"].discard(_norm_addr(who))
            return "tapback:removed"
        st["who"].add(_norm_addr(who))
        n = len(st["who"])
        level = None
        if self.s.haha_praise_at and n >= self.s.haha_praise_at and "praise" not in st["fired"]:
            level = "praise"
        elif self.s.haha_ack_at and n >= self.s.haha_ack_at and not st["fired"]:
            level = "ack"
        if level is None or self._muted(chat_guid):
            return "tapback:counted"
        st["fired"].add(level)
        log_event("haha.milestone", chat_guid=chat_guid, kind=level, count=n)
        self._spawn(self._haha_reply(chat_guid, target, level, n))
        return f"tapback:{level}"

    async def _haha_reply(self, chat_guid: str, target: str, level: str, n: int) -> None:
        try:
            msg = await self.bb.get_message(target)
        except Exception as exc:  # noqa: BLE001
            log_event("haha.lookup_failed", chat_guid=chat_guid, error=exc_name(exc))
            return
        if msg.get("isFromMe"):
            return  # the bot's own line got the laughs: no victory lap
        addr = (msg.get("handle") or {}).get("address") if isinstance(msg.get("handle"), dict) else None
        st = self._laughs.get(target)
        if st is not None and addr and _norm_addr(addr) in st["who"]:
            # laughing at your own message doesn't count
            n = len(st["who"]) - 1
            need = self.s.haha_praise_at if level == "praise" else self.s.haha_ack_at
            if n < need:
                st["fired"].discard(level)
                log_event("haha.self_laugh", chat_guid=chat_guid, kind=level)
                return
        name = await self.resolver.name_for(addr)
        said = " ".join(str(msg.get("text") or "").split())[:200] or "(a picture)"
        if level == "praise":
            nudge = (f"\n\n{name}'s message \"{said}\" just got a haha from every single person in the chat. That "
                     f"basically never happens. Give {name} one or two lines of huge, over-the-top, genuine praise for "
                     "it. Mean it: no roast, no backhanded stuff this time.")
        else:
            nudge = (f"\n\n{name}'s message \"{said}\" just got {n} haha reactions. Acknowledge it in one short, "
                     "dumb line: way too impressed, or a nonsense joke about it.")
        async with self._chat_lock(chat_guid):
            if self._muted(chat_guid):
                return
            try:
                await self._reply(chat_guid, self.context.get(chat_guid), use_cloud=False, reason="haha", nudge=nudge)
            except Exception as exc:  # noqa: BLE001
                log_event("handler.error", chat_guid=chat_guid, kind="haha", error=exc_name(exc))

    def _arm_unanswered(self, chat_guid: str, marker) -> None:
        """(Re)start the chat's 'nobody answered' clock for its newest message."""
        if self.gifs is None or self.s.unanswered_gif_minutes <= 0:
            return
        self._unanswered[chat_guid] = (marker, time.monotonic() + self.s.unanswered_gif_minutes * 60)
        t = self._unanswered_tasks.get(chat_guid)
        if t is None or t.done():
            # not in self._tasks: drain() must not wait half an hour on shutdown
            self._unanswered_tasks[chat_guid] = asyncio.create_task(self._unanswered_loop(chat_guid))

    async def _unanswered_loop(self, chat_guid: str) -> None:
        while True:
            entry = self._unanswered.get(chat_guid)
            if entry is None:
                return
            marker, due = entry
            wait = due - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
                continue  # a newer message may have moved the deadline
            self._unanswered.pop(chat_guid, None)
            history = self.context.get(chat_guid)
            if not history or history[-1] is not marker or self._muted(chat_guid):
                return  # someone (or the bot) said something after it, or he's muted
            if self.s.in_quiet_hours(_local_hour()):
                log_event("unanswered_gif.skipped", chat_guid=chat_guid, reason="quiet_hours")
                return
            log_event("trigger", chat_guid=chat_guid, kind="unanswered_gif")
            query = "crickets"
            try:
                async with self._llm_gate:
                    raw = await self.llm.reply(
                        UNANSWERED_GIF_SYSTEM.replace("{sender}", marker.sender).replace("{text}", strip_cards(marker.text)[:200]),
                        [{"role": "user", "content": "GIF search:"}])
                query = clean_query(raw.splitlines()[0] if raw else "", 40) or query
            except Exception as exc:  # noqa: BLE001
                log_event("handler.error", chat_guid=chat_guid, kind="unanswered_gif", error=exc_name(exc))
            async with self._chat_lock(chat_guid):
                h = self.context.get(chat_guid)
                if h and h[-1] is marker and not self._muted(chat_guid):
                    if not await self._send_gif(chat_guid, query) and query != "crickets":
                        await self._send_gif(chat_guid, "crickets")
            return

    async def _said_to_bot(self, chat_guid: str) -> bool:
        """Judgment call on a message that came right after the bot's, without its name."""
        history = self.context.get(chat_guid)
        if not history:
            return False
        try:
            async with self._llm_gate:
                structured = getattr(self.llm, "structured", None)
                system = ADDRESSEE_SYSTEM.replace("{bot_name}", self.s.bot_name)
                msgs = addressee_messages(_plain(history), self.s.bot_name)
                raw = await structured(system, msgs, ADDRESSEE_SCHEMA) if structured else await self.llm.reply(system, msgs)
        except Exception as exc:  # noqa: BLE001
            log_event("followup.error", chat_guid=chat_guid, error=exc_name(exc))
            return False
        to_bot = parse_addressee(raw)
        log_event("followup.judged", chat_guid=chat_guid, status="to_bot" if to_bot else "not_to_bot")
        return to_bot

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
            st = self._pending[chat_guid] = {"first": now, "last": now, "marker": marker, "pings": [], "n": 0, "cloud": False}
            self._spawn(self._answer_batch(chat_guid))
        st["last"], st["marker"] = now, marker
        st["n"] += 1
        st["cloud"] = st["cloud"] or force_cloud
        if marker is not None and all(m is not marker for m in st["pings"]):
            st["pings"].append(marker)

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
                await self._handle_mention(chat_guid, force_cloud=st["cloud"], marker=st["marker"], pings=st["pings"])
            except Exception as exc:  # noqa: BLE001
                log_event("handler.error", chat_guid=chat_guid, kind="mention", error=exc_name(exc))

    async def _handle_mention(self, chat_guid: str, force_cloud: bool = False, marker=None,
                              pings: list | None = None) -> None:
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
        if intent.action == "gif":
            log_event("intent", chat_guid=chat_guid, kind="gif")
            await self._handle_gif(chat_guid, intent.prompt)
            return
        if intent.action == "tweet":
            log_event("intent", chat_guid=chat_guid, kind="tweet")
            await self._handle_tweet(chat_guid, intent.prompt)
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
        nudge = target_nudge(pings or ([marker] if marker is not None else []))
        await self._reply(chat_guid, history, use_cloud=(intent.action == "lookup"),
                          reason="command" if force_cloud else "intent", nudge=nudge)

    async def _reply(self, chat_guid: str, history, use_cloud: bool, reason: str, nudge: str = "",
                     as_voice: bool = False) -> None:
        messages = build_messages(self.s.persona, self.s.bot_name, history)
        backend = self.llm
        system = self._system_for(chat_guid) + nudge
        facts = ""
        if use_cloud and self.cloud is not None:
            log_event("llm.escalate", chat_guid=chat_guid, reason=reason, backend=self.cloud.name)
            # Say so first: the lookup takes 10-20 s and silence reads as ignoring them.
            await self._say(chat_guid, random.choice(LOOKUP_NOTICES), remember=False)
            facts = await self._look_up(chat_guid, history, nudge)
            if facts:
                system += LOOKED_UP.replace("{facts}", facts)
        names = speaker_names(history, self.s.bot_name)
        stops = stop_sequences(names)

        async def ask(b, sys_prompt):
            if isinstance(b, OllamaBackend):
                return await b.reply(sys_prompt + CONSISTENCY_RULE + SELF_ONLY_RULE, messages, stop=stops)
            return await b.reply(sys_prompt + CONSISTENCY_RULE + SELF_ONLY_RULE, messages)

        started = time.monotonic()
        async with self._llm_gate:
            raw = await ask(backend, system)
            if facts:
                chat_text = "\n".join(m.text for m in history[-20:])
                if not facts_kept(clean_reply(raw, self.s.bot_name), facts, chat_text):
                    log_event("lookup.numbers_changed", chat_guid=chat_guid, attempt=1)
                    raw = await ask(backend, system + FACTS_STRICT)
                    if not facts_kept(clean_reply(raw, self.s.bot_name), facts, chat_text):
                        log_event("lookup.numbers_changed", chat_guid=chat_guid, attempt=2)
                        raw = facts  # the plain facts beat a garbled score
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
        if as_voice:
            await self._say_voice(chat_guid, reply)
        else:
            await self._say(chat_guid, reply)

    async def _look_up(self, chat_guid: str, history, nudge: str) -> str:
        """Plain facts from the cloud model (web search), or '' if it's unavailable."""
        recent = [m for m in history[-12:]]
        lines = "\n".join(f"{self.s.bot_name}: {strip_cards(m.text)}" if m.from_bot else f"{m.sender}: {strip_cards(m.text)}"
                          for m in recent)
        target = re.sub(r"\s+", " ", nudge).strip()
        system = FACT_FINDER_SYSTEM.replace("{bot_name}", self.s.bot_name).replace(
            "{target}", f" ({target.split('. Answer that')[0]}.)" if target else "")
        try:
            facts = await self.cloud.reply(system, [{"role": "user", "content": lines or "(no messages)"}])  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            log_event("llm.cloud_failed", chat_guid=chat_guid, backend=self.cloud.name, error=exc_name(exc))  # type: ignore[union-attr]
            return ""
        facts = strip_fake_notes(re.sub(r"https?://\S+", "", facts or "")).strip()[:900]
        log_event("lookup.facts", chat_guid=chat_guid, chars=len(facts))
        return facts

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
        last = strip_cards(history[-1].text)
        if intent.action == "gif" and not looks_like_gif_request(last):
            log_event("intent.vetoed", kind="gif")
            return Intent()
        if intent.action == "tweet" and not looks_like_tweet_request(last):
            log_event("intent.vetoed", kind="tweet")
            return Intent()
        if intent.action == "image" and looks_like_gif_request(last):
            return Intent("gif", intent.prompt)  # "send a gif of ..." is a GIF, not a generated picture
        if intent.action == "image" and not looks_like_image_request(last):
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
                raw = await self.llm.reply(self._system_for(chat_guid) + reaction_addendum(last, gifs=self.gifs is not None),
                                           build_messages(self.s.persona, self.s.bot_name, history))
            gif_query = gif_from_reaction(raw) if self.gifs is not None else None
            if gif_query:
                self._last_reaction_text[chat_guid] = "GIF: " + gif_query
                await self._send_gif(chat_guid, gif_query)
                log_event("reaction.sent", chat_guid=chat_guid, kind="gif")
                return
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

    # ------------------------------------------------------------------ NFL
    def _bookie_here(self, chat_guid: str) -> bool:
        return self.bookie is not None and self.espn is not None and chat_guid in self.s.sports_chat_guids

    def _fan_of(self, name: str) -> str | None:
        """Which followed team this person roots for: full name first, first name only if unambiguous."""
        full = (name or "").strip().lower()
        for abbr, fans in self.sports_fans.items():
            if any(f.lower() == full for f in fans):
                return abbr
        first = full.split(" ")[0] if full else ""
        hits = {abbr for abbr, fans in self.sports_fans.items() for f in fans if f.split(" ")[0].lower() == first}
        return hits.pop() if len(hits) == 1 else None

    async def _handle_book(self, chat_guid: str, kind: str, arg: str, name: str, address: str | None) -> None:
        if not self._bookie_here(chat_guid):
            await self._say(chat_guid, "the book's closed in here", remember=False)
            return
        who = _norm_addr(address) or name
        book = self.bookie
        if kind == "bank":
            p = book._player(chat_guid, who, name)
            riding = sum(b["stake"] for b in book.open_bets(chat_guid) if b["who"] == who)
            await self._say(chat_guid, f"{name}: {book.balance(chat_guid, who, name)} bucks"
                            + (f", {riding} riding" if riding else "") + f" ({p['won']}-{p['lost']})")
            return
        if kind == "bailout":
            ok, bal = book.bailout(chat_guid, who, name)
            await self._say(chat_guid, random.choice(BAILOUT_YES).replace("{n}", str(bal)) if ok else
                            random.choice(BAILOUT_NO))
            return
        if kind == "leaderboard":
            rows = book.leaderboard(chat_guid)
            if not rows:
                await self._say(chat_guid, "nobody's bet yet. cowards", remember=False)
                return
            lines = [f"{i}. {r['name']} {r['bucks']}" + (f" (+{r['at_risk']} riding)" if r["at_risk"] else "")
                     + f" {r['won']}-{r['lost']}" + (" 🪦" * min(r["bailouts"], 3)) for i, r in enumerate(rows, 1)]
            await self._say(chat_guid, "🏦 the book\n" + "\n".join(lines))
            return
        if kind == "bets":
            bets = book.open_bets(chat_guid)
            if not bets:
                await self._say(chat_guid, "nothing riding. boring", remember=False)
                return
            await self._say(chat_guid, "riding:\n" + "\n".join(
                f"{b['name']} {b['stake']} on {b['team_name']} ({fmt_odds(b['odds'])}) to win {b['to_win']}" for b in bets))
            return
        hint = league_hint(arg)
        if kind == "odds" and not arg.strip():
            hint = "nfl"
        try:
            games = await (self.espn.upcoming(hint) if hint else self.espn.upcoming_all())  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            log_event("bookie.odds_failed", error=exc_name(exc))
            await self._say(chat_guid, "can't reach the lines right now, try in a bit", remember=False)
            return
        idx = TeamIndex.from_games(games)
        if kind == "odds":
            team = self._pick_team(idx, arg) if arg.strip() and idx.candidates(arg) else None
            if isinstance(team, str):
                await self._say(chat_guid, team, remember=False)
                return
            mine = [g for g in games if team is None or (g.league == team[0] and g.team(team[1]))]
            live = [g for g in mine if g.state == "in" and self.s.bookie_live and not too_late_to_bet(g)]
            pre = sorted((g for g in mine if g.state == "pre"), key=lambda g: g.start)
            if team is not None:
                live, pre = live[:1], ([] if live else pre[:1])
            live = live[:8]
            lines = []
            for g, odds in zip(live, await asyncio.gather(*(live_odds(self.espn, g) for g in live))):
                if odds:
                    g.away.moneyline, g.home.moneyline = odds
                    lines.append(f"🔴 LIVE {g.detail}: {g.away.name} {g.away.score} {fmt_odds(odds[0])} @ "
                                 f"{g.home.name} {g.home.score} {fmt_odds(odds[1])}")
            lines += [odds_line(g) for g in pre[:16 - len(lines)]]
            if not lines:
                await self._say(chat_guid, "nothing on the board for that right now", remember=False)
                return
            title = f"{LEAGUES[mine[0].league][1]} lines (moneyline)" if team is None else "the line"
            more = "\n(/odds nba, nhl, mlb, cfb, cbb, wnba, or a team)" if not arg.strip() else ""
            await self._say(chat_guid, f"{title}:\n" + "\n".join(lines) + more)
            return
        parsed = parse_bet(arg) or parse_bet(arg, loose=True)
        if parsed is None:
            await self._say(chat_guid, "how much on who? like: /bet 50 jets", remember=False)
            return
        stake, team_text = parsed
        team = self._pick_team(idx, team_text)
        if isinstance(team, str):
            await self._say(chat_guid, team, remember=False)
            return
        upcoming = sorted((g for g in games if team and g.league == team[0] and g.team(team[1])), key=lambda g: g.start)
        # a game that's on right now is a live bet; otherwise their next game
        game = next((g for g in upcoming if g.state == "in"), None) or next((g for g in upcoming if g.state == "pre"),
                                                                             upcoming[0] if upcoming else None)
        if team is None or game is None:
            await self._say(chat_guid, "who? i don't have them on the board", remember=False)
            return
        live = game.state == "in"
        if live:
            if not self.s.bookie_live or too_late_to_bet(game):
                await self._say(chat_guid, f"{game.matchup} is too far gone, book's closed on it", remember=False)
                return
            odds = await live_odds(self.espn, game)  # type: ignore[arg-type]
            if odds is None:
                await self._say(chat_guid, "no live line on that one right now", remember=False)
                return
            game.away.moneyline, game.home.moneyline = odds
        bet = book.place(chat_guid, who, name, game, team[1], stake, live=live)
        if isinstance(bet, str):
            msg = {"started": f"{game.matchup} already started, book's closed on it",
                   "too_much": f"you've got {book.balance(chat_guid, who, name)} bucks, relax",
                   "broke": "you're broke. /bailout if you've got no shame",
                   "zero": "bet something real"}.get(bet, "can't take that one")
            await self._say(chat_guid, msg, remember=False)
            return
        quip = random.choice(BET_QUIPS)
        fan = self._fan_of(name)
        if fan and game.league == "nfl" and fan == game.other(team[1]).abbr:
            quip = "betting against your own team. cold blooded"
        when = f"LIVE ({game.detail}, {game.score_line()})" if live else game.start.astimezone().strftime("%a %-I:%M%p").lower()
        await self._say(chat_guid, f"locked: {name} {bet['stake']} on the {bet['team_name']} ({fmt_odds(bet['odds'])}) vs the "
                        f"{bet['opp_name']} {when}, pays {bet['to_win']}. {book.balance(chat_guid, who, name)} left. {quip}")
        log_event("bookie.bet", chat_guid=chat_guid, count=bet["stake"])

    def _pick_team(self, idx: TeamIndex, text: str) -> tuple[str, str] | str | None:
        """(league, abbr) for what someone typed; a question back if it could be several teams."""
        c = idx.candidates(text)
        hint = league_hint(text)
        if hint:
            c = {t for t in c if t[0] == hint}
        if len(c) > 1:
            followed = {t for t in c if t[0] == "nfl" and t[1] in self.sports_fans}
            if len(followed) == 1:
                return followed.pop()
            leagues = sorted({LEAGUES[t[0]][1] for t in c})
            if len(leagues) > 1:
                return f"which one, {' or '.join(leagues)}? say it like /bet 50 {sorted({t[0] for t in c})[0]} {text.strip()}"
        return next(iter(c)) if len(c) == 1 else None

    async def on_games(self, games: list[Game]) -> None:
        """Every sports poll: settle finished games that have bets on them (any league)."""
        if self.bookie is None:
            return
        have = {("nfl", f"w{g.week}:{g.season_type}") for g in games}
        extra = []
        for league, what in self.bookie.open_fetches() - have:
            try:
                if what.startswith("w"):
                    week, st = what[1:].split(":")
                    extra += await self.espn.scoreboard(int(week), int(st), league=league)  # type: ignore[union-attr]
                else:
                    extra += await self.espn.scoreboard(league=league, date=what[1:])  # type: ignore[union-attr]
            except Exception as exc:  # noqa: BLE001
                log_event("bookie.settle_fetch_failed", error=exc_name(exc))
        everything = {g.key: g for g in games + extra}
        await self._watch_bets(list(everything.values()))
        for chat_guid, results in self.bookie.settle(list(everything.values())).items():
            by_game: dict[str, list] = {}
            for r in results:
                by_game.setdefault(r.final, []).append(r)
            blocks = []
            for final, rs in by_game.items():
                blocks.append(f"💰 FINAL: {final}\n" + "\n".join(
                    ("✅" if r.result == "won" else "❌" if r.result == "lost" else "➖")
                    + f" {r.name} {'+' if r.delta > 0 else ''}{r.delta} on the {r.team} → {r.balance}" for r in rs))
            facts = "\n\n".join(blocks)
            if self._muted(chat_guid):
                await self._say(chat_guid, facts)  # money still gets reported, just no commentary
            else:
                comment = await self._sports_comment(chat_guid, facts, "Roast whoever lost the most and gas up whoever "
                                                     "won the most, by name. One short line.")
                await self._say(chat_guid, facts + (f"\n{comment}" if comment else ""))
            log_event("bookie.settled", chat_guid=chat_guid, count=len(results))

    async def _watch_bets(self, games: list[Game]) -> None:
        """Games people have money on: post at the half, and late if it's a sweat or a blowout."""
        for g in games:
            if g.state != "in":
                continue
            per_chat = self.bookie.bets_on(g.key)  # type: ignore[union-attr]
            if not per_chat:
                continue
            late = late_game(g)
            if late and self.bookie.mark(g.key, "late"):  # type: ignore[union-attr]
                self.bookie.mark(g.key, "half")  # type: ignore[union-attr]
                head = f"🚨 sweat alert, {g.detail}" if late == "close" else f"💀 blowout, {g.detail}"
                how = ("It's coming down to the wire. Hype the sweat for everyone with money on it." if late == "close" else
                       "It's over. Roast whoever's getting cooked, gloat for whoever's about to cash.")
            elif past_halfway(g) and self.bookie.mark(g.key, "half"):  # type: ignore[union-attr]
                head = f"📊 {g.detail if 'half' in g.detail.lower() else 'halfway, ' + g.detail}"
                how = "Say how it's looking for each person with money on it, by name."
            else:
                continue
            for chat_guid, bets in per_chat.items():
                if chat_guid not in self._allowed or self._muted(chat_guid):
                    continue
                riding = "; ".join(f"{b['name']} {b['stake']} on the {b['team_name']} (to win {b['to_win']})" for b in bets)
                facts = f"{head}: {g.score_line()}\nriding: {riding}"
                comment = await self._sports_comment(chat_guid, facts, how)
                await self._say(chat_guid, facts + (f"\n{comment}" if comment else ""))
                log_event("bookie.watch", chat_guid=chat_guid, kind=late or "half")

    async def on_alert(self, alert: Alert) -> None:
        """A live NFL alert: the fact line, plus the bot's one-line reaction."""
        g = alert.game
        fans = "; ".join(f"{(g.team(a) or g.home).name} fans in here: {', '.join(n)}" for a, n in alert.fans.items())
        scorer = g.team(alert.scorer).name if alert.scorer and g.team(alert.scorer) else None
        how = {"kickoff": "The game just started. Hype the fans in here.",
               "score": f"{scorer} just scored. Hype their fans in here if any, needle the other side's fans.",
               "half": "It's halftime. React to how it's going for the fans in here.",
               "final": f"Game over{', ' + scorer + ' won' if scorer else ', a tie'}. Celebrate with the winners' fans, "
                        "roast the losers' fans, by name."}[alert.kind]
        for chat_guid in self.s.sports_chat_guids:
            if chat_guid not in self._allowed or self._muted(chat_guid):
                continue
            comment = await self._sports_comment(chat_guid, alert.text, f"{how} {fans}.")
            await self._say(chat_guid, alert.text + (f"\n{comment}" if comment else ""))

    async def _sports_comment(self, chat_guid: str, facts: str, how: str) -> str:
        system = (self._system_for(chat_guid) + f"\n\nYou just posted this sports update in the chat:\n{facts}\n"
                  f"Now add ONE short line of your own reaction, like a friend watching. {how} Don't repeat the "
                  "score or any numbers, no links, under 140 characters.")
        try:
            async with self._llm_gate:
                raw = await self.llm.reply(system, [{"role": "user", "content": "(your one line)"}])
        except Exception as exc:  # noqa: BLE001
            log_event("sports.comment_failed", error=exc_name(exc))
            return ""
        line = strip_fake_notes(clean_reply(raw, self.s.bot_name)).split("\n")[0].strip()
        if not line or len(line) > 160 or is_refusal(line) or not afterword_ok(facts, line):
            return ""
        return line

    async def _handle_voice(self, chat_guid: str, text: str, marker) -> None:
        """'send a voice note' / '/voice <what>' (a new reply, spoken) or 'say that out loud' (the last one)."""
        if self.voice is None:
            await self._say(chat_guid, "no voice on me right now", remember=False)
            return
        async with self._chat_lock(chat_guid):
            history = self.context.get(chat_guid)
            if repeat_last(text):
                last = next((m.text for m in reversed(history) if m.from_bot and not m.text.startswith("[")), None)
                if last:
                    await self._say_voice(chat_guid, last, remember=False)
                    return
            await self._reply(chat_guid, history, use_cloud=False, reason="voice", as_voice=True,
                              nudge=target_nudge([marker] if marker is not None else [])
                              + "\n\nThis reply will be read out loud as a voice note: talk like you're leaving one, "
                                "two to four sentences, no emoji or links." + (
                                  " You can drop in one or two of these sound tags where they fit: "
                                  + ", ".join(f"<{t}>" for t in SOUND_TAGS[:6]) + " (e.g. '<laugh> no way')."
                                  if getattr(self.voice, "expressive", False) else ""))

    async def _say_voice(self, chat_guid: str, text: str, remember: bool = True) -> None:
        """Speak `text` and send it as audio; falls back to text if the voice fails."""
        text = strip_fake_notes(text)
        workdir = new_workdir()
        try:
            audio = await self.voice.synthesize(text, workdir) if self.voice is not None else None
            if audio is None:
                await self._say(chat_guid, strip_tags(text), remember=remember)
                return
            await self.bb.send_attachment(chat_guid, audio, name=f"{self.s.bot_name}.m4a", mime="audio/x-m4a")
            log_event("voice.sent", chat_guid=chat_guid, chars=len(text))
            if remember:
                self.context.add(chat_guid, self.s.bot_name, strip_tags(text), from_bot=True)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    async def _buddy_check(self, chat_guid: str, msg) -> None:
        """The designated person: encouragement when they're down, a "this u?" picture when they diss the bot."""
        text = strip_cards(msg.text)
        try:
            async with self._llm_gate:
                structured = getattr(self.llm, "structured", None)
                system = BUDDY_MOOD_SYSTEM.replace("{name}", msg.sender).replace("{bot_name}", self.s.bot_name)
                msgs = [{"role": "user", "content": f"{msg.sender}: {text}\n\nJSON only."}]
                raw = await structured(system, msgs, BUDDY_MOOD_SCHEMA) if structured else await self.llm.reply(system, msgs)
        except Exception as exc:  # noqa: BLE001
            log_event("buddy.error", chat_guid=chat_guid, error=exc_name(exc))
            return
        mood = parse_mood(raw)
        if mood == "jab" and not mentions_bots(text, self.s.bot_name):
            mood = "none"
        if mood not in ("down", "jab"):
            return
        cooldown = (self.s.buddy_cheer_cooldown_minutes if mood == "down" else self.s.buddy_jab_cooldown_minutes) * 60
        now = time.monotonic()
        if now - self._buddy_last.get((chat_guid, mood), -1e9) < cooldown or self._muted(chat_guid):
            return
        self._buddy_last[(chat_guid, mood)] = now
        log_event("buddy.trigger", chat_guid=chat_guid, kind=mood)
        if mood == "down":
            nudge = (f"\n\n{msg.sender} just said: \"{text[:200]}\". He sounds down. Check in on him with one or two lines "
                     "of real encouragement, like a friend who has his back. Warm and genuine, no roasting this time.")
            async with self._chat_lock(chat_guid):
                await self._reply(chat_guid, self.context.get(chat_guid), use_cloud=False, reason="buddy", nudge=nudge)
            return
        person = self.likenesses.match(msg.sender) if self.likenesses is not None else None
        if person is None or self.images is None or self.s.image_backend != "mflux":
            await self._say(chat_guid, random.choice(BUDDY_COMEBACKS).replace("{name}", msg.sender.lower()))
            return
        scene = random.choice(BUDDY_SCENES)
        job = ImageJob(chat_guid=chat_guid, prompt=person.prompt_for(f"{person.name} {scene}, candid photo"),
                       job_id=uuid.uuid4().hex[:8], loras=((str(person.lora), person.scale),),
                       caption=random.choice(BUDDY_THIS_U).replace("{name}", msg.sender.lower()))
        await self.images.submit(job)

    async def _handle_opt(self, chat_guid: str, address: str | None, opted_in: bool) -> None:
        person = self.likenesses.set_opt_in(address, opted_in) if self.likenesses is not None else None
        if person is None:
            await self._say(chat_guid, "you're not in my picture book, nothing to opt out of" if not opted_in
                            else "i don't have a trained likeness for you", remember=False)
            return
        await self._say(chat_guid, f"done, {person.name} is back in the pictures" if opted_in
                        else f"done, no more {person.name} in my pictures. /optin if you miss it")

    async def _handle_recap(self, chat_guid: str, name: str, address: str | None, arg: str) -> None:
        window = window_from(arg)
        await self._say(chat_guid, random.choice(RECAP_ACKS), remember=False)
        try:
            lines, since = await gather(self.bb, self.resolver, self.s.bot_name, chat_guid, address, window, norm=_norm_addr)
        except Exception as exc:  # noqa: BLE001
            log_event("recap.failed", chat_guid=chat_guid, error=exc_name(exc))
            await self._say(chat_guid, "can't pull up the history right now, scroll up like an animal", remember=False)
            return
        if len(lines) < 3:
            await self._say(chat_guid, "nothing happened. you didn't miss a thing", remember=False)
            return
        system = (RECAP_SYSTEM.replace("{persona}", self.s.persona).replace("{asker}", name).replace("{since}", since))
        try:
            async with self._llm_gate:
                raw = await self.llm.reply(system, [{"role": "user", "content": "\n".join(lines) + f"\n\nNow catch {name} up."}])
        except Exception as exc:  # noqa: BLE001
            log_event("recap.failed", chat_guid=chat_guid, error=exc_name(exc))
            raw = ""
        text = strip_fake_notes(clean_reply(raw, self.s.bot_name))
        if not text or is_refusal(text):
            await self._say(chat_guid, "too much went down, just scroll up", remember=False)
            return
        if len(text) > 900:
            text = text[:900].rsplit("\n", 1)[0]
        await self._say(chat_guid, text)
        log_event("recap.sent", chat_guid=chat_guid, count=len(lines), chars=len(text))

    async def _handle_gif(self, chat_guid: str, query: str) -> None:
        if self.gifs is None:
            await self._say(chat_guid, "no gifs, nobody set me up with giphy", remember=False)
            return
        if not query.strip():
            await self._say(chat_guid, "a gif of what", remember=False)
            return
        if not await self._send_gif(chat_guid, query):
            await self._say(chat_guid, "couldn't find a gif for that", remember=False)

    async def _send_gif(self, chat_guid: str, query: str) -> bool:
        try:
            gif = await self.gifs.find(query)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            log_event("gif.failed", chat_guid=chat_guid, error=exc_name(exc))
            return False
        if gif is None:
            log_event("gif.none", chat_guid=chat_guid)
            return False
        try:
            await self.bb.send_attachment(chat_guid, gif.path, name=f"{self.s.bot_name.lower()}.gif", mime="image/gif")
        finally:
            gif.path.unlink(missing_ok=True)
        log_event("gif.sent", chat_guid=chat_guid)
        self.context.add(chat_guid, self.s.bot_name, f"[sent a gif: {gif.query}]", from_bot=True)
        return True

    async def _handle_tweet(self, chat_guid: str, query: str) -> None:
        if self.tweets is None:
            await self._say(chat_guid, "can't pull tweets, that's not switched on", remember=False)
            return
        if not query.strip():
            await self._say(chat_guid, "a tweet about what", remember=False)
            return
        if not self.tweets.allowed():
            await self._say(chat_guid, "i'm tapped out on tweets for the hour", remember=False)
            return
        try:
            tweet = await self.tweets.find(query)
        except Exception as exc:  # noqa: BLE001
            log_event("tweet.failed", chat_guid=chat_guid, error=exc_name(exc))
            tweet = None
        if tweet is None:
            await self._say(chat_guid, "couldn't find anything good on that", remember=False)
            return
        await self._say(chat_guid, tweet.url, remember=False)
        log_event("tweet.sent", chat_guid=chat_guid)
        self.context.add(chat_guid, self.s.bot_name, describe_tweet(tweet), from_bot=True)

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
        loras: tuple[tuple[str, float], ...] = ()
        person = self.likenesses.match(prompt) if self.likenesses is not None and self.s.image_backend == "mflux" else None
        if person is not None:
            if blocked(prompt):
                log_event("likeness.blocked", chat_guid=chat_guid)
                await self._say(chat_guid, random.choice(LIKENESS_NO).replace("{name}", person.name), remember=False)
                return
            loras = ((str(person.lora), person.scale),)
            prompt = person.prompt_for(prompt)
            log_event("likeness.used", chat_guid=chat_guid)
        job = ImageJob(chat_guid=chat_guid, prompt=prompt, job_id=uuid.uuid4().hex[:8], loras=loras)
        ahead = self.images.qsize()
        pool = [a.replace("{n}", str(ahead)) for a in IMAGE_QUEUE_ACKS] if ahead else list(IMAGE_ACKS)
        last = self._last_image_ack.get(chat_guid)
        ack = random.choice([a for a in pool if a != last] or pool)
        self._last_image_ack[chat_guid] = ack
        await self._say(chat_guid, ack)
        await self.images.submit(job)

    async def send_image(self, job: ImageJob, png: Path) -> None:
        await self.bb.send_attachment(job.chat_guid, png, name=f"{self.s.bot_name.lower()}-{job.job_id}.png")
        log_event("image.sent", chat_guid=job.chat_guid, job_id=job.job_id)
        if job.caption:
            await self._say(job.chat_guid, job.caption)
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
        cleaned = strip_fake_notes(text)
        if cleaned != text:
            log_event("reply.fake_note_removed", chat_guid=chat_guid)
            text = cleaned
        if not text:
            return
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
        for t in self._unanswered_tasks.values():
            t.cancel()
        self._unanswered.clear()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)


def _local_hour() -> int:
    return datetime.now().hour


def _plain(history):
    """History without link cards, for the tool router and the memory notes."""
    return [replace(m, text=strip_cards(m.text)) if "[link: " in m.text else m for m in history]
