"""Catch someone up: summarize what happened in the chat since they last said something.

Reads the chat's recent history from BlueBubbles (the bot's in-memory context is wiped on
restart and capped), keeps real messages (no tapbacks), and has the local model write a
short recap. Nothing leaves the Mac.
"""

from __future__ import annotations

import re
import time
from typing import Any

from .triggers import parse_duration

_CHAT = r"(chat|convo|conversation|gc|group ?chat|thread|in here|here)"
RECAP_ASK = re.compile(
    r"\bcatch (me|us|him|her|them) up\b|\bwhat(?:'d|d| did) (i|we) miss\b|\btl;?dr\b|\bfill (me|us) in\b"
    r"|\brecap\b(?!\s+(?:of\s+)?(?:the\s+)?(?:game|match|fight|draft|season|week|episode|show))"
    rf"|\bsummar(?:y|ize|ise)\b.{{0,20}}\b{_CHAT}\b"
    rf"|\bwhat(?:'s| has)? (?:been )?(?:going on|happen(?:ed|ing)) {_CHAT}\b",
    re.IGNORECASE,
)
MAX_LINES = 150
DEFAULT_LOOKBACK = 24 * 3600
MIN_MISSED = 12  # fewer than this since their last message: recap the last stretch instead

RECAP_SYSTEM = (
    "{persona}\n\nYou're catching {asker} up on the group chat. Below is what was said{since}. Write a short "
    "recap in your own voice: 3 to 7 dash bullets, most important first. Cover who said or did what, plans and "
    "decisions (times, places, who's in), bets, arguments, and the best jokes or roasts, by name. Accuracy "
    "first: only what's in the messages, never invent, and skip filler. Lines marked (you) are yours. No intro "
    "paragraph and no sign-off; at most one short line before the bullets. Under 700 characters."
)


def wants_recap(text: str) -> bool:
    return bool(RECAP_ASK.search(text or ""))


def _body(m: dict[str, Any]) -> str:
    text = " ".join(str(m.get("text") or "").split())
    text = re.sub(r"https?://(?:www\.)?([^/\s]+)\S*", r"(a \1 link)", text)
    atts = m.get("attachments") or []
    if atts:
        kinds = {str(a.get("mimeType") or "").split("/")[0] for a in atts if isinstance(a, dict)}
        tag = "[photo]" if kinds == {"image"} else "[video]" if "video" in kinds else "[attachment]"
        text = f"{text} {tag}".strip()
    return text[:300]


async def gather(bb, resolver, bot_name: str, chat_guid: str, asker_addr: str | None, window: int | None,
                 norm=lambda a: a or "", now: float | None = None) -> tuple[list[str], str]:
    """(transcript lines oldest first, a phrase describing the span) for the recap."""
    now = now or time.time()
    lookback = window or DEFAULT_LOOKBACK
    raw = await bb.recent_messages(chat_guid, limit=400, after_ms=int((now - lookback) * 1000))
    msgs = [m for m in raw if not m.get("associatedMessageType") and not m.get("itemType") and _body(m)]
    msgs.sort(key=lambda m: m.get("dateCreated") or 0)
    if msgs and not msgs[-1].get("isFromMe") and norm(((msgs[-1].get("handle") or {}).get("address"))) == norm(asker_addr):
        msgs = msgs[:-1]  # the "catch me up" itself
    since = f" in the last {_span(window)}" if window else ""
    if not window and asker_addr:
        mine = [i for i, m in enumerate(msgs) if not m.get("isFromMe")
                and norm(((m.get("handle") or {}).get("address"))) == norm(asker_addr)]
        if mine and len(msgs) - 1 - mine[-1] >= MIN_MISSED:
            msgs = msgs[mine[-1] + 1:]
            since = " since they last said something"
        elif not mine:
            since = " in the last day"
        else:
            msgs = msgs[-60:]
            since = " most recently"
    msgs = msgs[-MAX_LINES:]
    names: dict[str, str] = {}
    lines = []
    for m in msgs:
        if m.get("isFromMe"):
            who = f"{bot_name} (you)"
        else:
            addr = (m.get("handle") or {}).get("address")
            if addr not in names:
                names[addr] = await resolver.name_for(addr)
            who = names[addr]
        lines.append(f"{who}: {_body(m)}")
    return lines, since


def _span(seconds: int) -> str:
    if seconds % 3600 == 0:
        h = seconds // 3600
        return f"{h} hour{'s' if h != 1 else ''}" if h < 48 else f"{h // 24} days"
    return f"{seconds // 60} minutes"


def window_from(text: str) -> int | None:
    """'catch me up on the last 2 hours' -> 7200; None for 'since I left'."""
    return parse_duration(text)
