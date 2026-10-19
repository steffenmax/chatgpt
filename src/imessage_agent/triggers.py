"""Pure functions: does a message address the bot, and what does it ask for?"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Kind = Literal["mention", "ask", "image", "help", "forget", "remind", "reminders", "none"]


@dataclass(frozen=True)
class Trigger:
    kind: Kind
    arg: str = ""  # image prompt, or the message with the mention left in for chat


_COMMANDS = {
    "/image": "image", "/img": "image", "/ask": "ask", "/help": "help", "/forget": "forget",
    "/remind": "remind", "/reminders": "reminders",
}


def mention_pattern(bot_name: str) -> re.Pattern[str]:
    name = re.escape(bot_name.strip())
    # whole-word, case-insensitive, optional leading @ ; tolerate trailing punctuation
    return re.compile(rf"(?<![\w@])@?{name}(?![\w])", re.IGNORECASE)


def is_mention(text: str, bot_name: str) -> bool:
    if not text or not bot_name:
        return False
    return mention_pattern(bot_name).search(text) is not None


def parse_command(text: str) -> tuple[str | None, str]:
    """Return (command kind, argument) if the message starts with a known slash command."""
    if not text:
        return None, ""
    stripped = text.lstrip()
    if not stripped.startswith("/"):
        return None, ""
    head, _, rest = stripped.partition(" ")
    kind = _COMMANDS.get(head.lower())
    if kind is None:
        return None, ""
    return kind, rest.strip()


def detect(text: str | None, bot_name: str) -> Trigger:
    text = (text or "").strip()
    if not text:
        return Trigger("none")
    cmd, arg = parse_command(text)
    if cmd:
        return Trigger(cmd, arg)  # type: ignore[arg-type]
    if is_mention(text, bot_name):
        return Trigger("mention", text)
    return Trigger("none")


def help_text(bot_name: str, image_enabled: bool, cloud_facts: bool = False, reminders: bool = False) -> str:
    where = "most of it runs on one mac" if cloud_facts else "everything i do runs on one mac, nothing leaves it"
    lines = [
        f"hey, i'm {bot_name}. {where}.",
        f"- say \"{bot_name}\" in a message and i'll reply",
    ]
    if cloud_facts:
        lines.append("- /ask <question> - i'll look it up online (that one leaves the mac)")
    if image_enabled:
        lines.append("- /image <prompt> - i'll make a picture (takes a bit)")
    if reminders:
        lines.append("- /remind in 20m <what>, /remind at 7pm <what>, /remind tomorrow 9am <what> - i'll post it here then")
        lines.append("- /reminders - what's pending, /remind cancel <n> - drop one")
    lines += [
        "- /forget - wipe what i remember from this chat",
        "- /help - this",
        "i only remember the last few messages here, and forget them after a while anyway.",
    ]
    return "\n".join(lines)
