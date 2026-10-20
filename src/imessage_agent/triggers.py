"""Pure functions: does a message address the bot, and what does it ask for?"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Kind = Literal["mention", "ask", "image", "help", "forget", "remind", "reminders", "task", "mute", "unmute", "gif", "tweet", "none"]


@dataclass(frozen=True)
class Trigger:
    kind: Kind
    arg: str = ""  # image prompt, or the message with the mention left in for chat


_COMMANDS = {
    "/image": "image", "/img": "image", "/ask": "ask", "/help": "help", "/forget": "forget",
    "/remind": "remind", "/reminders": "reminders", "/task": "task", "/mute": "mute", "/unmute": "unmute",
    "/gif": "gif", "/tweet": "tweet",
}

DEFAULT_MUTE_SECONDS = 2 * 60
MAX_MUTE_SECONDS = 24 * 3600
# "<name> chill", "<name> shut up for an hour": short messages only, so "ricky what's a chill bar" isn't one.
_MUTE_WORDS = re.compile(
    r"\b(chill|chill out|shut up|shut it|stfu|be quiet|quiet|hush|pipe down|zip it|stop talking|go away|"
    r"take a break|mute|enough|relax|calm down|knock it off)\b", re.IGNORECASE)
_UNMUTE_WORDS = re.compile(
    r"\b(unmute|you can talk|you can speak|come back|wake up|you'?re back|talk again|speak again)\b", re.IGNORECASE)


_FILLER = re.compile(r"^(?:(?:yo|bro|bruh|dude|man|ok|okay|alright|aight|please|pls|plz|just|can you|could you|"
                     r"can u|u|you|go|now|ayo)\b[\s,.!]*)+", re.IGNORECASE)


def parse_duration(text: str) -> int | None:
    """Seconds for '30m', '2 hours', 'an hour', 'half an hour', '1h30m'; None if there's no duration."""
    t = (text or "").lower()
    if re.search(r"\bhalf an? hour\b", t):
        return 1800
    if re.search(r"\b(an|a|one) hour\b", t):
        return 3600
    total, found = 0, False
    for n, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(d|days?|h|hrs?|hours?|m|mins?|minutes?)(?![a-z])", t):
        found = True
        total += int(float(n) * (86400 if unit.startswith("d") else 3600 if unit.startswith("h") else 60))
    if not found:
        return None
    return max(60, min(total, MAX_MUTE_SECONDS))


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
        if len(text.split()) <= 9 and "?" not in text:
            # The ask has to lead what's left once the name is out: "ricky chill", "shut up ricky",
            # "yo ricky you can talk" - not "ricky that was chill".
            raw = mention_pattern(bot_name).sub(" ", text).strip(" ,.!-")
            rests = (raw, _FILLER.sub("", raw).strip())
            if any(_UNMUTE_WORDS.match(r) for r in rests):
                return Trigger("unmute", text)
            if any(_MUTE_WORDS.match(r) for r in rests):
                return Trigger("mute", text)
        return Trigger("mention", text)
    return Trigger("none")


def help_text(bot_name: str, image_enabled: bool, cloud_facts: bool = False, reminders: bool = False, task: bool = False,
              gifs: bool = False, tweets: bool = False) -> str:
    where = "most of it runs on one mac" if cloud_facts else "everything i do runs on one mac, nothing leaves it"
    lines = [
        f"hey, i'm {bot_name}. {where}.",
        f"- say \"{bot_name}\" in a message and i'll reply",
    ]
    if cloud_facts:
        lines.append("- /ask <question> - i'll look it up online (that one leaves the mac)")
    if image_enabled:
        lines.append("- /image <prompt> - i'll make a picture (takes a bit)")
    if gifs:
        lines.append("- /gif <what> - i'll find a gif")
    if tweets:
        lines.append("- /tweet <topic> - i'll dig up the most-liked tweet about it (leaves the mac)")
    if task:
        lines.append("- /task <job> - i'll go research it properly and come back (owner only, takes a few min, leaves the mac)")
    if reminders:
        lines.append("- /remind in 20m <what>, /remind at 7pm <what>, /remind tomorrow 9am <what> - i'll post it here then")
        lines.append("- /reminders - what's pending, /remind cancel <n> - drop one")
    lines += [
        f"- \"{bot_name} chill\" or /mute 30m - i'll stay quiet in this chat (2 min unless you say how long), /unmute to bring me back",
        "- /forget - wipe what i remember from this chat",
        "- /help - this",
        "i only remember the last few messages here, and forget them after a while anyway.",
    ]
    return "\n".join(lines)
