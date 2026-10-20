"""Text generation backends.

Default: Ollama on localhost (POST /api/chat, stream=false).
Optional: Anthropic API, only when CLOUD_LLM_ENABLED=true. Text only, never images.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Any, Protocol

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


# A fresh handful of examples each time, so the model doesn't anchor on one ('lmaooo').
REACTION_EXAMPLES = (
    "lmaooo", "💀", "wait what", "W", "massive L", "nahhh", "be so fr", "who asked", "respectfully no",
    "this is huge", "say less", "cooked", "he's done", "screenshotted", "not again", "😭😭", "insane behavior",
    "we been knew", "receipts?", "that's crazy bro", "fraud alert", "the audacity", "absolute cinema",
    "seek help", "ok but why", "elite", "certified moment", "no shot", "tell me you didn't", "unreal",
)
REACTION_ADDENDUM = (
    "\n\nRight now you are NOT replying. You are only deciding whether the LAST message deserves a "
    "quick chime-in, the way a friend fires off a reaction or a one-liner, e.g. {examples}. The default "
    "is NONE: plans, logistics, questions, money, 'ok', 'see you', and anything ordinary get NONE. Chime "
    "in when the message is funny, wild, a win or an L, a confession, a hot take, or begging to be "
    "roasted. Mix it up: a reaction, a short roast, or a quick take in your own voice, specific to what "
    "they said, and never the same thing you said last time{last}. Answer with exactly the word NONE, or "
    "the chime-in itself: at most 12 words, nothing else, no quotes, no explanation."
)


def reaction_addendum(last: str | None = None, rng=random) -> str:
    examples = ", ".join(f"'{e}'" for e in rng.sample(REACTION_EXAMPLES, 6))
    return REACTION_ADDENDUM.replace("{examples}", examples).replace("{last}", f" ('{last}')" if last else "")


MEMORY_SYSTEM = (
    "You keep private notes for {bot_name} about one group chat so {bot_name} can talk to these people "
    "like it knows them. You get the current notes, the list of people who actually send messages here, "
    "and the most recent messages. Output the UPDATED notes as plain markdown and nothing else, under "
    "{max_chars} characters, with exactly these sections:\n"
    "## People: one bullet per real member, and ONLY the members listed. How they text, what they care "
    "about, their sense of humor, what they're known for in the group. Never add anyone else here.\n"
    "## Running bits: recurring jokes, nicknames, invented characters or stories (including ones "
    "{bot_name} started), labeled as bits, with who's in on them.\n"
    "## Group: dynamics, recurring topics, who ribs whom, plans that come up.\n"
    "Rules: 1) Only write what the messages show. Lines marked '(you)' are {bot_name}'s own words: they "
    "are never evidence about a person, only possible bits. 2) Something earns a bullet when it shows "
    "up more than once or someone states it about themselves; one-off noise stays out. 3) Record traits "
    "and jokes, not insults people threw at each other, and never slurs. 4) Keep what is still true, "
    "merge new tells, drop stale or trivial stuff. 5) Never include phone numbers, emails, addresses, or "
    "long quotes. If nothing new was learned, output the current notes unchanged."
)


def memory_messages(current_notes: str, history: list[ContextMessage], bot_name: str = "", members: list[str] | None = None) -> list[dict[str, str]]:
    lines = [f"{bot_name or m.sender} (you): {m.text}" if m.from_bot else f"{m.sender}: {m.text}" for m in history]
    who = ", ".join(members) if members else "(nobody has spoken yet)"
    content = (
        "Current notes:\n" + (current_notes.strip() or "(none yet)") +
        "\n\nPeople who actually send messages in this chat (the only names allowed under ## People): " + who +
        "\n\nRecent messages:\n" + "\n".join(lines) +
        "\n\nOutput the updated notes now."
    )
    return [{"role": "user", "content": content}]


def parse_reaction(text: str) -> str | None:
    t = _THINK_RE.sub("", text or "").strip().strip('"').strip("'").strip()
    if not t or t.upper().startswith("NONE") or "\n" in t or len(t) > 90 or len(t.split()) > 12:
        return None
    if re.match(r"^\s*\w+\s*:", t) or is_refusal(t):
        return None
    return t


# --- tool-aware intent -----------------------------------------------------------
INTENT_SYSTEM = (
    "You are the dispatcher for {bot_name}, a friend in a group chat who has a few tools. Read the "
    "recent chat; the LAST message is addressed to {bot_name}. Decide what {bot_name} should do and "
    "answer ONLY with JSON matching the schema.\n"
    "actions:\n"
    "- \"image\": ONLY when the LAST message directly asks {bot_name} to make/draw/generate/send a "
    "picture. Talking about pictures, reacting to one, joking that {bot_name} will make another, or "
    "asking what a picture looked like is NOT a request: that is chat. Put the requester's own "
    "description in \"prompt\", cleaned up (drop the bot's name and the asking words), without "
    "inventing extra scenes, styles or characters they did not mention.\n"
    "- \"remind\": they want to be reminded of something later. Put the time phrase exactly as said "
    "in \"when\" (e.g. \"in 20m\", \"at 7pm\", \"tomorrow 9am\", \"sat 3pm\") and the thing in \"what\".\n"
    "- \"lookup\": answering needs current or outside information (scores, schedules, news, prices, "
    "weather, hours, who won, precise stats/dates).\n"
    "- \"forget\": they explicitly ask {bot_name} to forget/wipe what it remembers.\n"
    "- \"chat\": anything else: banter, opinions, roasts, advice, general knowledge, questions "
    "{bot_name} can answer from memory. When unsure, choose chat."
)

INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["chat", "image", "remind", "lookup", "forget"]},
        "prompt": {"type": "string"},
        "when": {"type": "string"},
        "what": {"type": "string"},
    },
    "required": ["action"],
}


@dataclass(frozen=True)
class Intent:
    action: str = "chat"
    prompt: str = ""
    when: str = ""
    what: str = ""


_IMAGE_VERB = r"\b(make|draw|generate|create|render|paint|sketch|send|show|give|cook up|whip up|do|gimme|get)\b"
_IMAGE_NOUN = r"\b(pic|pics|picture|pictures|photo|photos|image|images|drawing|drawings|painting|art|portrait|render|meme|wallpaper|logo|poster)\b"


def looks_like_image_request(text: str) -> bool:
    """Deterministic gate on top of the dispatcher: an explicit ask for a picture.

    Requires a request verb and a picture noun in the same message, and refuses plain
    commentary ("that pic was bad", "it's gonna make another one").
    """
    t = (text or "").lower()
    drawing_verb = re.search(r"\b(draw|paint|sketch|illustrate)\b", t) is not None  # verbs that imply a picture
    if not drawing_verb and not (re.search(_IMAGE_NOUN, t) and re.search(_IMAGE_VERB, t)):
        return False
    if re.search(r"\b(gonna|going to|will|about to|another one|again\??$|that (pic|picture|photo|image)|the (pic|picture|photo|image) (was|is|looks))\b", t) and not re.search(r"\b(can|could|please|pls|plz|\?)", t):
        return False
    return True


# A plain message right after the bot spoke: was it said to the bot?
ADDRESSEE_SYSTEM = (
    "You watch a group chat for {bot_name}. Lines marked (you) are {bot_name}'s. The LAST message came right "
    "after {bot_name} spoke. Decide who the LAST message is for.\n"
    "\"bot\": it reacts to what {bot_name} just said: agrees, disagrees, laughs at it (lol, lmao, 💀), is hurt "
    "by it, argues with it, answers it, asks {bot_name} a follow-up, or roasts {bot_name}.\n"
    "\"other\": it names or answers someone else, changes the subject, or carries on a conversation between "
    "other people that has nothing to do with {bot_name}'s line.\n"
    "Examples:\n"
    "{bot_name} (you): giants by 10 / Tom: no way the jets got this -> bot\n"
    "{bot_name} (you): 8:15 on espn / Chris: jack you driving or me -> other\n"
    "{bot_name} (you): dan dresses like a sub teacher / Dan: ok that hurt -> bot\n"
    "{bot_name} (you): knicks in 6 / Joe: anyone tryna get food later -> other\n"
    "Answer ONLY with JSON matching the schema."
)
ADDRESSEE_SCHEMA = {"type": "object", "properties": {"to": {"type": "string", "enum": ["bot", "other"]}}, "required": ["to"]}


def addressee_messages(history: list[ContextMessage], bot_name: str) -> list[dict[str, str]]:
    lines = [f"{bot_name} (you): {m.text}" if m.from_bot else f"{m.sender}: {m.text}" for m in history[-8:]]
    return [{"role": "user", "content": "\n".join(lines) + "\n\nWho is the LAST message for? JSON only."}]


def parse_addressee(raw: str) -> bool:
    text = _THINK_RE.sub("", raw or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return False
    try:
        return json.loads(text[start : end + 1]).get("to") == "bot"
    except (ValueError, AttributeError):
        return False


def intent_messages(history: list[ContextMessage]) -> list[dict[str, str]]:
    lines = [f"{m.sender}: {m.text}" for m in history[-8:]]
    return [{"role": "user", "content": "\n".join(lines) + "\n\nJSON only."}]


def parse_intent(raw: str) -> Intent:
    text = _THINK_RE.sub("", raw or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return Intent()
    try:
        data = json.loads(text[start : end + 1])
    except ValueError:
        return Intent()
    if not isinstance(data, dict):
        return Intent()
    action = str(data.get("action") or "chat").lower().strip()
    if action not in ("chat", "image", "remind", "lookup", "forget"):
        action = "chat"
    g = lambda k: str(data.get(k) or "").strip()[:600]
    intent = Intent(action, g("prompt"), g("when"), g("what"))
    if intent.action == "image" and not intent.prompt:
        return Intent()
    if intent.action == "remind" and not (intent.when and intent.what):
        return Intent()
    return intent


VOICE_SYSTEM_SUFFIX = (
    "\n\nRight now you are just letting the chat know something, as ONE short text in your own "
    "voice, speaking as yourself. Keep every time, number and name exactly; add nothing new; no "
    "markdown, no quotes, no name prefix. The plain version is: "
)


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


SELF_ONLY_RULE = (
    "\n\nWrite only your own next message. Never write lines for anyone else in the chat, never "
    "start with a name and a colon, and never post links you haven't seen in the chat."
)

# Added to every chat reply: replies used to contradict earlier ones and grow to paragraphs.
CONSISTENCY_RULE = (
    "\n\nStay consistent with what you already said in this chat (your earlier lines): don't flip an "
    "answer, a take or a fact you gave; if you really changed your mind, say so. Don't repeat yourself. "
    "Keep it short like a text: one to three lines unless someone asked for detail."
)


def target_nudge(pings: list[ContextMessage]) -> str:
    """Name the message(s) the bot is answering. In a busy chat the newest line is often someone
    else's, and a small model answers whatever is newest."""
    from .linkreader import strip_cards

    def said(m: ContextMessage) -> str:
        t = " ".join(strip_cards(m.text).split())
        return t if len(t) <= 200 else t[:200].rsplit(" ", 1)[0] + "…"

    pings = [m for m in pings if not m.from_bot][-4:]
    if not pings:
        return ""
    rest = " The other messages are context: leave other conversations alone unless they're about this."
    if len(pings) == 1:
        return f"\n\nYou're answering {pings[0].sender}, who said to you: \"{said(pings[0])}\". Answer that.{rest}"
    lines = "\n".join(f"- {m.sender}: \"{said(m)}\"" for m in pings)
    return f"\n\nSeveral people pinged you at once. Answer them in ONE message, a quick line for each:\n{lines}\n{rest.strip()}"


def trim_reply(text: str, limit: int) -> str:
    """Cut an overlong reply at the last sentence end before the limit."""
    if limit <= 0 or len(text) <= limit:
        return text
    cut = text[:limit]
    ends = [cut.rfind(c) for c in (". ", "! ", "? ", "\n")]
    end = max(ends)
    if end >= limit // 2:
        return cut[: end + 1].strip()
    return cut.rsplit(" ", 1)[0].rstrip(",;:") + "…"


_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)


def speaker_names(history: list[ContextMessage], bot_name: str) -> list[str]:
    """First names of everyone in the history, plus the bot, lowercased."""
    names = {bot_name.strip().split()[0].lower()} if bot_name.strip() else set()
    for m in history:
        if not m.from_bot and m.sender and m.sender[0].isalpha():
            names.add(m.sender.strip().split()[0].lower())
    return sorted(names)


def stop_sequences(names: list[str]) -> list[str]:
    """Cut generation as soon as the model starts writing another speaker's line."""
    out = []
    for n in names:
        for v in (n, n.capitalize()):
            out.append(f"\n{v}:")
    return out[:20]


def fix_speaker(reply: str, names: list[str], bot_name: str) -> tuple[str, bool]:
    """Remove transcript-style lines written as other people.

    Returns (text, impersonated). `impersonated` is True when the reply itself opens
    as someone else ("max: ..."): the caller should retry rather than post it.
    """
    bot = bot_name.strip().split()[0].lower() if bot_name.strip() else ""
    others = [n for n in names if n != bot]
    if not others:
        return reply, False
    pat = re.compile(r"^\s*(" + "|".join(map(re.escape, others)) + r")\s*:", re.IGNORECASE)
    lines = reply.split("\n")
    if pat.match(lines[0]):
        return reply, True
    kept = []
    for line in lines:
        if pat.match(line):
            break  # everything after the first line written as someone else is a continuation
        kept.append(line)
    return "\n".join(kept).strip(), False


# Assistant-style refusals ("I can't create explicit content... anything else I can help
# with?") break character. They get swapped for a one- or two-word roast instead.
_REFUSAL_PATTERNS = re.compile(
    r"\bi\s*(?:can'?t|cannot|can not|won'?t|will not|am unable to|'?m unable to|'?m not able to|am not able to)\s+"
    r"(?:help|assist|create|generate|provide|write|engage|fulfill|comply|produce|make|do that|participate|discuss|"
    r"continue|describe|share)"
    r"|\b(?:sexually )?explicit (?:content|material|images?|stuff)\b|\bgraphic content\b|\bnsfw content\b"
    r"|\bas an ai\b|\bai (?:language )?model\b|\bi'?m (?:just )?(?:an ai|a chatbot|a bot|a language model)\b"
    r"|\b(?:against|violates?) (?:my|the) (?:guidelines|programming|policies|policy|rules)\b|\bcontent polic(?:y|ies)\b"
    r"|\bi must decline\b|\bnot comfortable\b|\bi'?d (?:rather|prefer) not\b"
    r"|\banything else i can (?:help|do)\b|\bcan i help you with something else\b",
    re.IGNORECASE,
)
REFUSAL_ROASTS = (
    "weirdo", "down bad", "seek help", "touch grass", "bro what", "freak", "nice try", "go outside",
    "lol no", "creep", "absolutely not", "pervert", "log off", "sick puppy",
)


def is_refusal(text: str) -> bool:
    return bool(_REFUSAL_PATTERNS.search((text or "").replace("\u2019", "'")))


def strip_unseen_urls(reply: str, seen_text: str) -> str:
    """Drop links the model invented (a local model can't browse)."""
    def repl(m):
        url = m.group(0).rstrip(".,!?)\"'")
        return m.group(0) if url in seen_text else ""
    out = _URL.sub(repl, reply)
    return re.sub(r"[ \t]{2,}", " ", out).strip()


AFTERWORD_SUFFIX = (
    "\n\nYour research guy just posted the results below in the chat, for {who}, who asked: \"{request}\". "
    "Now send ONE short text of your own reacting to it, in your voice: your pick, an opinion, what "
    "you'd do, or a jab at whoever asked. Don't repeat the details, numbers, times or links; they're "
    "right above. One or two sentences, no name prefix, no quotes, no markdown."
)

_LINKISH = re.compile(r"https?://|www\.|\.(com|net|org|io)\b", re.IGNORECASE)


def afterword_ok(results: str, text: str) -> bool:
    """A closing comment may react to the results but must not restate (and garble) them."""
    t = (text or "").strip()
    if not t or len(t) > 260 or t.count("\n") > 1 or _LINKISH.search(t):
        return False
    have = set(_NUM.findall(results)) if results else set()
    return all(n in have for n in _NUM.findall(t))


_NUM = re.compile(r"\d+(?:[:.,]\d+)*")


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

    async def reply(self, system: str, messages: list[dict[str, str]], stop: list[str] | None = None) -> str:
        return await self._chat(system, messages, stop=stop)

    async def structured(self, system: str, messages: list[dict[str, str]], schema: dict[str, Any]) -> str:
        """Ask for JSON matching `schema` (Ollama constrained decoding), deterministic."""
        return await self._chat(system, messages, fmt=schema, temperature=0.0, max_tokens=200)

    async def _chat(self, system, messages, fmt=None, temperature=None, max_tokens=None, stop=None) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
                "num_predict": self.max_tokens if max_tokens is None else max_tokens,
                "num_ctx": self.num_ctx,
            },
        }
        if fmt is not None:
            payload["format"] = fmt
        if stop:
            payload["options"]["stop"] = stop
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
