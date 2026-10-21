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


# The cloud model only fetches facts (no persona, so no "in character" second-guessing); the local
# model then answers in the bot's own voice from them.
FACT_FINDER_SYSTEM = (
    "You are the research tool behind a group-chat bot called {bot_name}. Below are the latest messages from "
    "the chat; someone asked {bot_name} something that needs current or precise information.{target} Use web "
    "search (one or two searches are enough) and reply with ONLY the facts that answer it: short plain lines "
    "with the specific names, numbers, scores, dates, times and prices. No opinions, no jokes, no greeting, "
    "nothing about the people in the chat, no links, no markdown. If something can't be confirmed, say that "
    "in one plain line. Under 600 characters."
)
LOOKED_UP = (
    "\n\nYou just looked this up, and it's accurate (your own knowledge is old, so trust this over it):\n{facts}\n"
    "Now answer in your own voice and style, jokes and all. If they asked for an opinion or a call (start or sit, "
    "who wins, is it worth it), make the call yourself from these facts, commit to it and give the reason in a few "
    "words. Keep every name, number, score, date and time exactly as above and don't add any facts or numbers "
    "of your own."
)
FACTS_STRICT = "\n\nYour last try changed or invented a number. Use ONLY the numbers in what you looked up."


def facts_kept(reply: str, facts: str, chat_text: str = "") -> bool:
    """Every number in the reply comes from the looked-up facts (or was already in the chat)."""
    have = set(_NUM.findall(facts or "")) | set(_NUM.findall(chat_text or ""))
    return all(n in have for n in _NUM.findall(reply or ""))


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


def reaction_addendum(last: str | None = None, rng=random, gifs: bool = False) -> str:
    examples = ", ".join(f"'{e}'" for e in rng.sample(REACTION_EXAMPLES, 6))
    out = REACTION_ADDENDUM.replace("{examples}", examples).replace("{last}", f" ('{last}')" if last else "")
    if gifs:
        out += (" Instead of words you may send a reaction GIF: answer 'GIF: ' and a 2-4 word search "
                "(e.g. 'GIF: michael scott no'). Use that only now and then.")
    return out


UNANSWERED_GIF_SYSTEM = (
    "{sender} said this in a group chat and nobody has answered for half an hour: \"{text}\"\n"
    "Pick a reaction GIF to drop on it: something like crickets, hello is anyone there, awkward silence, or a "
    "joke about what they said. Answer with ONLY a 2-4 word GIF search, nothing else."
)


BUDDY_MOOD_SYSTEM = (
    "You read one message {name} just sent in a group chat with his friends and a bot called {bot_name}. Pick ONE "
    "label.\n\"down\": he sounds genuinely sad, stressed, defeated, lonely, or like he's having a rough day (not a "
    "joke or sarcasm).\n\"jab\": he's trash-talking, insulting or mocking {bot_name}, bots, or AI (e.g. '{bot_name} is "
    "trash', 'AI is cringe', 'this bot sucks').\n\"fed_up\": he says he's tired of {bot_name} or the bot stuff, or wants "
    "it to stop.\n\"none\": anything else: normal chat, jokes, sports, plans, roasting other people.\nWhen unsure, answer "
    "none. Answer ONLY with JSON matching the schema."
)
BUDDY_MOOD_SCHEMA = {"type": "object", "properties": {"label": {"type": "string", "enum": ["down", "jab", "fed_up", "none"]}},
                     "required": ["label"]}


def parse_mood(raw: str) -> str:
    m = re.search(r'"label"\s*:\s*"(down|jab|fed_up|none)"', raw or "")
    return m.group(1) if m else "none"


def mentions_bots(text: str, bot_name: str) -> bool:
    """A jab only counts if it's actually about the bot or AI."""
    return bool(re.search(rf"\b({re.escape(bot_name)}|bots?|robots?|ai|a\.i\.|chatgpt|gpt|claude|llms?|this thing|"
                          r"artificial intelligence)\b", text or "", re.IGNORECASE))


# Backup for the router: questions that need this season's facts go to a lookup even if it says "chat".
_NEEDS_LOOKUP = re.compile(
    r"\b(start|sit|bench|fantasy|waivers?|waiver wire|flex|ppr|injur(?:y|ed|ies)|questionable|doubtful|out for|"
    r"trade(?:d)?|rookie|this (?:week|season|year)|tonight'?s|last night|who won|score|standings|playoffs?|odds|"
    r"stats?|depth chart|cut|signed|released)\b",
    re.IGNORECASE,
)


def needs_lookup(text: str) -> bool:
    """A question about this season's players, games or fantasy: the 2023-era model can't answer it."""
    return bool(_NEEDS_LOOKUP.search(text or "")) and asks_something(text)


def asks_something(text: str) -> bool:
    """A direct question (the kind of answer that can contradict itself)."""
    t = (text or "").strip()
    return "?" in t or bool(re.match(r"(?i)^\W*(?:\w+[,:]?\s+)?(should|who|what|which|when|where|why|how|is|are|do|does|did|"
                                      r"will|would|can|could)\b", t))


CONTRADICTION_SYSTEM = (
    "Check one reply from a group-chat bot. Does it contradict itself: says both yes and no to the same thing, "
    "or recommends doing something and also not doing it, or calls something good and bad in the same breath? "
    "Jokes, sarcasm and roasting are fine; only an answer that cancels itself out counts. Answer ONLY with JSON "
    "matching the schema."
)
CONTRADICTION_SCHEMA = {"type": "object", "properties": {"contradicts": {"type": "boolean"}}, "required": ["contradicts"]}
NO_CONTRADICTION = ("\n\nYour last answer contradicted itself. Give ONE clear answer, commit to it, and keep it "
                    "consistent from start to finish.")


def parse_contradiction(raw: str) -> bool:
    return bool(re.search(r'"contradicts"\s*:\s*true', raw or ""))


REMINDER_SYSTEM = (
    "{persona}\n\nIt's time to give a reminder you were asked to give in this group chat. {setup} The thing to "
    "remind them about: \"{what}\". Write the text you send now: address {target} at the start, say what "
    "they need to do in your own words and style (one or two short lines), and keep every name, number, amount, "
    "time and place from the reminder exactly. Words like 'me' and 'my' in it mean {setter}. Don't add anything "
    "that isn't in the reminder, no links, no emoji spam."
)


ENHANCE_ASK = re.compile(r"\benhance\b", re.IGNORECASE)
# "enhance tom's message", "enhance what tom said", "enhance tom"
ENHANCE_WHO = re.compile(
    r"\benhance\s+(?:what\s+)?(?!this\b|that\b|it\b|the\b|his\b|her\b|my\b|me\b|last\b|message\b|pls\b|please\b)"
    r"([a-z][a-z'’]+(?:\s+[a-z][a-z'’]+)?)", re.IGNORECASE)

ENHANCE_SYSTEM = (
    "You are the analyst behind a group-chat bot called {bot_name}. A friend asked {bot_name} to 'enhance' one "
    "message, CSI style: really dig into it. Below are the surrounding messages and the target message. Give a "
    "sharp, specific read: what they literally said; what they actually mean (subtext, tone, sarcasm, what's "
    "between the lines); any references, slang or callbacks explained (search the web if a reference needs it); "
    "how it fits the conversation around it; what it says about them or their situation; and what they're "
    "probably after. Be bold and specific, no hedging, no moralizing, no therapy-speak. Plain text, 4 to 6 "
    "dash bullets of ONE sentence each, under 700 characters total, no links."
)
ENHANCE_VOICE = (
    "\n\nSomeone asked you to 'enhance' {name}'s message, CSI style: \"{target}\". Your analyst's breakdown, which "
    "is accurate:\n{analysis}\nNow deliver it in your own voice like you're zooming in on evidence: open with a "
    "one-line 'enhance' bit, then the real insights as short lines (4 to 7), roasting allowed. Keep every insight's "
    "substance and every name and number exactly; don't invent anything that isn't in the breakdown."
)


def wants_enhance(text: str) -> bool:
    return bool(ENHANCE_ASK.search(text or ""))


def enhance_who(text: str) -> str | None:
    m = ENHANCE_WHO.search(text or "")
    if not m:
        return None
    who = re.sub(r"\s+(message|msg|text|said|says|take)$", "", m.group(1).strip(), flags=re.IGNORECASE).strip()
    who = re.sub(r"['’]s$", "", who).strip()
    return who or None


def looks_like_gif_request(text: str) -> bool:
    return bool(re.search(r"\bgifs?\b", text or "", re.IGNORECASE))


def looks_like_tweet_request(text: str) -> bool:
    return bool(re.search(r"\b(tweets?|twitter|x posts?|post from x|posts? on x)\b", text or "", re.IGNORECASE))


MEMORY_SYSTEM = (
    "You keep private notes for {bot_name} about one group chat so {bot_name} can talk to these people "
    "like it knows them. You get the current notes, the list of people in the chat, and the most recent "
    "messages the PEOPLE sent (whether or not they were talking to {bot_name}; {bot_name}'s own messages are "
    "left out on purpose). Output the UPDATED notes as plain markdown and nothing else, under {max_chars} "
    "characters, with exactly these sections:\n"
    "## People: one bullet per real member, and ONLY the members listed. What they actually say and do: how "
    "they text, what they care about, their teams, jobs, plans, what they're known for.\n"
    "## Running bits: jokes, nicknames and stories that the PEOPLE themselves keep bringing up.\n"
    "## Group: dynamics, recurring topics, who ribs whom, plans that come up.\n"
    "Rules: 1) Everything must come from what the people wrote. {bot_name}'s jokes, takes and made-up stories "
    "never go in, even if they're in the current notes; drop any bullet that sounds like {bot_name} invented "
    "it (a bit only the bot keeps bringing up, an invented character, a nickname nobody in the chat uses). "
    "2) A bullet earns its place when someone states it about themselves, it comes up from more than one "
    "person, or people keep repeating it; one-off noise stays out. 3) Prefer concrete facts (plans, teams, "
    "bets, events, preferences) over vibes. 4) Record traits and jokes, not insults people threw at each other, "
    "and never slurs. 5) Keep what is still true and still backed by the people, merge new things, drop stale "
    "or trivial stuff; shorter and accurate beats long. 6) Never include phone numbers, emails, addresses, or "
    "long quotes. If nothing new was learned, output the current notes unchanged."
)


def memory_messages(current_notes: str, history: list[ContextMessage], bot_name: str = "", members: list[str] | None = None) -> list[dict[str, str]]:
    # Only what people said: the bot's own lines are what turned its jokes into "facts" in a loop.
    lines = [f"{m.sender}: {m.text}" for m in history if not m.from_bot]
    who = ", ".join(members) if members else "(nobody has spoken yet)"
    content = (
        "Current notes:\n" + (current_notes.strip() or "(none yet)") +
        "\n\nPeople in this chat (the only names allowed under ## People): " + who +
        "\n\nRecent messages from the people:\n" + ("\n".join(lines) or "(none)") +
        "\n\nOutput the updated notes now."
    )
    return [{"role": "user", "content": content}]


def parse_reaction(text: str) -> str | None:
    t = strip_fake_notes(_THINK_RE.sub("", text or "")).strip().strip('"').strip("'").strip()
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
    "  Put who should get the reminder in \"who\": \"me\" if it's for the person asking, the person's name if "
    "it's for someone else (\"remind tom to...\" -> \"tom\"), or \"everyone\" for the whole chat. Keep \"what\" as "
    "they said it.\n"
    "- \"lookup\": answering needs current or outside information. {bot_name}'s own knowledge stops in 2023, so "
    "this includes: scores, schedules, news, prices, weather, hours, who won, precise stats/dates; anything about "
    "current players, rosters, injuries, trades, rookies or this season; fantasy sports (start/sit, waivers, "
    "trades, rankings); and any person, team, product or event that may be from after 2023. If {bot_name} might "
    "not know who or what they mean, choose lookup.\n"
    "- \"forget\": they explicitly ask {bot_name} to forget/wipe what it remembers.\n"
    "- \"gif\": they ask {bot_name} for a GIF. Put what the GIF should show in \"prompt\" (2-5 words).\n"
    "- \"tweet\": they ask {bot_name} to find or send a tweet/post from Twitter/X. Put the topic in "
    "\"prompt\" (2-5 words).\n"
    "- \"chat\": anything else: banter, opinions, roasts, advice, general knowledge, questions "
    "{bot_name} can answer from memory. When unsure, choose chat."
)

INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["chat", "image", "remind", "lookup", "forget", "gif", "tweet"]},
        "prompt": {"type": "string"},
        "when": {"type": "string"},
        "what": {"type": "string"},
        "who": {"type": "string"},
    },
    "required": ["action"],
}


@dataclass(frozen=True)
class Intent:
    action: str = "chat"
    prompt: str = ""
    when: str = ""
    what: str = ""
    who: str = ""  # remind: "me", a name from the chat, or "everyone"


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
    if action not in ("chat", "image", "remind", "lookup", "forget", "gif", "tweet"):
        action = "chat"
    g = lambda k: str(data.get(k) or "").strip()[:600]
    intent = Intent(action, g("prompt"), g("when"), g("what"), g("who")[:40])
    if intent.action in ("image", "gif", "tweet") and not intent.prompt:
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
    "start with a name and a colon, and never post links you haven't seen in the chat. Notes in square "
    "brackets like [link: ...], [photo: ...], [sent a picture: ...] or [sent a gif: ...] are added by the "
    "system to describe what happened; never write one yourself. Writing one doesn't send anything."
)

# The bracketed notes the system adds to context; the model sometimes writes its own fake ones
# ("[sent a picture: ...]", a made-up "[link: ...]"). They're cut from anything it says.
_FAKE_NOTE = re.compile(
    r"\[\s*(?:link|photo|picture|image|video|gif|tweet|thumbnail|sent (?:a |an )?(?:picture|pic|photo|image|gif|tweet|link|video))"
    r"\s*:[^\]]*(?:\]|$)",
    re.IGNORECASE,
)


def strip_fake_notes(text: str) -> str:
    out = _FAKE_NOTE.sub(" ", text or "")
    return re.sub(r"[ \t]{2,}", " ", out).strip()

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



# Insults and roasts aimed at the bot. It usually fires back; TAKE_IT_CHANCE of the time it just eats it.
_JAB = re.compile(
    r"\b(shut (up|it|the fuck up)|stfu|stf?u|nobody asked|who asked|fuck (you|off)|f off|eat (shit|a dick)|"
    r"(you|u|ur|you're|youre|he's|hes|ricky'?s?) (are |is |so |such |a |an |being |lowkey |really |actually )*"
    r"(trash|garbage|dumb|stupid|an? idiot|idiot|useless|mid|cringe|corny|lame|annoying|a clown|clown|a loser|loser|"
    r"a bum|bum|a bozo|bozo|washed|cooked|a joke|worthless|dogshit|ass|so bad|terrible|the worst)|"
    r"(you|u) suck|go away|log off|kys|ratio|L ricky|ricky L|take the L|nobody likes you|stick to)\b",
    re.IGNORECASE,
)

TAKE_IT = (
    "\n\nIf they're roasting or insulting you, this time just take it: no comeback, no 'shut up' back. Admit "
    "they got you, laugh at yourself, or agree in a few words (like a guy who knows he walked into that one)."
)


def jabs_at_bot(text: str) -> bool:
    return bool(_JAB.search(text or ""))


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
    return strip_fake_notes(text).strip().strip('"').strip()


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
