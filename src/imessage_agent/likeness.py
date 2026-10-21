"""Opted-in likenesses: a trained LoRA that lets image requests show a real person from the chat.

Each person has a folder in LIKENESS_DIR with their add-on and a likeness.json:
  {"name": "dziuba", "aliases": ["dziuba", "matt dziuba"], "lora": "dziuba.safetensors",
   "scale": 1.0, "owner": "+16095550100", "opted_in": true}
Rules, enforced here and not left to the model:
  * only people who opted in; the person (by phone number) can /optout or /optin any time;
  * a request that names them is blocked if it's sexual, nude, gory or about them dying;
  * the add-on files never leave the Mac.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .logsafe import exc_name, log_event

_BLOCKED = re.compile(
    r"\b(nude|nudes|naked|nsfw|sex|sexy|sexual|porn\w*|xxx|topless|bottomless|lingerie|underwear|thong|bikini|"
    r"speedo|strip(per|ping|tease)?|erotic|aroused|horny|orgasm|genital\w*|penis|dick|cock|boobs?|tits?|nipples?|"
    r"butt ?naked|onlyfans|explicit|kink\w*|bdsm|fetish|seductive|sensual|making out|in bed with|kissing|"
    r"dead|corpse|blood|bloody|gore|gory|murder\w*|killed|suicide|hanged|hanging|shot dead|decapitat\w*|"
    r"mugshot|arrested|in prison|behind bars)\b",
    re.IGNORECASE,
)


def blocked(prompt: str) -> bool:
    """Requests a real person's likeness must never be used for."""
    return bool(_BLOCKED.search(prompt or ""))


@dataclass
class Likeness:
    name: str
    aliases: list[str]
    lora: Path
    scale: float
    owner: str
    opted_in: bool
    meta_path: Path

    def mentioned_in(self, text: str) -> bool:
        t = " " + re.sub(r"[^a-z0-9 ]", " ", (text or "").lower()) + " "
        return any(f" {a.lower()} " in t for a in self.aliases)

    def prompt_for(self, prompt: str) -> str:
        """Lead with how the add-on was trained ('dziuba, a man') so it kicks in."""
        return f"{self.name}, a man. {prompt}"


class Likenesses:
    def __init__(self, directory: str, norm=lambda a: a or ""):
        self.dir = Path(directory).expanduser()
        self._norm = norm
        self.people: list[Likeness] = self._load()

    def _load(self) -> list[Likeness]:
        out = []
        for meta in sorted(self.dir.glob("*/likeness.json")):
            try:
                d = json.loads(meta.read_text(encoding="utf-8"))
                lora = meta.parent / d["lora"]
                if not lora.is_file():
                    continue
                out.append(Likeness(d["name"], [a for a in d.get("aliases", [d["name"]]) if a], lora,
                                    float(d.get("scale", 1.0)), str(d.get("owner", "")), bool(d.get("opted_in")), meta))
            except (OSError, ValueError, KeyError) as exc:
                log_event("likeness.load_failed", error=exc_name(exc))
        return out

    def match(self, prompt: str) -> Likeness | None:
        """The opted-in person the prompt names, if any."""
        return next((p for p in self.people if p.opted_in and p.mentioned_in(prompt)), None)

    def set_opt_in(self, address: str | None, opted_in: bool) -> Likeness | None:
        """Only the person themself (by number) can change it. Returns their likeness, or None."""
        who = self._norm(address)
        for p in self.people:
            if who and self._norm(p.owner) == who:
                p.opted_in = opted_in
                d = json.loads(p.meta_path.read_text(encoding="utf-8"))
                d["opted_in"] = opted_in
                fd, tmp = tempfile.mkstemp(prefix=".likeness-", dir=p.meta_path.parent)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(d, f, indent=1)
                os.chmod(tmp, 0o600)
                os.replace(tmp, p.meta_path)
                log_event("likeness.opt", kind="in" if opted_in else "out")
                return p
        return None
