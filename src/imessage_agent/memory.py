"""Per-chat notes about the people and the group, kept by the local model on disk.

This is the one place message-derived content is written to disk, and only when
MEMORY_DIR is set. Files are 0600 in a 0700 directory, one per chat, plain markdown
the owner can read and edit. The model rewrites the whole file every
MEMORY_UPDATE_EVERY stored messages; contents never reach the logs.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

from .context import ContextMessage
from .llm import MEMORY_SYSTEM, TextBackend, memory_messages
from .logsafe import exc_name, log_event

SECTION_ORDER = ["People", "Running bits", "Group"]


def _first_token(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.strip().split()[0].lower()) if name.strip() else ""


def _split_sections(notes: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current = None
    for raw in notes.splitlines():
        line = raw.rstrip()
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if m:
            current = m.group(1).strip()
            sections.setdefault(current, [])
            continue
        if current is not None and line.strip():
            sections[current].append(line)
    return sections


def _bullet_name(line: str) -> str:
    m = re.match(r"^\s*[-*]\s*\**([^:*]+?)\**\s*:", line)
    return m.group(1).strip() if m else ""


def ground_notes(notes: str, members: list[str]) -> str:
    """Enforce the one rule a prompt can't: ## People holds only real senders.

    Bullets about anyone else are moved to ## Running bits, tagged, so a character
    the bot or the chat invented stays a bit instead of becoming a member.
    """
    sections = _split_sections(notes)
    if "People" not in sections:
        return ""
    known = {_first_token(m) for m in members if m.strip()}
    keep, moved = [], []
    for line in sections["People"]:
        name = _bullet_name(line)
        if not name:
            continue  # stray prose under People
        if _first_token(name) in known:
            keep.append(line)
        else:
            body = line.split(":", 1)[1].strip() if ":" in line else line.strip("-* ")
            moved.append(f"- {name} (not a member, a bit): {body}")
    sections["People"] = keep
    if moved:
        sections.setdefault("Running bits", [])
        existing = {_bullet_name(l).lower() for l in sections["Running bits"]}
        sections["Running bits"] += [l for l in moved if _bullet_name(l).lower() not in existing]
    out = []
    for title in SECTION_ORDER + [t for t in sections if t not in SECTION_ORDER]:
        if title in sections or title in SECTION_ORDER:
            out.append(f"## {title}")
            out += sections.get(title, [])
            out.append("")
    return "\n".join(out).strip() + "\n"


class MemoryStore:
    def __init__(self, directory: str, max_chars: int = 3000, update_every: int = 15):
        self.dir = Path(directory).expanduser()
        self.max_chars = max_chars
        self.update_every = update_every
        self._counts: dict[str, int] = {}

    def path_for(self, chat_guid: str) -> Path:
        return self.dir / (re.sub(r"[^A-Za-z0-9]+", "-", chat_guid).strip("-") + ".md")

    def load(self, chat_guid: str) -> str:
        try:
            return self.path_for(chat_guid).read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""
        except OSError as exc:
            log_event("memory.read_failed", chat_guid=chat_guid, error=exc_name(exc))
            return ""

    def save(self, chat_guid: str, notes: str) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.dir, 0o700)
        target = self.path_for(chat_guid)
        fd, tmp = tempfile.mkstemp(prefix=".notes-", suffix=".md", dir=self.dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(notes.rstrip() + "\n")
            os.chmod(tmp, 0o600)
            os.replace(tmp, target)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def tick(self, chat_guid: str) -> bool:
        """Count a stored human message; True when it is time to refresh the notes."""
        n = self._counts.get(chat_guid, 0) + 1
        self._counts[chat_guid] = n
        return n % self.update_every == 0

    async def update(self, chat_guid: str, bot_name: str, history: list[ContextMessage], llm: TextBackend) -> bool:
        current = self.load(chat_guid)
        members: list[str] = []
        for m in history:
            if not m.from_bot and m.sender and m.sender not in members:
                members.append(m.sender)
        system = MEMORY_SYSTEM.replace("{bot_name}", bot_name).replace("{max_chars}", str(self.max_chars))
        raw = await llm.reply(system, memory_messages(current, history, bot_name, members))
        notes = (raw or "").strip()
        if notes.startswith("```"):
            notes = notes.strip("`").strip()
            if notes.lower().startswith("markdown"):
                notes = notes[8:].strip()
        notes = ground_notes(notes, members)
        if len(notes) > self.max_chars:
            notes = notes[: self.max_chars].rstrip()
        if not notes or notes.strip() == current.strip():
            return False
        self.save(chat_guid, notes)
        log_event("memory.updated", chat_guid=chat_guid, chars=len(notes))
        return True
