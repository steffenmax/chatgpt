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
        system = MEMORY_SYSTEM.replace("{bot_name}", bot_name).replace("{max_chars}", str(self.max_chars))
        raw = await llm.reply(system, memory_messages(current, history))
        notes = (raw or "").strip()
        if notes.startswith("```"):
            notes = notes.strip("`").strip()
            if notes.lower().startswith("markdown"):
                notes = notes[8:].strip()
        if len(notes) > self.max_chars:
            notes = notes[: self.max_chars].rstrip()
        if not notes or "## People" not in notes or notes == current.strip():
            return False
        self.save(chat_guid, notes)
        log_event("memory.updated", chat_guid=chat_guid, chars=len(notes))
        return True
