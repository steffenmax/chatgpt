"""Per-chat rolling conversation context. Memory only; never written to disk."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ContextMessage:
    sender: str
    text: str
    ts: float
    from_bot: bool = False


@dataclass
class _ChatContext:
    messages: deque[ContextMessage]
    last_activity: float


class ContextStore:
    def __init__(
        self,
        max_messages: int = 30,
        ttl_seconds: float = 6 * 3600,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.max_messages = max_messages
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._chats: dict[str, _ChatContext] = {}

    # -- public API --
    def add(self, chat_guid: str, sender: str, text: str, from_bot: bool = False) -> None:
        now = self._clock()
        ctx = self._chats.get(chat_guid)
        if ctx is None or self._expired(ctx, now):
            ctx = _ChatContext(messages=deque(maxlen=self.max_messages), last_activity=now)
            self._chats[chat_guid] = ctx
        ctx.messages.append(ContextMessage(sender=sender, text=text, ts=now, from_bot=from_bot))
        ctx.last_activity = now

    def get(self, chat_guid: str) -> list[ContextMessage]:
        ctx = self._chats.get(chat_guid)
        if ctx is None:
            return []
        if self._expired(ctx, self._clock()):
            self.forget(chat_guid)
            return []
        return list(ctx.messages)

    def forget(self, chat_guid: str) -> bool:
        return self._chats.pop(chat_guid, None) is not None

    def forget_all(self) -> None:
        self._chats.clear()

    def sweep(self) -> int:
        """Drop every chat idle longer than the TTL. Returns number wiped."""
        now = self._clock()
        dead = [g for g, c in self._chats.items() if self._expired(c, now)]
        for g in dead:
            self._chats.pop(g, None)
        return len(dead)

    def chat_count(self) -> int:
        return len(self._chats)

    async def run_sweeper(self, interval_seconds: float, on_sweep=None) -> None:
        while True:
            await asyncio.sleep(interval_seconds)
            n = self.sweep()
            if n and on_sweep:
                on_sweep(n)

    # -- internals --
    def _expired(self, ctx: _ChatContext, now: float) -> bool:
        return (now - ctx.last_activity) > self.ttl_seconds
