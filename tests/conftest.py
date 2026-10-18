from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from imessage_agent.config import Settings
from imessage_agent.context import ContextStore
from imessage_agent.handler import Agent, SenderResolver

ALLOWED = "iMessage;+;chat111"
OTHER = "iMessage;+;chat999"
SECRET = "s3cret-s3cret-s3cret-s3cret"
BB_PASSWORD = "bluebubbles-pass-123"
API_KEY = "sk-ant-test-key-000"


class FakeBB:
    def __init__(self):
        self.texts: list[tuple[str, str]] = []
        self.attachments: list[tuple[str, Path, bytes]] = []

    async def send_text(self, chat_guid, text):
        self.texts.append((chat_guid, text))
        return {"guid": "sent"}

    async def send_attachment(self, chat_guid, file_path: Path, name=None):
        self.attachments.append((chat_guid, file_path, file_path.read_bytes()))
        return {"guid": "sent-att"}

    async def ping(self):
        return True

    async def query_contacts(self, addresses):
        return []

    async def aclose(self):
        pass


class FakeLLM:
    name = "fake"

    def __init__(self, reply="yo"):
        self.reply_text = reply
        self.calls: list[list[dict]] = []

    async def reply(self, system, messages):
        self.calls.append(messages)
        return self.reply_text

    async def healthy(self):
        return True

    async def aclose(self):
        pass


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


def make_settings(**over) -> Settings:
    base = dict(
        _env_file=None,
        bot_name="Coco",
        allowed_chat_guids=[ALLOWED],
        webhook_secret=SECRET,
        bluebubbles_password=BB_PASSWORD,
        anthropic_api_key="",
        context_ttl_seconds=3600,
        context_max_messages=5,
        image_enabled=True,
    )
    base.update(over)
    return Settings(**base)


def message(text, chat=ALLOWED, sender="+15550001111", from_me=False, guid=None, **extra):
    data = {
        "guid": guid or f"g-{abs(hash((text, chat, sender)))}",
        "text": text,
        "isFromMe": from_me,
        "handle": {"address": sender, "service": "iMessage"},
        "chats": [{"guid": chat, "displayName": "Test Group", "style": 43}],
        "associatedMessageType": None,
        "itemType": 0,
        "dateCreated": 1700000000000,
    }
    data.update(extra)
    return {"type": "new-message", "data": data}


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def agent_parts(clock):
    s = make_settings()
    bb = FakeBB()
    llm = FakeLLM()
    ctx = ContextStore(max_messages=s.context_max_messages, ttl_seconds=s.context_ttl_seconds, clock=clock)
    resolver = SenderResolver(None, {"+15550001111": "Alex", "+15550002222": "Sam"}, use_contacts=False)
    agent = Agent(s, bb, llm, ctx, images=None, resolver=resolver)  # type: ignore[arg-type]
    return s, bb, llm, ctx, agent
