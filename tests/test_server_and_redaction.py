"""End-to-end through FastAPI: webhook auth, health, and the log-redaction guarantee."""

import logging

from fastapi.testclient import TestClient

from imessage_agent.context import ContextStore
from imessage_agent.handler import Agent, SenderResolver
from imessage_agent.main import create_app
from tests.conftest import ALLOWED, API_KEY, BB_PASSWORD, OTHER, SECRET, FakeBB, FakeLLM, make_settings, message

SENSITIVE = [
    "coco the password to the wifi is hunter2",
    "my salary is 123456 dollars coco",
    "/image a photo of my diagnosis results",
    "plain gossip about dave",
]


def _client(**over):
    s = make_settings(anthropic_api_key=API_KEY, **over)
    bb, llm = FakeBB(), FakeLLM(reply="lol ok")
    agent = Agent(s, bb, llm, ContextStore(5, 3600), images=None, resolver=SenderResolver(None, {}, use_contacts=False))  # type: ignore[arg-type]
    app = create_app(s, agent=agent)
    return s, bb, agent, app


def test_webhook_requires_secret():
    s, bb, agent, app = _client()
    with TestClient(app) as c:
        assert c.post("/webhook", json=message("coco hi")).status_code == 401
        assert c.post("/webhook?secret=wrong", json=message("coco hi")).status_code == 401
        assert c.post("/webhook", json=message("coco hi"), headers={"X-Webhook-Secret": "nope"}).status_code == 401
        r = c.post(f"/webhook?secret={SECRET}", json=message("coco hi"))
        assert r.status_code == 200 and r.json() == {"ok": True}
    assert bb.texts == [(ALLOWED, "lol ok")]


def test_health():
    s, bb, agent, app = _client()
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["bluebubbles"] is True
        assert body["cloud_llm_enabled"] is False
        assert body["allowlisted_chats"] == 1


def test_logs_never_contain_message_text_prompts_or_secrets(caplog):
    caplog.set_level(logging.DEBUG)
    s, bb, agent, app = _client(log_level="DEBUG")
    with TestClient(app) as c:
        for i, text in enumerate(SENSITIVE):
            c.post(f"/webhook?secret={SECRET}", json=message(text, guid=f"s{i}"))
            c.post(f"/webhook?secret={SECRET}", json=message(text, chat=OTHER, guid=f"o{i}"))
        c.post(f"/webhook?secret={SECRET}", json=message("/forget", guid="fg"))
        c.post(f"/webhook?secret={SECRET}", json=message("/help", guid="hp"))
        c.post("/webhook?secret=bad", json=message("coco secret stuff", guid="bad"))

    blob = caplog.text + "\n".join(r.getMessage() for r in caplog.records)
    for text in SENSITIVE:
        for word in ("hunter2", "salary", "diagnosis", "gossip", "dave"):
            assert word not in blob, f"log leaked content word {word!r}"
        assert text not in blob
    assert "lol ok" not in blob  # bot replies are content too
    assert SECRET not in blob
    assert BB_PASSWORD not in blob
    assert API_KEY not in blob
    # but operational metadata IS there
    assert "event=trigger" in blob
    assert ALLOWED in blob
    assert f"chat_guid={OTHER}" in blob  # non-allowlisted chat logged by GUID so it can be added


def test_secret_redaction_filter_scrubs_exception_urls(caplog):
    from imessage_agent.logsafe import SecretRedactingFilter

    logger = logging.getLogger("imessage_agent.test_redact")
    logger.addFilter(SecretRedactingFilter([BB_PASSWORD, SECRET]))
    caplog.set_level(logging.INFO)
    logger.info("httpx error for url http://127.0.0.1:1234/api/v1/message/text?password=%s", BB_PASSWORD)
    logger.info(f"webhook http://127.0.0.1:8765/webhook?secret={SECRET}")
    assert BB_PASSWORD not in caplog.text and SECRET not in caplog.text
    assert "password=***" in caplog.text


def test_server_refuses_non_loopback_bind():
    import pytest

    s = make_settings(host="0.0.0.0")
    bb, llm = FakeBB(), FakeLLM()
    agent = Agent(s, bb, llm, ContextStore(5, 3600), images=None, resolver=SenderResolver(None, {}, use_contacts=False))  # type: ignore[arg-type]
    app = create_app(s, agent=agent)
    with pytest.raises(Exception, match="fatal configuration"):
        with TestClient(app):
            pass
