from imessage_agent.config import Settings


def test_env_parsing(monkeypatch):
    monkeypatch.setenv("ALLOWED_CHAT_GUIDS", "iMessage;+;chat1, iMessage;+;chat2 ,")
    monkeypatch.setenv("SENDER_NAMES", "+15551234567=Alex, friend@example.com=Sam Q")
    monkeypatch.setenv("MFLUX_QUANTIZE", "")
    monkeypatch.setenv("CLOUD_LLM_ENABLED", "false")
    s = Settings(_env_file=None)
    assert s.allowed_chat_guids == ["iMessage;+;chat1", "iMessage;+;chat2"]
    assert s.sender_names == {"+15551234567": "Alex", "friend@example.com": "Sam Q"}
    assert s.mflux_quantize is None
    assert [h for _, h in s.outbound_hosts()] == ["127.0.0.1:1234", "127.0.0.1:11434"]


def test_env_example_loads(tmp_path):
    import shutil

    shutil.copy(".env.example", tmp_path / ".env")
    s = Settings(_env_file=str(tmp_path / ".env"))
    assert len(s.allowed_chat_guids) == 2
    assert s.cloud_llm_enabled is False
    assert s.host == "127.0.0.1"


def test_validate_runtime_flags_problems():
    s = Settings(_env_file=None, webhook_secret="short", bluebubbles_password="", host="0.0.0.0", allowed_chat_guids=[])
    msgs = " ".join(s.validate_runtime())
    for key in ("WEBHOOK_SECRET", "BLUEBUBBLES_PASSWORD", "HOST", "ALLOWED_CHAT_GUIDS"):
        assert key in msgs


def test_cloud_facts_needs_key_and_lists_host():
    from tests.conftest import make_settings

    s = make_settings(cloud_facts_enabled=True, anthropic_api_key="")
    assert any("CLOUD_FACTS_ENABLED" in p for p in s.validate_runtime())
    s = make_settings(cloud_facts_enabled=True, anthropic_api_key="sk-ant-x")
    assert not any("CLOUD_FACTS" in p for p in s.validate_runtime())
    hosts = dict(s.outbound_hosts())
    assert hosts["Anthropic API (fact lookups)"] == "api.anthropic.com:443"
    assert s.any_cloud
    off = make_settings()
    assert "Anthropic API (fact lookups)" not in dict(off.outbound_hosts()) and not off.any_cloud
