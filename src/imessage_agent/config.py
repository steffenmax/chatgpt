"""Settings, loaded from environment variables / .env.

Every setting is documented in .env.example. Nothing here is persisted anywhere.
"""

from __future__ import annotations

from typing import Annotated, Literal
from urllib.parse import urlparse

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- identity / behaviour ---
    bot_name: str = Field(default="Coco", min_length=1)
    bot_persona: str = (
        "You are {bot_name}, a friend in an iMessage group chat. Reply short and casual, "
        "like texting a friend. One to three sentences unless asked for more. No markdown, "
        "no bullet points, no emoji spam. You can only send text replies; images are made "
        "by a separate /image command, so never claim you drew, sent, searched, remembered "
        "long-term, set reminders, or did anything you cannot actually do. If you don't know, "
        "say so. Address people by the name shown before their message when it helps."
    )
    allowed_chat_guids: Annotated[list[str], NoDecode] = Field(default_factory=list)
    context_max_messages: int = Field(default=30, ge=1, le=500)
    context_ttl_seconds: int = Field(default=6 * 60 * 60, ge=1)
    context_sweep_seconds: int = Field(default=60, ge=5)
    sender_names: Annotated[dict[str, str], NoDecode] = Field(default_factory=dict)
    # A message ending in "?" that nobody answers within this many seconds is treated
    # as addressed to the bot. 0 disables.
    quiet_question_seconds: float = Field(default=0.0, ge=0.0)

    # --- webhook server (loopback only) ---
    host: str = "127.0.0.1"
    port: int = Field(default=8765, ge=1, le=65535)
    webhook_secret: str = Field(default="", description="Shared secret; required.")
    webhook_path: str = "/webhook"

    # --- BlueBubbles ---
    bluebubbles_url: str = "http://127.0.0.1:1234"
    bluebubbles_password: str = ""
    bluebubbles_send_method: Literal["apple-script", "private-api"] = "apple-script"
    bluebubbles_timeout_seconds: float = Field(default=30.0, gt=0)

    # --- Ollama (local LLM) ---
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    ollama_max_tokens: int = Field(default=300, ge=1)
    ollama_timeout_seconds: float = Field(default=120.0, gt=0)
    ollama_keep_alive: str = "30m"
    # Context window handed to Ollama (tokens). Must fit persona + CONTEXT_MAX_MESSAGES.
    ollama_num_ctx: int = Field(default=8192, ge=512)
    unload_llm_before_image: bool = False

    # --- image generation (mflux, always local) ---
    image_enabled: bool = True
    mflux_command: str = "mflux-generate"
    mflux_model: str = "schnell"
    mflux_steps: int = Field(default=4, ge=1, le=100)
    mflux_quantize: int | None = 8
    mflux_width: int = Field(default=768, ge=64, le=2048)
    mflux_height: int = Field(default=768, ge=64, le=2048)
    mflux_extra_args: str = ""
    image_timeout_seconds: float = Field(default=600.0, gt=0)
    image_prompt_max_chars: int = Field(default=600, ge=1)

    # --- optional cloud text mode (OFF by default) ---
    cloud_llm_enabled: bool = False
    # Cloud fact lookups: ONLY mentions that need outside/current information (as judged
    # by the local model, or explicit /ask) are answered by the Anthropic API with web
    # search. Everything else stays local. Off by default.
    cloud_facts_enabled: bool = False
    cloud_facts_max_searches: int = Field(default=3, ge=0, le=10)
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-opus-5-5"
    anthropic_max_tokens: int = Field(default=400, ge=1)
    anthropic_timeout_seconds: float = Field(default=60.0, gt=0)

    # --- logging ---
    log_level: str = "INFO"

    @field_validator("allowed_chat_guids", mode="before")
    @classmethod
    def _split_guids(cls, v):
        if isinstance(v, str):
            return [g.strip() for g in v.split(",") if g.strip()]
        return v

    @field_validator("sender_names", mode="before")
    @classmethod
    def _parse_sender_names(cls, v):
        # Format: "+15551234567=Alex,friend@example.com=Sam"
        if isinstance(v, str):
            out: dict[str, str] = {}
            for pair in v.split(","):
                pair = pair.strip()
                if not pair or "=" not in pair:
                    continue
                addr, name = pair.split("=", 1)
                out[addr.strip()] = name.strip()
            return out
        return v

    @field_validator("mflux_quantize", mode="before")
    @classmethod
    def _quantize_none(cls, v):
        if v in ("", None, "none", "None", "0", 0):
            return None
        return v

    @field_validator("bluebubbles_url", "ollama_url")
    @classmethod
    def _strip_slash(cls, v: str) -> str:
        return v.rstrip("/")

    # --- derived helpers ---
    @property
    def persona(self) -> str:
        return self.bot_persona.replace("{bot_name}", self.bot_name)

    def outbound_hosts(self) -> list[tuple[str, str]]:
        """Every (purpose, host) the running app may open a connection to."""
        hosts = [
            ("BlueBubbles", _host_of(self.bluebubbles_url)),
            ("Ollama", _host_of(self.ollama_url)),
        ]
        if self.cloud_llm_enabled:
            hosts.append(("Anthropic API (cloud text mode)", "api.anthropic.com:443"))
        elif self.cloud_facts_enabled:
            hosts.append(("Anthropic API (fact lookups)", "api.anthropic.com:443"))
        return hosts

    @property
    def any_cloud(self) -> bool:
        return self.cloud_llm_enabled or self.cloud_facts_enabled

    def validate_runtime(self) -> list[str]:
        """Return a list of fatal configuration problems (empty = OK)."""
        problems = []
        if len(self.webhook_secret) < 16:
            problems.append("WEBHOOK_SECRET must be set and at least 16 characters.")
        if not self.bluebubbles_password:
            problems.append("BLUEBUBBLES_PASSWORD must be set.")
        if self.host not in ("127.0.0.1", "::1", "localhost"):
            problems.append(f"HOST={self.host!r} is not loopback; refusing to start.")
        if self.cloud_llm_enabled and not self.anthropic_api_key:
            problems.append("CLOUD_LLM_ENABLED=true but ANTHROPIC_API_KEY is empty.")
        if self.cloud_facts_enabled and not self.anthropic_api_key:
            problems.append("CLOUD_FACTS_ENABLED=true but ANTHROPIC_API_KEY is empty.")
        if not self.allowed_chat_guids:
            problems.append(
                "ALLOWED_CHAT_GUIDS is empty; the bot will answer nowhere. "
                "Run `imessage-agent chats` to find GUIDs."
            )
        return problems


def _host_of(url: str) -> str:
    p = urlparse(url)
    port = p.port or (443 if p.scheme == "https" else 80)
    return f"{p.hostname}:{port}"


def is_loopback_host(hostport: str) -> bool:
    host = hostport.rsplit(":", 1)[0]
    return host in ("127.0.0.1", "localhost", "::1", "[::1]")
