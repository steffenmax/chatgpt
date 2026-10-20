"""Settings, loaded from environment variables / .env.

Every setting is documented in .env.example. Nothing here is persisted anywhere.
"""

from __future__ import annotations

import re

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
    # Mentions that arrive together are answered in one reply: wait until the chat has
    # been quiet this long (capped at REPLY_BATCH_MAX_SECONDS after the first ping).
    # Replies in one chat never overlap either way. 0 = answer each mention at once.
    reply_batch_seconds: float = Field(default=0.0, ge=0.0, le=30.0)
    reply_batch_max_seconds: float = Field(default=12.0, ge=0.0, le=60.0)
    # Soft cap on a reply's length (cut at a sentence end). 0 = no cap.
    reply_max_chars: int = Field(default=0, ge=0)
    # Occasional short text reactions ("lmaoo", "💀") to ordinary messages. Real iMessage
    # tapbacks need the BlueBubbles Private API, so these are plain texts. 0 = off.
    reaction_chance: float = Field(default=0.0, ge=0.0, le=1.0)
    reaction_cooldown_seconds: float = Field(default=300.0, ge=0.0)
    # Haha tapbacks on one message: at HAHA_ACK_AT different people the bot says something dumb
    # about it, at HAHA_PRAISE_AT it compliments whoever sent it. 0 = off.
    haha_ack_at: int = Field(default=0, ge=0)
    haha_praise_at: int = Field(default=0, ge=0)
    # Per-chat notes the local model keeps about the people and the group, as markdown
    # files on disk. Empty = off (nothing message-derived is ever written to disk).
    memory_dir: str = ""
    # /remind support. The JSON file holds reminder text, so this is opt-in like MEMORY_DIR.
    reminders_file: str = ""
    # Phrase command acknowledgements ("on it", "got it, 7pm") in the bot's own voice via
    # the local model (one short extra call). False = fixed wording.
    voice_acks: bool = True
    # Local vision model (Ollama) used to caption photos people send, so the bot can see
    # them. Loaded only when a photo arrives, unloaded after VISION_KEEP_ALIVE. Empty = off.
    vision_model: str = ""
    vision_max_width: int = Field(default=1024, ge=256, le=4096)
    vision_keep_alive: str = "2m"
    memory_update_every: int = Field(default=15, ge=1)
    memory_max_chars: int = Field(default=3000, ge=200)

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
    # "mflux" = local FLUX on this Mac (default). "openai" = OpenAI's image API:
    # fast and good, but every /image prompt leaves the Mac. Needs OPENAI_API_KEY.
    image_backend: Literal["mflux", "openai"] = "mflux"
    openai_api_key: str = ""
    openai_image_model: str = "gpt-image-1"
    openai_image_size: str = "1024x1024"
    openai_image_quality: Literal["low", "medium", "high", "auto"] = "medium"
    openai_timeout_seconds: float = Field(default=120.0, gt=0)
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
    # /task: multi-step research jobs run by Claude (Agent SDK) with web search, web fetch
    # and a sandbox folder. Only from the chats listed in TASK_CHAT_GUIDS. Off by default.
    task_enabled: bool = False
    task_chat_guids: Annotated[list[str], NoDecode] = Field(default_factory=list)
    # --- NFL alerts and the bookie (sports.py, alerts.py, bookie.py) ---
    # Chats that get live alerts and can bet. Followed teams and their fans, for context:
    # "NYG=Max,Joe;NYJ=Chris Oddo". ESPN's public API, no key.
    sports_chat_guids: Annotated[list[str], NoDecode] = Field(default_factory=list)
    sports_teams: str = ""
    sports_alerts: bool = False
    sports_field_goals: bool = True
    bookie_enabled: bool = False
    bookie_start: int = Field(default=1000, ge=1)
    bookie_bailout: int = Field(default=100, ge=1)
    # Bets on games already in progress, at live odds (closes in the last two minutes / OT).
    bookie_live: bool = True
    # Who may start a task and answer its questions (phone numbers / emails as iMessage
    # reports them). Empty = anyone in the task chats.
    task_owner_addresses: Annotated[list[str], NoDecode] = Field(default_factory=list)
    task_dir: str = ""
    task_model: str = ""  # empty = ANTHROPIC_MODEL
    task_max_turns: int = Field(default=25, ge=1, le=200)
    task_max_budget_usd: float = Field(default=1.0, gt=0, le=20)
    task_timeout_seconds: float = Field(default=600.0, gt=0)
    task_context_messages: int = Field(default=15, ge=0, le=100)
    # Read-only headless Chromium for JavaScript-heavy sites (playwright install chromium).
    task_browser: bool = False
    # The agent may stop and ask the requester up to N short questions in the chat.
    task_max_questions: int = Field(default=2, ge=0, le=5)
    task_ask_timeout_seconds: float = Field(default=240.0, ge=10)
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-opus-5-5"
    anthropic_max_tokens: int = Field(default=400, ge=1)
    anthropic_timeout_seconds: float = Field(default=60.0, gt=0)

    # --- daily culture brief (OFF by default) ---
    # Once a day the task agent web-searches what people are joking about (no chat content
    # is sent) and the brief is added to the bot's prompt. Needs ANTHROPIC_API_KEY.
    brief_enabled: bool = False
    brief_time: str = "08:00"
    brief_topics: str = ""
    brief_file: str = ""
    brief_max_budget_usd: float = Field(default=0.5, gt=0, le=5)
    # Where the brief comes from. auto = X pulls when X_BEARER_TOKEN is set (plus web
    # search when there is an Anthropic key), else the web. See resolved_brief_source.
    brief_source: Literal["auto", "web", "x", "both"] = "auto"

    # --- X (Twitter) API pulls for the brief (OFF until X_BEARER_TOKEN is set) ---
    # Pay-per-use: $0.005 per post returned. Each query asks for exactly X_POSTS_PER_QUERY
    # posts and a refresh never requests more than X_MAX_USD_PER_REFRESH worth.
    x_bearer_token: str = ""
    # One search per ';'-separated item. Empty = derived from BRIEF_TOPICS.
    x_queries: str = ""
    # Appended to every query unless it already sets them. Add e.g. "-has:links -has:hashtags"
    # to cut betting touts and spam, or "is:verified" for bigger accounts only.
    x_query_filters: str = "lang:en -is:retweet -is:reply"
    x_posts_per_query: int = Field(default=15, ge=10, le=100)
    # Keep only posts with at least this many likes (applied after the pull: the API has no
    # likes filter on pay-per-use, so filtered-out posts are still billed). min_faves:N in a
    # single search raises it for that search.
    x_min_likes: int = Field(default=0, ge=0)
    x_lookback_hours: int = Field(default=48, ge=1, le=168)
    x_max_usd_per_refresh: float = Field(default=0.75, gt=0, le=20)

    # --- links people share (linkreader.py) ---
    # Open links posted in allowlisted chats and add what's behind them (post text, top
    # replies, page title and opening text) to the in-memory context. Fetches go out from
    # this Mac to the linked site; X posts go through api.fxtwitter.com when LINK_X_REPLIES
    # is on (needed for replies), otherwise X's own embed endpoint (post only).
    links_enabled: bool = False
    # A link nobody responds to for this long gets the bot's own take (0 = off).
    link_quiet_seconds: float = Field(default=0.0, ge=0.0)
    link_x_replies: bool = True
    link_max_chars: int = Field(default=900, ge=200, le=3000)

    # --- GIFs and tweets (media.py) ---
    # GIFs come from GIPHY (free key at developers.giphy.com); empty key = no GIFs.
    giphy_api_key: str = ""
    gif_rating: Literal["g", "pg", "pg-13", "r"] = "pg-13"
    # Funny tweets: an X recent search on what the chat is talking about (uses X_BEARER_TOKEN,
    # billed per post returned, search words come from the chat). Off unless TWEETS_ENABLED.
    tweets_enabled: bool = False
    tweet_min_likes: int = Field(default=200, ge=0)
    tweet_posts_per_search: int = Field(default=10, ge=10, le=100)
    tweets_per_hour: int = Field(default=6, ge=1, le=60)
    # A message nobody (the bot included) answers for this many minutes gets a reaction GIF.
    # Needs GIPHY_API_KEY. 0 = off.
    unanswered_gif_minutes: float = Field(default=0.0, ge=0.0)
    # Local hours with no unanswered GIFs, "start-end" (e.g. "1-9" = 1:00 to 8:59). Empty = none.
    unanswered_gif_quiet_hours: str = ""

    def in_quiet_hours(self, hour: int) -> bool:
        m = re.fullmatch(r"\s*(\d{1,2})\s*-\s*(\d{1,2})\s*", self.unanswered_gif_quiet_hours or "")
        if not m:
            return False
        start, end = int(m.group(1)) % 24, int(m.group(2)) % 24
        return start <= hour < end if start <= end else (hour >= start or hour < end)  # wraps midnight

    # --- dashboard (loopback only, same port as the webhook) ---
    # http://127.0.0.1:PORT/dashboard : health, what the bot can see and connect to,
    # memory notes (editable), reminders, live events. Off by default.
    dashboard_enabled: bool = False

    # --- logging ---
    log_level: str = "INFO"

    @field_validator("allowed_chat_guids", "task_chat_guids", "task_owner_addresses", "sports_chat_guids", mode="before")
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

    @property
    def resolved_brief_source(self) -> str:
        """BRIEF_SOURCE with 'auto' resolved: web / x / both."""
        if self.brief_source != "auto":
            return self.brief_source
        if not self.x_bearer_token:
            return "web"
        return "both" if self.anthropic_api_key else "x"

    @property
    def brief_uses_anthropic(self) -> bool:
        src = self.resolved_brief_source
        return self.brief_enabled and (src != "x" or bool(self.anthropic_api_key))

    def x_query_list(self) -> list[str]:
        from .xsearch import parse_queries, topics_to_queries

        return parse_queries(self.x_queries) or topics_to_queries(self.brief_topics)

    @property
    def x_query_origin(self) -> str:
        from .xsearch import parse_queries

        if parse_queries(self.x_queries):
            return "X_QUERIES"
        return "BRIEF_TOPICS" if self.brief_topics.strip() else "defaults"

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
        if self.image_enabled and self.image_backend == "openai":
            hosts.append(("OpenAI API (image generation)", "api.openai.com:443"))
        if self.brief_uses_anthropic and not self.task_enabled:
            hosts.append(("Anthropic API (daily brief)", "api.anthropic.com:443"))
        if self.brief_enabled and self.resolved_brief_source in ("x", "both"):
            hosts.append(("X API (brief post pulls)", "api.x.com:443"))
        if self.links_enabled:
            hosts.append(("Links shared in allowlisted chats", "any website:443"))
            hosts.append(("X posts and replies (shared X links)",
                          "api.fxtwitter.com:443" if self.link_x_replies else "cdn.syndication.twimg.com:443"))
        if self.giphy_api_key:
            hosts.append(("GIPHY (GIF search, words from the chat)", "api.giphy.com:443, media*.giphy.com:443"))
        if self.tweets_enabled and self.x_bearer_token:
            hosts.append(("X API (tweet search, words from the chat)", "api.x.com:443"))
        if (self.sports_alerts or self.bookie_enabled) and self.sports_chat_guids:
            hosts.append(("ESPN scores and odds (no chat content)", "site.api.espn.com:443"))
        if self.task_enabled:
            hosts.append(("Anthropic API (/task agent)", "api.anthropic.com:443"))
            hosts.append(("Web pages fetched during /task", "any website:443"))
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
        if self.brief_enabled and not self.anthropic_api_key and self.resolved_brief_source != "x":
            problems.append(
                "BRIEF_ENABLED=true but ANTHROPIC_API_KEY is empty (set it, or set X_BEARER_TOKEN "
                "and BRIEF_SOURCE=x to build the brief from X with the local model)."
            )
        if self.brief_enabled and self.resolved_brief_source in ("x", "both") and not self.x_bearer_token:
            problems.append(f"BRIEF_SOURCE={self.brief_source} but X_BEARER_TOKEN is empty.")
        if self.brief_enabled and self.x_bearer_token and self.x_posts_per_query * 0.005 > self.x_max_usd_per_refresh:
            problems.append("X_MAX_USD_PER_REFRESH is too low for even one query of X_POSTS_PER_QUERY posts.")
        if self.task_enabled and not self.anthropic_api_key:
            problems.append("TASK_ENABLED=true but ANTHROPIC_API_KEY is empty.")
        if self.task_enabled and not self.task_chat_guids:
            problems.append("TASK_ENABLED=true but TASK_CHAT_GUIDS is empty; nobody can use /task.")
        if self.image_enabled and self.image_backend == "openai" and not self.openai_api_key:
            problems.append("IMAGE_BACKEND=openai but OPENAI_API_KEY is empty.")
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


def env_quote(value: str) -> str:
    """A .env value that python-dotenv reads back exactly: double-quoted, with escapes."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("\r\n", "\n").replace("\n", "\\n") + '"'


def set_env_value(path, key: str, value: str) -> None:
    """Replace KEY=... in a .env file (or append it), atomically, keeping it private."""
    import os
    import re
    import tempfile
    from pathlib import Path

    path = Path(path)
    lines = path.read_text(encoding="utf-8").split("\n") if path.exists() else []
    new = f"{key}={env_quote(value)}"
    pat = re.compile(rf"^\s*(export\s+)?{re.escape(key)}\s*=", re.IGNORECASE)
    for i, line in enumerate(lines):
        if pat.match(line):
            lines[i] = new
            break
    else:
        lines.append(new)
    fd, tmp = tempfile.mkstemp(prefix=".env-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
