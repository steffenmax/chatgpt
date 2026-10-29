"""Settings, read from environment variables only."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TOKEN_DIR = "~/.garminconnect"


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    token_dir: Path
    email: str | None
    password: str | None
    is_cn: bool
    allow_writes: bool

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            token_dir=Path(os.getenv("GARMINTOKENS", DEFAULT_TOKEN_DIR)).expanduser(),
            email=os.getenv("GARMIN_EMAIL") or None,
            password=os.getenv("GARMIN_PASSWORD") or None,
            is_cn=_truthy(os.getenv("GARMIN_IS_CN")),
            allow_writes=_truthy(os.getenv("GARMIN_MCP_ALLOW_WRITES")),
        )
