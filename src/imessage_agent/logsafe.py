"""Privacy-preserving logging.

Rules enforced here:
  * Application code logs through `log_event`, which only accepts an allow-list of
    metadata keys (event type, chat GUID, timings, error class...). Passing any other
    key raises, so message text or image prompts cannot be logged by accident.
  * A redaction filter is attached to the root logger so that secrets (BlueBubbles
    password, webhook secret, API key) are scrubbed from every record, including
    exceptions raised by HTTP libraries that embed the full URL.
  * uvicorn's access log is disabled by the server (it would print query strings).
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone

LOGGER_NAME = "imessage_agent"
log = logging.getLogger(LOGGER_NAME)
_handler: logging.Handler | None = None

SAFE_KEYS = frozenset(
    {
        "event",
        "chat_guid",
        "message_guid",
        "allowed",
        "command",
        "kind",
        "status",
        "reason",
        "error",  # exception class name only, never str(exc)
        "host",
        "purpose",
        # daily brief / X pulls: counts and dollars only, never post text or queries
        "source",
        "posts",
        "queries_ok",
        "queries_failed",
        "errors",
        "model",
        "backend",
        "duration_ms",
        "queue_len",
        "count",
        "job_id",
        "ttl_seconds",
        "path",
        "code",
        "attempt",
        "chars",  # length only
        "cost_cents",
        "turns",
        "files",
    }
)


class SecretRedactingFilter(logging.Filter):
    """Replaces every configured secret in log messages/args with '***'."""

    def __init__(self, secrets: list[str]):
        super().__init__()
        self._secrets = [s for s in secrets if s and len(s) >= 4]

    def _scrub(self, s: str) -> str:
        for sec in self._secrets:
            if sec in s:
                s = s.replace(sec, "***")
        return s

    def _scrub_arg(self, a):
        # Keep non-string args (ints for %d, objects for %r) unless their text form
        # contains a secret, e.g. an httpx URL object carrying ?secret=...
        text = a if isinstance(a, str) else str(a)
        scrubbed = self._scrub(text)
        return scrubbed if (isinstance(a, str) or scrubbed != text) else a

    def filter(self, record: logging.LogRecord) -> bool:
        if not self._secrets:
            return True
        try:
            if isinstance(record.msg, str):
                record.msg = self._scrub(record.msg)
            if record.args:
                if isinstance(record.args, dict):
                    record.args = {k: self._scrub_arg(v) for k, v in record.args.items()}
                else:
                    record.args = tuple(self._scrub_arg(a) for a in record.args)
            if record.exc_text:
                record.exc_text = self._scrub(record.exc_text)
        except Exception:  # never let logging crash the app
            pass
        return True


class _RingHandler(logging.Handler):
    """Keeps the last N formatted (already redacted) lines in memory for the dashboard."""

    def __init__(self, capacity: int = 400):
        super().__init__()
        from collections import deque

        self.lines = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.lines.append(self.format(record))
        except Exception:  # noqa: BLE001
            pass


_ring: _RingHandler | None = None


def recent_lines(n: int = 200) -> list[str]:
    if _ring is None:
        return []
    return list(_ring.lines)[-n:]


def install_logging(level: str = "INFO", secrets: list[str] | None = None) -> None:
    global _handler, _ring
    root = logging.getLogger()
    root.setLevel(level.upper())
    if _handler is not None:
        root.removeHandler(_handler)
    handler = logging.StreamHandler(sys.stderr)
    _handler = handler
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    redactor = SecretRedactingFilter(secrets or [])
    handler.addFilter(redactor)
    root.addHandler(handler)
    if _ring is not None:
        root.removeHandler(_ring)
    _ring = _RingHandler()
    _ring.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    _ring.addFilter(redactor)
    root.addHandler(_ring)
    # Make sure third-party loggers (uvicorn, httpx) also go through redaction.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "httpx", "httpcore", "httpx2", "httpcore2"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True
    logging.getLogger("uvicorn.access").disabled = True
    for name in ("httpx", "httpcore", "httpx2", "httpcore2"):
        logging.getLogger(name).setLevel(logging.WARNING)


def log_event(event: str, level: int = logging.INFO, **fields) -> None:
    """Structured, allow-listed log line. Raises on any non-allow-listed key."""
    bad = set(fields) - SAFE_KEYS
    if bad:
        raise ValueError(f"refusing to log non-allow-listed fields: {sorted(bad)}")
    parts = [f"event={event}"]
    for k, v in fields.items():
        if v is None:
            continue
        if isinstance(v, BaseException):
            v = type(v).__name__
        parts.append(f"{k}={v}")
    parts.append("ts=" + datetime.now(timezone.utc).isoformat(timespec="seconds"))
    log.log(level, " ".join(parts))


def exc_name(exc: BaseException) -> str:
    """Only the exception class name is ever logged (messages may embed content)."""
    return type(exc).__name__
