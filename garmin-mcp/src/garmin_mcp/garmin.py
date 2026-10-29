"""Thin wrapper around `garminconnect.Garmin`: lazy login, token reuse, friendly errors."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from typing import Any

from garminconnect import (
    Garmin,
    GarminConnectAuthenticationError,
    GarminConnectConnectionError,
    GarminConnectNotFoundError,
    GarminConnectTooManyRequestsError,
)

from .config import Settings

log = logging.getLogger(__name__)

LOGIN_HINT = "Run `garmin-mcp login` in a terminal to sign in (handles MFA) and save tokens."


class GarminError(Exception):
    """A failure whose message is safe and useful to show to the model/user."""


class MfaRequired(GarminError):
    pass


def _no_interactive_mfa() -> str:
    raise MfaRequired(
        "Garmin asked for an MFA code, which the MCP server can't prompt for. " + LOGIN_HINT
    )


class GarminSession:
    """One authenticated Garmin client per process, created on first use.

    `garminconnect` is synchronous (requests/curl_cffi), so calls run in a worker
    thread; a lock serialises them because the client refreshes tokens in place.
    """

    def __init__(self, settings: Settings, factory: Callable[..., Any] = Garmin) -> None:
        self.settings = settings
        self._factory = factory
        self._api: Any | None = None
        self._lock = threading.Lock()

    def _login(self) -> Any:
        s = self.settings
        api = self._factory(
            email=s.email,
            password=s.password,
            is_cn=s.is_cn,
            prompt_mfa=_no_interactive_mfa,
        )
        try:
            api.login(tokenstore=str(s.token_dir))
        except GarminError:
            raise
        except GarminConnectTooManyRequestsError as e:
            raise GarminError(f"Garmin is rate-limiting logins, wait a few minutes: {e}") from e
        except (GarminConnectAuthenticationError, GarminConnectConnectionError) as e:
            # An MfaRequired raised from our prompt arrives here wrapped by garminconnect.
            raise GarminError(f"Garmin login failed: {e}\n{LOGIN_HINT}") from e
        log.info("Logged in to Garmin Connect as %s", getattr(api, "display_name", "?"))
        return api

    def api(self) -> Any:
        if self._api is None:
            self._api = self._login()
        return self._api

    def call_sync(self, fn: Callable[[Any], Any]) -> Any:
        with self._lock:
            try:
                return fn(self.api())
            except GarminError:
                raise
            except GarminConnectAuthenticationError as e:
                # Tokens were rejected mid-session: drop the client so the next call re-logs in.
                self._api = None
                raise GarminError(f"Garmin rejected the saved session: {e}\n{LOGIN_HINT}") from e
            except GarminConnectTooManyRequestsError as e:
                raise GarminError(f"Garmin rate limit hit, try again in a few minutes: {e}") from e
            except GarminConnectNotFoundError as e:
                raise GarminError(f"Not found on Garmin Connect: {e}") from e
            except GarminConnectConnectionError as e:
                raise GarminError(f"Garmin Connect request failed: {e}") from e
            except ValueError as e:
                # garminconnect validates arguments (dates, ids, limits) with ValueError.
                raise GarminError(f"Invalid argument: {e}") from e

    async def call(self, fn: Callable[[Any], Any]) -> Any:
        return await asyncio.to_thread(self.call_sync, fn)


def interactive_login(settings: Settings, email: str, password: str, prompt_mfa: Callable[[], str]) -> Any:
    """Full credential login (with MFA prompt) that writes tokens to `settings.token_dir`."""
    api = Garmin(email=email, password=password, is_cn=settings.is_cn, prompt_mfa=prompt_mfa)
    api.login(tokenstore=str(settings.token_dir))
    return api
