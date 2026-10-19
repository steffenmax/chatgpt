"""Thin client for the BlueBubbles Server REST API (local only).

Endpoints used (verified against bluebubbles-server source, packages/server/src/server/api/http/api/v1):
  GET  /api/v1/ping
  GET  /api/v1/server/info
  POST /api/v1/message/text          JSON: chatGuid, tempGuid, message, method
  POST /api/v1/message/attachment    multipart: chatGuid, tempGuid, name, method, attachment=<file>
  POST /api/v1/chat/query            JSON: limit, offset, with, sort
  GET  /api/v1/chat/:guid?with=participants
  GET  /api/v1/webhook, POST /api/v1/webhook {url, events}, DELETE /api/v1/webhook/:id
  POST /api/v1/contact/query         JSON: addresses

Authentication is a query parameter (`password`); BlueBubbles does not read headers.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import httpx


class BlueBubblesError(RuntimeError):
    pass


class BlueBubblesClient:
    def __init__(
        self,
        base_url: str,
        password: str,
        send_method: str = "apple-script",
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self._password = password
        self.send_method = send_method
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    # -- helpers --
    def _params(self, **extra: Any) -> dict[str, Any]:
        p = {"password": self._password}
        p.update({k: v for k, v in extra.items() if v is not None})
        return p

    @staticmethod
    def _unwrap(resp: httpx.Response) -> Any:
        try:
            body = resp.json()
        except ValueError as e:
            raise BlueBubblesError(f"non-JSON response (HTTP {resp.status_code})") from e
        if resp.status_code >= 400 or (isinstance(body, dict) and body.get("status", 200) >= 400):
            # Do not include body text: it may embed the message content.
            raise BlueBubblesError(f"BlueBubbles API error (HTTP {resp.status_code})")
        return body.get("data") if isinstance(body, dict) else body

    # -- attachments --
    async def download_attachment(self, guid: str, max_bytes: int = 25 * 1024 * 1024) -> bytes:
        """Raw bytes of an attachment (held in RAM only by the caller)."""
        r = await self._client.get(f"/api/v1/attachment/{guid}/download", params=self._params())
        if r.status_code != 200:
            raise BlueBubblesError(f"attachment download failed (HTTP {r.status_code})")
        if len(r.content) > max_bytes:
            raise BlueBubblesError("attachment too large")
        return r.content

    # -- server --
    async def ping(self) -> bool:
        r = await self._client.get("/api/v1/ping", params=self._params())
        return r.status_code == 200

    async def server_info(self) -> dict[str, Any]:
        r = await self._client.get("/api/v1/server/info", params=self._params())
        return self._unwrap(r) or {}

    # -- messages --
    async def send_text(self, chat_guid: str, text: str) -> dict[str, Any]:
        payload = {
            "chatGuid": chat_guid,
            "tempGuid": str(uuid.uuid4()),
            "message": text,
            "method": self.send_method,
        }
        r = await self._client.post("/api/v1/message/text", params=self._params(), json=payload)
        return self._unwrap(r) or {}

    async def send_attachment(self, chat_guid: str, file_path: Path, name: str | None = None) -> dict[str, Any]:
        name = name or file_path.name
        data = {
            "chatGuid": chat_guid,
            "tempGuid": str(uuid.uuid4()),
            "name": name,
            "method": self.send_method,
        }
        with file_path.open("rb") as fh:
            files = {"attachment": (name, fh, "image/png")}
            r = await self._client.post(
                "/api/v1/message/attachment",
                params=self._params(),
                data=data,
                files=files,
                timeout=httpx.Timeout(120.0),
            )
        return self._unwrap(r) or {}

    # -- chats --
    async def list_chats(self, limit: int = 200, offset: int = 0) -> list[dict[str, Any]]:
        payload = {"limit": limit, "offset": offset, "with": ["participants", "lastMessage"], "sort": "lastmessage"}
        r = await self._client.post("/api/v1/chat/query", params=self._params(), json=payload)
        return self._unwrap(r) or []

    async def get_chat(self, chat_guid: str) -> dict[str, Any]:
        r = await self._client.get(f"/api/v1/chat/{chat_guid}", params=self._params(**{"with": "participants"}))
        return self._unwrap(r) or {}

    # -- contacts (optional, for sender display names) --
    async def query_contacts(self, addresses: list[str]) -> list[dict[str, Any]]:
        r = await self._client.post("/api/v1/contact/query", params=self._params(), json={"addresses": addresses})
        return self._unwrap(r) or []

    # -- webhooks --
    async def list_webhooks(self) -> list[dict[str, Any]]:
        r = await self._client.get("/api/v1/webhook", params=self._params())
        return self._unwrap(r) or []

    async def create_webhook(self, url: str, events: list[str]) -> dict[str, Any]:
        r = await self._client.post("/api/v1/webhook", params=self._params(), json={"url": url, "events": events})
        return self._unwrap(r) or {}

    async def delete_webhook(self, webhook_id: int) -> None:
        r = await self._client.delete(f"/api/v1/webhook/{webhook_id}", params=self._params())
        self._unwrap(r)
