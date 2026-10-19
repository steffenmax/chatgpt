"""Read-only browser and ask-the-chat tools for the /task agent.

The browser is a fresh private Chromium context per task (nothing on disk), with:
  * read-only tools: open a URL, follow a link by number. No typing, clicking forms,
    uploads, downloads, or agent-supplied JavaScript;
  * every request checked: http(s) only, and nothing that resolves to loopback,
    private, link-local or reserved addresses (so it can't reach BlueBubbles,
    Ollama, the dashboard or the home network);
  * no anti-bot workarounds: if a site blocks a plain headless browser, it's blocked.
"""

from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
from typing import Any, Awaitable, Callable
from urllib.parse import urlparse

MAX_TEXT = 12_000
MAX_LINKS = 60
MAX_PAGES = 20

_CREDENTIALS = re.compile(
    r"\b(pass ?words?|passcodes?|pass ?phrases?|log ?ins?|logging in|sign ?in|credentials?|usernames?|user ?names?|"
    r"2fa|two.factor|verification codes?|one.time (code|password)s?|otps?|auth(entication)? codes?|security codes?|"
    r"card numbers?|credit cards?|debit cards?|cvv|cvc|expir(y|ation) dates?|ssn|social security|bank accounts?|"
    r"routing numbers?|pins?|api keys?|tokens?)\b",
    re.IGNORECASE,
)


def asks_for_credentials(question: str) -> bool:
    return bool(_CREDENTIALS.search(question or ""))


def _ip_blocked(ip: str) -> bool:
    try:
        a = ipaddress.ip_address(ip)
    except ValueError:
        return True
    return a.is_private or a.is_loopback or a.is_link_local or a.is_reserved or a.is_multicast or a.is_unspecified


async def url_blocked(url: str, cache: dict[str, bool] | None = None) -> str | None:
    """None if the URL may be fetched, else a short reason."""
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        return "only http(s) pages"
    host = (p.hostname or "").lower()
    if not host:
        return "no host"
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal", ".lan", ".home")):
        return "local addresses are off limits"
    if cache is not None and host in cache:
        return "local addresses are off limits" if cache[host] else None
    try:
        ipaddress.ip_address(host)
        addrs = [host]
    except ValueError:
        try:
            infos = await asyncio.get_running_loop().getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
            addrs = [i[4][0] for i in infos]
        except OSError:
            return "can't resolve that host"
    blocked = any(_ip_blocked(a.split("%")[0]) for a in addrs) or not addrs
    if cache is not None:
        cache[host] = blocked
    return "local addresses are off limits" if blocked else None


_EXTRACT_LINKS = """
() => {
  const seen = new Set(), out = [];
  for (const a of document.querySelectorAll('a[href]')) {
    const href = a.href, text = (a.innerText || a.getAttribute('aria-label') || '').trim().replace(/\\s+/g, ' ');
    if (!href.startsWith('http') || seen.has(href) || !text) continue;
    const r = a.getBoundingClientRect();
    if (r.width === 0 && r.height === 0) continue;
    seen.add(href); out.push([text.slice(0, 80), href]);
    if (out.length >= %d) break;
  }
  return out;
}
""" % MAX_LINKS


class ReadOnlyBrowser:
    def __init__(self):
        self._pw = None
        self._browser = None
        self._context = None
        self._page = None
        self._links: list[list[str]] = []
        self._hosts: dict[str, bool] = {}
        self.pages_opened = 0

    async def _ensure(self):
        if self._page is not None:
            return
        from playwright.async_api import async_playwright

        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(headless=True)
        self._context = await self._browser.new_context(
            accept_downloads=False, service_workers="block", viewport={"width": 1280, "height": 900}
        )

        async def route(r):
            req = r.request
            if req.resource_type in ("media", "font"):
                return await r.abort()
            if await url_blocked(req.url, self._hosts):
                return await r.abort()
            await r.continue_()

        await self._context.route("**/*", route)
        self._page = await self._context.new_page()
        self._page.set_default_timeout(25_000)

    async def open(self, url: str) -> str:
        reason = await url_blocked(url, self._hosts)
        if reason:
            return f"Blocked: {reason}."
        if self.pages_opened >= MAX_PAGES:
            return f"Page limit reached ({MAX_PAGES} per task). Work with what you have."
        await self._ensure()
        self.pages_opened += 1
        page = self._page
        try:
            resp = await page.goto(url, wait_until="domcontentloaded")
        except Exception as exc:  # noqa: BLE001
            # A failed navigation can leave the tab mid-load; start the next page in a fresh tab.
            try:
                await page.close()
            except Exception:  # noqa: BLE001
                pass
            self._page = await self._context.new_page()
            self._page.set_default_timeout(25_000)
            return f"Could not load {url}: {type(exc).__name__}. The site may block automated browsers; don't retry it."
        try:
            await page.wait_for_load_state("networkidle", timeout=8_000)
        except Exception:  # noqa: BLE001
            pass
        for _ in range(3):  # trigger lazy-loaded content
            await page.mouse.wheel(0, 2500)
            await page.wait_for_timeout(400)
        status = resp.status if resp else "?"
        title = (await page.title())[:200]
        text = await page.evaluate("() => document.body ? document.body.innerText : ''")
        text = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text or "")).strip()
        self._links = await page.evaluate(_EXTRACT_LINKS)
        links = "\n".join(f"[{i}] {t} -> {h}" for i, (t, h) in enumerate(self._links))
        blocked_hint = ""
        if status in (401, 403, 429) or len(text) < 200:
            blocked_hint = "\n(Little or no content: the site may be blocking automated browsers. Don't retry it.)"
        return (
            f"URL: {page.url}\nStatus: {status}\nTitle: {title}{blocked_hint}\n\n"
            f"{text[:MAX_TEXT]}{' …(truncated)' if len(text) > MAX_TEXT else ''}\n\nLinks:\n{links or '(none)'}"
        )

    async def follow(self, n: int) -> str:
        if not self._links or not (0 <= n < len(self._links)):
            return "No such link number on the current page. Open a page first."
        return await self.open(self._links[n][1])

    async def close(self):
        for obj in (self._context, self._browser):
            try:
                if obj is not None:
                    await obj.close()
            except Exception:  # noqa: BLE001
                pass
        if self._pw is not None:
            try:
                await self._pw.stop()
            except Exception:  # noqa: BLE001
                pass
        self._pw = self._browser = self._context = self._page = None


def build_server(browser: ReadOnlyBrowser | None, ask: Callable[[str], Awaitable[str | None]] | None, max_questions: int):
    """In-process MCP server 'task' exposing the browser and ask-the-chat tools."""
    from claude_agent_sdk import create_sdk_mcp_server, tool

    def text(s: str, error: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {"content": [{"type": "text", "text": s}]}
        if error:
            out["is_error"] = True
        return out

    tools = []
    if browser is not None:

        @tool("browser_open", "Open a web page in a real (headless, read-only) browser that runs JavaScript, "
              "and get its visible text plus numbered links. Use for sites WebFetch can't render. "
              "You can't type, click buttons, log in or submit forms.", {"url": str})
        async def browser_open(args):
            return text(await browser.open(str(args.get("url", ""))))

        @tool("browser_follow", "Open link number n from the page you last opened.", {"n": int})
        async def browser_follow(args):
            try:
                n = int(args.get("n", -1))
            except (TypeError, ValueError):
                n = -1
            return text(await browser.follow(n))

        tools += [browser_open, browser_follow]

    if ask is not None:
        asked = {"n": 0}

        @tool("ask_chat", "Ask the person who gave you the task one short question in the group chat and wait "
              "for their reply (a few minutes at most). Only for details you truly need, like a date, party "
              "size, budget or which option they prefer. Never ask for passwords, logins, codes or payment "
              "details.", {"question": str})
        async def ask_chat(args):
            q = " ".join(str(args.get("question", "")).split())[:300]
            if not q:
                return text("Empty question.", error=True)
            if asks_for_credentials(q):
                return text("Not allowed: never ask the chat for passwords, logins, codes or payment details. "
                            "Finish what you can and tell them to do that part themselves.", error=True)
            if asked["n"] >= max_questions:
                return text(f"Question limit reached ({max_questions}). Make a sensible assumption and say what you assumed.")
            asked["n"] += 1
            reply = await ask(q)
            if reply is None:
                return text("No reply in time. Make a sensible assumption and say what you assumed.")
            return text(f"Their reply: {reply}")

        tools.append(ask_chat)

    return create_sdk_mcp_server("task", "1.0.0", tools=tools) if tools else None
