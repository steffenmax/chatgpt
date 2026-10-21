"""Read links people share in the chat, so the bot can talk about what's behind them.

Each link becomes a short card appended to the message in context, e.g.
  [link: X post by Name (@handle), 12K likes: "..." | top replies: @a (2K): "..."]
Sources, all read without logging in to anything:
  * X/Twitter posts: FxTwitter's public API (post + top replies), or X's own embed
    endpoint (post only) if replies are off or FxTwitter is down;
  * YouTube and TikTok: their public oEmbed endpoints (title/caption and channel);
  * anything else: the page itself, plain HTTP, title + description + opening paragraphs.
Every hop is checked against loopback/private addresses (same rule as the /task browser),
pages are size-capped, and nothing here logs in, posts, or runs page JavaScript.
Cards live only in the in-memory chat context.
"""

from __future__ import annotations

import asyncio
import math
import re
import time
from collections import OrderedDict
from html.parser import HTMLParser
from urllib.parse import quote, urljoin, urlparse

import httpx

from .logsafe import exc_name, log_event
from .taskbrowser import url_blocked

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 "
      "(KHTML, like Gecko) Version/17.0 Safari/605.1.15")
MAX_LINKS_PER_MESSAGE = 2
MAX_PAGE_BYTES = 2_000_000
MAX_IMAGE_BYTES = 5_000_000
CACHE_SIZE = 200
CACHE_TTL = 1800.0

_URL = re.compile(r"https?://[^\s<>\"]+", re.IGNORECASE)
_X_STATUS = re.compile(
    r"^(?:www\.|mobile\.)?(?:x|twitter|fxtwitter|vxtwitter|fixupx|fixvx)\.com/(?:[\w]+|i/web)/status(?:es)?/(\d+)", re.IGNORECASE)
_YOUTUBE = re.compile(r"^(?:www\.|m\.)?(?:youtube\.com/(?:watch|shorts/|live/)|youtu\.be/)", re.IGNORECASE)
_TIKTOK = re.compile(r"^(?:www\.|vm\.|vt\.|m\.)?tiktok\.com/", re.IGNORECASE)
_CARD = re.compile(r"\s*\[link: [^\]]*\]")


def find_urls(text: str) -> list[str]:
    out: list[str] = []
    for m in _URL.finditer(text or ""):
        url = m.group(0).rstrip(".,;:!?)'’”")
        if url not in out:
            out.append(url)
        if len(out) >= MAX_LINKS_PER_MESSAGE:
            break
    return out


def strip_cards(text: str) -> str:
    """The message as the person typed it, without link cards."""
    return _CARD.sub("", text or "").strip()


def _clean(s: str, limit: int) -> str:
    s = " ".join((s or "").replace("[", "(").replace("]", ")").split())
    s = re.sub(r"https?://t\.co/\S+", "", s).strip()
    return s if len(s) <= limit else s[:limit].rsplit(" ", 1)[0] + "…"


def _count(n) -> str:
    try:
        n = int(n)
    except (TypeError, ValueError):
        return "?"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M".replace(".0M", "M")
    if n >= 10_000:
        return f"{n // 1000}K"
    if n >= 1000:
        return f"{n / 1000:.1f}K".replace(".0K", "K")
    return str(n)


def _syndication_token(tweet_id: str) -> str:
    """The token X's embed widget sends: (id / 1e15 * pi) in base 36, without zeros and the dot."""
    n = int(tweet_id) / 1e15 * math.pi
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    whole, frac = int(n), n - int(n)
    s = ""
    while whole:
        s, whole = digits[whole % 36] + s, whole // 36
    f = ""
    for _ in range(12):
        frac *= 36
        f += digits[int(frac)]
        frac -= int(frac)
    return re.sub(r"(0+|\.)", "", (s or "0") + "." + f)


class _PageText(HTMLParser):
    """Title, meta description and paragraph text, skipping scripts, nav and footers."""

    SKIP = {"script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg", "template"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, str] = {}
        self.title = ""
        self.paras: list[str] = []
        self._skip = 0
        self._in_title = False
        self._p: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in self.SKIP:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag == "meta":
            key = (a.get("property") or a.get("name") or "").lower()
            if key in ("og:title", "og:description", "og:site_name", "description", "twitter:title",
                       "twitter:description") and a.get("content"):
                self.meta.setdefault(key, a["content"])
        elif tag == "p" and not self._skip:
            self._p = []

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False
        elif tag == "p" and self._p is not None:
            p = " ".join(re.sub(r"\[(\d+|[a-z])\]", "", "".join(self._p)).split())
            if len(p) >= 60:
                self.paras.append(p)
            self._p = None

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif self._p is not None and not self._skip:
            self._p.append(data)


class LinkReader:
    def __init__(self, x_replies: bool = True, max_chars: int = 900, vision=None, gate: asyncio.Semaphore | None = None,
                 timeout: float = 10.0, client: httpx.AsyncClient | None = None, clock=time.monotonic):
        self.x_replies = x_replies
        self.max_chars = max_chars
        # Optional OllamaVision: the first photo (or video thumbnail) in a post gets a caption.
        self.vision = vision
        self._gate = gate
        self._client = client or httpx.AsyncClient(timeout=timeout, headers={"User-Agent": UA}, follow_redirects=False)
        self._clock = clock
        self._cache: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._hosts: dict[str, bool] = {}

    async def aclose(self) -> None:
        await self._client.aclose()

    async def read(self, url: str) -> str:
        """A '[link: ...]' card for the URL ('' if there's nothing worth adding). Never raises."""
        hit = self._cache.get(url)
        if hit and self._clock() - hit[0] < CACHE_TTL:
            return hit[1]
        started = time.monotonic()
        kind = "page"
        try:
            rest = re.sub(r"^https?://", "", url, flags=re.IGNORECASE)
            if m := _X_STATUS.match(rest):
                kind, body = "x", await self._x(m.group(1))
            elif _YOUTUBE.match(rest):
                kind, body = "youtube", await self._oembed("YouTube video", "https://www.youtube.com/oembed?format=json&url=", url)
            elif _TIKTOK.match(rest):
                kind, body = "tiktok", await self._oembed("TikTok", "https://www.tiktok.com/oembed?url=", url)
            else:
                body = ""
            if not body:  # also when an oEmbed had nothing
                body = await self._page(url) or "couldn't open it"
        except Exception as exc:  # noqa: BLE001
            log_event("link.failed", kind=kind, error=exc_name(exc))
            body = "couldn't open it"
        card = f"[link: {_clean(body, self.max_chars)}]" if body else ""
        log_event("link.read", kind=kind, chars=len(card), duration_ms=int((time.monotonic() - started) * 1000))
        self._cache[url] = (self._clock(), card)
        while len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
        return card

    # -- HTTP --
    async def _get(self, url: str, limit: int = MAX_PAGE_BYTES, accept: tuple[str, ...] = ()) -> tuple[httpx.Response, bytes, str]:
        """GET with redirects followed by hand so every hop is checked. Returns (response, body, final url)."""
        for _ in range(6):
            reason = await url_blocked(url, self._hosts)
            if reason:
                raise PermissionError(reason)
            async with self._client.stream("GET", url) as r:
                if r.is_redirect and r.headers.get("location"):
                    url = urljoin(url, r.headers["location"])
                    continue
                r.raise_for_status()
                ctype = r.headers.get("content-type", "").lower()
                if accept and not ctype.startswith(accept):
                    raise ValueError("unsupported content type")
                body = b""
                async for chunk in r.aiter_bytes():
                    body += chunk
                    if len(body) > limit:
                        break
                return r, body[:limit], url
        raise ValueError("too many redirects")

    async def _json(self, url: str) -> dict:
        import json

        _, body, _ = await self._get(url, accept=("application/json", "text/json", "application/javascript"))
        return json.loads(body.decode("utf-8", "replace"))

    # -- sources --
    async def _x(self, tweet_id: str) -> str:
        if self.x_replies:
            try:
                j = await self._json(f"https://api.fxtwitter.com/2/conversation/{tweet_id}")
                if isinstance(j.get("status"), dict):
                    return await self._x_card(j["status"], j.get("replies") or [])
            except Exception as exc:  # noqa: BLE001
                log_event("link.x_fallback", error=exc_name(exc))
        j = await self._json(f"https://cdn.syndication.twimg.com/tweet-result?id={tweet_id}&lang=en&token={_syndication_token(tweet_id)}")
        user = j.get("user") or {}
        photo = ((j.get("photos") or [{}])[0].get("url")) or ((j.get("video") or {}).get("poster"))
        status = {
            "text": j.get("text", ""), "author": {"name": user.get("name", ""), "screen_name": user.get("screen_name", "")},
            "likes": j.get("favorite_count"), "replies": j.get("conversation_count"),
            "quote": j.get("quoted_tweet"), "_photo": photo,
        }
        return await self._x_card(status, [])

    async def _x_card(self, s: dict, replies: list) -> str:
        a = s.get("author") or {}
        head = f"X post by {a.get('name') or '?'} (@{a.get('screen_name') or '?'})"
        stats = ", ".join(x for x in (
            f"{_count(s['likes'])} likes" if s.get("likes") is not None else "",
            f"{_count(s['replies'])} replies" if s.get("replies") else "",
        ) if x)
        parts = [f"{head}{', ' + stats if stats else ''}: \"{_clean(s.get('text', ''), 400)}\""]
        q = s.get("quote")
        if isinstance(q, dict) and q.get("text"):
            qa = (q.get("author") or q.get("user") or {}).get("screen_name", "?")
            parts.append(f"quoting @{qa}: \"{_clean(q['text'], 200)}\"")
        media = s.get("media") or {}
        photo = s.get("_photo") or ((media.get("photos") or [{}])[0].get("url")) or ((media.get("videos") or [{}])[0].get("thumbnail_url"))
        is_video = bool(media.get("videos")) or bool(s.get("_photo") and "video" in str(s.get("_photo")))
        if photo:
            cap = await self._caption(photo)
            label = "video" if is_video else "photo"
            parts.append(f"{label}: {cap}" if cap else f"has a {label}")
        author = (a.get("screen_name") or "").lower()
        others = [r for r in replies if isinstance(r, dict) and r.get("text")
                  and ((r.get("author") or {}).get("screen_name") or "").lower() != author]
        others.sort(key=lambda r: int(r.get("likes") or 0), reverse=True)
        top = []
        for r in others[:5]:
            text = re.sub(r"^(@\w+\s*)+", "", r.get("text", ""))
            if not text.strip():
                continue
            top.append(f"@{(r.get('author') or {}).get('screen_name', '?')} ({_count(r.get('likes'))}): \"{_clean(text, 140)}\"")
        if top:
            parts.append("top replies: " + "; ".join(top))
        return " | ".join(parts)

    async def _oembed(self, label: str, endpoint: str, url: str) -> str:
        j = await self._json(endpoint + quote(url, safe=""))
        title = _clean(j.get("title", ""), 300)
        author = _clean(j.get("author_name", ""), 80)
        out = f"{label}{' by ' + author if author else ''}: \"{title}\"" if title else ""
        thumb = j.get("thumbnail_url")
        if out and thumb and self.vision is not None:
            cap = await self._caption(thumb)
            if cap:
                out += f" | thumbnail: {cap}"
        return out

    async def _page(self, url: str) -> str:
        r, body, final = await self._get(url, accept=("text/html", "application/xhtml"))
        charset = r.charset_encoding or "utf-8"
        p = _PageText()
        p.feed(body.decode(charset, "replace"))
        site = p.meta.get("og:site_name") or (urlparse(final).hostname or "").removeprefix("www.")
        title = p.meta.get("og:title") or p.meta.get("twitter:title") or p.title
        desc = p.meta.get("og:description") or p.meta.get("description") or p.meta.get("twitter:description") or ""
        if not (desc or p.paras) and (not title or title.strip().lower() == site.lower()):
            return ""  # a bare app shell or a block page
        out = f"{_clean(site, 60)} page: \"{_clean(title, 200)}\""
        if desc:
            out += f" | {_clean(desc, 300)}"
        text = " ".join(x for x in p.paras[:6] if x not in desc)
        if text:
            out += f" | {_clean(text, 500)}"
        return out

    async def _caption(self, image_url: str) -> str:
        if self.vision is None:
            return ""
        try:
            _, data, _ = await self._get(image_url, limit=MAX_IMAGE_BYTES, accept=("image/",))
            if len(data) >= MAX_IMAGE_BYTES:
                return ""
            if self._gate is not None:
                async with self._gate:
                    return await self.vision.caption(data)
            return await self.vision.caption(data)
        except Exception as exc:  # noqa: BLE001
            log_event("link.caption_failed", error=exc_name(exc))
            return ""
