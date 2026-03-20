"""browser_engine.py - Async multi-page scraper built on Playwright.

Designed for ephemeral Jupyter runtimes (Colab / Kaggle).  All persistent
state (browser sessions, URL caches) is written under ``/tmp/`` so nothing
survives a runtime restart unless the caller explicitly copies it.

Key capabilities
----------------
* **Stealth evasion** via ``playwright-stealth``.
* **Dynamic capture** - human-like scrolling, viewport adjustment, full HTML
  including ``<iframe>`` content and ``srcset`` media URLs.
* **Multi-page spidering** with ``asyncio.Semaphore``-bounded concurrency.
* **Persistent context** for cookie / session reuse across sequential runs.
* **Remote CDP** connection for outbound anti-detect browser WebSockets.
* **Async hooks** (``on_page_load``, ``on_extract``) for user-injected logic.
* **SQLite disk cache** (``/tmp/opus_cache/cache.db``) to skip redundant fetches.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Sequence
from urllib.parse import urljoin, urlparse

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)
from playwright_stealth import stealth_async  # type: ignore[import-untyped]

logger = logging.getLogger("opus_crawl.browser_engine")

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class PageResult:
    """Immutable snapshot of a single fetched page."""

    url: str
    status: int
    html: str
    title: str
    media_urls: list[str] = field(default_factory=list)
    iframe_htmls: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0
    from_cache: bool = False


# ---------------------------------------------------------------------------
# Ephemeral SQLite cache
# ---------------------------------------------------------------------------

_DEFAULT_CACHE_DIR = "/tmp/opus_cache"
_DEFAULT_CACHE_TTL = 3600  # seconds


class _DiskCache:
    """Lightweight SQLite URL cache stored under ``/tmp``."""

    def __init__(self, cache_dir: str = _DEFAULT_CACHE_DIR, ttl: int = _DEFAULT_CACHE_TTL) -> None:
        self._ttl = ttl
        os.makedirs(cache_dir, exist_ok=True)
        db_path = os.path.join(cache_dir, "cache.db")
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS page_cache "
            "(url_hash TEXT PRIMARY KEY, url TEXT, status INTEGER, html TEXT, "
            "title TEXT, media TEXT, iframes TEXT, ts REAL)"
        )
        self._conn.commit()

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _hash(url: str) -> str:
        return hashlib.sha256(url.encode()).hexdigest()

    # -- public API -------------------------------------------------------

    def get(self, url: str) -> Optional[PageResult]:
        """Return cached *PageResult* or ``None`` if missing / expired."""
        row = self._conn.execute(
            "SELECT url, status, html, title, media, iframes, ts "
            "FROM page_cache WHERE url_hash = ?",
            (self._hash(url),),
        ).fetchone()
        if row is None:
            return None
        _url, status, html, title, media_json, iframes_json, ts = row
        if time.time() - ts > self._ttl:
            self._conn.execute("DELETE FROM page_cache WHERE url_hash = ?", (self._hash(url),))
            self._conn.commit()
            return None
        return PageResult(
            url=_url,
            status=status,
            html=html,
            title=title,
            media_urls=json.loads(media_json),
            iframe_htmls=json.loads(iframes_json),
            from_cache=True,
        )

    def put(self, result: PageResult) -> None:
        """Insert or replace a page result in the cache."""
        self._conn.execute(
            "INSERT OR REPLACE INTO page_cache "
            "(url_hash, url, status, html, title, media, iframes, ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                self._hash(result.url),
                result.url,
                result.status,
                result.html,
                result.title,
                json.dumps(result.media_urls),
                json.dumps(result.iframe_htmls),
                time.time(),
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Hook type aliases
# ---------------------------------------------------------------------------

OnPageLoadHook = Callable[[Page], Awaitable[None]]
OnExtractHook = Callable[[str], Awaitable[str]]


# ---------------------------------------------------------------------------
# BrowserEngine
# ---------------------------------------------------------------------------


class BrowserEngine:
    """Async Playwright-based scraper with stealth, caching, and spidering.

    Parameters
    ----------
    headless : bool
        Launch the browser in headless mode (default ``True``).
    session_dir : str | None
        Path for ``launch_persistent_context``.  Defaults to
        ``/tmp/opus_session``.  Set to ``None`` to use a throwaway context.
    cache_dir : str
        Directory for the SQLite URL cache.
    cache_ttl : int
        Seconds before a cached page expires.
    concurrency : int
        Maximum number of pages fetched in parallel during ``crawl()``.
    viewport : dict
        Viewport dimensions passed to Playwright.
    user_agent : str | None
        Override the default user-agent string.
    on_page_load : OnPageLoadHook | None
        Async callback invoked right after every page navigation.
    on_extract : OnExtractHook | None
        Async callback that receives raw HTML and must return
        (possibly transformed) HTML.
    extra_headers : dict[str, str] | None
        Additional HTTP headers sent with every request.
    """

    def __init__(
        self,
        *,
        headless: bool = True,
        session_dir: Optional[str] = "/tmp/opus_session",
        cache_dir: str = _DEFAULT_CACHE_DIR,
        cache_ttl: int = _DEFAULT_CACHE_TTL,
        concurrency: int = 5,
        viewport: Optional[dict[str, int]] = None,
        user_agent: Optional[str] = None,
        on_page_load: Optional[OnPageLoadHook] = None,
        on_extract: Optional[OnExtractHook] = None,
        extra_headers: Optional[dict[str, str]] = None,
    ) -> None:
        self._headless = headless
        self._session_dir = session_dir
        self._cache = _DiskCache(cache_dir, cache_ttl)
        self._concurrency = concurrency
        self._viewport = viewport or {"width": 1920, "height": 1080}
        self._user_agent = user_agent
        self._on_page_load = on_page_load
        self._on_extract = on_extract
        self._extra_headers = extra_headers or {}

        # Playwright objects - initialised lazily via ``_ensure_browser``.
        self._pw: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    async def _ensure_browser(self) -> BrowserContext:
        """Lazily launch Playwright and return a *BrowserContext*."""
        if self._context is not None:
            return self._context

        self._pw = await async_playwright().start()

        launch_args = [
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
        ]

        if self._session_dir:
            os.makedirs(self._session_dir, exist_ok=True)
            self._context = await self._pw.chromium.launch_persistent_context(
                self._session_dir,
                headless=self._headless,
                viewport=self._viewport,
                user_agent=self._user_agent or "",
                args=launch_args,
                extra_http_headers=self._extra_headers,
                ignore_https_errors=True,
            )
        else:
            self._browser = await self._pw.chromium.launch(
                headless=self._headless,
                args=launch_args,
            )
            self._context = await self._browser.new_context(
                viewport=self._viewport,
                user_agent=self._user_agent or "",
                extra_http_headers=self._extra_headers,
                ignore_https_errors=True,
            )

        return self._context

    async def connect_over_cdp(self, ws_endpoint: str) -> BrowserContext:
        """Connect to a remote browser via Chrome DevTools Protocol.

        Parameters
        ----------
        ws_endpoint : str
            WebSocket URL exposed by the remote browser
            (e.g. ``ws://127.0.0.1:9222/devtools/browser/...``).

        Returns
        -------
        BrowserContext
            The default context of the remote browser.
        """
        if self._pw is None:
            self._pw = await async_playwright().start()

        self._browser = await self._pw.chromium.connect_over_cdp(ws_endpoint)
        contexts = self._browser.contexts
        self._context = contexts[0] if contexts else await self._browser.new_context(
            viewport=self._viewport,
            user_agent=self._user_agent or "",
            extra_http_headers=self._extra_headers,
            ignore_https_errors=True,
        )
        logger.info("Connected to remote CDP endpoint: %s", ws_endpoint)
        return self._context

    async def close(self) -> None:
        """Tear down browser, context, Playwright, and the disk cache."""
        if self._context is not None:
            await self._context.close()
            self._context = None
        if self._browser is not None:
            await self._browser.close()
            self._browser = None
        if self._pw is not None:
            await self._pw.stop()
            self._pw = None
        self._cache.close()

    # ------------------------------------------------------------------
    # Page-level helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _human_scroll(page: Page, pause: float = 0.25) -> None:
        """Scroll the full page height in increments to trigger lazy loads."""
        viewport_height: int = page.viewport_size["height"] if page.viewport_size else 1080
        current = 0
        prev_height = 0
        max_iterations = 50  # safety cap

        for _ in range(max_iterations):
            total_height: int = await page.evaluate("document.body.scrollHeight")
            if current >= total_height:
                break
            step = min(viewport_height, total_height - current)
            current += step
            await page.evaluate(f"window.scrollTo(0, {current})")
            await asyncio.sleep(pause)
            new_height: int = await page.evaluate("document.body.scrollHeight")
            if new_height == prev_height:
                break
            prev_height = new_height

        # Scroll back to top so subsequent interactions start at origin.
        await page.evaluate("window.scrollTo(0, 0)")

    @staticmethod
    async def _collect_media(page: Page) -> list[str]:
        """Extract media URLs from ``<img>``, ``<source>``, and ``srcset``."""
        urls: list[str] = []
        # Regular src attributes
        for tag in await page.query_selector_all("img[src], source[src], video[src], audio[src]"):
            src = await tag.get_attribute("src")
            if src:
                urls.append(src)
        # srcset attributes (each entry is "url descriptor")
        for tag in await page.query_selector_all("[srcset]"):
            srcset = await tag.get_attribute("srcset")
            if srcset:
                for entry in srcset.split(","):
                    parts = entry.strip().split()
                    if parts:
                        urls.append(parts[0])
        return urls

    @staticmethod
    async def _collect_iframes(page: Page) -> list[str]:
        """Return the inner HTML of every accessible ``<iframe>``."""
        htmls: list[str] = []
        for frame in page.frames:
            if frame == page.main_frame:
                continue
            try:
                content = await frame.content()
                htmls.append(content)
            except Exception:  # noqa: BLE001 - iframes may be cross-origin
                pass
        return htmls

    # ------------------------------------------------------------------
    # Single-page fetch
    # ------------------------------------------------------------------

    async def fetch(
        self,
        url: str,
        *,
        wait_until: str = "networkidle",
        timeout_ms: int = 30_000,
        scroll: bool = True,
        use_cache: bool = True,
    ) -> PageResult:
        """Fetch a single URL and return a :class:`PageResult`.

        Parameters
        ----------
        url : str
            Absolute URL to fetch.
        wait_until : str
            Playwright ``wait_until`` strategy (``domcontentloaded``,
            ``load``, ``networkidle``).
        timeout_ms : int
            Navigation timeout in milliseconds.
        scroll : bool
            Whether to perform human-like scrolling after load.
        use_cache : bool
            Check / populate the disk cache.
        """
        if use_cache:
            cached = self._cache.get(url)
            if cached is not None:
                logger.debug("Cache hit: %s", url)
                return cached

        ctx = await self._ensure_browser()
        page = await ctx.new_page()
        await stealth_async(page)

        t0 = time.monotonic()
        try:
            resp = await page.goto(url, wait_until=wait_until, timeout=timeout_ms)
            status = resp.status if resp else 0

            if self._on_page_load:
                await self._on_page_load(page)

            if scroll:
                await self._human_scroll(page)

            html = await page.content()

            if self._on_extract:
                html = await self._on_extract(html)

            title = await page.title()
            media = await self._collect_media(page)
            iframes = await self._collect_iframes(page)
            elapsed = (time.monotonic() - t0) * 1000

            result = PageResult(
                url=url,
                status=status,
                html=html,
                title=title,
                media_urls=media,
                iframe_htmls=iframes,
                elapsed_ms=round(elapsed, 2),
            )

            if use_cache:
                self._cache.put(result)

            return result

        finally:
            await page.close()

    # ------------------------------------------------------------------
    # Multi-page spider
    # ------------------------------------------------------------------

    @staticmethod
    def _same_origin(base: str, candidate: str) -> bool:
        """Return ``True`` if *candidate* shares the same origin as *base*."""
        return urlparse(base).netloc == urlparse(candidate).netloc

    @staticmethod
    def _normalise(url: str) -> str:
        """Strip fragment and trailing slash for dedup purposes."""
        parsed = urlparse(url)
        path = parsed.path.rstrip("/") or "/"
        return f"{parsed.scheme}://{parsed.netloc}{path}"

    async def _extract_links(self, page_html: str, base_url: str) -> list[str]:
        """Parse internal links from raw HTML (no extra browser round-trip)."""
        from bs4 import BeautifulSoup  # local import to keep top-level lean

        soup = BeautifulSoup(page_html, "lxml")
        links: list[str] = []
        for anchor in soup.find_all("a", href=True):
            href: str = anchor["href"]
            absolute = urljoin(base_url, href)
            if self._same_origin(base_url, absolute):
                links.append(self._normalise(absolute))
        return links

    async def crawl(
        self,
        start_url: str,
        *,
        max_depth: int = 2,
        max_pages: int = 50,
        wait_until: str = "networkidle",
        timeout_ms: int = 30_000,
        scroll: bool = True,
        use_cache: bool = True,
    ) -> list[PageResult]:
        """Spider internal links starting from *start_url*.

        Parameters
        ----------
        start_url : str
            The seed URL.
        max_depth : int
            Maximum link-follow depth (0 = only the seed page).
        max_pages : int
            Hard cap on total pages fetched.
        wait_until / timeout_ms / scroll / use_cache
            Forwarded to :meth:`fetch`.

        Returns
        -------
        list[PageResult]
            All successfully fetched pages.
        """
        sem = asyncio.Semaphore(self._concurrency)
        visited: set[str] = set()
        results: list[PageResult] = []
        lock = asyncio.Lock()

        async def _visit(url: str, depth: int) -> None:
            norm = self._normalise(url)
            async with lock:
                if norm in visited or len(results) >= max_pages:
                    return
                visited.add(norm)

            async with sem:
                try:
                    result = await self.fetch(
                        url,
                        wait_until=wait_until,
                        timeout_ms=timeout_ms,
                        scroll=scroll,
                        use_cache=use_cache,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to fetch %s: %s", url, exc)
                    return

                async with lock:
                    if len(results) >= max_pages:
                        return
                    results.append(result)

            if depth < max_depth:
                child_links = await self._extract_links(result.html, url)
                tasks: list[asyncio.Task[None]] = []
                for link in child_links:
                    async with lock:
                        if self._normalise(link) in visited or len(results) >= max_pages:
                            continue
                    tasks.append(asyncio.create_task(_visit(link, depth + 1)))
                if tasks:
                    await asyncio.gather(*tasks)

        await _visit(start_url, 0)
        logger.info("Crawl complete: %d pages fetched from %s", len(results), start_url)
        return results

    # ------------------------------------------------------------------
    # Convenience context-manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "BrowserEngine":
        await self._ensure_browser()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()
