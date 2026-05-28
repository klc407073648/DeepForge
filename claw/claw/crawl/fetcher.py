"""HTTP fetcher with retries, rate limiting, and robots.txt support."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
from robotexclusionrulesparser import RobotExclusionRulesParser
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from claw.config import CrawlConfig


@dataclass
class FetchResult:
    url: str
    status_code: int
    content_type: str
    html: str
    final_url: str


class RobotsCache:
    """Cache robots.txt rules per origin."""

    def __init__(self, user_agent: str) -> None:
        self.user_agent = user_agent
        self._cache: dict[str, RobotExclusionRulesParser | None] = {}
        self._lock = asyncio.Lock()

    async def can_fetch(self, client: httpx.AsyncClient, url: str, ignore: bool) -> bool:
        if ignore:
            return True
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        async with self._lock:
            if origin not in self._cache:
                robots_url = f"{origin}/robots.txt"
                parser = RobotExclusionRulesParser()
                try:
                    resp = await client.get(robots_url, timeout=10.0)
                    if resp.status_code >= 400:
                        self._cache[origin] = None
                    else:
                        parser.parse(resp.text)
                        self._cache[origin] = parser
                except Exception:
                    self._cache[origin] = None
            rules = self._cache[origin]
        if rules is None:
            return True
        return rules.is_allowed(self.user_agent, url)


class Fetcher:
    def __init__(self, config: CrawlConfig) -> None:
        self.config = config
        self.robots = RobotsCache(config.user_agent)
        self._delay_lock = asyncio.Lock()
        self._last_request_at = 0.0

    async def _throttle(self) -> None:
        delay = self.config.request_delay_ms / 1000.0
        if delay <= 0:
            return
        async with self._delay_lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            wait = self._last_request_at + delay - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request_at = loop.time()

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.TransportError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    async def _get(self, client: httpx.AsyncClient, url: str) -> httpx.Response:
        await self._throttle()
        return await client.get(
            url,
            headers={"User-Agent": self.config.user_agent},
            follow_redirects=True,
            timeout=self.config.timeout_seconds,
        )

    async def fetch(self, client: httpx.AsyncClient, url: str) -> FetchResult:
        if not await self.robots.can_fetch(client, url, self.config.ignore_robots):
            raise PermissionError(f"Blocked by robots.txt: {url}")

        response = await self._get(client, url)
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                await asyncio.sleep(int(retry_after))
            response = await self._get(client, url)

        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            raise ValueError(f"Unsupported content type: {content_type}")

        encoding = response.encoding
        if not encoding:
            import chardet

            detected = chardet.detect(response.content)
            encoding = detected.get("encoding") or "utf-8"

        html = response.content.decode(encoding, errors="replace")
        return FetchResult(
            url=url,
            status_code=response.status_code,
            content_type=content_type,
            html=html,
            final_url=str(response.url),
        )
