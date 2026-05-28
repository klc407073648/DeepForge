"""URL normalization and recursive link filtering."""

from __future__ import annotations

import fnmatch
import re
from urllib.parse import urljoin, urlparse, urlunparse

HTTP_SCHEMES = {"http", "https"}


def normalize_url(url: str, base: str | None = None) -> str:
    """Resolve relative URLs, strip fragments, and normalize scheme/host casing."""
    resolved = urljoin(base or url, url)
    parsed = urlparse(resolved)
    if not parsed.scheme or not parsed.netloc:
        return ""
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    normalized = urlunparse((scheme, netloc, path, parsed.params, parsed.query, ""))
    return normalized


def registrable_domain(netloc: str) -> str:
    """Return a coarse domain key for same-site checks."""
    host = netloc.split("@")[-1].split(":")[0].lower()
    parts = host.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host


def is_http_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme.lower() in HTTP_SCHEMES and bool(parsed.netloc)


def matches_any_pattern(url: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        if not pattern:
            continue
        if pattern.startswith("re:"):
            if re.search(pattern[3:], url):
                return True
        elif fnmatch.fnmatch(url, pattern):
            return True
    return False


class LinkFilter:
    """Decide whether a discovered link should be enqueued for crawling."""

    def __init__(
        self,
        root_url: str,
        *,
        max_depth: int,
        same_domain_only: bool,
        exclude_patterns: list[str],
        include_patterns: list[str],
    ) -> None:
        self.root_url = normalize_url(root_url)
        self.root_domain = registrable_domain(urlparse(self.root_url).netloc)
        self.max_depth = max_depth
        self.same_domain_only = same_domain_only
        self.exclude_patterns = exclude_patterns
        self.include_patterns = include_patterns

    def should_follow(self, url: str, depth: int, visited: set[str]) -> bool:
        normalized = normalize_url(url, self.root_url)
        if not normalized:
            return False
        if depth > self.max_depth:
            return False
        if normalized in visited:
            return False
        if not is_http_url(normalized):
            return False
        if self.same_domain_only:
            domain = registrable_domain(urlparse(normalized).netloc)
            if domain != self.root_domain:
                return False
        if matches_any_pattern(normalized, self.exclude_patterns):
            return False
        if self.include_patterns and not matches_any_pattern(normalized, self.include_patterns):
            return False
        return True
