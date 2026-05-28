"""HTML parsing: title, main content, and outbound links."""

from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup, NavigableString, Tag

from claw.crawl.link_filter import normalize_url
from claw.crawl.rules import SiteRule

REMOVE_TAGS = {"script", "style", "nav", "footer", "header", "noscript", "iframe", "svg"}
DEFAULT_CONTENT_SELECTORS = ["main", "article", "[role=main]", ".markdown-body", "#content", "body"]


@dataclass
class ParsedPage:
    url: str
    title: str
    content_html: str
    links: list[str]
    text_length: int = 0


def _clean_tree(node: Tag, extra_remove: list[str] | None = None) -> None:
    for tag in node.find_all(REMOVE_TAGS):
        tag.decompose()
    for selector in extra_remove or []:
        for tag in node.select(selector):
            tag.decompose()


def _extract_title(soup: BeautifulSoup, title_selector: str | None = None) -> str:
    if title_selector:
        el = soup.select_one(title_selector)
        if el and el.get_text(strip=True):
            return el.get_text(strip=True)
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    if h1:
        text = h1.get_text(strip=True)
        if text:
            return text
    return "Untitled"


def _cleanup_title(title: str, pattern: str | None) -> str:
    if not pattern:
        return title
    return re.sub(pattern, "", title).strip()


def _find_content_root(soup: BeautifulSoup, selectors: list[str]) -> Tag:
    for selector in selectors:
        found = soup.select_one(selector)
        if found and found.get_text(strip=True):
            return found
    return soup.body or soup


def extract_links(root: Tag, base_url: str) -> list[str]:
    links: list[str] = []
    seen: set[str] = set()
    for anchor in root.find_all("a", href=True):
        href = anchor.get("href", "").strip()
        if not href or href.startswith("#") or href.lower().startswith(("mailto:", "javascript:", "tel:")):
            continue
        normalized = normalize_url(href, base_url)
        if normalized and normalized not in seen:
            seen.add(normalized)
            links.append(normalized)
    return links


def parse_html(html: str, url: str, rule: SiteRule | None = None) -> ParsedPage:
    soup = BeautifulSoup(html, "html.parser")
    title_selector = rule.title_selector if rule else None
    title_cleanup = rule.title_cleanup if rule else None
    remove_selectors = list(rule.remove_selectors) if rule and rule.remove_selectors else []

    title = _cleanup_title(_extract_title(soup, title_selector), title_cleanup)

    selectors = (
        rule.content_selectors
        if rule and rule.content_selectors
        else DEFAULT_CONTENT_SELECTORS
    )
    content_root = _find_content_root(soup, selectors)
    _clean_tree(content_root, remove_selectors)

    content_html = "".join(str(child) for child in content_root.children if isinstance(child, (Tag, NavigableString)))
    links = extract_links(content_root, url)
    visible_text = content_root.get_text(separator=" ", strip=True)
    return ParsedPage(
        url=url,
        title=title,
        content_html=content_html,
        links=links,
        text_length=len(visible_text),
    )
