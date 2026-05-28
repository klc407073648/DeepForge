"""HTML parsing: title, main content, and outbound links."""

from __future__ import annotations

from dataclasses import dataclass

from bs4 import BeautifulSoup, NavigableString, Tag

from claw.crawl.link_filter import normalize_url

REMOVE_TAGS = {"script", "style", "nav", "footer", "header", "noscript", "iframe", "svg"}
CONTENT_SELECTORS = ["main", "article", "[role=main]", ".markdown-body", "#content", "body"]


@dataclass
class ParsedPage:
    url: str
    title: str
    content_html: str
    links: list[str]


def _clean_tree(node: Tag) -> None:
    for tag in node.find_all(REMOVE_TAGS):
        tag.decompose()


def _extract_title(soup: BeautifulSoup) -> str:
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    if h1:
        text = h1.get_text(strip=True)
        if text:
            return text
    return "Untitled"


def _find_content_root(soup: BeautifulSoup) -> Tag:
    for selector in CONTENT_SELECTORS:
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


def parse_html(html: str, url: str) -> ParsedPage:
    soup = BeautifulSoup(html, "html.parser")
    title = _extract_title(soup)
    content_root = _find_content_root(soup)
    _clean_tree(content_root)

    # Clone content to avoid mutating shared soup references.
    content_html = "".join(str(child) for child in content_root.children if isinstance(child, (Tag, NavigableString)))
    links = extract_links(content_root, url)
    return ParsedPage(url=url, title=title, content_html=content_html, links=links)
