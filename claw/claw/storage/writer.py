"""Write per-page Markdown files with YAML front matter."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from claw.crawl.html_to_md import html_to_markdown, truncate_markdown
from claw.crawl.parser import ParsedPage
from claw.storage.manifest import PageRecord


def slug_from_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.strip("/") or "index"
    slug = re.sub(r"[^a-zA-Z0-9/_-]+", "-", path)
    slug = slug.replace("/", "__")
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug or "index"


def is_empty_page(text_length: int, min_content_chars: int) -> bool:
    """Return True when visible text is shorter than the minimum threshold."""
    return text_length < min_content_chars


def write_page_markdown(
    output_dir: Path,
    page: ParsedPage,
    *,
    depth: int,
    parent_url: str | None,
    no_images: bool,
    max_content_chars: int,
) -> tuple[Path | None, PageRecord]:
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = slug_from_url(page.url)
    filename = f"{slug}.md"
    counter = 1
    while (output_dir / filename).exists():
        filename = f"{slug}-{counter}.md"
        counter += 1

    body = html_to_markdown(page.content_html, base_url=page.url, no_images=no_images)
    body = truncate_markdown(body, max_content_chars)
    fetched_at = datetime.now(UTC).isoformat()

    safe_title = page.title.replace("\n", " ").replace('"', '\\"')
    front_matter = (
        "---\n"
        f"source_url: {page.url}\n"
        f'title: "{safe_title}"\n'
        f"depth: {depth}\n"
        f"parent_url: {parent_url or ''}\n"
        f"fetched_at: {fetched_at}\n"
        "---\n\n"
    )
    content = front_matter + f"# {page.title}\n\n{body}\n"
    path = output_dir / filename
    path.write_text(content, encoding="utf-8")

    record = PageRecord(
        path=filename,
        source_url=page.url,
        title=page.title,
        depth=depth,
        parent_url=parent_url,
        links_to=page.links,
        status="ok",
        text_length=page.text_length,
    )
    return path, record
