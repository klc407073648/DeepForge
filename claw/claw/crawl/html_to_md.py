"""Convert HTML fragments to Markdown."""

from __future__ import annotations

from markdownify import markdownify as md


def html_to_markdown(html: str, *, base_url: str = "", no_images: bool = False) -> str:
    options = {
        "heading_style": "ATX",
        "bullets": "-",
        "strip": ["script", "style"],
    }
    if no_images:
        options["convert"] = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "table", "tr", "td", "th", "pre", "code", "blockquote", "a", "strong", "em", "br", "div", "span"]

    text = md(html, **options)
    lines = [line.rstrip() for line in text.splitlines()]
    cleaned: list[str] = []
    blank = False
    for line in lines:
        if not line.strip():
            if not blank:
                cleaned.append("")
            blank = True
        else:
            cleaned.append(line)
            blank = False
    return "\n".join(cleaned).strip()


def truncate_markdown(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - len("\n\n[truncated]")] + "\n\n[truncated]"
