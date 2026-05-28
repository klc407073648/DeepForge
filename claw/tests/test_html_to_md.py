"""Tests for HTML to Markdown conversion."""

from claw.crawl.html_to_md import html_to_markdown, truncate_markdown


def test_html_to_markdown_headings_and_code():
    html = "<h1>Title</h1><p>Hello <strong>world</strong></p><pre><code>print(1)</code></pre>"
    md = html_to_markdown(html)
    assert "# Title" in md
    assert "**world**" in md
    assert "print(1)" in md


def test_truncate_markdown():
    text = "a" * 100
    result = truncate_markdown(text, 50)
    assert len(result) <= 50
    assert "[truncated]" in result
