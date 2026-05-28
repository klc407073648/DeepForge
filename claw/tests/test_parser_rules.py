"""Tests for parser with site rules."""

from pathlib import Path

from claw.crawl.parser import parse_html
from claw.crawl.rules import load_rules, match_rule


def test_parse_html_with_site_rule():
    rules_dir = Path(__file__).parent / "fixtures" / "rules"
    rules = load_rules(rules_dir)
    rule = match_rule("https://example.com/article/1", rules)
    html = (Path(__file__).parent / "fixtures" / "juejin-article.html").read_text(encoding="utf-8")
    parsed = parse_html(html, "https://example.com/article/1", rule)
    assert parsed.title == "Article Title"
    assert "main article body" in parsed.content_html or parsed.text_length > 50
    assert "sidebar" not in parsed.content_html.lower() or parsed.text_length < 500
