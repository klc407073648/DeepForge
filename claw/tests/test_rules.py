"""Tests for site rule loading and matching."""

from pathlib import Path

from claw.config import CrawlConfig
from claw.crawl.rules import (
    apply_site_rule_to_crawl,
    load_rules,
    match_rule,
    rule_matches_url,
)


def test_load_rules_from_fixtures():
    rules_dir = Path(__file__).parent / "fixtures" / "rules"
    rules = load_rules(rules_dir)
    assert len(rules) == 1
    assert rules[0].name == "testsite"


def test_match_rule_by_host():
    rules_dir = Path(__file__).parent / "fixtures" / "rules"
    rules = load_rules(rules_dir)
    matched = match_rule("https://example.com/article/1", rules)
    assert matched is not None
    assert matched.name == "testsite"


def test_match_rule_force_name():
    rules_dir = Path(__file__).parent / "fixtures" / "rules"
    rules = load_rules(rules_dir)
    matched = match_rule("https://other.com/", rules, force_name="testsite")
    assert matched is not None
    assert matched.name == "testsite"


def test_rule_matches_url_host_glob():
    from claw.crawl.rules import SiteRule

    rule = SiteRule(name="j", match=["*.example.com"])
    assert rule_matches_url(rule, "https://docs.example.com/page")


def test_apply_site_rule_overrides_crawl_config():
    rules_dir = Path(__file__).parent / "fixtures" / "rules"
    rules = load_rules(rules_dir)
    matched = match_rule("https://example.com/", rules)
    cfg = CrawlConfig()
    apply_site_rule_to_crawl(cfg, matched)
    assert cfg.content_selectors == [".article-content"]
    assert cfg.include_patterns == ["*/article/*"]
    assert cfg.min_content_chars == 50
