"""Tests for URL normalization and link filtering."""

from claw.crawl.link_filter import LinkFilter, normalize_url, registrable_domain


def test_normalize_url_strips_fragment():
    assert normalize_url("https://Example.com/path/#section") == "https://example.com/path"


def test_normalize_url_resolves_relative():
    base = "https://example.com/spec/"
    assert normalize_url("feature-a.html", base) == "https://example.com/spec/feature-a.html"


def test_registrable_domain():
    assert registrable_domain("www.example.com") == "example.com"


def test_link_filter_same_domain():
    root = "https://example.com/spec"
    filt = LinkFilter(root, max_depth=2, same_domain_only=True, exclude_patterns=[], include_patterns=[])
    visited: set[str] = set()
    assert filt.should_follow("https://example.com/spec/a", 1, visited)
    assert not filt.should_follow("https://other.com/page", 1, visited)


def test_link_filter_exclude_pattern():
    root = "https://example.com/"
    filt = LinkFilter(
        root,
        max_depth=2,
        same_domain_only=True,
        exclude_patterns=["*.pdf"],
        include_patterns=[],
    )
    visited: set[str] = set()
    assert not filt.should_follow("https://example.com/doc.pdf", 1, visited)


def test_link_filter_max_depth():
    root = "https://example.com/"
    filt = LinkFilter(root, max_depth=2, same_domain_only=True, exclude_patterns=[], include_patterns=[])
    visited: set[str] = set()
    assert filt.should_follow("https://example.com/a", 2, visited)
    assert not filt.should_follow("https://example.com/a", 3, visited)
