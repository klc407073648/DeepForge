"""Site-specific crawl and plan rules loaded from TOML files."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from claw.crawl.link_filter import matches_any_pattern


@dataclass
class SitePlanRule:
    system_prompt_file: str | None = None
    user_prompt_file: str | None = None
    sections: list[str] = field(default_factory=list)


@dataclass
class SiteRule:
    name: str
    match: list[str] = field(default_factory=list)
    priority: int = 0
    content_selectors: list[str] = field(default_factory=list)
    title_selector: str | None = None
    remove_selectors: list[str] = field(default_factory=list)
    include_patterns: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=list)
    min_content_chars: int | None = None
    title_cleanup: str | None = None
    plan: SitePlanRule = field(default_factory=SitePlanRule)


def _parse_site_rule(data: dict[str, Any], fallback_name: str) -> SiteRule:
    crawl = data.get("crawl", {})
    plan_data = data.get("plan", {})
    plan = SitePlanRule(
        system_prompt_file=plan_data.get("system_prompt_file"),
        user_prompt_file=plan_data.get("user_prompt_file"),
        sections=plan_data.get("sections", []),
    )
    match = data.get("match", [])
    if isinstance(match, str):
        match = [match]
    return SiteRule(
        name=data.get("name", fallback_name),
        match=list(match),
        priority=int(data.get("priority", 0)),
        content_selectors=list(crawl.get("content_selectors", [])),
        title_selector=crawl.get("title_selector"),
        remove_selectors=list(crawl.get("remove_selectors", [])),
        include_patterns=list(crawl.get("include_patterns", [])),
        exclude_patterns=list(crawl.get("exclude_patterns", [])),
        min_content_chars=crawl.get("min_content_chars"),
        title_cleanup=crawl.get("title_cleanup"),
        plan=plan,
    )


def load_rules(rules_dir: Path | None = None) -> list[SiteRule]:
    """Load all *.toml rule files from rules_dir (default: .claw/rules)."""
    directory = rules_dir or Path(".claw/rules")
    if not directory.is_dir():
        return []

    rules: list[SiteRule] = []
    for path in sorted(directory.glob("*.toml")):
        if path.name.endswith(".example"):
            continue
        with path.open("rb") as f:
            data = tomllib.load(f)
        rules.append(_parse_site_rule(data, path.stem))
    return rules


def rule_matches_url(rule: SiteRule, url: str) -> bool:
    if not rule.match:
        return False
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    for pattern in rule.match:
        if matches_any_pattern(url, [pattern]) or matches_any_pattern(host, [pattern]):
            return True
    return False


def match_rule(
    url: str,
    rules: list[SiteRule],
    *,
    force_name: str | None = None,
) -> SiteRule | None:
    """Return highest-priority matching rule, or forced rule by name."""
    if force_name:
        for rule in rules:
            if rule.name == force_name:
                return rule
        return None

    sorted_rules = sorted(rules, key=lambda r: r.priority, reverse=True)
    for rule in sorted_rules:
        if rule_matches_url(rule, url):
            return rule
    return None


def apply_site_rule_to_crawl(crawl_cfg: Any, rule: SiteRule | None) -> None:
    """Mutate CrawlConfig in place with site rule overrides."""
    if rule is None:
        return
    if rule.content_selectors:
        crawl_cfg.content_selectors = rule.content_selectors
    if rule.title_selector:
        crawl_cfg.title_selector = rule.title_selector
    if rule.remove_selectors:
        crawl_cfg.remove_selectors = rule.remove_selectors
    if rule.include_patterns:
        crawl_cfg.include_patterns = rule.include_patterns
    if rule.exclude_patterns:
        crawl_cfg.exclude_patterns = rule.exclude_patterns
    if rule.min_content_chars is not None:
        crawl_cfg.min_content_chars = rule.min_content_chars


def apply_site_rule_to_plan(plan_cfg: Any, rule: SiteRule | None) -> None:
    """Mutate PlanConfig in place with site rule plan overrides."""
    if rule is None or not rule.plan:
        return
    p = rule.plan
    if p.system_prompt_file:
        plan_cfg.system_prompt_file = p.system_prompt_file
    if p.user_prompt_file:
        plan_cfg.user_prompt_file = p.user_prompt_file
    if p.sections:
        plan_cfg.sections = p.sections
