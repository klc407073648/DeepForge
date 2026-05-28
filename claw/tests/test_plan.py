"""Tests for plan prompt assembly."""

from pathlib import Path

import pytest

from claw.config import Settings, build_plan_config
from claw.plan.aggregator import aggregate_requirements
from claw.plan.prompts import build_user_prompt
from claw.storage.manifest import Manifest, PageRecord, new_manifest


def test_build_user_prompt_includes_sections():
    prompt = build_user_prompt("# Req\n\ncontent", repo_context="# README")
    assert "需求文档" in prompt
    assert "目标仓库上下文" in prompt
    assert "分步实施任务" in prompt


def test_aggregate_requirements(tmp_path: Path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "page.md").write_text("# Page\n\nBody", encoding="utf-8")
    manifest = new_manifest("https://example.com/")
    manifest.add_page(
        PageRecord(
            path="page.md",
            source_url="https://example.com/",
            title="Page",
            depth=0,
            parent_url=None,
        )
    )
    manifest.save(cache)

    combined = aggregate_requirements(cache, max_chars=10_000)
    assert "Page" in combined
    assert "Body" in combined


def test_aggregate_requirements_empty_raises(tmp_path: Path):
    cache = tmp_path / "empty"
    cache.mkdir()
    manifest = Manifest(root_url="https://example.com/", created_at="now", pages=[])
    manifest.save(cache)
    with pytest.raises(ValueError):
        aggregate_requirements(cache, max_chars=1000)


def test_build_plan_config_uses_env_model(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CLAW_CHAT_MODEL", "deepseek-chat")
    cfg = Settings()
    plan_cfg = build_plan_config(app_settings=cfg)
    assert plan_cfg.model == "deepseek-chat"
