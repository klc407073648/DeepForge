"""Tests for target repo context loading."""

from pathlib import Path

import pytest

from claw.plan.aggregator import (
    load_repo_context,
    load_repo_context_dir,
    resolve_repo_context,
)


def test_load_repo_context_dir_empty(tmp_path: Path):
    ctx_dir = tmp_path / "context"
    ctx_dir.mkdir()
    assert load_repo_context_dir(ctx_dir) is None


def test_load_repo_context_dir_skips_example(tmp_path: Path):
    ctx_dir = tmp_path / "context"
    ctx_dir.mkdir()
    (ctx_dir / "README.md.example").write_text("example only", encoding="utf-8")
    (ctx_dir / "README.md").write_text("# Target Project\n\nPortal service.", encoding="utf-8")
    result = load_repo_context_dir(ctx_dir)
    assert result is not None
    assert "Target Project" in result
    assert "example only" not in result


def test_load_repo_context_dir_multiple_files(tmp_path: Path):
    ctx_dir = tmp_path / "context"
    ctx_dir.mkdir()
    (ctx_dir / "README.md").write_text("# README", encoding="utf-8")
    (ctx_dir / "architecture.md").write_text("# Architecture", encoding="utf-8")
    result = load_repo_context_dir(ctx_dir)
    assert "文件: README.md" in result
    assert "文件: architecture.md" in result


def test_resolve_repo_context_file_priority(tmp_path: Path):
    ctx_dir = tmp_path / "context"
    ctx_dir.mkdir()
    (ctx_dir / "README.md").write_text("from dir", encoding="utf-8")
    single = tmp_path / "single.md"
    single.write_text("from file", encoding="utf-8")
    text, source = resolve_repo_context(single, ctx_dir)
    assert text == "from file"
    assert source.startswith("file:")


def test_resolve_repo_context_dir_fallback(tmp_path: Path):
    ctx_dir = tmp_path / "context"
    ctx_dir.mkdir()
    (ctx_dir / "README.md").write_text("from dir", encoding="utf-8")
    text, source = resolve_repo_context(None, ctx_dir)
    assert "from dir" in (text or "")
    assert source.startswith("dir:")


def test_resolve_repo_context_none(tmp_path: Path):
    ctx_dir = tmp_path / "empty"
    ctx_dir.mkdir()
    text, source = resolve_repo_context(None, ctx_dir)
    assert text is None
    assert source == "none"


def test_load_repo_context_truncates(tmp_path: Path):
    p = tmp_path / "big.md"
    p.write_text("x" * 100, encoding="utf-8")
    result = load_repo_context(p, max_chars=50)
    assert result is not None
    assert "[truncated]" in result


def test_load_repo_context_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_repo_context(tmp_path / "missing.md")
