"""Tests for custom prompt loading and rendering."""

from pathlib import Path

import pytest

from claw.plan.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    build_user_prompt,
    load_prompt_file,
    render_template,
    resolve_system_prompt,
    resolve_user_prompt_template,
)


def test_render_template():
    result = render_template("Hello {{name}}!", {"name": "Claw"})
    assert result == "Hello Claw!"


def test_build_user_prompt_with_custom_template():
    tpl = "## Req\n\n{{requirements}}\n\n{{repo_context_block}}"
    prompt = build_user_prompt("# Doc", repo_context="# README", template=tpl)
    assert "# Doc" in prompt
    assert "目标仓库上下文" in prompt


def test_load_prompt_file(tmp_path: Path):
    p = tmp_path / "system.md"
    p.write_text("Custom system", encoding="utf-8")
    assert load_prompt_file(p) == "Custom system"


def test_resolve_system_prompt_priority(tmp_path: Path):
    cli_file = tmp_path / "cli.md"
    cfg_file = tmp_path / "cfg.md"
    cli_file.write_text("CLI prompt", encoding="utf-8")
    cfg_file.write_text("Config prompt", encoding="utf-8")
    assert resolve_system_prompt(cli_path=str(cli_file), config_path=str(cfg_file)) == "CLI prompt"


def test_resolve_system_prompt_default():
    assert resolve_system_prompt() == DEFAULT_SYSTEM_PROMPT


def test_resolve_user_prompt_template(tmp_path: Path):
    p = tmp_path / "user.md"
    p.write_text("{{requirements}}", encoding="utf-8")
    assert resolve_user_prompt_template(cli_path=str(p)) == "{{requirements}}"


def test_load_prompt_file_missing():
    with pytest.raises(FileNotFoundError):
        load_prompt_file("/nonexistent/prompt.md")
