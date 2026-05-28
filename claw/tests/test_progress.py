"""Tests for step logger."""

from io import StringIO
from pathlib import Path

from rich.console import Console

from claw.progress import PlanInputSummary, StepLogger, default_log_path


def test_step_logger_prints_steps():
    buf = StringIO()
    log = StepLogger(Console(file=buf, width=120), verbose=False, enabled=True)
    log.step("加载文档")
    log.info("1000 字符")
    output = buf.getvalue()
    assert "步骤 1" in output
    assert "加载文档" in output
    assert "1000 字符" in output


def test_step_logger_quiet():
    buf = StringIO()
    log = StepLogger(Console(file=buf, width=120), enabled=False)
    log.step("不应显示")
    assert buf.getvalue() == ""


def test_step_logger_verbose_panel():
    buf = StringIO()
    log = StepLogger(Console(file=buf, width=120), verbose=True, enabled=True)
    log.panel("System 提示词", "hello system")
    assert "System 提示词" in buf.getvalue()
    assert "hello system" in buf.getvalue()


def test_plan_input_summary_table():
    buf = StringIO()
    log = StepLogger(Console(file=buf, width=120), verbose=False, enabled=True)
    log.print_plan_inputs(
        PlanInputSummary(
            model="deepseek-chat",
            temperature=0.2,
            api_base="https://api.deepseek.com/v1",
            cache_dir="/cache",
            output_path="/out/plan.md",
            context_source="dir:.claw/context",
            requirements_chars=5000,
            repo_context_chars=1000,
            system_prompt_chars=200,
            user_prompt_chars=6000,
            system_prompt="sys",
            user_prompt="user",
        )
    )
    output = buf.getvalue()
    assert "deepseek-chat" in output
    assert "5,000" in output or "5000" in output


def test_step_logger_writes_log_file(tmp_path: Path):
    buf = StringIO()
    log_path = default_log_path(tmp_path, "plan", "20260528T120000Z")
    log = StepLogger(
        Console(file=buf, width=120),
        verbose=True,
        enabled=True,
        log_path=log_path,
    )
    log.step("加载文档")
    log.info("1000 字符")
    log.panel("System 提示词", "hello system")
    log.success("Plan written to /out/plan.md")
    log.close()

    text = log_path.read_text(encoding="utf-8")
    assert "步骤 1 加载文档" in text
    assert "→ 1000 字符" in text
    assert "=== System 提示词 ===" in text
    assert "hello system" in text
    assert "Plan written to /out/plan.md" in text


def test_step_logger_quiet_skips_log_steps(tmp_path: Path):
    buf = StringIO()
    log_path = default_log_path(tmp_path, "fetch", "20260528T120000Z")
    log = StepLogger(
        Console(file=buf, width=120),
        enabled=False,
        log_path=log_path,
    )
    log.step("不应记录")
    log.success("done")
    log.close()

    assert "不应记录" not in buf.getvalue()
    assert "done" in buf.getvalue()
    text = log_path.read_text(encoding="utf-8")
    assert "不应记录" not in text
    assert "done" in text
