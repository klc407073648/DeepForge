"""Console step logging for CLI workflows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


@dataclass
class PlanInputSummary:
    model: str
    temperature: float
    api_base: str
    cache_dir: str
    output_path: str
    context_source: str
    requirements_chars: int
    repo_context_chars: int
    system_prompt_chars: int
    user_prompt_chars: int
    system_prompt: str
    user_prompt: str


def default_log_path(logs_dir: Path, command: str, timestamp: str | None = None) -> Path:
    ts = timestamp or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return logs_dir / f"{ts}-{command}.log"


class StepLogger:
    """Print numbered workflow steps; verbose mode shows full LLM inputs."""

    def __init__(
        self,
        console: Console,
        *,
        verbose: bool = False,
        enabled: bool = True,
        log_path: Path | None = None,
    ) -> None:
        self.console = console
        self.verbose = verbose
        self.enabled = enabled
        self.log_path = log_path
        self._step = 0
        self._log_handle = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = log_path.open("w", encoding="utf-8")
            started = datetime.now(UTC).isoformat()
            self._write_log(f"# claw log started at {started}")

    def close(self) -> None:
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def _write_log(self, text: str) -> None:
        if self._log_handle is None:
            return
        self._log_handle.write(text)
        if not text.endswith("\n"):
            self._log_handle.write("\n")
        self._log_handle.flush()

    @staticmethod
    def _render_kv_table(title: str, rows: list[tuple[str, str]]) -> str:
        lines = [f"=== {title} ===", ""]
        if rows:
            width = max(len(key) for key, _ in rows)
            for key, value in rows:
                lines.append(f"{key.ljust(width)}  {value}")
        return "\n".join(lines)

    def step(self, message: str) -> None:
        if not self.enabled:
            return
        self._step += 1
        plain = f"步骤 {self._step} {message}"
        self.console.print(f"[bold cyan]步骤 {self._step}[/bold cyan] {message}")
        self._write_log(plain)

    def info(self, message: str) -> None:
        if not self.enabled:
            return
        self.console.print(f"  [dim]→[/dim] {message}")
        self._write_log(f"  → {message}")

    def panel(self, title: str, content: str, *, max_chars: int = 12_000) -> None:
        if not self.enabled or not self.verbose:
            return
        if len(content) > max_chars:
            body = content[:max_chars] + f"\n\n[dim]... 已截断，共 {len(content):,} 字符[/dim]"
            log_body = content[:max_chars] + f"\n\n... 已截断，共 {len(content):,} 字符"
        else:
            body = content
            log_body = content
        self.console.print(Panel(body, title=title, border_style="blue"))
        self._write_log(f"=== {title} ===\n\n{log_body}")

    def print_kv_table(self, title: str, rows: list[tuple[str, str]]) -> None:
        if not self.enabled:
            return
        table = Table(title=title, show_header=True, header_style="bold")
        table.add_column("项")
        table.add_column("值")
        for key, value in rows:
            table.add_row(key, value)
        self.console.print(table)
        self._write_log(self._render_kv_table(title, rows))

    def print_plan_inputs(self, summary: PlanInputSummary) -> None:
        if not self.enabled:
            return

        rows = [
            ("模型", summary.model),
            ("Temperature", str(summary.temperature)),
            ("API Base", summary.api_base),
            ("需求文档目录", summary.cache_dir),
            ("计划输出", summary.output_path),
            ("仓库上下文来源", summary.context_source),
            ("需求文档字符数", f"{summary.requirements_chars:,}"),
            ("仓库上下文字符数", f"{summary.repo_context_chars:,}"),
            ("System 提示词字符数", f"{summary.system_prompt_chars:,}"),
            ("User 提示词字符数", f"{summary.user_prompt_chars:,}"),
        ]
        self.print_kv_table("LLM 调用参数", rows)

        self.panel("System 提示词", summary.system_prompt)
        self.panel("User 提示词", summary.user_prompt)

    def print_crawl_config(
        self,
        *,
        url: str,
        output: str,
        depth: int,
        max_pages: int,
        same_domain: bool,
        min_content_chars: int,
        dry_run: bool,
    ) -> None:
        if not self.enabled:
            return
        self.print_kv_table(
            "抓取配置",
            [
                ("URL", url),
                ("输出目录", output),
                ("最大深度", str(depth)),
                ("最多页数", str(max_pages)),
                ("仅同域", "是" if same_domain else "否"),
                ("空页阈值", str(min_content_chars)),
                ("Dry run", "是" if dry_run else "否"),
            ],
        )

    def error(self, message: str) -> None:
        if not self.enabled:
            return
        self.console.print(f"[red]{message}[/red]")
        self._write_log(f"ERROR: {message}")

    def success(self, message: str) -> None:
        self.console.print(f"[green]{message}[/green]")
        self._write_log(message)

    def note(self, message: str) -> None:
        self.console.print(f"[dim]{message}[/dim]")
        self._write_log(message)
