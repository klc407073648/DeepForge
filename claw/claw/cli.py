"""Typer CLI entry point."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from claw.config import Settings, build_crawl_config, build_plan_config, settings
from claw.crawl.crawler import crawl, default_output_dir
from claw.plan.generator import generate_plan
from claw.progress import StepLogger, default_log_path

app = typer.Typer(
    name="claw",
    help="Crawl requirement pages and generate code plans.",
    no_args_is_help=True,
)
console = Console()


def _resolve_config(config: Path | None) -> Path | None:
    if config is not None:
        return config
    default = Path(".claw.toml")
    return default if default.is_file() else None


def _utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _make_logger(
    *,
    command: str,
    verbose: bool,
    quiet: bool,
    timestamp: str | None = None,
) -> StepLogger:
    return StepLogger(
        console,
        verbose=verbose,
        enabled=not quiet,
        log_path=default_log_path(settings.logs_dir, command, timestamp),
    )


@app.command("fetch")
def fetch_cmd(
    url: Annotated[str, typer.Argument(help="Root requirement page URL")],
    out: Annotated[Optional[Path], typer.Option("--out", help="Output directory")] = None,
    depth: Annotated[Optional[int], typer.Option("--depth", help="Max crawl depth")] = None,
    max_pages: Annotated[Optional[int], typer.Option("--max-pages", help="Max pages to fetch")] = None,
    any_domain: Annotated[bool, typer.Option("--any-domain", help="Follow links on other domains")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Discover links without writing files")] = False,
    ignore_robots: Annotated[bool, typer.Option("--ignore-robots", help="Ignore robots.txt")] = False,
    no_images: Annotated[bool, typer.Option("--no-images", help="Strip images from markdown")] = False,
    min_content_chars: Annotated[
        Optional[int], typer.Option("--min-content-chars", help="Skip pages with fewer visible text chars")
    ] = None,
    config: Annotated[Optional[Path], typer.Option("--config", help="Path to .claw.toml")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Print steps and detailed inputs")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Suppress step output")] = False,
) -> None:
    """Recursively crawl a requirement page and save Markdown files."""
    timestamp = _utc_timestamp()
    log = _make_logger(command="fetch", verbose=verbose, quiet=quiet, timestamp=timestamp)
    try:
        cfg_path = _resolve_config(config)
        crawl_cfg = build_crawl_config(
            cfg_path,
            max_depth=depth,
            max_pages=max_pages,
            same_domain_only=False if any_domain else None,
            ignore_robots=ignore_robots,
            no_images=no_images,
            min_content_chars=min_content_chars,
        )
        output = out or default_output_dir(settings.cache_dir, url, run_id=timestamp)

        log.step("准备抓取配置")
        log.print_crawl_config(
            url=url,
            output=str(output),
            depth=crawl_cfg.max_depth,
            max_pages=crawl_cfg.max_pages,
            same_domain=crawl_cfg.same_domain_only,
            min_content_chars=crawl_cfg.min_content_chars,
            dry_run=dry_run,
        )

        log.step("递归抓取页面")
        result = asyncio.run(crawl(url, output, crawl_cfg, dry_run=dry_run))

        log.step("抓取完成")
        rows = [
            ("Pages fetched", str(result.pages_fetched)),
            ("Pages saved", str(result.pages_saved)),
            ("Pages empty", str(result.pages_empty)),
            ("Errors", str(len(result.manifest.errors))),
        ]
        if not dry_run:
            rows.append(("Output", str(result.output_dir)))
        log.print_kv_table("Crawl Summary", rows)

        if result.manifest.errors and verbose:
            for err in result.manifest.errors:
                log.error(err)

        if log.log_path is not None:
            log.note(f"Log written to {log.log_path}")
    finally:
        log.close()


@app.command("plan")
def plan_cmd(
    cache_dir: Annotated[Path, typer.Argument(help="Directory with crawled markdown + manifest")],
    out: Annotated[Optional[Path], typer.Option("--out", help="Plan output path")] = None,
    repo_context: Annotated[
        Optional[Path],
        typer.Option("--repo-context", help="Single file for target repo context (overrides .claw/context/)"),
    ] = None,
    context_dir: Annotated[
        Optional[Path], typer.Option("--context-dir", help="Directory of target project context markdown files")
    ] = None,
    model: Annotated[Optional[str], typer.Option("--model", help="Chat model override")] = None,
    system_prompt: Annotated[Optional[Path], typer.Option("--system-prompt", help="Custom system prompt file")] = None,
    user_prompt: Annotated[Optional[Path], typer.Option("--user-prompt", help="Custom user prompt template file")] = None,
    config: Annotated[Optional[Path], typer.Option("--config", help="Path to .claw.toml")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Print full system/user prompts")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Suppress step output")] = False,
) -> None:
    """Generate a code plan from crawled markdown."""
    timestamp = _utc_timestamp()
    log = _make_logger(command="plan", verbose=verbose, quiet=quiet, timestamp=timestamp)
    try:
        cfg_path = _resolve_config(config)
        plan_cfg = build_plan_config(cfg_path, app_settings=settings, model=model, context_dir=context_dir)
        output = out or (settings.plans_dir / f"{timestamp}-plan.md")

        log.step("初始化计划生成")
        log.info(f"模型: {plan_cfg.model}，temperature: {plan_cfg.temperature}")

        async def _run() -> Path:
            return await generate_plan(
                cache_dir,
                output,
                settings=settings,
                plan_config=plan_cfg,
                repo_context_path=repo_context,
                system_prompt_path=str(system_prompt) if system_prompt else None,
                user_prompt_path=str(user_prompt) if user_prompt else None,
                logger=log,
            )

        path = asyncio.run(_run())
        log.success(f"Plan written to {path}")
        if log.log_path is not None:
            log.note(f"Log written to {log.log_path}")
    finally:
        log.close()


@app.command("run")
def run_cmd(
    url: Annotated[str, typer.Argument(help="Root requirement page URL")],
    depth: Annotated[Optional[int], typer.Option("--depth")] = None,
    max_pages: Annotated[Optional[int], typer.Option("--max-pages")] = None,
    out: Annotated[Optional[Path], typer.Option("--out", help="Cache output directory")] = None,
    plan_out: Annotated[Optional[Path], typer.Option("--plan-out", help="Plan output path")] = None,
    repo_context: Annotated[
        Optional[Path],
        typer.Option("--repo-context", help="Single file for target repo context (overrides .claw/context/)"),
    ] = None,
    context_dir: Annotated[Optional[Path], typer.Option("--context-dir", help="Target project context directory")] = None,
    model: Annotated[Optional[str], typer.Option("--model")] = None,
    system_prompt: Annotated[Optional[Path], typer.Option("--system-prompt")] = None,
    user_prompt: Annotated[Optional[Path], typer.Option("--user-prompt")] = None,
    config: Annotated[Optional[Path], typer.Option("--config", help="Path to .claw.toml")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Print full system/user prompts")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Suppress step output")] = False,
) -> None:
    """Fetch requirements and generate a code plan in one step."""
    timestamp = _utc_timestamp()
    log = _make_logger(command="run", verbose=verbose, quiet=quiet, timestamp=timestamp)
    try:
        cfg_path = _resolve_config(config)
        crawl_cfg = build_crawl_config(cfg_path, max_depth=depth, max_pages=max_pages)
        cache_output = out or default_output_dir(settings.cache_dir, url, run_id=timestamp)
        plan_cfg = build_plan_config(cfg_path, app_settings=settings, model=model, context_dir=context_dir)
        plan_path = plan_out or (settings.plans_dir / f"{timestamp}-plan.md")

        log.step("[1/2] 准备抓取需求页面")
        log.print_crawl_config(
            url=url,
            output=str(cache_output),
            depth=crawl_cfg.max_depth,
            max_pages=crawl_cfg.max_pages,
            same_domain=crawl_cfg.same_domain_only,
            min_content_chars=crawl_cfg.min_content_chars,
            dry_run=False,
        )

        log.step("[1/2] 递归抓取页面")
        result = asyncio.run(crawl(url, cache_output, crawl_cfg))
        log.info(
            f"抓取 {result.pages_fetched} 页（保存 {result.pages_saved}，空页 {result.pages_empty}）"
        )

        log.step("[2/2] 初始化计划生成")
        log.info(f"模型: {plan_cfg.model}，输出: {plan_path}")

        async def _plan() -> Path:
            return await generate_plan(
                cache_output,
                plan_path,
                settings=settings,
                plan_config=plan_cfg,
                repo_context_path=repo_context,
                system_prompt_path=str(system_prompt) if system_prompt else None,
                user_prompt_path=str(user_prompt) if user_prompt else None,
                logger=log,
            )

        written = asyncio.run(_plan())
        log.success(f"Plan written to {written}")
        if log.log_path is not None:
            log.note(f"Log written to {log.log_path}")
    finally:
        log.close()


if __name__ == "__main__":
    app()
