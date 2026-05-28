"""Typer CLI entry point."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from claw.config import Settings, build_crawl_config, build_plan_config, settings
from claw.crawl.crawler import crawl, default_output_dir
from claw.plan.generator import generate_plan

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
    config: Annotated[Optional[Path], typer.Option("--config", help="Path to .claw.toml")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Recursively crawl a requirement page and save Markdown files."""
    cfg_path = _resolve_config(config)
    crawl_cfg = build_crawl_config(
        cfg_path,
        max_depth=depth,
        max_pages=max_pages,
        same_domain_only=False if any_domain else None,
        ignore_robots=ignore_robots,
        no_images=no_images,
    )
    output = out or default_output_dir(settings.cache_dir, url)
    if verbose:
        console.print(f"[dim]Output: {output}[/dim]")
        console.print(f"[dim]Depth: {crawl_cfg.max_depth}, max_pages: {crawl_cfg.max_pages}[/dim]")

    result = asyncio.run(crawl(url, output, crawl_cfg, dry_run=dry_run))

    table = Table(title="Crawl Summary")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Pages", str(result.pages_fetched))
    table.add_row("Errors", str(len(result.manifest.errors)))
    if not dry_run:
        table.add_row("Output", str(result.output_dir))
    console.print(table)

    if result.manifest.errors and verbose:
        for err in result.manifest.errors:
            console.print(f"[red]{err}[/red]")


@app.command("plan")
def plan_cmd(
    cache_dir: Annotated[Path, typer.Argument(help="Directory with crawled markdown + manifest")],
    out: Annotated[Optional[Path], typer.Option("--out", help="Plan output path")] = None,
    repo_context: Annotated[Optional[Path], typer.Option("--repo-context", help="README or repo context file")] = None,
    model: Annotated[Optional[str], typer.Option("--model", help="Chat model override")] = None,
    config: Annotated[Optional[Path], typer.Option("--config", help="Path to .claw.toml")] = None,
) -> None:
    """Generate a code plan from crawled markdown."""
    cfg_path = _resolve_config(config)
    plan_cfg = build_plan_config(cfg_path, model=model)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = out or (settings.plans_dir / f"{timestamp}-plan.md")

    async def _run() -> Path:
        return await generate_plan(
            cache_dir,
            output,
            settings=settings,
            plan_config=plan_cfg,
            repo_context_path=repo_context,
        )

    path = asyncio.run(_run())
    console.print(f"[green]Plan written to[/green] {path}")


@app.command("run")
def run_cmd(
    url: Annotated[str, typer.Argument(help="Root requirement page URL")],
    depth: Annotated[Optional[int], typer.Option("--depth")] = None,
    max_pages: Annotated[Optional[int], typer.Option("--max-pages")] = None,
    out: Annotated[Optional[Path], typer.Option("--out", help="Cache output directory")] = None,
    plan_out: Annotated[Optional[Path], typer.Option("--plan-out", help="Plan output path")] = None,
    repo_context: Annotated[Optional[Path], typer.Option("--repo-context")] = None,
    model: Annotated[Optional[str], typer.Option("--model")] = None,
    config: Annotated[Optional[Path], typer.Option("--config")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Fetch requirements and generate a code plan in one step."""
    cfg_path = _resolve_config(config)
    crawl_cfg = build_crawl_config(cfg_path, max_depth=depth, max_pages=max_pages)
    cache_output = out or default_output_dir(settings.cache_dir, url)

    if verbose:
        console.print(f"[dim]Fetching {url} -> {cache_output}[/dim]")

    result = asyncio.run(crawl(url, cache_output, crawl_cfg))
    console.print(f"[green]Fetched[/green] {result.pages_fetched} pages -> {cache_output}")

    plan_cfg = build_plan_config(cfg_path, model=model)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    plan_path = plan_out or (settings.plans_dir / f"{timestamp}-plan.md")

    async def _plan() -> Path:
        return await generate_plan(
            cache_output,
            plan_path,
            settings=settings,
            plan_config=plan_cfg,
            repo_context_path=repo_context,
        )

    written = asyncio.run(_plan())
    console.print(f"[green]Plan written to[/green] {written}")


if __name__ == "__main__":
    app()
