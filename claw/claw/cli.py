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
from claw.crawl.rules import load_rules, match_rule
from claw.plan.generator import generate_plan
from claw.storage.manifest import Manifest

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


def _resolve_site_rule_for_cache(
    cache_dir: Path,
    rules_dir: Path | None,
    force_rule: str | None = None,
):
    rules = load_rules(rules_dir)
    if force_rule:
        return match_rule("", rules, force_name=force_rule)
    try:
        manifest = Manifest.load(cache_dir)
    except Exception:
        return None
    if manifest.matched_rule:
        return match_rule("", rules, force_name=manifest.matched_rule)
    return match_rule(manifest.root_url, rules)


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
    rule: Annotated[Optional[str], typer.Option("--rule", help="Force a named site rule")] = None,
    rules_dir: Annotated[Optional[Path], typer.Option("--rules-dir", help="Directory of site rule TOML files")] = None,
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
        min_content_chars=min_content_chars,
        rules_dir=rules_dir,
    )
    output = out or default_output_dir(settings.cache_dir, url)
    if verbose:
        console.print(f"[dim]Output: {output}[/dim]")
        console.print(f"[dim]Depth: {crawl_cfg.max_depth}, max_pages: {crawl_cfg.max_pages}[/dim]")

    result = asyncio.run(
        crawl(
            url,
            output,
            crawl_cfg,
            dry_run=dry_run,
            force_rule_name=rule,
            rules_dir=rules_dir,
        )
    )

    if verbose and result.matched_rule:
        console.print(f"[dim]Matched rule: {result.matched_rule}[/dim]")

    table = Table(title="Crawl Summary")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Pages fetched", str(result.pages_fetched))
    table.add_row("Pages saved", str(result.pages_saved))
    table.add_row("Pages empty", str(result.pages_empty))
    table.add_row("Errors", str(len(result.manifest.errors)))
    if result.matched_rule:
        table.add_row("Matched rule", result.matched_rule)
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
    system_prompt: Annotated[Optional[Path], typer.Option("--system-prompt", help="Custom system prompt file")] = None,
    user_prompt: Annotated[Optional[Path], typer.Option("--user-prompt", help="Custom user prompt template file")] = None,
    rule: Annotated[Optional[str], typer.Option("--rule", help="Force a named site rule for plan prompts")] = None,
    rules_dir: Annotated[Optional[Path], typer.Option("--rules-dir", help="Directory of site rule TOML files")] = None,
    config: Annotated[Optional[Path], typer.Option("--config", help="Path to .claw.toml")] = None,
) -> None:
    """Generate a code plan from crawled markdown."""
    cfg_path = _resolve_config(config)
    plan_cfg = build_plan_config(cfg_path, app_settings=settings, model=model)
    site_rule = _resolve_site_rule_for_cache(cache_dir, rules_dir, force_rule=rule)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = out or (settings.plans_dir / f"{timestamp}-plan.md")

    async def _run() -> Path:
        return await generate_plan(
            cache_dir,
            output,
            settings=settings,
            plan_config=plan_cfg,
            repo_context_path=repo_context,
            site_rule=site_rule,
            system_prompt_path=str(system_prompt) if system_prompt else None,
            user_prompt_path=str(user_prompt) if user_prompt else None,
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
    system_prompt: Annotated[Optional[Path], typer.Option("--system-prompt")] = None,
    user_prompt: Annotated[Optional[Path], typer.Option("--user-prompt")] = None,
    rule: Annotated[Optional[str], typer.Option("--rule", help="Force a named site rule")] = None,
    rules_dir: Annotated[Optional[Path], typer.Option("--rules-dir")] = None,
    config: Annotated[Optional[Path], typer.Option("--config")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Fetch requirements and generate a code plan in one step."""
    cfg_path = _resolve_config(config)
    crawl_cfg = build_crawl_config(cfg_path, max_depth=depth, max_pages=max_pages, rules_dir=rules_dir)
    cache_output = out or default_output_dir(settings.cache_dir, url)

    if verbose:
        console.print(f"[dim]Fetching {url} -> {cache_output}[/dim]")

    result = asyncio.run(
        crawl(url, cache_output, crawl_cfg, force_rule_name=rule, rules_dir=rules_dir)
    )
    if verbose and result.matched_rule:
        console.print(f"[dim]Matched rule: {result.matched_rule}[/dim]")
    console.print(
        f"[green]Fetched[/green] {result.pages_fetched} pages "
        f"({result.pages_saved} saved, {result.pages_empty} empty) -> {cache_output}"
    )

    plan_cfg = build_plan_config(cfg_path, app_settings=settings, model=model)
    site_rule = match_rule(url, load_rules(rules_dir or crawl_cfg.rules_dir), force_name=rule)
    if site_rule is None and result.matched_rule:
        site_rule = match_rule("", load_rules(rules_dir or crawl_cfg.rules_dir), force_name=result.matched_rule)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    plan_path = plan_out or (settings.plans_dir / f"{timestamp}-plan.md")

    async def _plan() -> Path:
        return await generate_plan(
            cache_output,
            plan_path,
            settings=settings,
            plan_config=plan_cfg,
            repo_context_path=repo_context,
            site_rule=site_rule,
            system_prompt_path=str(system_prompt) if system_prompt else None,
            user_prompt_path=str(user_prompt) if user_prompt else None,
        )

    written = asyncio.run(_plan())
    console.print(f"[green]Plan written to[/green] {written}")


if __name__ == "__main__":
    app()
