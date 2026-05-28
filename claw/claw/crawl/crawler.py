"""BFS recursive crawler."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx

from claw.config import CrawlConfig
from claw.crawl.fetcher import Fetcher
from claw.crawl.link_filter import LinkFilter, normalize_url
from claw.crawl.parser import parse_html
from claw.storage.manifest import Manifest, PageRecord, new_manifest
from claw.storage.writer import is_empty_page, write_page_markdown


@dataclass
class CrawlTask:
    url: str
    depth: int
    parent_url: str | None


@dataclass
class CrawlResult:
    output_dir: Path
    manifest: Manifest
    pages_fetched: int
    pages_saved: int = 0
    pages_empty: int = 0


async def crawl(
    root_url: str,
    output_dir: Path,
    config: CrawlConfig,
    *,
    dry_run: bool = False,
) -> CrawlResult:
    normalized_root = normalize_url(root_url)
    if not normalized_root:
        raise ValueError(f"Invalid URL: {root_url}")

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    manifest = new_manifest(normalized_root)

    link_filter = LinkFilter(
        normalized_root,
        max_depth=config.max_depth,
        same_domain_only=config.same_domain_only,
        exclude_patterns=config.exclude_patterns,
        include_patterns=config.include_patterns,
    )

    visited: set[str] = set()
    queue: deque[CrawlTask] = deque([CrawlTask(normalized_root, 0, None)])
    pages_fetched = 0
    pages_saved = 0
    pages_empty = 0
    fetcher = Fetcher(config)
    semaphore = asyncio.Semaphore(config.max_concurrency)

    async with httpx.AsyncClient() as client:

        async def process(task: CrawlTask) -> None:
            nonlocal pages_fetched, pages_saved, pages_empty
            async with semaphore:
                try:
                    result = await fetcher.fetch(client, task.url)
                    parsed = parse_html(result.html, result.final_url)
                    pages_fetched += 1

                    if dry_run:
                        status = "empty" if is_empty_page(parsed.text_length, config.min_content_chars) else "dry-run"
                        manifest.add_page(
                            PageRecord(
                                path="",
                                source_url=task.url,
                                title=parsed.title,
                                depth=task.depth,
                                parent_url=task.parent_url,
                                links_to=parsed.links,
                                status=status,
                                text_length=parsed.text_length,
                            )
                        )
                        if status == "empty":
                            pages_empty += 1
                        else:
                            next_depth = task.depth + 1
                            for link in parsed.links:
                                if link_filter.should_follow(link, next_depth, visited):
                                    queue.append(CrawlTask(link, next_depth, task.url))
                        return

                    if is_empty_page(parsed.text_length, config.min_content_chars):
                        pages_empty += 1
                        manifest.add_page(
                            PageRecord(
                                path="",
                                source_url=task.url,
                                title=parsed.title,
                                depth=task.depth,
                                parent_url=task.parent_url,
                                links_to=parsed.links,
                                status="empty",
                                text_length=parsed.text_length,
                            )
                        )
                        return

                    _, record = write_page_markdown(
                        output_dir,
                        parsed,
                        depth=task.depth,
                        parent_url=task.parent_url,
                        no_images=config.no_images,
                        max_content_chars=config.max_content_chars,
                    )
                    manifest.add_page(record)
                    pages_saved += 1

                    next_depth = task.depth + 1
                    for link in parsed.links:
                        if link_filter.should_follow(link, next_depth, visited):
                            queue.append(CrawlTask(link, next_depth, task.url))
                except Exception as exc:
                    manifest.add_error(f"{task.url}: {exc}")
                    manifest.add_page(
                        PageRecord(
                            path="",
                            source_url=task.url,
                            title="",
                            depth=task.depth,
                            parent_url=task.parent_url,
                            status="failed",
                            error=str(exc),
                        )
                    )

        while queue and pages_fetched < config.max_pages:
            batch: list[CrawlTask] = []
            while queue and len(batch) < config.max_concurrency and pages_fetched + len(batch) < config.max_pages:
                task = queue.popleft()
                normalized = normalize_url(task.url, normalized_root)
                if not normalized or normalized in visited:
                    continue
                if task.depth > 0 and not link_filter.should_follow(normalized, task.depth, visited):
                    continue
                visited.add(normalized)
                batch.append(CrawlTask(normalized, task.depth, task.parent_url))

            if not batch:
                continue
            await asyncio.gather(*(process(t) for t in batch))

    if not dry_run:
        manifest.save(output_dir)
        if manifest.errors:
            (output_dir / "errors.log").write_text("\n".join(manifest.errors) + "\n", encoding="utf-8")

    return CrawlResult(
        output_dir=output_dir,
        manifest=manifest,
        pages_fetched=pages_fetched,
        pages_saved=pages_saved,
        pages_empty=pages_empty,
    )


def default_output_dir(base: Path, url: str, run_id: str | None = None) -> Path:
    from datetime import UTC, datetime

    parsed = urlparse(url)
    host = parsed.netloc.replace(":", "_") or "unknown"
    run = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return base / host / run
