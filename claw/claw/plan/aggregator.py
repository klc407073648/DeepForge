"""Aggregate crawled Markdown for plan generation."""

from __future__ import annotations

from pathlib import Path

from claw.storage.manifest import Manifest


def load_repo_context(path: Path | None) -> str | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"Repo context not found: {path}")
    text = path.read_text(encoding="utf-8")
    if len(text) > 20_000:
        return text[:20_000] + "\n\n[truncated]"
    return text


def aggregate_requirements(cache_dir: Path, max_chars: int) -> str:
    manifest = Manifest.load(cache_dir)
    pages = sorted(
        [p for p in manifest.pages if p.status == "ok" and p.path],
        key=lambda p: (p.depth, p.path),
    )

    sections: list[str] = []
    total = 0
    for page in pages:
        md_path = cache_dir / page.path
        if not md_path.is_file():
            continue
        content = md_path.read_text(encoding="utf-8")
        header = f"<!-- source: {page.source_url} | depth: {page.depth} -->\n"
        block = header + content + "\n"
        if total + len(block) > max_chars:
            remaining = max_chars - total
            if remaining > 500:
                sections.append(block[:remaining] + "\n\n[truncated remaining pages]\n")
            break
        sections.append(block)
        total += len(block)

    if not sections:
        raise ValueError(f"No markdown pages found in {cache_dir}")

    index_lines = ["# 需求文档索引", ""]
    for page in pages:
        index_lines.append(f"- depth {page.depth}: {page.title} ({page.source_url})")
    index = "\n".join(index_lines) + "\n\n---\n\n"
    combined = index + "\n\n---\n\n".join(sections)
    if len(combined) > max_chars:
        return combined[:max_chars] + "\n\n[truncated]"
    return combined
