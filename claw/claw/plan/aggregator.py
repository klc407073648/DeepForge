"""Aggregate crawled Markdown for plan generation."""

from __future__ import annotations

from pathlib import Path

from claw.storage.manifest import Manifest


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[truncated]"


def load_repo_context(path: Path | None, *, max_chars: int = 20_000) -> str | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"Repo context not found: {path}")
    text = path.read_text(encoding="utf-8")
    return _truncate(text, max_chars)


def load_repo_context_dir(context_dir: Path, *, max_chars: int = 20_000) -> str | None:
    if not context_dir.is_dir():
        return None

    md_files = sorted(
        p for p in context_dir.glob("*.md") if p.is_file() and not p.name.endswith(".example")
    )
    if not md_files:
        return None

    parts: list[str] = []
    total = 0
    for md_path in md_files:
        content = md_path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        block = f"# 文件: {md_path.name}\n\n{content}\n"
        if total + len(block) > max_chars:
            remaining = max_chars - total
            if remaining > 100:
                parts.append(block[:remaining] + "\n\n[truncated]")
            break
        parts.append(block)
        total += len(block)

    if not parts:
        return None
    return "\n---\n\n".join(parts)


def resolve_repo_context(
    single_file: Path | None,
    context_dir: Path,
    *,
    max_chars: int = 20_000,
) -> tuple[str | None, str]:
    """Return (context text, source description)."""
    if single_file is not None:
        return load_repo_context(single_file, max_chars=max_chars), f"file:{single_file}"

    dir_context = load_repo_context_dir(context_dir, max_chars=max_chars)
    if dir_context is not None:
        return dir_context, f"dir:{context_dir}"

    return None, "none"


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
