"""Manifest models for crawl output."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class PageRecord:
    path: str
    source_url: str
    title: str
    depth: int
    parent_url: str | None
    links_to: list[str] = field(default_factory=list)
    status: str = "ok"
    error: str | None = None
    text_length: int = 0


@dataclass
class Manifest:
    root_url: str
    created_at: str
    pages: list[PageRecord] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def add_page(self, page: PageRecord) -> None:
        self.pages.append(page)

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def save(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "manifest.json"
        payload = {
            "root_url": self.root_url,
            "created_at": self.created_at,
            "pages": [asdict(p) for p in self.pages],
            "errors": self.errors,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, output_dir: Path) -> Manifest:
        path = output_dir / "manifest.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        pages = [PageRecord(**p) for p in data.get("pages", [])]
        return cls(
            root_url=data["root_url"],
            created_at=data.get("created_at", ""),
            pages=pages,
            errors=data.get("errors", []),
        )


def new_manifest(root_url: str) -> Manifest:
    return Manifest(
        root_url=root_url,
        created_at=datetime.now(UTC).isoformat(),
    )
