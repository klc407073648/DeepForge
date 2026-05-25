"""知识库文档列表与删除。"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _guess_format(source: str) -> str:
    ext = Path(source).suffix.lower().lstrip(".")
    return ext or "unknown"


def list_indexed_documents(collection: Any) -> list[dict[str, Any]]:
    result = collection.get(include=["metadatas"])
    metadatas = result.get("metadatas") or []
    grouped: dict[str, int] = {}
    for meta in metadatas:
        if not meta:
            continue
        source = str(meta.get("source", "")).strip()
        if not source:
            continue
        grouped[source] = grouped.get(source, 0) + 1

    now = datetime.now(timezone.utc).isoformat()
    docs: list[dict[str, Any]] = []
    for source, chunk_count in sorted(grouped.items()):
        docs.append(
            {
                "source": source,
                "format": _guess_format(source),
                "chunk_count": chunk_count,
                "uploaded_at": now,
                "status": "indexed",
            }
        )
    return docs


def get_document_detail(collection: Any, source: str) -> dict[str, Any] | None:
    result = collection.get(
        where={"source": source},
        include=["documents", "metadatas"],
    )
    ids = result.get("ids") or []
    if not ids:
        return None

    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    chunks: list[dict[str, Any]] = []
    for i, doc_id in enumerate(ids):
        meta = metadatas[i] if i < len(metadatas) else {}
        text = documents[i] if i < len(documents) else ""
        chunks.append(
            {
                "id": doc_id,
                "chunk_index": int(meta.get("chunk_index", 0)),
                "text": text,
                "source": source,
            }
        )
    chunks.sort(key=lambda c: c["chunk_index"])

    return {
        "source": source,
        "format": _guess_format(source),
        "chunk_count": len(chunks),
        "status": "indexed",
        "chunks": chunks,
    }


def delete_document(collection: Any, source: str) -> int:
    before = collection.get(where={"source": source}, include=[])
    ids = before.get("ids") or []
    count = len(ids)
    if count == 0:
        return 0
    collection.delete(where={"source": source})
    return count
