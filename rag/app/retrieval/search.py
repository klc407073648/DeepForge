"""把用户问题变成向量 → 在 Chroma 里做相似度检索 → 变成一条条带元数据的片段。"""
from dataclasses import dataclass
from typing import Any

from app.config import Settings
from app.generation.llm import OpenAICompatibleClient


@dataclass
class RetrievedChunk:
    id: str
    text: str
    source: str
    chunk_index: int
    distance: float

async def search_relevant_chunks(
    question: str,
    collection: Any,
    client: OpenAICompatibleClient,
    settings: Settings,
) -> list[RetrievedChunk]:
    vectors = await client.embed_texts([question])
    if not vectors:
        return []

    res = collection.query(
        query_embeddings=vectors,
        n_results=settings.top_k,
        include=["documents", "metadatas", "distances"],
    )

    ids = res.get("ids", [[]])[0] or []
    documents = res.get("documents", [[]])[0] or []
    metadatas = res.get("metadatas", [[]])[0] or []
    distances = res.get("distances", [[]])[0] or []

    out: list[RetrievedChunk] = []
    for i, doc_id in enumerate(ids):
        meta = metadatas[i] or {}
        text = documents[i] if i < len(documents) else ""
        dist = float(distances[i]) if i < len(distances) else 0.0
        out.append(
            RetrievedChunk(
                id=str(doc_id),
                text=text or "",
                source=str(meta.get("source", "")),
                chunk_index=int(meta.get("chunk_index", 0)),
                distance=dist,
            )
        )
    return out
