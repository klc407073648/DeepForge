import time
import uuid
from pathlib import Path

from app.config import Settings
from app.generation.llm import OpenAICompatibleClient
from app.ingestion.chunking import TextChunk, chunk_text
from app.ingestion.loaders import load_document, safe_source_name


def _delete_source_chunks(collection: object, source: str) -> None:
    collection.delete(where={"source": source})


async def _embed_in_batches(
    client: OpenAICompatibleClient,
    texts: list[str],
    batch_size: int = 64,
) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        part = await client.embed_texts(batch)
        out.extend(part)
    return out


async def ingest_paths(
    paths: list[Path],
    collection: object,
    client: OpenAICompatibleClient,
    settings: Settings,
) -> tuple[int, list[str], float]:
    t0 = time.perf_counter()
    total_chunks = 0
    sources_out: list[str] = []

    for path in paths:
        if not path.is_file():
            continue
        source = safe_source_name(path.name)
        text = load_document(path)
        chunks: list[TextChunk] = chunk_text(
            text,
            source=source,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        if not chunks:
            continue

        _delete_source_chunks(collection, source)

        texts = [c.text for c in chunks]
        embeddings = await _embed_in_batches(client, texts, batch_size=settings.embedding_ingest_batch_size)

        ids = [f"{uuid.uuid4().hex}_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "source": c.source,
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        total_chunks += len(chunks)
        sources_out.append(source)

    elapsed = time.perf_counter() - t0
    return total_chunks, sources_out, elapsed


async def ingest_documents(
    paths: list[Path],
    collection: object,
    client: OpenAICompatibleClient,
    settings: Settings,
) -> tuple[int, list[str], float]:
    return await ingest_paths(paths, collection, client, settings)

