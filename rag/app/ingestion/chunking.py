from dataclasses import dataclass


@dataclass
class TextChunk:
    text: str
    chunk_index: int
    source: str


def chunk_text(
    text: str,
    source: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be in [0, chunk_size)")

    cleaned = text.strip()
    if not cleaned:
        return []

    chunks: list[TextChunk] = []
    start = 0
    idx = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(TextChunk(text=piece, chunk_index=idx, source=source))
            idx += 1
        if end >= len(cleaned):
            break
        new_start = end - chunk_overlap
        if new_start <= start:
            new_start = end
        start = new_start

    return chunks
