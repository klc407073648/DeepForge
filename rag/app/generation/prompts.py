from app.config import Settings
from app.retrieval.search import RetrievedChunk


def build_context_block(
    chunks: list[RetrievedChunk],
    settings: Settings,
) -> tuple[str, list[tuple[int, RetrievedChunk]]]:
    """Returns numbered context string and mapping of citation id -> chunk."""
    lines: list[str] = []
    mapping: list[tuple[int, RetrievedChunk]] = []
    used = 0
    cid = 1
    for ch in chunks:
        block = f"[{cid}] (source: {ch.source})\n{ch.text}"
        if used + len(block) + 2 > settings.max_context_chars:
            break
        lines.append(block)
        mapping.append((cid, ch))
        used += len(block) + 2
        cid += 1
    return "\n\n".join(lines), mapping


SYSTEM_PROMPT = """You are a careful assistant. Answer using ONLY the numbered context passages below. \
When you use a fact from a passage, cite it with the same number in square brackets, e.g. [1]. \
If the context does not contain enough information, say you cannot find it in the knowledge base."""


def build_messages(
    question: str,
    chunks: list[RetrievedChunk],
    settings: Settings,
) -> tuple[list[dict[str, str]], list[tuple[int, RetrievedChunk]]]:
    context_str, mapping = build_context_block(chunks, settings)
    user_content = (
        "Context passages:\n"
        f"{context_str}\n\n"
        f"Question: {question}\n\n"
        "Answer with citations where applicable."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return messages, mapping
