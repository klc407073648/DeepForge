"""将检索片段格式化为带编号上下文，并组装 chat API 的 messages。"""
from app.config import Settings
from app.retrieval.search import RetrievedChunk


def build_context_block(
    chunks: list[RetrievedChunk],
    settings: Settings,
) -> tuple[str, list[tuple[int, RetrievedChunk]]]:
    """按序号拼接上下文块，受 max_context_chars 限制；返回上下文字符串与 citation 编号到片段的映射。"""
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


# 系统提示：仅依据编号片段作答，并沿用 [n] 形式的引用。
SYSTEM_PROMPT = """You are a careful assistant. Answer using ONLY the numbered context passages below. \
When you use a fact from a passage, cite it with the same number in square brackets, e.g. [1]. \
If the context does not contain enough information, say you cannot find it in the knowledge base."""


def build_messages(
    question: str,
    chunks: list[RetrievedChunk],
    settings: Settings,
) -> tuple[list[dict[str, str]], list[tuple[int, RetrievedChunk]]]:
    """生成 system+user 两条 message，以及用于 API 响应里 Citation 对齐的编号映射。"""
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
