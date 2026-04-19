"""
自检：本地嵌入能否加载并完成一次 encode（在 rag 目录执行）:
  python scripts/verify_local_embedding.py

可通过环境变量覆盖模型，例如:
  set LOCAL_EMBEDDING_MODEL=sentence-transformers/paraphrase-MiniLM-L6-v2
  python scripts/verify_local_embedding.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# 保证从 rag 根目录加载 app
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# 强制覆盖当前 shell / .env 中的 http 设置，便于单独测 local
os.environ["EMBEDDING_BACKEND"] = "local"
if "LOCAL_EMBEDDING_MODEL" not in os.environ:
    os.environ["LOCAL_EMBEDDING_MODEL"] = "sentence-transformers/paraphrase-MiniLM-L6-v2"


async def main() -> None:
    from app.config import Settings
    from app.generation.llm import OpenAICompatibleClient

    s = Settings()
    client = OpenAICompatibleClient(s)
    vecs = await client.embed_texts(["hello", "world"])
    print("dims:", len(vecs), "x", len(vecs[0]) if vecs else 0)
    print("ok")


if __name__ == "__main__":
    asyncio.run(main())
