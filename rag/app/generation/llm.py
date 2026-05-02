"""与 OpenAI 兼容的 Embedding / Chat HTTP 客户端，以及可选的本地 sentence-transformers 嵌入。"""
import asyncio
from typing import Any
import httpx
from app.config import Settings, normalize_openai_v1_base

class OpenAICompatibleClient:
    """封装 embed 与 chat 调用；embedding 可走远端 API 或 EMBEDDING_BACKEND=local 本地模型。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._embed_base = normalize_openai_v1_base(settings.resolved_embedding_base_url)
        self._chat_base = normalize_openai_v1_base(settings.resolved_chat_base_url)
        self._local_model: Any = None
        self._local_lock = asyncio.Lock()

    def _embedding_headers(self) -> dict[str, str]:
        """构造 /embeddings 请求的 HTTP 头（含 Bearer）。"""
        h: dict[str, str] = {"Content-Type": "application/json"}
        key = self._settings.resolved_embedding_api_key
        if key:
            h["Authorization"] = f"Bearer {key}"
        return h

    def _chat_headers(self) -> dict[str, str]:
        """构造 /chat/completions 请求的 HTTP 头（含 Bearer）。"""
        h: dict[str, str] = {"Content-Type": "application/json"}
        key = self._settings.resolved_chat_api_key
        if key:
            h["Authorization"] = f"Bearer {key}"
        return h

    async def _ensure_local_model(self) -> Any:
        """懒加载本地 SentenceTransformer，单例 + 异步锁避免并发重复加载。"""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            msg = "EMBEDDING_BACKEND=local requires sentence-transformers and torch (pip install sentence-transformers)"
            raise RuntimeError(msg) from e

        async with self._local_lock:
            if self._local_model is None:
                mid = self._settings.resolved_local_embedding_model_id
                if not mid:
                    raise ValueError(
                        "Set LOCAL_EMBEDDING_MODEL or EMBEDDING_MODEL for local embeddings",
                    )

                dev = self._settings.local_embedding_device.strip()
                kwargs: dict[str, Any] = {}
                if dev:
                    kwargs["device"] = dev

                def load() -> Any:
                    return SentenceTransformer(mid, **kwargs)

                self._local_model = await asyncio.to_thread(load)
        return self._local_model

    async def _embed_local(self, texts: list[str]) -> list[list[float]]:
        """使用本地模型对多段文本编码；在线程池中执行 encode 以免阻塞事件循环。"""
        model = await self._ensure_local_model()
        bs = self._settings.local_embedding_batch_size

        def encode_all() -> list[list[float]]:
            arr = model.encode(
                texts,
                batch_size=min(bs, len(texts)),
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return arr.tolist()

        return await asyncio.to_thread(encode_all)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """返回与输入顺序一致的向量列表；local 走 sentence-transformers，否则走 OpenAI 式 /embeddings。"""
        if not texts:
            return []
        if self._settings.embedding_backend == "local":
            return await self._embed_local(texts)

        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(
                f"{self._embed_base}/embeddings",
                headers=self._embedding_headers(),
                json={
                    "model": self._settings.embedding_model,
                    "input": texts,
                },
            )
            r.raise_for_status()
            data: Any = r.json()
        items = data.get("data", [])
        items_sorted = sorted(items, key=lambda x: x.get("index", 0))
        return [item["embedding"] for item in items_sorted]

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
    ) -> str:
        """调用 /chat/completions，返回助手回复正文（strip 后）。"""
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(
                f"{self._chat_base}/chat/completions",
                headers=self._chat_headers(),
                json={
                    "model": self._settings.chat_model,
                    "messages": messages,
                    "temperature": temperature,
                },
            )
            r.raise_for_status()
            data: Any = r.json()
        return data["choices"][0]["message"]["content"].strip()
