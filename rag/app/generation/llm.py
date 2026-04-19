from typing import Any

import httpx

from app.config import Settings


class OpenAICompatibleClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        base = settings.openai_base_url.rstrip("/")
        self._base = base if base.endswith("/v1") else f"{base}/v1"

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Content-Type": "application/json"}
        if self._settings.openai_api_key:
            h["Authorization"] = f"Bearer {self._settings.openai_api_key}"
        return h

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(
                f"{self._base}/embeddings",
                headers=self._headers(),
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
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(
                f"{self._base}/chat/completions",
                headers=self._headers(),
                json={
                    "model": self._settings.chat_model,
                    "messages": messages,
                    "temperature": temperature,
                },
            )
            r.raise_for_status()
            data: Any = r.json()
        return data["choices"][0]["message"]["content"].strip()
