"""OpenAI-compatible chat client."""

from __future__ import annotations

from typing import Any

import httpx

from claw.config import Settings, normalize_openai_v1_base


class ChatClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._base = normalize_openai_v1_base(settings.resolved_chat_base_url)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = self._settings.resolved_chat_api_key
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.2,
    ) -> str:
        model_name = model or self._settings.claw_chat_model
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                f"{self._base}/chat/completions",
                headers=self._headers(),
                json={
                    "model": model_name,
                    "messages": messages,
                    "temperature": temperature,
                },
            )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                body = exc.response.text[:800]
                raise RuntimeError(
                    f"Chat API error {exc.response.status_code} (model={model_name}): {body}"
                ) from exc
            data: Any = response.json()
        content = data["choices"][0]["message"]["content"]
        return (content or "").strip()
