"""运行时配置持久化：切片参数与 API Key 覆盖 .env 默认值。"""
from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any

from app.config import Settings, settings

RUNTIME_PATH = Path("data/runtime_config.json")
_lock = Lock()

KEY_FIELDS: dict[str, str] = {
    "OPENAI_API_KEY": "openai_api_key",
    "OPENAI_BASE_URL": "openai_base_url",
    "EMBEDDING_API_KEY": "embedding_api_key",
    "EMBEDDING_BASE_URL": "embedding_base_url",
    "CHAT_API_KEY": "chat_api_key",
    "CHAT_BASE_URL": "chat_base_url",
    "EMBEDDING_MODEL": "embedding_model",
    "CHAT_MODEL": "chat_model",
}

KEY_LABELS: dict[str, str] = {
    "OPENAI_API_KEY": "通用 API Key",
    "OPENAI_BASE_URL": "通用 Base URL",
    "EMBEDDING_API_KEY": "Embedding Key",
    "EMBEDDING_BASE_URL": "Embedding Base URL",
    "CHAT_API_KEY": "Chat Key",
    "CHAT_BASE_URL": "Chat Base URL",
    "EMBEDDING_MODEL": "Embedding 模型",
    "CHAT_MODEL": "Chat 模型",
}


def _empty_runtime() -> dict[str, Any]:
    return {"chunk_size": None, "chunk_overlap": None, "keys": {}, "chat_models": []}


def load_runtime_raw() -> dict[str, Any]:
    with _lock:
        if not RUNTIME_PATH.exists():
            return _empty_runtime()
        data = json.loads(RUNTIME_PATH.read_text(encoding="utf-8"))
        base = _empty_runtime()
        base.update({k: v for k, v in data.items() if k in base or k == "keys"})
        if "keys" not in base or not isinstance(base["keys"], dict):
            base["keys"] = {}
        return base


def save_runtime_raw(data: dict[str, Any]) -> None:
    with _lock:
        RUNTIME_PATH.parent.mkdir(parents=True, exist_ok=True)
        RUNTIME_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def mask_secret(value: str) -> str:
    v = value.strip()
    if not v:
        return ""
    if len(v) <= 8:
        return "*" * len(v)
    return f"{v[:4]}{'*' * (len(v) - 8)}{v[-4:]}"


def get_effective_settings() -> Settings:
    raw = load_runtime_raw()
    overrides: dict[str, Any] = {}
    if raw.get("chunk_size") is not None:
        overrides["chunk_size"] = int(raw["chunk_size"])
    if raw.get("chunk_overlap") is not None:
        overrides["chunk_overlap"] = int(raw["chunk_overlap"])
    for env_name, field_name in KEY_FIELDS.items():
        val = raw.get("keys", {}).get(env_name)
        if val is not None and str(val).strip():
            overrides[field_name] = str(val).strip()
    if not overrides:
        return settings
    return settings.model_copy(update=overrides)


def get_chunking_settings() -> dict[str, int]:
    s = get_effective_settings()
    return {"chunk_size": s.chunk_size, "chunk_overlap": s.chunk_overlap}


def update_chunking_settings(chunk_size: int, chunk_overlap: int) -> dict[str, int]:
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be in [0, chunk_size)")
    raw = load_runtime_raw()
    raw["chunk_size"] = chunk_size
    raw["chunk_overlap"] = chunk_overlap
    save_runtime_raw(raw)
    return get_chunking_settings()


def list_key_configs() -> list[dict[str, Any]]:
    s = get_effective_settings()
    raw = load_runtime_raw()
    runtime_keys = raw.get("keys", {})
    items: list[dict[str, Any]] = []
    for env_name, field_name in KEY_FIELDS.items():
        env_val = getattr(settings, field_name, "")
        runtime_val = runtime_keys.get(env_name, "")
        effective = getattr(s, field_name, "")
        configured = bool(str(effective).strip())
        display = str(runtime_val).strip() or str(env_val).strip()
        is_secret = "KEY" in env_name
        items.append(
            {
                "name": env_name,
                "label": KEY_LABELS.get(env_name, env_name),
                "masked_value": mask_secret(display) if is_secret else display,
                "purpose": KEY_LABELS.get(env_name, env_name),
                "configured": configured,
                "source": "runtime" if str(runtime_val).strip() else ("env" if configured else "none"),
            }
        )
    return items


def upsert_key_config(name: str, value: str) -> dict[str, Any]:
    if name not in KEY_FIELDS:
        raise ValueError(f"Unknown key config: {name}")
    raw = load_runtime_raw()
    keys = raw.setdefault("keys", {})
    if not value.strip():
        keys.pop(name, None)
    else:
        keys[name] = value.strip()
    save_runtime_raw(raw)
    for item in list_key_configs():
        if item["name"] == name:
            return item
    raise RuntimeError("Failed to upsert key config")


def delete_key_config(name: str) -> None:
    if name not in KEY_FIELDS:
        raise ValueError(f"Unknown key config: {name}")
    raw = load_runtime_raw()
    raw.setdefault("keys", {}).pop(name, None)
    save_runtime_raw(raw)


AVAILABLE_CHAT_MODELS = [
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "deepseek-chat",
]


def list_available_models() -> list[str]:
    return list(AVAILABLE_CHAT_MODELS)
