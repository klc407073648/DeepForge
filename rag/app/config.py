from pathlib import Path
from typing import Literal

from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

MAX_UPLOAD_BYTES = 15 * 1024 * 1024

def normalize_openai_v1_base(url: str) -> str:
    b = url.strip().rstrip("/")
    if not b:
        return "https://api.openai.com/v1"
    return b if b.endswith("/v1") else f"{b}/v1"

# 从 .env 读变量；embedding/chat 可分 URL 与 Key，空的则回落到 OPENAI_*。
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        validation_alias="OPENAI_BASE_URL",
    )

    embedding_base_url: str = Field(default="", validation_alias="EMBEDDING_BASE_URL")
    embedding_api_key: str = Field(default="", validation_alias="EMBEDDING_API_KEY")
    chat_base_url: str = Field(default="", validation_alias="CHAT_BASE_URL")
    chat_api_key: str = Field(default="", validation_alias="CHAT_API_KEY")

    embedding_model: str = Field(
        default="text-embedding-3-small",
        validation_alias="EMBEDDING_MODEL",
    )
    chat_model: str = Field(default="gpt-4o-mini", validation_alias="CHAT_MODEL")

    embedding_backend: Literal["http", "local"] = Field(
        default="http",
        validation_alias="EMBEDDING_BACKEND",
    )
    local_embedding_model: str = Field(
        default="",
        validation_alias="LOCAL_EMBEDDING_MODEL",
    )
    local_embedding_device: str = Field(default="", validation_alias="LOCAL_EMBEDDING_DEVICE")
    local_embedding_batch_size: int = Field(
        default=32,
        ge=1,
        validation_alias="LOCAL_EMBEDDING_BATCH_SIZE",
    )

    chroma_persist_dir: Path = Field(
        default=Path("data/chroma"),
        validation_alias="CHROMA_PERSIST_DIR",
    )
    collection_name: str = Field(default="rag_docs", validation_alias="COLLECTION_NAME")

    chunk_size: int = Field(default=512, validation_alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=128, validation_alias="CHUNK_OVERLAP")

    top_k: int = Field(default=5, validation_alias="TOP_K")
    max_context_chars: int = Field(
        default=8000,
        validation_alias="MAX_CONTEXT_CHARS",
    )
    max_retrieval_distance: float | None = Field(
        default=1.25,
        validation_alias="MAX_RETRIEVAL_DISTANCE",
    )

    @field_validator("embedding_backend", mode="before")
    @classmethod
    def normalize_embedding_backend(cls, v: object) -> str:
        if v is None or v == "":
            return "http"
        s = str(v).strip().lower()
        if s not in ("http", "local"):
            raise ValueError("EMBEDDING_BACKEND must be http or local")
        return s

    @field_validator("max_retrieval_distance", mode="before")
    @classmethod
    def empty_distance_to_none(cls, v: object) -> object:
        if v == "" or v is None:
            return None
        return v

    app_name: str = Field(default="rag-service", validation_alias="APP_NAME")
    app_version: str = Field(default="0.1.0", validation_alias="APP_VERSION")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def resolved_embedding_base_url(self) -> str:
        s = self.embedding_base_url.strip()
        return s if s else self.openai_base_url

    @computed_field  # type: ignore[prop-decorator]
    @property
    def resolved_chat_base_url(self) -> str:
        s = self.chat_base_url.strip()
        return s if s else self.openai_base_url

    @computed_field  # type: ignore[prop-decorator]
    @property
    def resolved_embedding_api_key(self) -> str:
        return self.embedding_api_key if self.embedding_api_key.strip() else self.openai_api_key

    @computed_field  # type: ignore[prop-decorator]
    @property
    def resolved_chat_api_key(self) -> str:
        return self.chat_api_key if self.chat_api_key.strip() else self.openai_api_key

    @computed_field  # type: ignore[prop-decorator]
    @property
    def resolved_local_embedding_model_id(self) -> str:
        if self.local_embedding_model.strip():
            return self.local_embedding_model.strip()
        return self.embedding_model.strip()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def embedding_ingest_batch_size(self) -> int:
        if self.embedding_backend == "local":
            return self.local_embedding_batch_size
        return 64


settings = Settings()
