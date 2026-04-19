from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    embedding_model: str = Field(
        default="text-embedding-3-small",
        validation_alias="EMBEDDING_MODEL",
    )
    chat_model: str = Field(default="gpt-4o-mini", validation_alias="CHAT_MODEL")

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
    # Chroma cosine distance: lower is more similar; disable check if None
    max_retrieval_distance: float | None = Field(
        default=1.25,
        validation_alias="MAX_RETRIEVAL_DISTANCE",
    )

    @field_validator("max_retrieval_distance", mode="before")
    @classmethod
    def empty_distance_to_none(cls, v: object) -> object:
        if v == "" or v is None:
            return None
        return v

    app_name: str = Field(default="rag-service", validation_alias="APP_NAME")
    app_version: str = Field(default="0.1.0", validation_alias="APP_VERSION")


settings = Settings()
