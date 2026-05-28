"""Application settings and crawl/plan configuration."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


def normalize_openai_v1_base(url: str) -> str:
    base = url.strip().rstrip("/")
    if not base:
        return "https://api.openai.com/v1"
    return base if base.endswith("/v1") else f"{base}/v1"


class CrawlConfig:
    """Crawl-related settings loaded from defaults, env, or .claw.toml."""

    def __init__(
        self,
        *,
        max_depth: int = 2,
        max_pages: int = 50,
        same_domain_only: bool = True,
        max_concurrency: int = 5,
        request_delay_ms: int = 200,
        timeout_seconds: float = 30.0,
        user_agent: str = "ClawBot/0.1 (+https://github.com/deepforge/claw)",
        exclude_patterns: list[str] | None = None,
        include_patterns: list[str] | None = None,
        ignore_robots: bool = False,
        no_images: bool = False,
        max_content_chars: int = 100_000,
    ) -> None:
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.same_domain_only = same_domain_only
        self.max_concurrency = max_concurrency
        self.request_delay_ms = request_delay_ms
        self.timeout_seconds = timeout_seconds
        self.user_agent = user_agent
        self.exclude_patterns = exclude_patterns or [
            "*/login*",
            "*/logout*",
            "*.pdf",
            "*.zip",
            "*.png",
            "*.jpg",
            "*.jpeg",
            "*.gif",
            "*.svg",
        ]
        self.include_patterns = include_patterns or []
        self.ignore_robots = ignore_robots
        self.no_images = no_images
        self.max_content_chars = max_content_chars


class PlanConfig:
    """Plan generation settings."""

    def __init__(
        self,
        *,
        model: str = "deepseek-chat",
        max_input_chars: int = 120_000,
        temperature: float = 0.2,
    ) -> None:
        self.model = model
        self.max_input_chars = max_input_chars
        self.temperature = temperature


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    claw_chat_api_key: str = Field(default="", validation_alias="CLAW_CHAT_API_KEY")
    claw_chat_base_url: str = Field(
        default="https://api.openai.com/v1",
        validation_alias="CLAW_CHAT_BASE_URL",
    )
    claw_chat_model: str = Field(default="deepseek-chat", validation_alias="CLAW_CHAT_MODEL")

    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        validation_alias="OPENAI_BASE_URL",
    )

    cache_dir: Path = Field(default=Path(".claw/cache"), validation_alias="CLAW_CACHE_DIR")
    plans_dir: Path = Field(default=Path(".claw/plans"), validation_alias="CLAW_PLANS_DIR")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def resolved_chat_api_key(self) -> str:
        return self.claw_chat_api_key.strip() or self.openai_api_key.strip()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def resolved_chat_base_url(self) -> str:
        url = self.claw_chat_base_url.strip() or self.openai_base_url.strip()
        return normalize_openai_v1_base(url)


def load_toml_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


def build_crawl_config(settings_path: Path | None = None, **overrides: Any) -> CrawlConfig:
    data = load_toml_config(settings_path)
    crawl = data.get("crawl", {})
    params = {
        "max_depth": crawl.get("max_depth", 2),
        "max_pages": crawl.get("max_pages", 50),
        "same_domain_only": crawl.get("same_domain_only", True),
        "max_concurrency": crawl.get("max_concurrency", 5),
        "request_delay_ms": crawl.get("request_delay_ms", 200),
        "timeout_seconds": crawl.get("timeout_seconds", 30.0),
        "user_agent": crawl.get("user_agent", CrawlConfig().user_agent),
        "exclude_patterns": crawl.get("exclude_patterns"),
        "include_patterns": crawl.get("include_patterns"),
        "ignore_robots": crawl.get("ignore_robots", False),
        "no_images": crawl.get("no_images", False),
        "max_content_chars": crawl.get("max_content_chars", 100_000),
    }
    params.update({k: v for k, v in overrides.items() if v is not None})
    return CrawlConfig(**params)


def build_plan_config(
    settings_path: Path | None = None,
    *,
    app_settings: Settings | None = None,
    **overrides: Any,
) -> PlanConfig:
    data = load_toml_config(settings_path)
    plan = data.get("plan", {})
    cfg = app_settings or settings
    params = {
        "model": plan.get("model", cfg.claw_chat_model),
        "max_input_chars": plan.get("max_input_chars", 120_000),
        "temperature": plan.get("temperature", 0.2),
    }
    params.update({k: v for k, v in overrides.items() if v is not None})
    return PlanConfig(**params)


settings = Settings()
