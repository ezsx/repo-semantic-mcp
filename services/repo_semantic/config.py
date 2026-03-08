"""Конфигурация repo semantic MCP."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_repo_root() -> str:
    """Вернуть корень репозитория из расположения текущего файла."""

    return str(Path(__file__).resolve().parents[2])


class SemanticMcpSettings(BaseSettings):
    """Настройки semantic MCP, Qdrant и indexing pipeline."""

    SEMANTIC_MCP_REPO_ROOT: str = Field(default_factory=_default_repo_root)
    SEMANTIC_MCP_QDRANT_URL: str = "http://qdrant:6333"
    SEMANTIC_MCP_QDRANT_API_KEY: str | None = None
    SEMANTIC_MCP_COLLECTION_PREFIX: str = "repo_semantic"
    SEMANTIC_MCP_INDEX_SCHEMA_VERSION: int = 1

    SEMANTIC_MCP_EMBEDDING_BACKEND: str = "fastembed_local"
    SEMANTIC_MCP_EMBEDDING_MODEL: str = "BAAI/bge-m3"
    SEMANTIC_MCP_TEI_URL: str | None = None

    SEMANTIC_MCP_TRANSPORT: str = "stdio"
    SEMANTIC_MCP_HTTP_HOST: str = "0.0.0.0"
    SEMANTIC_MCP_HTTP_PORT: int = 8011

    SEMANTIC_MCP_WATCH_ENABLED: bool = False
    SEMANTIC_MCP_WATCH_DEBOUNCE_SEC: int = 3
    SEMANTIC_MCP_AUTO_INDEX_ON_START: bool = True
    SEMANTIC_MCP_EMBED_BATCH_DOCS: int = 24
    SEMANTIC_MCP_EMBED_BATCH_CHARS: int = 12000
    SEMANTIC_MCP_MAX_CHUNK_CHARS: int = 800

    SEMANTIC_MCP_INCLUDE_GLOBS: list[str] = Field(
        default_factory=lambda: [
            "apps/**",
            "services/**",
            "libs/**",
            "docs/**",
            "deploy/**",
            "scripts/**",
            "tools/**",
            "agent_context/**",
            "AGENTS.md",
            "CLAUDE.md",
            "STRUCTURE.md",
        ]
    )
    SEMANTIC_MCP_EXCLUDE_GLOBS: list[str] = Field(
        default_factory=lambda: [
            ".git/**",
            ".venv/**",
            "venv/**",
            "**/__pycache__/**",
            "node_modules/**",
            "artifacts/**",
            "tmp/**",
            "temp/**",
            "**/*.png",
            "**/*.jpg",
            "**/*.jpeg",
            "**/*.gif",
            "**/*.webp",
            "**/*.ico",
            "**/*.svg",
            "**/*.zip",
            "**/*.gz",
            "**/*.tar",
            "**/*.7z",
            "**/*.pdf",
            "**/*.db",
            "**/*.sqlite",
            "**/.env*",
            "**/*.pem",
            "**/*.key",
            "**/*.crt",
            "**/*.pfx",
            "**/*.kdbx",
        ]
    )
    SEMANTIC_MCP_LOG_LEVEL: str = "info"

    @field_validator("SEMANTIC_MCP_INCLUDE_GLOBS", "SEMANTIC_MCP_EXCLUDE_GLOBS", mode="before")
    @classmethod
    def _split_csv_lists(cls, value):
        """Разрешить список globs через CSV-строку в env."""

        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @property
    def repo_root(self) -> Path:
        """Вернуть корень репозитория как Path."""

        return Path(self.SEMANTIC_MCP_REPO_ROOT).resolve()

    @property
    def collection_code(self) -> str:
        """Имя коллекции для code corpus."""

        return (
            f"{self.SEMANTIC_MCP_COLLECTION_PREFIX}_code_v"
            f"{self.SEMANTIC_MCP_INDEX_SCHEMA_VERSION}"
        )

    @property
    def collection_docs(self) -> str:
        """Имя коллекции для docs corpus."""

        return (
            f"{self.SEMANTIC_MCP_COLLECTION_PREFIX}_docs_v"
            f"{self.SEMANTIC_MCP_INDEX_SCHEMA_VERSION}"
        )

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=True,
        enable_decoding=False,
    )
