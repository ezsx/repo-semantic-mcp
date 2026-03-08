"""Модели semantic MCP и индексируемых чанков."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

SearchScope = Literal["all", "code", "docs"]
ChunkScope = Literal["code", "docs"]


@dataclass(slots=True)
class ChunkRecord:
    """Нормализованный чанк, который индексируется в vector store."""

    point_id: str
    scope: ChunkScope
    relative_path: str
    language: str
    chunk_type: str
    text: str
    start_line: int
    end_line: int
    content_hash: str
    source_mtime: float
    symbol_path: str | None = None
    heading_path: str | None = None
    domain_tags: list[str] = field(default_factory=list)
    is_generated: bool = False
    extra: dict[str, str] = field(default_factory=dict)


class SearchResult(BaseModel):
    """Результат semantic/hybrid поиска."""

    chunk_id: str
    scope: ChunkScope
    relative_path: str
    language: str
    chunk_type: str
    start_line: int
    end_line: int
    snippet: str
    symbol_path: str | None = None
    heading_path: str | None = None
    domain_tags: list[str] = Field(default_factory=list)
    score: float
    dense_score: float | None = None
    lexical_score: float | None = None


class ReadChunkResult(BaseModel):
    """Полный текст индексированного чанка."""

    chunk_id: str
    scope: ChunkScope
    relative_path: str
    language: str
    chunk_type: str
    start_line: int
    end_line: int
    symbol_path: str | None = None
    heading_path: str | None = None
    text: str


class IndexCollectionStatus(BaseModel):
    """Статус конкретной коллекции индекса."""

    scope: ChunkScope
    collection_name: str
    points_count: int
    lexical_documents: int


class IndexStatusResult(BaseModel):
    """Итоговый статус semantic индекса."""

    repo_root: str
    index_profile: str
    embedding_backend: str
    embedding_model: str
    qdrant_url: str
    schema_version: int
    watch_enabled: bool
    watch_running: bool
    last_full_build_ts: str | None = None
    collections: list[IndexCollectionStatus]
