"""Internal retrieval execution types."""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import models as qdrant_models

from services.repo_semantic.contracts.common import ChunkRecord
from services.repo_semantic.contracts.search import (
    PayloadCapabilities,
    SearchDiagnostics,
    SearchResult,
    SearchWarning,
)


@dataclass(slots=True)
class _LexicalCache:
    """Кэш текстов и токенов для локального BM25 по конкретной коллекции."""

    chunks: list[ChunkRecord]
    tokens: list[list[str]]


@dataclass(slots=True)
class _FilterPlan:
    """Qdrant pushdown plus client-side filter diagnostics for one scope."""

    qdrant_filter: qdrant_models.Filter | None
    pushed_families: list[str]
    post_filter_families: list[str]
    required_payload_fields: list[str]
    missing_payload_fields: list[str]
    best_effort: bool
    warning_codes: list[str]
    payload_capabilities: PayloadCapabilities


@dataclass(slots=True)
class _DenseSearchExecution:
    """Dense retrieval candidates plus execution metadata."""

    candidates: dict[str, tuple[ChunkRecord, float]]
    candidate_limit: int
    candidates_scanned: int
    filtered_candidates: int
    kept_candidates: int
    dense_best_effort: bool
    filter_plans: list[_FilterPlan]


@dataclass(slots=True)
class _SparseSearchExecution:
    """Sparse retrieval candidates plus availability metadata."""

    candidates: dict[str, tuple[ChunkRecord, float]]
    branch_available: bool
    partial_unavailable: bool
    manifest_seen: bool
    stats_stale: bool
    manifest_hash: str | None
    vocabulary_hash: str | None
    corpus_stats_hash: str | None
    unavailable_scopes: list[str]
    unavailable_codes: list[str]
    filter_plans: list[_FilterPlan]


@dataclass(slots=True)
class _LegacyBm25Execution:
    """Legacy local BM25 fallback candidates plus execution metadata."""

    candidates: dict[str, tuple[ChunkRecord, float]]
    lexical_cache_used: bool
    legacy_fallback_used: bool


@dataclass(slots=True)
class _SearchExecution:
    """Final search rows with response-level metadata."""

    results: list[SearchResult]
    diagnostics: SearchDiagnostics
    warnings: list[SearchWarning]
    branch_ranks_by_chunk: dict[str, dict[str, int]]
