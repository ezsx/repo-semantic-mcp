"""Search result and diagnostic public contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from services.repo_semantic.contracts.common import ChunkScope, MatchType, SearchMode


class LineMatch(BaseModel):
    """A single line-level lexical hit inside a returned chunk."""

    line: int
    text: str
    matched_terms: list[str] = Field(default_factory=list)


class SearchResult(BaseModel):
    """Результат semantic/hybrid поиска."""

    chunk_id: str
    repo_root: str | None = None
    scope: ChunkScope
    relative_path: str
    language: str
    chunk_type: str
    start_line: int
    end_line: int
    line_range: str | None = None
    file_extension: str | None = None
    snippet: str
    snippet_start_line: int | None = None
    snippet_end_line: int | None = None
    symbol_path: str | None = None
    heading_path: str | None = None
    domain_tags: list[str] = Field(default_factory=list)
    score: float
    final_score: float | None = None
    match_type: MatchType | None = None
    dense_score: float | None = None
    lexical_score: float | None = None
    matched_terms: list[str] = Field(default_factory=list)
    why_matched: list[str] = Field(default_factory=list)
    line_matches: list[LineMatch] = Field(default_factory=list)


class SearchFilters(BaseModel):
    """Normalized search filters used by semantic and hybrid retrieval."""

    path_prefix: str | None = None
    include_paths: list[str] = Field(default_factory=list)
    exclude_paths: list[str] = Field(default_factory=list)
    file_extensions: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    chunk_types: list[str] = Field(default_factory=list)
    domain_tags: list[str] = Field(default_factory=list)


class SearchWarning(BaseModel):
    """Response-level warning for agent-grade search envelopes."""

    code: str
    severity: Literal["info", "warning"]
    detail: str | None = None


class ExactAnchorCandidate(BaseModel):
    """Detected exact-looking query anchor that needs working-tree verification."""

    surface: str
    canonical: str
    start: int
    end: int
    anchor_type: Literal[
        "env_var",
        "route",
        "path",
        "file_name",
        "symbol",
        "sql_identifier",
        "cli_flag",
        "quoted_literal",
        "error_literal",
        "config_key",
    ]
    confidence: Literal["high", "medium", "low"]
    requires_verification: bool = True


class ExactAnchorAnalysis(BaseModel):
    """Query-time exact-anchor diagnostics and safe local-rg guidance."""

    anchors: list[ExactAnchorCandidate] = Field(default_factory=list)
    exact_anchor_heavy: bool = False
    exact_fallback_recommended: bool = False
    local_rg_authority: Literal["local_rg"] = "local_rg"
    cwd_hint: Literal["repo_root"] = "repo_root"
    argv_hints: list[list[str]] = Field(default_factory=list)


class PayloadCapabilities(BaseModel):
    """Payload-field capability summary for one collection."""

    scope: ChunkScope
    collection_name: str
    checked: bool
    available_fields: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)


class BranchDiagnostics(BaseModel):
    """Per-retrieval-branch execution diagnostics."""

    available: bool
    candidate_limit: int | None = None
    candidates_returned: int = 0
    warning_codes: list[str] = Field(default_factory=list)


class SearchDiagnostics(BaseModel):
    """Machine-readable diagnostics for a search execution."""

    candidate_limit: int | None = None
    candidates_scanned: int | None = None
    filtered_candidates: int | None = None
    final_results: int
    dense_best_effort: bool = False
    lexical_cache_used: bool = False
    retrieval_backend: str | None = None
    fusion_method: str | None = None
    rrf_k: int | None = None
    branch_weights: dict[str, float] = Field(default_factory=dict)
    branch_diagnostics: dict[str, BranchDiagnostics] = Field(default_factory=dict)
    sparse_available: bool | None = None
    sparse_contract_compatible: bool | None = None
    sparse_manifest_hash: str | None = None
    sparse_vocabulary_hash: str | None = None
    sparse_corpus_stats_hash: str | None = None
    sparse_stats_stale: bool = False
    sparse_unavailable_codes: list[str] = Field(default_factory=list)
    exact_anchor_analysis: ExactAnchorAnalysis | None = None
    filter_pushdown_supported: bool = False
    filter_pushed_families: list[str] = Field(default_factory=list)
    filter_post_families: list[str] = Field(default_factory=list)
    filter_best_effort: bool = False
    filter_payload_capabilities: list[PayloadCapabilities] = Field(default_factory=list)


class SearchResponse(BaseModel):
    """Agent-grade search response with result rows plus metadata."""

    contract_version: str = "search.v2"
    query: str
    mode: SearchMode
    results: list[SearchResult] = Field(default_factory=list)
    warnings: list[SearchWarning] = Field(default_factory=list)
    diagnostics: SearchDiagnostics | None = None
