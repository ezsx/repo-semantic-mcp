"""Repo context search public contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from services.repo_semantic.contracts.common import (
    ChunkScope,
    EvidenceReason,
    GraphEdgeType,
    GraphMode,
    GraphNodeType,
    GraphState,
    OriginBranch,
    RepoContextRoute,
)
from services.repo_semantic.contracts.search import (
    BranchDiagnostics,
    ExactAnchorAnalysis,
    ExactAnchorCandidate,
    SearchDiagnostics,
    SearchResult,
    SearchWarning,
)


class QueryUsage(BaseModel):
    """One query used by repo_context_search."""

    query: str
    role: Literal["original", "subquery"]
    route_used: Literal["semantic", "hybrid", "exact_handoff"]
    weight: float
    result_count: int = 0
    warning_codes: list[str] = Field(default_factory=list)


class ExactAnchorUsage(BaseModel):
    """Exact anchor plus the query that produced it."""

    query: str
    query_role: Literal["original", "subquery"]
    anchor: ExactAnchorCandidate


class VerificationHint(BaseModel):
    """Whether a returned context result needs explicit working-tree verification."""

    required: bool = False
    reason: str | None = None
    recommended_action_codes: list[str] = Field(default_factory=list)


class EvidencePathStep(BaseModel):
    """One compact step in graph evidence."""

    node_id: str | None = None
    node_type: GraphNodeType | None = None
    key: str | None = None
    relative_path: str | None = None
    chunk_id: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    edge_type: GraphEdgeType | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class EvidencePath(BaseModel):
    """Why graph expansion connected a seed chunk to a result chunk."""

    reason: EvidenceReason
    seed_chunk_id: str | None = None
    seed_relative_path: str | None = None
    path: list[EvidencePathStep] = Field(default_factory=list)
    confidence: float | None = None
    stale: bool = False


class RepoContextResult(SearchResult):
    """Search result enriched for agent context retrieval."""

    final_rank: int
    final_score: float
    origin_branches: list[OriginBranch] = Field(default_factory=list)
    origin_queries: list[str] = Field(default_factory=list)
    branch_scores: dict[str, float] = Field(default_factory=dict)
    branch_ranks: dict[str, int] = Field(default_factory=dict)
    uncovered_terms: list[str] = Field(default_factory=list)
    evidence_paths: list[EvidencePath] = Field(default_factory=list)
    verification: VerificationHint = Field(default_factory=VerificationHint)


class FileContextGroup(BaseModel):
    """Grouped context summary for one file."""

    relative_path: str
    scope: ChunkScope
    language: str | None = None
    best_rank: int
    best_score: float
    chunk_count: int
    chunk_ids: list[str] = Field(default_factory=list)
    line_ranges: list[str] = Field(default_factory=list)
    matched_terms: list[str] = Field(default_factory=list)
    origin_queries: list[str] = Field(default_factory=list)
    origin_branches: list[str] = Field(default_factory=list)
    recommended_action_codes: list[str] = Field(default_factory=list)


class FileDistributionEntry(BaseModel):
    """Compact result distribution for one file."""

    relative_path: str
    scope: ChunkScope
    result_count: int
    best_rank: int
    best_score: float


class RecommendedAction(BaseModel):
    """Small operational next step for the caller."""

    code: Literal[
        "read_file_range",
        "run_local_rg",
        "rebuild_index",
        "build_graph",
        "rebuild_graph",
        "update_graph",
        "retry_with_graph_expand",
        "retry_with_hybrid",
        "retry_with_semantic",
        "retry_with_path_filter",
        "retry_with_docs_scope",
        "retry_with_code_scope",
    ]
    severity: Literal["info", "warning"]
    title: str
    detail: str | None = None
    relative_path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    argv_hint: list[str] | None = None
    cwd_hint: Literal["repo_root"] | None = None
    authority: Literal["local_rg"] | None = None
    tool_hint: str | None = None


class GraphBranchDiagnostics(BaseModel):
    """Machine-readable graph branch diagnostics for repo_context_search."""

    requested_mode: GraphMode
    effective_mode: Literal["off", "auto", "expand"]
    available: bool
    used: bool
    status_state: GraphState | None = None
    expansion_allowed: bool = False
    skipped_reason: str | None = None
    seed_count: int = 0
    bound_seed_count: int = 0
    expanded_node_count: int = 0
    graph_candidate_count: int = 0
    final_graph_result_count: int = 0
    graph_candidates_before_filter: int = 0
    graph_candidates_after_filter: int = 0
    graph_filter_post_families: list[str] = Field(default_factory=list)
    degraded: bool = False
    seed_path_states: dict[str, str] = Field(default_factory=dict)
    expanded_path_states: dict[str, str] = Field(default_factory=dict)
    max_hops: int = 1
    edge_types_used: list[GraphEdgeType] = Field(default_factory=list)
    warning_codes: list[str] = Field(default_factory=list)


class RepoContextDiagnostics(BaseModel):
    """Machine-readable repo_context_search diagnostics."""

    fusion_method: Literal["single_query", "multi_query_weighted_rrf", "exact_handoff"]
    rrf_k: int | None = None
    query_count: int
    per_query: list[SearchDiagnostics] = Field(default_factory=list)
    branch_diagnostics: dict[str, BranchDiagnostics] = Field(default_factory=dict)
    filter_pushed_families: list[str] = Field(default_factory=list)
    filter_post_families: list[str] = Field(default_factory=list)
    filter_best_effort: bool = False
    exact_anchor_analysis: ExactAnchorAnalysis | None = None
    exact_anchor_usages: list[ExactAnchorUsage] = Field(default_factory=list)
    scope_distribution: dict[str, int] = Field(default_factory=dict)
    file_distribution: list[FileDistributionEntry] = Field(default_factory=list)
    matched_terms: list[str] = Field(default_factory=list)
    uncovered_terms: list[str] = Field(default_factory=list)
    sparse_available: bool | None = None
    sparse_unavailable_codes: list[str] = Field(default_factory=list)
    graph_available: bool = False
    graph_used: bool = False
    graph_diagnostics: GraphBranchDiagnostics | None = None
    rerank_used: bool = False
    colbert_used: bool = False


class RepoContextSearchResponse(BaseModel):
    """Agent-facing context retrieval envelope."""

    contract_version: Literal["repo_context_search.v1"] = "repo_context_search.v1"
    query: str
    queries_used: list[QueryUsage]
    route_requested: RepoContextRoute
    route_used: Literal["semantic", "hybrid", "exact_handoff"]
    graph_mode_requested: GraphMode
    graph_mode_effective: Literal["off", "auto", "expand"] = "off"
    results: list[RepoContextResult] = Field(default_factory=list)
    file_groups: list[FileContextGroup] = Field(default_factory=list)
    warnings: list[SearchWarning] = Field(default_factory=list)
    diagnostics: RepoContextDiagnostics | None = None
    recommended_next_actions: list[RecommendedAction] = Field(default_factory=list)
