"""Graph artifact public contracts."""

from __future__ import annotations

from pydantic import BaseModel, Field

from services.repo_semantic.contracts.common import GraphState


class GraphStatusSummary(BaseModel):
    """Compact graph readiness summary embedded into index_status.v2."""

    available: bool
    state: GraphState
    expansion_allowed: bool
    schema_version: int | None = None
    built_commit: str | None = None
    built_at: str | None = None
    extractor_versions_hash: str | None = None
    file_coverage_ratio: float | None = None
    stale_path_count: int | None = None
    invalidated_paths_preview: list[str] = Field(default_factory=list)
    update_available: bool = False
    update_required: bool = False
    update_safe_auto_run: bool = False
    update_blocked_reason: str | None = None
    update_path_count: int | None = None
    update_paths_preview: list[str] = Field(default_factory=list)
    rebuild_required: bool = False
    warning_codes: list[str] = Field(default_factory=list)


class GraphCounts(BaseModel):
    """Graph artifact table counts and type distributions."""

    files: int = 0
    nodes: int = 0
    edges: int = 0
    node_terms: int = 0
    node_chunks: int = 0
    node_type_counts: dict[str, int] = Field(default_factory=dict)
    edge_type_counts: dict[str, int] = Field(default_factory=dict)
    file_state_counts: dict[str, int] = Field(default_factory=dict)


class GraphStatusResult(BaseModel):
    """Detailed graph artifact status for explicit graph_status()."""

    contract_version: str = "graph_status.v1"
    repo_root: str
    profile: str
    graph_path: str
    available: bool
    state: GraphState
    expansion_allowed: bool
    schema_version: int | None = None
    built_commit: str | None = None
    built_at: str | None = None
    graph_contract_hash: str | None = None
    extractor_versions_hash: str | None = None
    source_index_contract: dict[str, object] | None = None
    file_coverage_ratio: float | None = None
    stale_path_count: int = 0
    invalidated_paths_preview: list[str] = Field(default_factory=list)
    update_available: bool = False
    update_required: bool = False
    update_safe_auto_run: bool = False
    update_blocked_reason: str | None = None
    update_path_count: int | None = None
    update_paths_preview: list[str] = Field(default_factory=list)
    rebuild_required: bool = False
    counts: GraphCounts = Field(default_factory=GraphCounts)
    warning_codes: list[str] = Field(default_factory=list)
    error_code: str | None = None
    error_detail: str | None = None


class GraphBuildResult(BaseModel):
    """Result of explicit build_graph/rebuild_graph lifecycle operation."""

    contract_version: str = "graph_build.v1"
    repo_root: str
    profile: str
    graph_path: str
    built: bool
    rebuilt: bool = False
    state: GraphState
    built_at: str | None = None
    built_commit: str | None = None
    files_count: int = 0
    chunks_count: int = 0
    nodes_count: int = 0
    edges_count: int = 0
    warning_codes: list[str] = Field(default_factory=list)
    status: GraphStatusResult | None = None
    reason: str | None = None


class GraphUpdateResult(BaseModel):
    """Result of explicit bounded update_graph lifecycle operation."""

    contract_version: str = "graph_update.v1"
    repo_root: str
    profile: str
    graph_path: str
    operation_id: str | None = None
    lease_id: str | None = None
    audit_id: str | None = None
    idempotency_key: str | None = None
    idempotent_retry: bool = False
    allow_large_update: bool = False
    operation_in_progress: bool = False
    updated: bool
    skipped: bool = False
    skip_reason: str | None = None
    paths_requested: int = 0
    paths_discovered: int = 0
    paths_normalized: int = 0
    paths_required: int = 0
    reverse_dependent_paths_added: int = 0
    paths_updated: int = 0
    paths_deleted: int = 0
    zero_chunk_paths: int = 0
    paths_missing_from_index: int = 0
    paths_skipped: int = 0
    chunks_scanned_total: int = 0
    chunks_for_updated_paths: int = 0
    nodes_deleted: int = 0
    edges_deleted: int = 0
    nodes_upserted: int = 0
    edges_upserted: int = 0
    source_contract_updated: bool = False
    source_contract_revision_before: str | None = None
    source_contract_revision_after: str | None = None
    graph_state_after: str | None = None
    expansion_allowed_after: bool = False
    update_plan_hash: str | None = None
    metadata_only_update: bool = False
    bounds_exceeded: list[str] = Field(default_factory=list)
    warning_codes: list[str] = Field(default_factory=list)
    recommended_actions: list[dict[str, object]] = Field(default_factory=list)
    status: GraphStatusResult | None = None
