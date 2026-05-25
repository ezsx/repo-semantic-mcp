"""Index status, readiness, and freshness public contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from services.repo_semantic.contracts.common import (
    ChunkScope,
    CompatibilityState,
    FreshnessSeverity,
    RepoStatus,
    StatusSeverity,
)
from services.repo_semantic.contracts.graph import GraphStatusSummary


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
    lexical_documents: int | None = None


class StatusAction(BaseModel):
    """Machine-readable next action for agents/operators."""

    code: str
    severity: StatusSeverity
    title: str
    detail: str | None = None
    tool_hint: str | None = None
    host_command_hint: str | None = None


class StatusWarning(BaseModel):
    """Non-blocking status warning."""

    code: str
    severity: Literal["info", "warning"]
    detail: str | None = None


class StatusAvailability(BaseModel):
    """Readiness and search availability verdict."""

    runtime_ready: bool
    dependencies_ready: bool | None = None
    search_available: bool
    search_unavailable_codes: list[str] = Field(default_factory=list)
    host_action_required: bool = False
    next_actions: list[StatusAction] = Field(default_factory=list)


class RuntimeStatusSummary(BaseModel):
    """Runtime identity summary for status consumers."""

    bootstrap_phase: str | None = None
    bootstrap_error: str | None = None
    mounted_repo_root: str
    logical_repo_root: str
    active_repo_root: str | None = None
    runtime_switch_required: bool
    runtime_switch_command_hint: str | None = None


class BackendStatusSummary(BaseModel):
    """Embedding backend summary without forcing expensive health checks."""

    runtime_embedding_backend_id: str | None = None
    selected_embedding_backend_id: str | None = None
    backend_type: str | None = None
    model_name: str | None = None
    compatible: bool
    switch_required: bool
    healthy: bool | None = None
    health_error: str | None = None
    managed_by_service: bool | None = None
    autostart_policy: str | None = None
    location_type: str | None = None
    device_type: str | None = None
    action_plan_tool: str = "get_backend_action_plan"


class IndexCollectionContract(BaseModel):
    """Stored-vs-runtime compatibility for one collection."""

    scope: ChunkScope
    collection_name: str
    exists: bool
    points_count: int
    lexical_documents: int | None = None
    stored_embedding_backend: str | None = None
    stored_embedding_model: str | None = None
    stored_schema_version: int | Literal["unknown"] | None = None
    query_template_hash: str | None = None
    document_prefix_hash: str | None = None
    compatibility: CompatibilityState
    blocking: bool
    warning_codes: list[str] = Field(default_factory=list)


class IndexContractStatus(BaseModel):
    """Index compatibility contract for current runtime settings."""

    profile: str
    embedding_backend: str
    embedding_model: str
    schema_version: int
    query_template_hash: str | None = None
    document_prefix_hash: str | None = None
    compatibility: CompatibilityState
    blocking: bool
    incompatibility_codes: list[str] = Field(default_factory=list)
    collections: list[IndexCollectionContract] = Field(default_factory=list)


class PayloadIndexStatus(BaseModel):
    """Expected payload-index status for one collection field."""

    scope: ChunkScope
    collection_name: str
    field_name: str
    expected: bool = True
    present: bool | None = None
    warning_code: str | None = None


class RetrievalScopeStatus(BaseModel):
    """Dense/sparse retrieval readiness for one logical collection."""

    scope: ChunkScope
    collection_name: str
    state: Literal[
        "ready",
        "intentionally_empty",
        "legacy_dense_only",
        "missing",
        "incompatible",
        "unknown",
    ]
    points_count: int = 0
    dense_available: bool = False
    dense_schema_kind: Literal["named", "unnamed_legacy", "missing", "unknown"] = "unknown"
    sparse_available: bool = False
    sparse_contract_compatible: bool = False
    sparse_stats_stale: bool = False
    sparse_manifest_hash: str | None = None
    sparse_vocabulary_hash: str | None = None
    sparse_corpus_stats_hash: str | None = None
    expected_lexical_analyzer_version: str | None = None
    stored_lexical_analyzer_version: str | None = None
    expected_sparse_encoder_kind: str | None = None
    stored_sparse_encoder_kind: str | None = None
    payload_indexes: list[PayloadIndexStatus] = Field(default_factory=list)
    unavailable_codes: list[str] = Field(default_factory=list)


class RetrievalContractStatus(BaseModel):
    """Current retrieval route capability summary for status consumers."""

    dense_available: bool
    sparse_available: bool
    sparse_contract_compatible: bool
    fusion_methods_supported: list[str] = Field(default_factory=lambda: ["weighted_rrf"])
    payload_filter_pushdown_supported: bool = False
    legacy_lexical_fallback_enabled: bool = True
    unavailable_codes: list[str] = Field(default_factory=list)
    sparse_unavailable_codes: list[str] = Field(default_factory=list)
    sparse_manifest_hash: str | None = None
    sparse_vocabulary_hash: str | None = None
    sparse_corpus_stats_hash: str | None = None
    sparse_stats_stale: bool = False
    expected_lexical_analyzer_version: str | None = None
    stored_lexical_analyzer_version: str | None = None
    expected_sparse_encoder_kind: str | None = None
    stored_sparse_encoder_kind: str | None = None
    scope_statuses: list[RetrievalScopeStatus] = Field(default_factory=list)


class IndexCounts(BaseModel):
    """Cheap index counts optimized for agent preflight."""

    code_points: int
    docs_points: int
    total_points: int
    code_lexical_documents: int | None = None
    docs_lexical_documents: int | None = None


class LifecycleStatus(BaseModel):
    """Lifecycle status for active runtime repo."""

    state: RepoStatus
    action_in_progress: bool
    operation_id: str | None = None
    phase: str | None = None
    started_at: str | None = None
    updated_at: str | None = None
    progress: dict[str, int | float | str] = Field(default_factory=dict)
    last_error: str | None = None
    last_error_at: str | None = None


class FreshnessPolicy(BaseModel):
    """Policy for interpreting git freshness diagnostics."""

    search_allowed_when_stale: bool = True
    stale_severity: FreshnessSeverity = "none"
    requires_rebuild: bool = False
    exact_fallback_recommended: bool = False


class WatcherStatusSummary(BaseModel):
    """Watcher status summary."""

    enabled: bool
    running: bool
    policy: str = "single_active_repo"
    runtime_available: bool = False
    autostart_policy: str = "manual"
    start_required: bool = False
    start_blocked_reason: str | None = None
    last_event_at: str | None = None
    last_error: str | None = None
    last_incremental_update_ts: str | None = None


class GitWorktreeStatus(BaseModel):
    """Git-состояние repo_root на момент status-запроса."""

    is_git_repo: bool
    branch: str | None = None
    head_commit: str | None = None
    worktree_dirty: bool = False
    changed_files_count: int = 0
    untracked_files_count: int = 0
    changed_indexable_files_count: int = 0
    untracked_indexable_files_count: int = 0
    changed_indexable_files_sample: list[str] = Field(default_factory=list)
    untracked_indexable_files_sample: list[str] = Field(default_factory=list)
    error: str | None = None


class IndexFreshnessStatus(BaseModel):
    """Machine-readable freshness diagnostics for agent preflight."""

    indexed_branch: str | None = None
    indexed_commit_hash: str | None = None
    current_branch: str | None = None
    current_head_commit: str | None = None
    primary_state: str = "fresh"
    states: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    head_mismatch: bool = False
    indexable_worktree_changes: bool = False
    untracked_indexable_files: bool = False
    stale: bool = False
    freshness_unknown: bool = False
    stale_reasons: list[str] = Field(default_factory=list)
    host_side_hint: str | None = None
    policy: FreshnessPolicy | None = None


class PathFreshnessSummary(BaseModel):
    """Path-level freshness coverage backed by the per-index manifest."""

    coverage_complete: bool = False
    coverage_error_code: str | None = None
    manifest_available: bool = False
    manifest_path: str | None = None
    manifest_schema_version: int | None = None
    identity_compatible: bool = False
    changed_indexable_paths_count: int = 0
    untracked_indexable_paths_count: int = 0
    missing_from_index_count: int = 0
    stale_indexed_paths_count: int = 0
    deleted_indexed_paths_count: int = 0
    path_error_count: int = 0
    stale_paths_preview: list[str] = Field(default_factory=list)
    deleted_paths_preview: list[str] = Field(default_factory=list)
    error_paths_preview: list[str] = Field(default_factory=list)


class IndexStatusResult(BaseModel):
    """Итоговый статус semantic индекса."""

    contract_version: str = "index_status.v2"
    status_kind: str = "runtime_repo_status"
    repo_root: str
    mounted_repo_root: str
    active_repo_root: str | None = None
    runtime_switch_required: bool = False
    repo_key: str
    repo_state: RepoStatus
    active: bool
    search_available: bool
    reason_if_unavailable: str | None = None
    index_profile: str
    embedding_backend: str
    embedding_model: str
    embedding_backend_type: str | None = None
    embedding_backend_id: str | None = None
    selected_embedding_backend_id: str | None = None
    backend_switch_required: bool = False
    runtime_switch_command_hint: str | None = None
    backend_switch_command_hint: str | None = None
    qdrant_url: str
    schema_version: int
    watch_enabled: bool
    watch_running: bool
    include_globs: list[str] = Field(default_factory=list)
    doc_prefixes: list[str] = Field(default_factory=list)
    last_full_build_ts: str | None = None
    last_incremental_update_ts: str | None = None
    active_branch: str | None = None
    head_commit_hash: str | None = None
    indexed_branch: str | None = None
    indexed_commit_hash: str | None = None
    worktree_dirty: bool = False
    index_stale: bool = False
    git: GitWorktreeStatus | None = None
    freshness: IndexFreshnessStatus | None = None
    path_freshness: PathFreshnessSummary | None = None
    availability: StatusAvailability | None = None
    runtime: RuntimeStatusSummary | None = None
    backend: BackendStatusSummary | None = None
    index_contract: IndexContractStatus | None = None
    retrieval: RetrievalContractStatus | None = None
    graph: GraphStatusSummary | None = None
    counts: IndexCounts | None = None
    lifecycle: LifecycleStatus | None = None
    watcher: WatcherStatusSummary | None = None
    warnings: list[StatusWarning] = Field(default_factory=list)
    collections: list[IndexCollectionStatus]
