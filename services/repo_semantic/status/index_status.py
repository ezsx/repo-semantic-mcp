"""Index status v2 assembly."""

from __future__ import annotations

from collections.abc import Callable

from services.repo_semantic.git_status import inspect_git_worktree
from services.repo_semantic.models import (
    BackendStatusSummary,
    GraphStatusSummary,
    IndexCollectionStatus,
    IndexContractStatus,
    IndexCounts,
    IndexStatusResult,
    LifecycleStatus,
    PathFreshnessSummary,
    RuntimeStatusSummary,
    StatusAction,
    StatusAvailability,
    StatusWarning,
    WatcherStatusSummary,
)
from services.repo_semantic.status import freshness as status_freshness
from services.repo_semantic.status import index_contract as status_index_contract
from services.repo_semantic.status import readiness as status_readiness
from services.repo_semantic.status import retrieval_status as status_retrieval


def build_index_status(
    *,
    settings,
    embedding_provider,
    store,
    indexer,
    watcher,
    registry,
    read_collection_exists: Callable[[str], bool],
    collection_contract: Callable[..., object],
    retrieval_status_for_points: Callable[[dict[str, int]], object],
    graph_status_summary: Callable[[], GraphStatusSummary] | None,
    sha256_text: Callable[[str], str],
) -> IndexStatusResult:
    """Build the full index_status.v2 payload."""

    repo_entry = registry.get_repo(settings.logical_repo_identity)
    active_repo = registry.get_active_repo()
    selected_embedding_backend = registry.get_role_backend("embedding")
    selected_embedding_backend_id = (
        selected_embedding_backend.backend_id
        if selected_embedding_backend is not None
        else None
    )
    runtime_embedding_backend_id = settings.embedding_backend_id
    runtime_index_profile = embedding_provider.index_profile()
    runtime_repo_is_active = bool(
        repo_entry and active_repo and active_repo.repo_root == repo_entry.repo_root
    )
    contract_issue = status_index_contract.index_contract_issue(
        store=store,
        settings=settings,
        embedding_backend=embedding_provider.backend_name(),
        embedding_model=embedding_provider.model_name(),
    )
    search_unavailable_codes = status_readiness.search_unavailable_codes(
        repo_entry,
        selected_backend_id=selected_embedding_backend_id,
        runtime_backend_id=runtime_embedding_backend_id,
        runtime_repo_is_active=runtime_repo_is_active,
        contract_issue=contract_issue,
    )
    reason_if_unavailable = status_readiness.reason_for_unavailable_codes(
        search_unavailable_codes,
        repo_entry,
        contract_issue=contract_issue,
    )
    effective_repo_state = repo_entry.status if repo_entry else "registered"
    points_by_scope: dict[str, int] = {}
    collections: list[IndexCollectionStatus] = []
    for scope in ("code", "docs"):
        points_count = store.count(scope)
        points_by_scope[scope] = points_count
        collections.append(
            IndexCollectionStatus(
                scope=scope,
                collection_name=store.collection_name(scope),
                points_count=points_count,
                lexical_documents=points_count,
            )
        )

    git_snapshot = inspect_git_worktree(
        settings.repo_root,
        include_globs=list(repo_entry.include_globs)
        if repo_entry
        else list(settings.effective_include_globs),
        exclude_globs=list(repo_entry.exclude_globs)
        if repo_entry
        else list(settings.SEMANTIC_MCP_EXCLUDE_GLOBS),
    )
    path_freshness = None
    path_manifest = getattr(indexer, "path_manifest", None)
    if path_manifest is not None:
        try:
            try:
                path_iter = indexer.iter_indexable_paths(
                    max_paths_scanned=getattr(settings, "SEMANTIC_MCP_STATUS_MAX_SCAN_PATHS", None),
                    max_duration_ms=getattr(settings, "SEMANTIC_MCP_STATUS_MAX_DURATION_MS", None),
                )
            except TypeError:
                path_iter = indexer.iter_indexable_paths()
            path_freshness = path_manifest.summarize(
                path_iter,
                max_duration_ms=getattr(settings, "SEMANTIC_MCP_STATUS_MAX_DURATION_MS", None),
                max_hashed_paths=getattr(settings, "SEMANTIC_MCP_STATUS_MAX_HASHED_PATHS", None),
            )
            path_freshness.changed_indexable_paths_count = git_snapshot.changed_indexable_files_count
            path_freshness.untracked_indexable_paths_count = git_snapshot.untracked_indexable_files_count
        except Exception as exc:  # noqa: BLE001
            coverage_error_code = (
                "path_manifest_status_bounds_exceeded"
                if str(exc) == "indexable_path_scan_bounds_exceeded"
                or getattr(exc, "summary", {}).get("warning_codes") == ["startup_reconcile_bounds_exceeded"]
                else "path_manifest_summary_failed"
            )
            path_freshness = PathFreshnessSummary(
                coverage_complete=False,
                coverage_error_code=coverage_error_code,
                manifest_available=bool(
                    getattr(path_manifest, "exists", lambda: False)()
                ),
                manifest_path=str(getattr(path_manifest, "path", "")) or None,
                error_paths_preview=[str(exc)[:240]],
            )
    freshness_build = status_freshness.build_freshness_status(
        git_snapshot=git_snapshot,
        repo_entry=repo_entry,
        repo_root=settings.repo_root,
        path_freshness=path_freshness,
    )
    git_status = freshness_build.git_status
    freshness = freshness_build.freshness
    stale_reasons = freshness_build.stale_reasons
    indexed_branch = freshness_build.indexed_branch
    indexed_commit_hash = freshness_build.indexed_commit_hash
    warnings = list(freshness_build.warnings)
    runtime_switch_required = bool(
        active_repo and active_repo.repo_root != settings.logical_repo_identity
    )
    runtime_switch_command_hint = (
        "pwsh -File scripts/agents/start_repo_semantic_for_project.ps1 "
        f"-RepoPath '{active_repo.repo_root}'"
        if runtime_switch_required and active_repo
        else None
    )
    backend_switch_required = bool(
        selected_embedding_backend_id
        and selected_embedding_backend_id != runtime_embedding_backend_id
    )
    backend_switch_command_hint = (
        "Call get_backend_action_plan(role='embedding') for the exact host-side command."
        if backend_switch_required
        else None
    )
    watcher_enabled = bool(settings.SEMANTIC_MCP_WATCH_ENABLED)
    watcher_runtime_available = watcher is not None
    watcher_running = bool(watcher and watcher.is_running)
    watcher_start_blocked_reason: str | None = None
    if not watcher_enabled:
        watcher_start_blocked_reason = "disabled_by_config"
    elif not watcher_runtime_available:
        watcher_start_blocked_reason = "runtime_watcher_unavailable"
    elif watcher_running:
        watcher_start_blocked_reason = None
    elif not runtime_repo_is_active:
        watcher_start_blocked_reason = "runtime_repo_not_active"
    elif effective_repo_state != "indexed":
        watcher_start_blocked_reason = "repo_not_indexed"
    elif search_unavailable_codes:
        watcher_start_blocked_reason = "search_unavailable"
    watcher_start_required = (
        watcher_enabled
        and watcher_runtime_available
        and not watcher_running
        and watcher_start_blocked_reason is None
    )
    next_actions = [
        status_readiness.status_action_for_code(code)
        for code in search_unavailable_codes
    ]
    if stale_reasons:
        next_actions.append(
            StatusAction(
                code="use_exact_search_fallback",
                severity="warning",
                title="Verify stale semantic results with exact search",
                detail="Git freshness is stale; use local rg or exact file reads before editing.",
            )
        )
    if (
        watcher_enabled
        and watcher_runtime_available
        and not watcher_running
        and runtime_repo_is_active
        and effective_repo_state == "indexed"
    ):
        warnings.append(
            StatusWarning(
                code="watcher_enabled_but_not_running",
                severity="warning",
                detail=(
                    "Incremental watcher is configured but not running; "
                    "new file changes will not be indexed automatically."
                ),
            )
        )
    if watcher_start_required:
        next_actions.append(
            StatusAction(
                code="start_watcher",
                severity="info",
                title="Start incremental watcher",
                detail=(
                    "Call start_watcher only after explicit user approval if live "
                    "incremental indexing is desired."
                ),
                tool_hint="start_watcher",
            )
        )

    collection_contracts = [
        collection_contract(
            scope,
            points_count=points_by_scope[scope],
            lexical_documents=points_by_scope[scope],
            runtime_embedding_backend=embedding_provider.backend_name(),
            runtime_embedding_model=embedding_provider.model_name(),
            runtime_query_template_hash=sha256_text(settings.SEMANTIC_MCP_QUERY_TEMPLATE),
            runtime_document_prefix_hash=sha256_text(settings.SEMANTIC_MCP_DOCUMENT_PREFIX),
        )
        for scope in ("code", "docs")
    ]
    retrieval_status = retrieval_status_for_points(points_by_scope)
    graph_summary = graph_status_summary() if graph_status_summary is not None else None
    retrieval_warnings, retrieval_actions = status_retrieval.retrieval_warnings_and_actions(
        retrieval_status=retrieval_status,
        total_points=points_by_scope["code"] + points_by_scope["docs"],
    )
    warnings.extend(retrieval_warnings)
    next_actions.extend(retrieval_actions)
    active_lifecycle_lease = None
    if repo_entry is not None:
        try:
            active_lifecycle_lease = registry.get_active_lifecycle_lease(
                settings.logical_repo_identity,
                index_profile=runtime_index_profile,
                backend_id=runtime_embedding_backend_id,
            )
        except Exception:  # noqa: BLE001
            active_lifecycle_lease = None
    if graph_summary is not None and graph_summary.state != "ready":
        warning_severity = "warning" if graph_summary.state in {"stale", "incompatible", "error"} else "info"
        warnings.append(
            StatusWarning(
                code=f"graph_{graph_summary.state}",
                severity=warning_severity,
            )
        )
        if active_lifecycle_lease is not None:
            pass
        elif graph_summary.state == "missing":
            next_actions.append(
                StatusAction(
                    code="build_graph",
                    severity="info",
                    title="Build graph artifact",
                    detail="Graph expansion is unavailable until build_graph is run explicitly.",
                    tool_hint="build_graph",
                )
            )
        elif graph_summary.state == "stale" and graph_summary.expansion_allowed:
            pass
        elif graph_summary.state == "stale" and graph_summary.update_required:
            next_actions.append(
                StatusAction(
                    code="update_graph",
                    severity="warning",
                    title="Update graph artifact",
                    detail="Graph artifact is behind the current source index and can be repaired incrementally.",
                    tool_hint="update_graph",
                )
            )
        elif (
            graph_summary.state == "stale"
            and "graph_invalidated_paths_unknown" in graph_summary.warning_codes
        ):
            pass
        elif graph_summary.state in {"stale", "incompatible", "error"}:
            next_actions.append(
                StatusAction(
                    code="rebuild_graph",
                    severity="warning",
                    title="Rebuild graph artifact",
                    detail="Graph expansion should stay disabled until the graph artifact is rebuilt.",
                    tool_hint="rebuild_graph",
                )
            )
    contract_summary = status_index_contract.summarize_collection_contracts(collection_contracts)
    for warning_code in contract_summary.warning_codes:
        warnings.append(
            StatusWarning(
                code=warning_code,
                severity="warning" if warning_code.endswith("mismatch") else "info",
            )
        )
    freshness_extra_states: list[str] = []
    freshness_extra_reasons: list[str] = []
    if retrieval_status.sparse_stats_stale:
        freshness_extra_states.append("sparse_stats_stale")
        freshness_extra_reasons.append("sparse_stats_stale")
    if graph_summary is not None and graph_summary.state == "stale" and graph_summary.expansion_allowed:
        freshness_extra_states.append("graph_degraded")
        freshness_extra_reasons.append("graph_stale")
        freshness_extra_reasons.extend(graph_summary.warning_codes)
    if freshness_extra_states or freshness_extra_reasons:
        freshness = status_freshness.add_freshness_states(
            freshness,
            states=freshness_extra_states,
            reason_codes=freshness_extra_reasons,
            exact_fallback_recommended=False,
        )

    return IndexStatusResult(
        repo_root=settings.logical_repo_identity,
        mounted_repo_root=str(settings.repo_root),
        active_repo_root=active_repo.repo_root if active_repo else None,
        runtime_switch_required=runtime_switch_required,
        repo_key=repo_entry.repo_key if repo_entry else settings.repo_key_slug,
        repo_state=effective_repo_state,
        active=runtime_repo_is_active,
        search_available=not search_unavailable_codes,
        reason_if_unavailable=reason_if_unavailable,
        index_profile=repo_entry.index_profile if repo_entry else embedding_provider.index_profile(),
        embedding_backend=embedding_provider.backend_name(),
        embedding_model=embedding_provider.model_name(),
        embedding_backend_type=embedding_provider.backend_name(),
        embedding_backend_id=runtime_embedding_backend_id,
        selected_embedding_backend_id=selected_embedding_backend_id or runtime_embedding_backend_id,
        backend_switch_required=backend_switch_required,
        runtime_switch_command_hint=runtime_switch_command_hint,
        backend_switch_command_hint=backend_switch_command_hint,
        qdrant_url=settings.SEMANTIC_MCP_QDRANT_URL,
        schema_version=settings.SEMANTIC_MCP_INDEX_SCHEMA_VERSION,
        watch_enabled=watcher_enabled,
        watch_running=watcher_running,
        include_globs=list(repo_entry.include_globs)
        if repo_entry
        else list(settings.effective_include_globs),
        doc_prefixes=list(repo_entry.doc_prefixes)
        if repo_entry
        else list(settings.effective_doc_prefixes),
        last_full_build_ts=repo_entry.last_full_build_ts
        if repo_entry
        else indexer.last_full_build_ts,
        last_incremental_update_ts=repo_entry.last_incremental_update_ts
        if repo_entry
        else indexer.last_incremental_update_ts,
        active_branch=git_status.branch,
        head_commit_hash=git_status.head_commit,
        indexed_branch=indexed_branch,
        indexed_commit_hash=indexed_commit_hash,
        worktree_dirty=git_status.worktree_dirty,
        index_stale=freshness.stale,
        git=git_status,
        freshness=freshness,
        path_freshness=path_freshness,
        availability=StatusAvailability(
            runtime_ready=True,
            dependencies_ready=None,
            search_available=not search_unavailable_codes,
            search_unavailable_codes=search_unavailable_codes,
            host_action_required=any(action.severity == "blocking" for action in next_actions),
            next_actions=next_actions,
        ),
        runtime=RuntimeStatusSummary(
            mounted_repo_root=str(settings.repo_root),
            logical_repo_root=settings.logical_repo_identity,
            active_repo_root=active_repo.repo_root if active_repo else None,
            runtime_switch_required=runtime_switch_required,
            runtime_switch_command_hint=runtime_switch_command_hint,
        ),
        backend=BackendStatusSummary(
            runtime_embedding_backend_id=runtime_embedding_backend_id,
            selected_embedding_backend_id=selected_embedding_backend_id or runtime_embedding_backend_id,
            backend_type=selected_embedding_backend.backend_type
            if selected_embedding_backend is not None
            else settings.SEMANTIC_MCP_EMBEDDING_BACKEND,
            model_name=selected_embedding_backend.model_name
            if selected_embedding_backend is not None
            else settings.SEMANTIC_MCP_EMBEDDING_MODEL,
            compatible=not backend_switch_required,
            switch_required=backend_switch_required,
            healthy=None,
            managed_by_service=selected_embedding_backend.managed_by_service
            if selected_embedding_backend is not None
            else settings.embedding_backend_managed_by_service,
            autostart_policy=selected_embedding_backend.autostart_policy
            if selected_embedding_backend is not None
            else settings.SEMANTIC_MCP_EMBEDDING_AUTOSTART_POLICY,
            location_type=selected_embedding_backend.location_type
            if selected_embedding_backend is not None
            else settings.embedding_backend_location_type,
            device_type=selected_embedding_backend.device_type
            if selected_embedding_backend is not None
            else settings.embedding_backend_device_type,
        ),
        index_contract=IndexContractStatus(
            profile=repo_entry.index_profile if repo_entry else embedding_provider.index_profile(),
            embedding_backend=embedding_provider.backend_name(),
            embedding_model=embedding_provider.model_name(),
            schema_version=settings.SEMANTIC_MCP_INDEX_SCHEMA_VERSION,
            query_template_hash=sha256_text(settings.SEMANTIC_MCP_QUERY_TEMPLATE),
            document_prefix_hash=sha256_text(settings.SEMANTIC_MCP_DOCUMENT_PREFIX),
            compatibility=contract_summary.compatibility,  # type: ignore[arg-type]
            blocking=contract_summary.blocking,
            incompatibility_codes=contract_summary.incompatibility_codes,
            collections=collection_contracts,
        ),
        retrieval=retrieval_status,
        graph=graph_summary,
        counts=IndexCounts(
            code_points=points_by_scope["code"],
            docs_points=points_by_scope["docs"],
            total_points=points_by_scope["code"] + points_by_scope["docs"],
        ),
        lifecycle=LifecycleStatus(
            state=effective_repo_state,  # type: ignore[arg-type]
            action_in_progress=effective_repo_state == "indexing" or active_lifecycle_lease is not None,
            operation_id=active_lifecycle_lease.lease_id if active_lifecycle_lease else None,
            phase=active_lifecycle_lease.operation if active_lifecycle_lease else None,
            started_at=active_lifecycle_lease.acquired_at if active_lifecycle_lease else None,
            updated_at=(
                active_lifecycle_lease.heartbeat_at
                if active_lifecycle_lease
                else repo_entry.updated_at
                if repo_entry
                else None
            ),
            last_error=repo_entry.last_error if repo_entry else None,
        ),
        watcher=WatcherStatusSummary(
            enabled=watcher_enabled,
            running=watcher_running,
            runtime_available=watcher_runtime_available,
            start_required=watcher_start_required,
            start_blocked_reason=watcher_start_blocked_reason,
            last_incremental_update_ts=repo_entry.last_incremental_update_ts
            if repo_entry
            else indexer.last_incremental_update_ts,
        ),
        warnings=warnings,
        collections=collections,
    )
