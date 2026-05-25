"""repo_context_search orchestration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from services.repo_semantic.context import actions as context_actions
from services.repo_semantic.context import assembler as context_assembler
from services.repo_semantic.context import diagnostics as context_diagnostics
from services.repo_semantic.models import (
    BranchDiagnostics,
    ChunkRecord,
    ExactAnchorAnalysis,
    ExactAnchorUsage,
    GraphBranchDiagnostics,
    GraphMode,
    IndexStatusResult,
    QueryUsage,
    RepoContextRoute,
    RepoContextSearchResponse,
    RecommendedAction,
    SearchDiagnostics,
    SearchFilters,
    SearchResult,
    SearchScope,
    SearchWarning,
    SnippetMode,
)
from services.repo_semantic.graph.types import GraphExpansionResult
from services.repo_semantic.retrieval.constants import (
    MAX_CONTEXT_FILE_GROUPS,
    RRF_K,
    MAX_TOP_K,
)
from services.repo_semantic.retrieval.types import _SearchExecution

ContextExecution = Callable[..., _SearchExecution]
GraphContextExpansion = Callable[..., GraphExpansionResult]
GraphResultBuilder = Callable[..., SearchResult]


def repo_context_search_response(
    *,
    query: str,
    subqueries: list[str] | None,
    route: RepoContextRoute,
    graph_mode: GraphMode,
    scope: SearchScope,
    path_prefix: str | None,
    include_paths: list[str] | None,
    exclude_paths: list[str] | None,
    file_extensions: list[str] | None,
    languages: list[str] | None,
    chunk_types: list[str] | None,
    domain_tags: list[str] | None,
    top_k: int,
    max_results_per_file: int | None,
    snippet_mode: SnippetMode,
    include_diagnostics: bool,
    include_explanations: bool,
    normalize_top_k: Callable[[int], int],
    normalize_context_queries: Callable[[str, list[str] | None], tuple[list[tuple[str, str, float]], int]],
    context_route_used: Callable[[RepoContextRoute], Literal["semantic", "hybrid", "exact_handoff"]],
    aggregate_exact_anchors: Callable[
        [list[tuple[str, str, float]]],
        tuple[ExactAnchorAnalysis, list[ExactAnchorUsage]],
    ],
    context_execution: ContextExecution,
    graph_context_expansion: GraphContextExpansion | None,
    graph_result_builder: GraphResultBuilder,
    build_filters: Callable[..., SearchFilters],
    index_status: Callable[[], IndexStatusResult],
) -> RepoContextSearchResponse:
    """Build the agent-facing repo context envelope."""

    if graph_mode not in {"off", "auto", "expand"}:
        raise ValueError("graph_mode must be one of: off, auto, expand")
    if snippet_mode not in {"chunk_start", "query_centered"}:
        raise ValueError("snippet_mode must be 'chunk_start' or 'query_centered'")
    route_used = context_route_used(route)
    top_k = normalize_top_k(top_k)
    if top_k == 0:
        top_k = 0
    query_items, dropped_subqueries = normalize_context_queries(query, subqueries)
    exact_anchor_analysis, exact_anchor_usages = aggregate_exact_anchors(query_items)

    warnings: list[SearchWarning] = []
    if dropped_subqueries:
        warnings.append(
            SearchWarning(
                code="multi_query_subquery_dropped",
                severity="info",
                detail=f"{dropped_subqueries} subquery value(s) were empty, duplicate, or over the cap.",
            )
        )
    status = None
    path_freshness = None
    stale_index = False
    try:
        status = index_status()
        path_freshness = getattr(status, "path_freshness", None)
        freshness = getattr(status, "freshness", None)
        policy = getattr(freshness, "policy", None)
        stale_index = bool(getattr(policy, "exact_fallback_recommended", False))
        if stale_index:
            warnings.append(
                SearchWarning(
                    code="stale_index_verify_with_exact_search",
                    severity="warning",
                    detail="Index freshness is stale; verify candidates with exact search before editing.",
                )
            )
    except Exception:  # noqa: BLE001
        status = None

    query_usages: list[QueryUsage] = []
    per_query_diagnostics: list[SearchDiagnostics] = []
    executions: list[tuple[str, str, float, _SearchExecution]] = []
    graph_diagnostics = None
    graph_actions = []
    graph_mode_effective = "off"
    evidence_paths_by_chunk = {}
    branch_score_overrides_by_chunk = {}
    origin_queries_by_chunk = {}
    origin_roles_by_chunk = {}
    graph_branch_diagnostics = None

    if route_used == "exact_handoff":
        if not exact_anchor_analysis.anchors:
            warnings.append(
                SearchWarning(
                    code="exact_anchor_not_detected",
                    severity="warning",
                    detail="No exact-looking anchor was detected; retry with hybrid or semantic retrieval.",
                )
            )
        query_usages = [
            QueryUsage(
                query=query_text,
                role=role,  # type: ignore[arg-type]
                route_used="exact_handoff",
                weight=weight,
                result_count=0,
                warning_codes=[],
            )
            for query_text, role, weight in query_items
        ]
    else:
        internal_top_k = min(MAX_TOP_K, max(top_k * 4, 50))
        for query_text, role, weight in query_items:
            execution = context_execution(
                query=query_text,
                route_used=route_used,
                scope=scope,
                path_prefix=path_prefix,
                include_paths=include_paths,
                exclude_paths=exclude_paths,
                file_extensions=file_extensions,
                languages=languages,
                chunk_types=chunk_types,
                domain_tags=domain_tags,
                top_k=internal_top_k,
                snippet_mode=snippet_mode,
                include_explanations=include_explanations,
            )
            query_warnings = list(execution.warnings)
            if not execution.results:
                query_warnings.append(
                    SearchWarning(
                        code="empty_result_best_effort",
                        severity="info",
                        detail="No indexed chunks matched the query and filters.",
                    )
                )
            warnings.extend(query_warnings)
            per_query_diagnostics.append(execution.diagnostics)
            executions.append((query_text, role, weight, execution))
            query_usages.append(
                QueryUsage(
                    query=query_text,
                    role=role,  # type: ignore[arg-type]
                    route_used=route_used,
                    weight=weight,
                    result_count=len(execution.results),
                    warning_codes=sorted({warning.code for warning in query_warnings}),
                )
            )

    if route_used != "exact_handoff":
        filters = build_filters(
            path_prefix=path_prefix,
            include_paths=include_paths,
            exclude_paths=exclude_paths,
            file_extensions=file_extensions,
            languages=languages,
            chunk_types=chunk_types,
            domain_tags=domain_tags,
        )
        try:
            if stale_index and graph_mode != "off":
                graph_expansion = GraphExpansionResult(
                    mode_effective="off",
                    diagnostics=GraphBranchDiagnostics(
                        requested_mode=graph_mode,
                        effective_mode="off",
                        available=False,
                        used=False,
                        expansion_allowed=False,
                        skipped_reason="stale_index",
                        seed_count=sum(
                            len(execution.results)
                            for _query, _role, _weight, execution in executions
                        ),
                        warning_codes=["graph_skipped_stale_index"],
                    ),
                    warnings=[
                        SearchWarning(
                            code="graph_skipped_stale_index",
                            severity="warning",
                            detail="Graph expansion was skipped because index freshness is stale.",
                        )
                    ],
                )
            else:
                graph_expansion = (
                    graph_context_expansion(
                        graph_mode=graph_mode,
                        route_used=route_used,
                        query_items=query_items,
                        executions=executions,
                        exact_anchor_analysis=exact_anchor_analysis,
                        filters=filters,
                        path_freshness=path_freshness,
                    )
                    if graph_context_expansion is not None
                    else GraphExpansionResult(
                        mode_effective="off",
                        diagnostics=None,
                        warnings=[
                            SearchWarning(
                                code="graph_branch_unavailable",
                                severity="info",
                                detail="Graph expansion is unavailable because no read-only graph provider is configured.",
                            )
                        ]
                        if graph_mode == "expand"
                        else [],
                    )
                )
        except Exception as exc:  # noqa: BLE001
            graph_error_warning = SearchWarning(
                code="graph_error",
                severity="warning",
                detail=f"Graph expansion failed closed; dense/sparse results were preserved. Error: {exc}",
            )
            graph_expansion = GraphExpansionResult(
                mode_effective="off",
                diagnostics=GraphBranchDiagnostics(
                    requested_mode=graph_mode,
                    effective_mode="off",
                    available=False,
                    used=False,
                    status_state="error",
                    expansion_allowed=False,
                    skipped_reason="graph_error",
                    seed_count=sum(len(execution.results) for _query, _role, _weight, execution in executions),
                    warning_codes=["graph_error"],
                ),
                warnings=[graph_error_warning],
                actions=[
                    RecommendedAction(
                        code="rebuild_graph",
                        severity="warning",
                        title="Rebuild graph artifact",
                        detail="Graph expansion failed while reading graph state; lifecycle remains explicit.",
                        tool_hint="rebuild_graph",
                    )
                ]
                if graph_mode == "expand"
                else [],
            )
        graph_mode_effective = graph_expansion.mode_effective
        graph_diagnostics = graph_expansion.diagnostics
        graph_actions = graph_expansion.actions
        warnings.extend(graph_expansion.warnings)
        if graph_expansion.candidates:
            graph_weight = 1.0 if graph_expansion.mode_effective == "expand" else 0.7
            graph_results = []
            graph_branch_ranks = {}
            graph_branch_diagnostics = BranchDiagnostics(
                available=True,
                candidate_limit=None,
                candidates_returned=len(graph_expansion.candidates),
                warning_codes=list(graph_diagnostics.warning_codes) if graph_diagnostics is not None else [],
            )
            for candidate in graph_expansion.candidates:
                fused_score = graph_weight / (RRF_K + max(1, candidate.branch_rank))
                graph_results.append(
                    graph_result_builder(
                        chunk=candidate.chunk,
                        score=fused_score,
                        matched_terms=candidate.matched_terms,
                        snippet_mode=snippet_mode,
                        include_explanations=include_explanations,
                    )
                )
                graph_branch_ranks[candidate.chunk_id] = {"graph": candidate.branch_rank}
                evidence_paths_by_chunk[candidate.chunk_id] = list(candidate.evidence_paths)
                branch_score_overrides_by_chunk[candidate.chunk_id] = {"graph": candidate.branch_score}
                origin_queries_by_chunk[candidate.chunk_id] = list(candidate.origin_queries)
                origin_roles_by_chunk[candidate.chunk_id] = dict(candidate.origin_query_roles)
            executions.append(
                (
                    query_items[0][0],
                    "subquery",
                    graph_weight,
                    _SearchExecution(
                        results=graph_results,
                        diagnostics=SearchDiagnostics(
                            final_results=len(graph_results),
                            retrieval_backend="sqlite_graph",
                            fusion_method="graph_seed_expansion",
                            branch_diagnostics={"graph": graph_branch_diagnostics},
                        ),
                        warnings=[],
                        branch_ranks_by_chunk=graph_branch_ranks,
                    ),
                )
            )

    results, global_matched_terms, uncovered_terms = context_assembler.merge_context_results(
        executions=executions,
        query_items=query_items,
        route_used=route_used,
        top_k=top_k,
        max_results_per_file=max_results_per_file,
        exact_anchor_analysis=exact_anchor_analysis,
        stale_index=stale_index,
        warnings=warnings,
        evidence_paths_by_chunk=evidence_paths_by_chunk,
        branch_score_overrides_by_chunk=branch_score_overrides_by_chunk,
        origin_queries_by_chunk=origin_queries_by_chunk,
        origin_roles_by_chunk=origin_roles_by_chunk,
        force_rrf_fusion=bool(branch_score_overrides_by_chunk),
    )

    file_groups = context_assembler.context_file_groups(results, top_k=top_k or MAX_CONTEXT_FILE_GROUPS)
    actions, sparse_codes = context_actions.build_context_actions(
        results=results,
        file_groups=file_groups,
        exact_anchor_analysis=exact_anchor_analysis,
        per_query_diagnostics=per_query_diagnostics,
        status=status,
        route_used=route_used,
        scope=scope,
        path_prefix=path_prefix,
        include_paths=include_paths,
        query_items=query_items,
        top_k=top_k,
        global_matched_terms=global_matched_terms,
    )
    actions.extend(graph_actions)

    warnings = context_assembler.merge_context_warnings(warnings)
    diagnostics = context_diagnostics.build_context_diagnostics(
        route_used=route_used,
        query_items=query_items,
        per_query_diagnostics=per_query_diagnostics,
        exact_anchor_analysis=exact_anchor_analysis,
        exact_anchor_usages=exact_anchor_usages,
        results=results,
        file_groups=file_groups,
        global_matched_terms=global_matched_terms,
        uncovered_terms=uncovered_terms,
        sparse_codes=sparse_codes,
        graph_diagnostics=graph_diagnostics,
        graph_branch_diagnostics=graph_branch_diagnostics,
        graph_fusion_used=bool(branch_score_overrides_by_chunk),
        include_diagnostics=include_diagnostics,
    )
    return RepoContextSearchResponse(
        query=query,
        queries_used=query_usages,
        route_requested=route,
        route_used=route_used,
        graph_mode_requested=graph_mode,
        graph_mode_effective=graph_mode_effective,  # type: ignore[arg-type]
        results=results,
        file_groups=file_groups,
        warnings=warnings,
        diagnostics=diagnostics if include_diagnostics else None,
        recommended_next_actions=context_actions.dedupe_context_actions(actions),
    )
