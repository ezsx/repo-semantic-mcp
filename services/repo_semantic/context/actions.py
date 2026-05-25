"""Recommended-action helpers for repo_context_search."""

from __future__ import annotations

from services.repo_semantic.contracts.common import SearchScope
from services.repo_semantic.contracts.repo_context import FileContextGroup, RecommendedAction, RepoContextResult
from services.repo_semantic.contracts.search import ExactAnchorAnalysis, SearchDiagnostics
from services.repo_semantic.context.query_plan import ContextQueryItem
from services.repo_semantic.retrieval.constants import CONTEXT_REBUILD_CODES, MAX_CONTEXT_READ_ACTIONS


def context_read_actions(results: list[RepoContextResult]) -> list[RecommendedAction]:
    actions: list[RecommendedAction] = []
    seen_paths: set[tuple[str, int | None, int | None]] = set()
    for result in results:
        start_line = result.snippet_start_line or result.start_line
        end_line = result.snippet_end_line or result.end_line
        key = (result.relative_path, start_line, end_line)
        if key in seen_paths:
            continue
        seen_paths.add(key)
        actions.append(
            RecommendedAction(
                code="read_file_range",
                severity="info",
                title="Read indexed file range",
                relative_path=result.relative_path,
                start_line=start_line,
                end_line=end_line,
            )
        )
        if len(actions) >= MAX_CONTEXT_READ_ACTIONS:
            break
    return actions


def dedupe_context_actions(actions: list[RecommendedAction]) -> list[RecommendedAction]:
    seen: set[tuple[str, str | None, int | None, int | None, tuple[str, ...] | None]] = set()
    result: list[RecommendedAction] = []
    for action in actions:
        key = (
            action.code,
            action.relative_path,
            action.start_line,
            action.end_line,
            tuple(action.argv_hint) if action.argv_hint else None,
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(action)
    return result


def collect_sparse_codes(
    *,
    per_query_diagnostics: list[SearchDiagnostics],
    status,
    route_used: str,
) -> set[str]:
    """Collect sparse rebuild/status codes from per-query diagnostics and status."""

    sparse_codes = {
        code
        for diagnostics in per_query_diagnostics
        for code in diagnostics.sparse_unavailable_codes
    }
    if any(diagnostics.sparse_stats_stale for diagnostics in per_query_diagnostics):
        sparse_codes.add("sparse_stats_stale")
    if route_used != "exact_handoff":
        retrieval = getattr(status, "retrieval", None)
        if retrieval is not None:
            sparse_codes.update(getattr(retrieval, "sparse_unavailable_codes", []) or [])
            if getattr(retrieval, "sparse_stats_stale", False):
                sparse_codes.add("sparse_stats_stale")
    return sparse_codes


def build_context_actions(
    *,
    results: list[RepoContextResult],
    file_groups: list[FileContextGroup],
    exact_anchor_analysis: ExactAnchorAnalysis,
    per_query_diagnostics: list[SearchDiagnostics],
    status,
    route_used: str,
    scope: SearchScope,
    path_prefix: str | None,
    include_paths: list[str] | None,
    query_items: list[ContextQueryItem],
    top_k: int,
    global_matched_terms: list[str],
) -> tuple[list[RecommendedAction], set[str]]:
    """Build repo_context_search recommended actions and return sparse status codes."""

    actions: list[RecommendedAction] = context_read_actions(results)
    for argv_hint in exact_anchor_analysis.argv_hints:
        actions.append(
            RecommendedAction(
                code="run_local_rg",
                severity="warning" if exact_anchor_analysis.exact_anchor_heavy else "info",
                title="Verify exact anchor in working tree",
                argv_hint=argv_hint,
                cwd_hint="repo_root",
                authority="local_rg",
            )
        )
    sparse_codes = collect_sparse_codes(
        per_query_diagnostics=per_query_diagnostics,
        status=status,
        route_used=route_used,
    )
    if sparse_codes.intersection(CONTEXT_REBUILD_CODES):
        actions.append(
            RecommendedAction(
                code="rebuild_index",
                severity="warning",
                title="Rebuild index to refresh dense/sparse retrieval state",
                detail="Sparse retrieval is unavailable or stale; lifecycle remains explicit.",
                tool_hint="build_index(force_rebuild=true)",
            )
        )

    docs_count = sum(1 for result in results if result.scope == "docs")
    code_count = sum(1 for result in results if result.scope == "code")
    active_path_filter = bool(path_prefix or include_paths)
    query_text_lc = " ".join(item[0] for item in query_items).lower()
    if len(file_groups) >= min(10, top_k or 10) and not active_path_filter:
        actions.append(
            RecommendedAction(
                code="retry_with_path_filter",
                severity="info",
                title="Retry with a narrower path filter",
            )
        )
    if scope == "all" and docs_count == 0 and any(
        term in query_text_lc
        for term in ("docs", "spec", "specification", "policy", "guide", "readme")
    ):
        actions.append(RecommendedAction(code="retry_with_docs_scope", severity="info", title="Retry in docs scope"))
    if scope == "all" and code_count == 0 and any(
        term in query_text_lc
        for term in ("implementation", "runtime", "handler", "endpoint", "test", "function", "class", "service")
    ):
        actions.append(RecommendedAction(code="retry_with_code_scope", severity="info", title="Retry in code scope"))
    if route_used == "hybrid" and not exact_anchor_analysis.anchors and results and not global_matched_terms:
        actions.append(RecommendedAction(code="retry_with_semantic", severity="info", title="Retry with semantic route"))
    if route_used == "semantic" and exact_anchor_analysis.anchors:
        actions.append(RecommendedAction(code="retry_with_hybrid", severity="info", title="Retry with hybrid route"))
    if route_used == "exact_handoff" and not exact_anchor_analysis.anchors:
        actions.append(RecommendedAction(code="retry_with_hybrid", severity="info", title="Retry with hybrid route"))

    return actions, sparse_codes
