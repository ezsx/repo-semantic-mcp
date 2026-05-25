"""Diagnostics assembly helpers for repo_context_search."""

from __future__ import annotations

from services.repo_semantic.contracts.repo_context import (
    ExactAnchorUsage,
    FileContextGroup,
    FileDistributionEntry,
    GraphBranchDiagnostics,
    RepoContextDiagnostics,
    RepoContextResult,
)
from services.repo_semantic.contracts.search import BranchDiagnostics, ExactAnchorAnalysis, SearchDiagnostics
from services.repo_semantic.context.query_plan import ContextQueryItem
from services.repo_semantic.retrieval.constants import RRF_K


def context_file_distribution(file_groups: list[FileContextGroup]) -> list[FileDistributionEntry]:
    return [
        FileDistributionEntry(
            relative_path=group.relative_path,
            scope=group.scope,
            result_count=group.chunk_count,
            best_rank=group.best_rank,
            best_score=group.best_score,
        )
        for group in file_groups
    ]


def merge_branch_diagnostics(per_query_diagnostics: list[SearchDiagnostics]) -> dict[str, BranchDiagnostics]:
    branch_diagnostics: dict[str, BranchDiagnostics] = {}
    for diagnostics in per_query_diagnostics:
        for branch, branch_diagnostic in diagnostics.branch_diagnostics.items():
            existing = branch_diagnostics.get(branch)
            warning_codes = set(branch_diagnostic.warning_codes)
            if existing is None:
                branch_diagnostics[branch] = BranchDiagnostics(
                    available=branch_diagnostic.available,
                    candidate_limit=branch_diagnostic.candidate_limit,
                    candidates_returned=branch_diagnostic.candidates_returned,
                    warning_codes=sorted(warning_codes),
                )
            else:
                existing.available = existing.available or branch_diagnostic.available
                existing.candidates_returned += branch_diagnostic.candidates_returned
                existing.warning_codes = sorted(set(existing.warning_codes).union(warning_codes))
                if existing.candidate_limit is None:
                    existing.candidate_limit = branch_diagnostic.candidate_limit
                elif branch_diagnostic.candidate_limit is not None:
                    existing.candidate_limit = max(existing.candidate_limit, branch_diagnostic.candidate_limit)
    return branch_diagnostics


def build_context_diagnostics(
    *,
    route_used: str,
    query_items: list[ContextQueryItem],
    per_query_diagnostics: list[SearchDiagnostics],
    exact_anchor_analysis: ExactAnchorAnalysis,
    exact_anchor_usages: list[ExactAnchorUsage],
    results: list[RepoContextResult],
    file_groups: list[FileContextGroup],
    global_matched_terms: list[str],
    uncovered_terms: list[str],
    sparse_codes: set[str],
    graph_diagnostics: GraphBranchDiagnostics | None = None,
    graph_branch_diagnostics: BranchDiagnostics | None = None,
    graph_fusion_used: bool = False,
    include_diagnostics: bool = True,
) -> RepoContextDiagnostics:
    docs_count = sum(1 for result in results if result.scope == "docs")
    code_count = sum(1 for result in results if result.scope == "code")
    branch_diagnostics = merge_branch_diagnostics(per_query_diagnostics)
    if graph_branch_diagnostics is not None:
        branch_diagnostics["graph"] = graph_branch_diagnostics
    sparse_availability_values = [
        diagnostics.sparse_available
        for diagnostics in per_query_diagnostics
        if diagnostics.sparse_available is not None
    ]
    return RepoContextDiagnostics(
        fusion_method=(
            "exact_handoff"
            if route_used == "exact_handoff"
            else (
                "multi_query_weighted_rrf"
                if graph_fusion_used or len(query_items) > 1
                else "single_query"
            )
        ),
        rrf_k=None if route_used == "exact_handoff" or (len(query_items) == 1 and not graph_fusion_used) else RRF_K,
        query_count=len(query_items),
        per_query=per_query_diagnostics if include_diagnostics else [],
        branch_diagnostics=branch_diagnostics,
        filter_pushed_families=sorted(
            {family for diagnostics in per_query_diagnostics for family in diagnostics.filter_pushed_families}
        ),
        filter_post_families=sorted(
            {family for diagnostics in per_query_diagnostics for family in diagnostics.filter_post_families}
        ),
        filter_best_effort=any(diagnostics.filter_best_effort for diagnostics in per_query_diagnostics),
        exact_anchor_analysis=exact_anchor_analysis,
        exact_anchor_usages=exact_anchor_usages,
        scope_distribution={
            "code": code_count,
            "docs": docs_count,
        },
        file_distribution=context_file_distribution(file_groups),
        matched_terms=global_matched_terms,
        uncovered_terms=uncovered_terms,
        sparse_available=(
            None
            if route_used != "hybrid" or not sparse_availability_values
            else all(bool(value) for value in sparse_availability_values)
        ),
        sparse_unavailable_codes=sorted(sparse_codes),
        graph_available=bool(graph_diagnostics and graph_diagnostics.available),
        graph_used=bool(graph_diagnostics and graph_diagnostics.used),
        graph_diagnostics=graph_diagnostics,
        rerank_used=False,
        colbert_used=False,
    )
