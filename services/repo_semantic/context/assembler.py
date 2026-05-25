"""Response assembly helpers for repo_context_search."""

from __future__ import annotations

from typing import Literal

from services.repo_semantic.contracts.common import OriginBranch
from services.repo_semantic.contracts.repo_context import (
    EvidencePath,
    FileContextGroup,
    RepoContextResult,
    VerificationHint,
)
from services.repo_semantic.contracts.search import ExactAnchorAnalysis, SearchDiagnostics, SearchResult, SearchWarning
from services.repo_semantic.context.query_plan import ContextQueryItem
from services.repo_semantic.retrieval.constants import (
    MAX_CONTEXT_FILE_GROUPS,
    MAX_CONTEXT_GROUP_CHUNK_IDS,
    MAX_CONTEXT_GROUP_LINE_RANGES,
    MAX_CONTEXT_UNCOVERED_TERMS,
    MAX_EXPLANATIONS,
    RRF_K,
    UNCOVERED_STOPWORDS,
)
from services.repo_semantic.retrieval.types import _SearchExecution
from services.repo_semantic.lexical import analyze_sparse_text, sparse_terms

ContextExecutionItem = tuple[str, str, float, _SearchExecution]
ContextRouteUsed = Literal["semantic", "hybrid", "exact_handoff"]


def context_origin_branches(
    branch_ranks: dict[str, int],
    diagnostics: SearchDiagnostics,
    route_used: Literal["semantic", "hybrid"],
) -> list[OriginBranch]:
    branches: list[OriginBranch] = []
    if "dense" in branch_ranks:
        branches.append("dense")
    if route_used == "hybrid" and "sparse" in branch_ranks:
        if diagnostics.retrieval_backend == "legacy_dense_local_bm25":
            branches.append("legacy_lexical")
        else:
            branches.append("sparse")
    if "graph" in branch_ranks:
        branches.append("graph")
    return branches


def merge_context_warnings(warnings: list[SearchWarning]) -> list[SearchWarning]:
    seen: set[tuple[str, str, str | None]] = set()
    merged: list[SearchWarning] = []
    for warning in warnings:
        key = (warning.code, warning.severity, warning.detail)
        if key in seen:
            continue
        seen.add(key)
        merged.append(warning)
    return merged


def context_uncovered_terms(
    query_items: list[ContextQueryItem],
    exact_anchor_analysis: ExactAnchorAnalysis,
    matched_terms: list[str],
) -> list[str]:
    matched = {term.lower() for term in matched_terms}
    covered_anchor_terms: set[str] = set()
    for anchor in exact_anchor_analysis.anchors:
        if anchor.surface.lower() not in matched:
            continue
        covered_anchor_terms.update(sparse_terms(anchor.surface))
        covered_anchor_terms.add(anchor.surface.lower())
    candidates: list[str] = [anchor.surface for anchor in exact_anchor_analysis.anchors]
    for query_text, _role, _weight in query_items:
        for token in analyze_sparse_text(query_text):
            term = token.surface if token.surface and token.surface != token.canonical else token.canonical
            if len(term) >= 3:
                candidates.append(term)
    uncovered: list[str] = []
    for term in candidates:
        normalized = term.strip()
        lowered = normalized.lower()
        if not normalized or lowered in matched or lowered in UNCOVERED_STOPWORDS:
            continue
        if lowered in covered_anchor_terms:
            continue
        if lowered.isdigit():
            continue
        if lowered in {item.lower() for item in uncovered}:
            continue
        uncovered.append(normalized)
        if len(uncovered) >= MAX_CONTEXT_UNCOVERED_TERMS:
            break
    return uncovered


def context_file_groups(results: list[RepoContextResult], top_k: int) -> list[FileContextGroup]:
    groups: dict[tuple[str, str], dict[str, object]] = {}
    for result in results:
        key = (result.scope, result.relative_path)
        group = groups.setdefault(
            key,
            {
                "relative_path": result.relative_path,
                "scope": result.scope,
                "language": result.language,
                "best_rank": result.final_rank,
                "best_score": result.final_score,
                "chunk_count": 0,
                "chunk_ids": [],
                "line_ranges": [],
                "matched_terms": [],
                "origin_queries": [],
                "origin_branches": [],
                "recommended_action_codes": ["read_file_range"],
            },
        )
        group["chunk_count"] = int(group["chunk_count"]) + 1
        group["best_rank"] = min(int(group["best_rank"]), result.final_rank)
        group["best_score"] = max(float(group["best_score"]), result.final_score)
        chunk_ids = group["chunk_ids"]
        assert isinstance(chunk_ids, list)
        if len(chunk_ids) < MAX_CONTEXT_GROUP_CHUNK_IDS and result.chunk_id not in chunk_ids:
            chunk_ids.append(result.chunk_id)
        line_ranges = group["line_ranges"]
        assert isinstance(line_ranges, list)
        if (
            result.line_range
            and len(line_ranges) < MAX_CONTEXT_GROUP_LINE_RANGES
            and result.line_range not in line_ranges
        ):
            line_ranges.append(result.line_range)
        for field_name, values in (
            ("matched_terms", result.matched_terms),
            ("origin_queries", result.origin_queries),
            ("origin_branches", result.origin_branches),
        ):
            target = group[field_name]
            assert isinstance(target, list)
            for value in values:
                if value not in target:
                    target.append(value)
    max_groups = min(MAX_CONTEXT_FILE_GROUPS, top_k)
    rows = [
        FileContextGroup(
            relative_path=str(group["relative_path"]),
            scope=str(group["scope"]),  # type: ignore[arg-type]
            language=str(group["language"]) if group["language"] is not None else None,
            best_rank=int(group["best_rank"]),
            best_score=float(group["best_score"]),
            chunk_count=int(group["chunk_count"]),
            chunk_ids=list(group["chunk_ids"]),
            line_ranges=list(group["line_ranges"]),
            matched_terms=list(group["matched_terms"])[:MAX_EXPLANATIONS],
            origin_queries=list(group["origin_queries"]),
            origin_branches=list(group["origin_branches"]),
            recommended_action_codes=list(group["recommended_action_codes"]),
        )
        for group in groups.values()
    ]
    return sorted(rows, key=lambda group: group.best_rank)[:max_groups]


def merge_context_results(
    *,
    executions: list[ContextExecutionItem],
    query_items: list[ContextQueryItem],
    route_used: ContextRouteUsed,
    top_k: int,
    max_results_per_file: int | None,
    exact_anchor_analysis: ExactAnchorAnalysis,
    stale_index: bool,
    warnings: list[SearchWarning],
    evidence_paths_by_chunk: dict[str, list[EvidencePath]] | None = None,
    branch_score_overrides_by_chunk: dict[str, dict[str, float]] | None = None,
    origin_queries_by_chunk: dict[str, list[str]] | None = None,
    origin_roles_by_chunk: dict[str, dict[str, str]] | None = None,
    force_rrf_fusion: bool = False,
) -> tuple[list[RepoContextResult], list[str], list[str]]:
    """Merge per-query search executions into ranked repo-context results."""

    merged: dict[str, dict[str, object]] = {}
    for query_text, role, weight, execution in executions:
        for rank, result in enumerate(execution.results, start=1):
            entry = merged.setdefault(
                result.chunk_id,
                {
                    "result": result,
                    "final_score": 0.0,
                    "origin_queries": [],
                    "origin_branches": [],
                    "branch_scores": {},
                    "branch_ranks": {},
                    "matched_terms": [],
                    "evidence_paths": [],
                    "best_original_rank": None,
                    "best_rank": rank,
                    "best_sort": (rank, 0 if role == "original" else 1),
                },
            )
            branch_ranks = execution.branch_ranks_by_chunk.get(result.chunk_id, {})
            is_graph_execution = "graph" in branch_ranks
            if force_rrf_fusion:
                entry["final_score"] = float(entry["final_score"]) + weight / (RRF_K + rank)
            elif is_graph_execution:
                entry["final_score"] = float(entry["final_score"]) + float(
                    result.final_score if result.final_score is not None else result.score
                )
            elif len(query_items) == 1:
                entry["final_score"] = max(
                    float(entry["final_score"]),
                    float(result.final_score if result.final_score is not None else result.score),
                )
            else:
                entry["final_score"] = float(entry["final_score"]) + weight / (RRF_K + rank)
            best_sort = entry["best_sort"]
            assert isinstance(best_sort, tuple)
            chunk_origin_roles = (origin_roles_by_chunk or {}).get(result.chunk_id, {}) if is_graph_execution else {}
            effective_role = "original" if "original" in chunk_origin_roles.values() else role
            current_sort = (rank, 0 if effective_role == "original" else 1)
            if current_sort < best_sort:
                entry["result"] = result
                entry["best_sort"] = current_sort
            entry["best_rank"] = min(int(entry["best_rank"]), rank)
            if effective_role == "original":
                existing = entry["best_original_rank"]
                entry["best_original_rank"] = rank if existing is None else min(int(existing), rank)
            origin_queries = entry["origin_queries"]
            assert isinstance(origin_queries, list)
            origin_query_values = (
                (origin_queries_by_chunk or {}).get(result.chunk_id, [query_text])
                if is_graph_execution
                else [query_text]
            )
            for origin_query in origin_query_values:
                if origin_query not in origin_queries:
                    origin_queries.append(origin_query)
            branches: list[OriginBranch] = []
            if route_used != "exact_handoff":
                branches = context_origin_branches(branch_ranks, execution.diagnostics, route_used)
            origin_branches = entry["origin_branches"]
            assert isinstance(origin_branches, list)
            for branch in branches:
                if branch not in origin_branches:
                    origin_branches.append(branch)
            branch_scores = entry["branch_scores"]
            assert isinstance(branch_scores, dict)
            if "dense" in branches and result.dense_score is not None:
                branch_scores["dense"] = max(float(branch_scores.get("dense", 0.0)), float(result.dense_score))
            lexical_branch = "legacy_lexical" if "legacy_lexical" in branches else "sparse"
            if lexical_branch in branches and result.lexical_score is not None:
                branch_scores[lexical_branch] = max(
                    float(branch_scores.get(lexical_branch, 0.0)),
                    float(result.lexical_score),
                )
            score_overrides = (branch_score_overrides_by_chunk or {}).get(result.chunk_id, {})
            for branch, branch_score in score_overrides.items():
                if branch in branches:
                    branch_scores[branch] = max(
                        float(branch_scores.get(branch, 0.0)),
                        float(branch_score),
                    )
            stored_branch_ranks = entry["branch_ranks"]
            assert isinstance(stored_branch_ranks, dict)
            for branch, branch_rank in branch_ranks.items():
                branch_name = "legacy_lexical" if branch == "sparse" and "legacy_lexical" in branches else branch
                previous = stored_branch_ranks.get(branch_name)
                stored_branch_ranks[branch_name] = branch_rank if previous is None else min(int(previous), branch_rank)
            matched_terms = entry["matched_terms"]
            assert isinstance(matched_terms, list)
            for term in result.matched_terms:
                if term not in matched_terms and len(matched_terms) < MAX_EXPLANATIONS:
                    matched_terms.append(term)
            evidence_paths = entry["evidence_paths"]
            assert isinstance(evidence_paths, list)
            for evidence_path in (evidence_paths_by_chunk or {}).get(result.chunk_id, []):
                if evidence_path not in evidence_paths:
                    evidence_paths.append(evidence_path)

    ranked_entries = sorted(
        merged.values(),
        key=lambda entry: (
            -float(entry["final_score"]),
            int(entry["best_original_rank"]) if entry["best_original_rank"] is not None else 1_000_000,
            int(entry["best_rank"]),
            str(entry["result"].relative_path),  # type: ignore[union-attr]
            int(entry["result"].start_line),  # type: ignore[union-attr]
            str(entry["result"].chunk_id),  # type: ignore[union-attr]
        ),
    )
    kept_entries: list[dict[str, object]] = []
    per_file: dict[str, int] = {}
    if max_results_per_file is not None and max_results_per_file <= 0:
        ranked_entries = []
    for entry in ranked_entries:
        if top_k <= 0:
            break
        result = entry["result"]
        assert isinstance(result, SearchResult)
        if max_results_per_file is not None and per_file.get(result.relative_path, 0) >= max_results_per_file:
            continue
        kept_entries.append(entry)
        per_file[result.relative_path] = per_file.get(result.relative_path, 0) + 1
        if len(kept_entries) >= top_k:
            break

    global_matched_terms: list[str] = []
    for entry in kept_entries:
        matched_terms = entry["matched_terms"]
        assert isinstance(matched_terms, list)
        for term in matched_terms:
            if term not in global_matched_terms and len(global_matched_terms) < MAX_EXPLANATIONS:
                global_matched_terms.append(str(term))
    uncovered_terms = context_uncovered_terms(query_items, exact_anchor_analysis, global_matched_terms)

    verification_codes: list[str] = []
    verification_reason = None
    if exact_anchor_analysis.anchors:
        verification_codes.append("run_local_rg")
        verification_reason = "exact_anchor_detected"
    if stale_index:
        verification_reason = verification_reason or "stale_index"
    if any(
        warning.code in {"legacy_lexical_fallback_used", "stale_index_verify_with_exact_search"}
        for warning in warnings
    ):
        verification_reason = verification_reason or "degraded_search"
    results: list[RepoContextResult] = []
    for final_rank, entry in enumerate(kept_entries, start=1):
        base = entry["result"]
        assert isinstance(base, SearchResult)
        base_payload = base.model_dump()
        base_payload.pop("final_score", None)
        result_uncovered = [
            term
            for term in uncovered_terms
            if term.lower() not in {matched.lower() for matched in base.matched_terms}
        ][:MAX_CONTEXT_UNCOVERED_TERMS]
        results.append(
            RepoContextResult(
                **base_payload,
                final_rank=final_rank,
                final_score=float(entry["final_score"]),
                origin_branches=list(entry["origin_branches"]),
                origin_queries=list(entry["origin_queries"]),
                branch_scores={str(key): float(value) for key, value in dict(entry["branch_scores"]).items()},
                branch_ranks={str(key): int(value) for key, value in dict(entry["branch_ranks"]).items()},
                uncovered_terms=result_uncovered,
                evidence_paths=list(entry["evidence_paths"]),
                verification=VerificationHint(
                    required=bool(verification_codes or stale_index or verification_reason),
                    reason=verification_reason,
                    recommended_action_codes=verification_codes,
                ),
            )
        )

    return results, global_matched_terms, uncovered_terms
