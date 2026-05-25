"""Dense retrieval branch execution."""

from __future__ import annotations

from collections.abc import Callable

from services.repo_semantic.contracts.common import ChunkRecord
from services.repo_semantic.contracts.search import SearchFilters
from services.repo_semantic.retrieval.constants import DENSE_FILTERED_CANDIDATE_CAP
from services.repo_semantic.retrieval.diversity import apply_max_results_per_file
from services.repo_semantic.retrieval.types import _DenseSearchExecution, _FilterPlan


def dense_candidates(
    *,
    store,
    query_vector: list[float],
    search_scopes: list[str],
    filter_plans_by_scope: dict[str, _FilterPlan],
    filters: SearchFilters,
    target_results: int,
    max_results_per_file: int | None,
    point_to_chunk: Callable[[object], ChunkRecord],
    matches_filters: Callable[[ChunkRecord, SearchFilters], bool],
) -> _DenseSearchExecution:
    """Fetch dense candidates, expanding the unfiltered shortlist under filters."""

    filters_active = any(plan.best_effort for plan in filter_plans_by_scope.values())
    base_limit = max(target_results * 6, 30)
    if filters_active:
        base_limit = max(base_limit, 80)
    candidate_limit = min(base_limit, DENSE_FILTERED_CANDIDATE_CAP)
    max_limit = DENSE_FILTERED_CANDIDATE_CAP if filters_active else candidate_limit

    final_candidates: dict[str, tuple[ChunkRecord, float]] = {}
    candidates_scanned = 0
    filtered_candidates = 0
    kept_candidates = 0
    while True:
        scoped_candidates: dict[str, tuple[ChunkRecord, float]] = {}
        current_scanned = 0
        current_filtered = 0
        for concrete_scope in search_scopes:
            filter_plan = filter_plans_by_scope[concrete_scope]
            points = store.search(
                concrete_scope,
                query_vector,
                limit=candidate_limit,
                query_filter=filter_plan.qdrant_filter,
            )
            current_scanned += len(points)
            for point in points:
                chunk = point_to_chunk(point)
                if not matches_filters(chunk, filters):
                    continue
                current_filtered += 1
                score = float(point.score)
                existing = scoped_candidates.get(chunk.point_id)
                if existing is None or score > existing[1]:
                    scoped_candidates[chunk.point_id] = (chunk, score)
        final_candidates = scoped_candidates
        candidates_scanned = current_scanned
        filtered_candidates = current_filtered
        ranked_preview = sorted(final_candidates.values(), key=lambda item: item[1], reverse=True)
        kept_preview = apply_max_results_per_file(
            ranked_preview,
            top_k=target_results,
            max_results_per_file=max_results_per_file,
            relative_path=lambda item: item[0].relative_path,
        )
        kept_candidates = len(kept_preview)
        if not filters_active or len(kept_preview) >= target_results or candidate_limit >= max_limit:
            break
        candidate_limit = min(candidate_limit * 2, max_limit)

    dense_best_effort = filters_active and candidate_limit >= max_limit and kept_candidates < target_results
    return _DenseSearchExecution(
        candidates=final_candidates,
        candidate_limit=candidate_limit,
        candidates_scanned=candidates_scanned,
        filtered_candidates=filtered_candidates,
        kept_candidates=kept_candidates,
        dense_best_effort=dense_best_effort,
        filter_plans=list(filter_plans_by_scope.values()),
    )
