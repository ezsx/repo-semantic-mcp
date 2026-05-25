"""Search execution assembly for semantic and hybrid retrieval."""

from __future__ import annotations

from collections.abc import Callable

from services.repo_semantic.contracts.common import ChunkRecord, MatchType, SnippetMode
from services.repo_semantic.contracts.search import (
    BranchDiagnostics,
    ExactAnchorAnalysis,
    PayloadCapabilities,
    SearchDiagnostics,
    SearchResult,
    SearchWarning,
)
from services.repo_semantic.retrieval.constants import RRF_DENSE_WEIGHT, RRF_K, RRF_SPARSE_WEIGHT
from services.repo_semantic.retrieval.diversity import apply_max_results_per_file
from services.repo_semantic.retrieval.fusion import weighted_rrf
from services.repo_semantic.retrieval.types import (
    _DenseSearchExecution,
    _FilterPlan,
    _SearchExecution,
    _SparseSearchExecution,
)

FilterPlanMerger = Callable[[list[_FilterPlan]], tuple[list[str], list[str], bool, list[PayloadCapabilities]]]
SearchResultFactory = Callable[..., SearchResult]


def build_semantic_search_execution(
    *,
    dense_execution: _DenseSearchExecution,
    top_k: int,
    max_results_per_file: int | None,
    query_tokens: list[str],
    exact_anchor_analysis: ExactAnchorAnalysis,
    snippet_mode: SnippetMode,
    include_explanations: bool,
    to_search_result: SearchResultFactory,
    merge_filter_plans: FilterPlanMerger,
) -> _SearchExecution:
    """Assemble a semantic search execution from dense branch candidates."""

    ranked = sorted(dense_execution.candidates.values(), key=lambda item: item[1], reverse=True)
    kept = apply_max_results_per_file(
        ranked,
        top_k=top_k,
        max_results_per_file=max_results_per_file,
        relative_path=lambda item: item[0].relative_path,
    )
    results = [
        to_search_result(
            chunk=chunk,
            score=dense_score,
            dense_score=dense_score,
            query_tokens=query_tokens,
            snippet_mode=snippet_mode,
            include_explanations=include_explanations,
            match_type="semantic",
        )
        for chunk, dense_score in kept
    ]
    branch_ranks_by_chunk = {
        chunk.point_id: {"dense": rank}
        for rank, (chunk, _dense_score) in enumerate(kept, start=1)
    }
    warnings: list[SearchWarning] = []
    if dense_execution.dense_best_effort:
        warnings.append(
            SearchWarning(
                code="filtered_dense_search_best_effort",
                severity="warning",
                detail="Dense search used an expanded unfiltered shortlist; verify exact matches when filters are narrow.",
            )
        )
    filter_pushed, filter_post, filter_best_effort, payload_capabilities = merge_filter_plans(
        dense_execution.filter_plans,
    )
    if filter_pushed:
        warnings.append(
            SearchWarning(
                code="qdrant_filter_pushdown_used",
                severity="info",
                detail="Qdrant-compatible filter families were pushed down before retrieval.",
            )
        )
    if any(capability.missing_fields for capability in payload_capabilities):
        warnings.append(
            SearchWarning(
                code="qdrant_filter_pushdown_partial",
                severity="warning",
                detail="Some requested filter families could not be pushed down for a legacy or incompatible collection.",
            )
        )
    if filter_best_effort:
        warnings.append(
            SearchWarning(
                code="qdrant_filter_post_filter_best_effort",
                severity="warning",
                detail="Some filters are applied after bounded retrieval; narrow filters can under-return.",
            )
        )
    if exact_anchor_analysis.anchors:
        warnings.append(
            SearchWarning(
                code="exact_anchor_verify_with_local_rg",
                severity="warning" if exact_anchor_analysis.exact_anchor_heavy else "info",
                detail="Query contains exact-looking anchors; verify candidates against the working tree before editing.",
            )
        )
    diagnostics = SearchDiagnostics(
        candidate_limit=dense_execution.candidate_limit,
        candidates_scanned=dense_execution.candidates_scanned,
        filtered_candidates=dense_execution.filtered_candidates,
        final_results=len(results),
        dense_best_effort=dense_execution.dense_best_effort,
        lexical_cache_used=False,
        retrieval_backend="qdrant_dense",
        exact_anchor_analysis=exact_anchor_analysis,
        filter_pushdown_supported=True,
        filter_pushed_families=filter_pushed,
        filter_post_families=filter_post,
        filter_best_effort=filter_best_effort,
        filter_payload_capabilities=payload_capabilities,
    )
    return _SearchExecution(
        results=results,
        diagnostics=diagnostics,
        warnings=warnings,
        branch_ranks_by_chunk=branch_ranks_by_chunk,
    )


def build_hybrid_search_execution(
    *,
    dense_execution: _DenseSearchExecution,
    sparse_execution: _SparseSearchExecution,
    sparse_candidates: dict[str, tuple[ChunkRecord, float]],
    lexical_cache_used: bool,
    legacy_fallback_used: bool,
    top_k: int,
    sparse_limit: int,
    max_results_per_file: int | None,
    exact_anchor_analysis: ExactAnchorAnalysis,
    result_query_tokens: list[str],
    snippet_mode: SnippetMode,
    include_explanations: bool,
    to_search_result: SearchResultFactory,
    merge_filter_plans: FilterPlanMerger,
) -> _SearchExecution:
    """Assemble a hybrid search execution from dense and sparse branches."""

    dense_ranked = sorted(dense_execution.candidates.values(), key=lambda item: item[1], reverse=True)
    sparse_ranked = sorted(sparse_candidates.values(), key=lambda item: item[1], reverse=True)
    branch_weights = {
        "dense": 0.8 if exact_anchor_analysis.exact_anchor_heavy else RRF_DENSE_WEIGHT,
        "sparse": 1.4 if exact_anchor_analysis.exact_anchor_heavy else RRF_SPARSE_WEIGHT,
    }
    ranked = weighted_rrf(
        branches={
            "dense": dense_ranked,
            "sparse": sparse_ranked,
        },
        weights=branch_weights,
    )
    kept = apply_max_results_per_file(
        ranked,
        top_k=top_k,
        max_results_per_file=max_results_per_file,
        relative_path=lambda item: item[0].relative_path,
    )
    results = [
        to_search_result(
            chunk=chunk,
            score=score,
            dense_score=dense_score,
            lexical_score=sparse_score,
            query_tokens=result_query_tokens,
            snippet_mode=snippet_mode,
            include_explanations=include_explanations,
            match_type=_hybrid_match_type(dense_score, sparse_score),
        )
        for chunk, score, dense_score, sparse_score, _branch_ranks in kept
    ]
    branch_ranks_by_chunk = {
        chunk.point_id: dict(_branch_ranks)
        for chunk, _score, _dense_score, _sparse_score, _branch_ranks in kept
    }
    dense_filter_pushed, dense_filter_post, dense_filter_best_effort, dense_payload_capabilities = (
        merge_filter_plans(dense_execution.filter_plans)
    )
    sparse_filter_pushed, sparse_filter_post, sparse_filter_best_effort, sparse_payload_capabilities = (
        merge_filter_plans(sparse_execution.filter_plans)
    )
    filter_pushed = sorted(set(dense_filter_pushed).union(sparse_filter_pushed))
    filter_post = sorted(set(dense_filter_post).union(sparse_filter_post))
    filter_best_effort = dense_filter_best_effort or sparse_filter_best_effort
    payload_capabilities = dense_payload_capabilities + [
        capability
        for capability in sparse_payload_capabilities
        if capability not in dense_payload_capabilities
    ]
    warnings = _hybrid_warnings(
        filter_pushed=filter_pushed,
        payload_capabilities=payload_capabilities,
        filter_best_effort=filter_best_effort,
        exact_anchor_analysis=exact_anchor_analysis,
        dense_execution=dense_execution,
        sparse_execution=sparse_execution,
        lexical_cache_used=lexical_cache_used,
        legacy_fallback_used=legacy_fallback_used,
    )
    diagnostics = SearchDiagnostics(
        candidate_limit=dense_execution.candidate_limit,
        candidates_scanned=dense_execution.candidates_scanned + len(sparse_candidates),
        filtered_candidates=dense_execution.filtered_candidates + len(sparse_candidates),
        final_results=len(results),
        dense_best_effort=dense_execution.dense_best_effort,
        lexical_cache_used=lexical_cache_used,
        retrieval_backend="qdrant_dense_sparse_rrf"
        if sparse_execution.branch_available
        else "legacy_dense_local_bm25",
        fusion_method="weighted_rrf",
        rrf_k=RRF_K,
        branch_weights=branch_weights,
        branch_diagnostics={
            "dense": BranchDiagnostics(
                available=True,
                candidate_limit=dense_execution.candidate_limit,
                candidates_returned=len(dense_ranked),
                warning_codes=["filtered_dense_search_best_effort"]
                if dense_execution.dense_best_effort
                else [],
            ),
            "sparse": BranchDiagnostics(
                available=sparse_execution.branch_available,
                candidate_limit=sparse_limit,
                candidates_returned=len(sparse_ranked),
                warning_codes=(
                    ["sparse_stats_stale"]
                    if sparse_execution.stats_stale
                    else []
                )
                + (
                    ["sparse_branch_unavailable"]
                    if not sparse_execution.branch_available
                    else []
                )
                + sparse_execution.unavailable_codes
                + (
                    ["qdrant_filter_post_filter_best_effort"]
                    if sparse_filter_best_effort
                    else []
                ),
            ),
        },
        sparse_available=sparse_execution.branch_available,
        sparse_contract_compatible=sparse_execution.branch_available,
        sparse_manifest_hash=sparse_execution.manifest_hash,
        sparse_vocabulary_hash=sparse_execution.vocabulary_hash,
        sparse_corpus_stats_hash=sparse_execution.corpus_stats_hash,
        sparse_stats_stale=sparse_execution.stats_stale,
        sparse_unavailable_codes=sparse_execution.unavailable_codes,
        exact_anchor_analysis=exact_anchor_analysis,
        filter_pushdown_supported=True,
        filter_pushed_families=filter_pushed,
        filter_post_families=filter_post,
        filter_best_effort=filter_best_effort,
        filter_payload_capabilities=payload_capabilities,
    )
    return _SearchExecution(
        results=results,
        diagnostics=diagnostics,
        warnings=warnings,
        branch_ranks_by_chunk=branch_ranks_by_chunk,
    )


def _hybrid_match_type(dense_score: float, sparse_score: float) -> MatchType:
    if dense_score > 0 and sparse_score > 0:
        return "hybrid"
    if sparse_score > 0:
        return "lexical"
    return "semantic"


def _hybrid_warnings(
    *,
    filter_pushed: list[str],
    payload_capabilities: list[PayloadCapabilities],
    filter_best_effort: bool,
    exact_anchor_analysis: ExactAnchorAnalysis,
    dense_execution: _DenseSearchExecution,
    sparse_execution: _SparseSearchExecution,
    lexical_cache_used: bool,
    legacy_fallback_used: bool,
) -> list[SearchWarning]:
    warnings: list[SearchWarning] = []
    if filter_pushed:
        warnings.append(
            SearchWarning(
                code="qdrant_filter_pushdown_used",
                severity="info",
                detail="Qdrant-compatible filter families were pushed down before retrieval.",
            )
        )
    if any(capability.missing_fields for capability in payload_capabilities):
        warnings.append(
            SearchWarning(
                code="qdrant_filter_pushdown_partial",
                severity="warning",
                detail="Some requested filter families could not be pushed down for a legacy or incompatible collection.",
            )
        )
    if filter_best_effort:
        warnings.append(
            SearchWarning(
                code="qdrant_filter_post_filter_best_effort",
                severity="warning",
                detail="Some filters are applied after bounded retrieval; narrow filters can under-return.",
            )
        )
    if exact_anchor_analysis.anchors:
        warnings.append(
            SearchWarning(
                code="exact_anchor_verify_with_local_rg",
                severity="warning" if exact_anchor_analysis.exact_anchor_heavy else "info",
                detail="Query contains exact-looking anchors; verify candidates against the working tree before editing.",
            )
        )
    if dense_execution.dense_best_effort:
        warnings.append(
            SearchWarning(
                code="filtered_dense_search_best_effort",
                severity="warning",
                detail="Dense side of hybrid search used an expanded unfiltered shortlist under active filters.",
            )
        )
    if not sparse_execution.branch_available:
        warnings.append(
            SearchWarning(
                code="sparse_branch_unavailable",
                severity="warning" if sparse_execution.manifest_seen else "info",
                detail="Hybrid search could not use Qdrant sparse vectors; legacy lexical fallback may be used for compatibility.",
            )
        )
        for code in sparse_execution.unavailable_codes:
            warnings.append(
                SearchWarning(
                    code=code,
                    severity="warning",
                    detail="Sparse branch is unavailable for this reason; run index_status for full retrieval state and rebuild guidance.",
                )
            )
    if sparse_execution.partial_unavailable:
        warnings.append(
            SearchWarning(
                code="sparse_scope_partial_unavailable",
                severity="warning",
                detail="Some requested non-empty scopes lack compatible sparse vectors; legacy lexical fallback was used to avoid silent under-recall.",
            )
        )
    if sparse_execution.stats_stale:
        warnings.append(
            SearchWarning(
                code="sparse_stats_stale",
                severity="warning",
                detail="Sparse BM25 corpus statistics are stale after incremental changes; explicit rebuild refreshes lexical weights.",
            )
        )
    if lexical_cache_used:
        warnings.append(
            SearchWarning(
                code="legacy_lexical_fallback_used" if legacy_fallback_used else "lexical_cache_loaded",
                severity="info",
                detail="Hybrid search loaded indexed chunks for legacy BM25 lexical scoring.",
            )
        )
    return warnings
