"""Sparse vector retrieval branch execution."""

from __future__ import annotations

from collections.abc import Callable

from services.repo_semantic.contracts.common import ChunkRecord
from services.repo_semantic.contracts.search import SearchFilters
from services.repo_semantic.retrieval.types import _FilterPlan, _SparseSearchExecution
from services.repo_semantic.lexical import SparseManifest, encode_sparse_query


def sparse_query_candidates(
    *,
    store,
    sparse_manifests,
    query: str,
    search_scopes: list[str],
    requested_scopes: list[str],
    filter_plans_by_scope: dict[str, _FilterPlan],
    filters: SearchFilters,
    limit: int,
    sparse_manifest_for_scope: Callable[[str], SparseManifest | None],
    sparse_manifest_unavailable_codes: Callable[[SparseManifest | None], list[str]],
    store_sparse_available: Callable[[str], bool],
    point_to_chunk: Callable[[object], ChunkRecord],
    matches_filters: Callable[[ChunkRecord, SearchFilters], bool],
) -> _SparseSearchExecution:
    """Fetch candidates from Qdrant sparse vectors and report branch readiness."""

    candidates: dict[str, tuple[ChunkRecord, float]] = {}
    ready_non_empty_scopes: set[str] = set()
    unavailable_scopes: list[str] = []
    unavailable_codes: list[str] = []
    any_stale = False
    any_manifest = False
    manifest_hash = None
    vocabulary_hash = None
    corpus_stats_hash = None
    search_sparse = getattr(store, "search_sparse", None)
    if search_sparse is None:
        return _SparseSearchExecution(
            candidates=candidates,
            branch_available=False,
            partial_unavailable=False,
            manifest_seen=False,
            stats_stale=False,
            manifest_hash=None,
            vocabulary_hash=None,
            corpus_stats_hash=None,
            unavailable_scopes=[],
            unavailable_codes=["sparse_search_unsupported"],
            filter_plans=[],
        )

    for concrete_scope in search_scopes:
        scope_count = store.count(concrete_scope)
        raw_manifest = sparse_manifests.load(concrete_scope)
        manifest = sparse_manifest_for_scope(concrete_scope)
        if manifest is None:
            if scope_count > 0:
                unavailable_scopes.append(concrete_scope)
                unavailable_codes.extend(sparse_manifest_unavailable_codes(raw_manifest))
            continue
        any_manifest = True
        manifest_hash = manifest_hash or manifest.manifest_content_hash
        vocabulary_hash = vocabulary_hash or manifest.vocabulary_hash
        corpus_stats_hash = corpus_stats_hash or manifest.corpus_stats_hash
        if manifest.sparse_stats_stale:
            any_stale = True
        if manifest.intentionally_empty and scope_count == 0:
            continue
        if not store_sparse_available(concrete_scope):
            if scope_count > 0:
                unavailable_scopes.append(concrete_scope)
                unavailable_codes.append("sparse_vector_missing")
            continue
        if scope_count > 0:
            ready_non_empty_scopes.add(concrete_scope)
        query_vector = encode_sparse_query(query, manifest)
        if not query_vector.indices:
            continue
        points = search_sparse(
            concrete_scope,
            query_vector.indices,
            query_vector.values,
            limit,
            query_filter=filter_plans_by_scope[concrete_scope].qdrant_filter,
        )
        for point in points:
            chunk = point_to_chunk(point)
            if not matches_filters(chunk, filters):
                continue
            score = float(point.score)
            existing = candidates.get(chunk.point_id)
            if existing is None or score > existing[1]:
                candidates[chunk.point_id] = (chunk, score)

    requested_non_empty_scopes = {
        concrete_scope
        for concrete_scope in requested_scopes
        if store.count(concrete_scope) > 0
    }
    branch_available = bool(requested_non_empty_scopes) and not unavailable_scopes
    partial_unavailable = bool(ready_non_empty_scopes and unavailable_scopes)
    return _SparseSearchExecution(
        candidates=candidates,
        branch_available=branch_available,
        partial_unavailable=partial_unavailable,
        manifest_seen=any_manifest,
        stats_stale=any_stale,
        manifest_hash=manifest_hash,
        vocabulary_hash=vocabulary_hash,
        corpus_stats_hash=corpus_stats_hash,
        unavailable_scopes=sorted(set(unavailable_scopes)),
        unavailable_codes=sorted(set(unavailable_codes)),
        filter_plans=list(filter_plans_by_scope.values()),
    )
