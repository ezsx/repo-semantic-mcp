"""Legacy local BM25 fallback helpers."""

from __future__ import annotations

from collections.abc import Callable

from rank_bm25 import BM25Okapi

from services.repo_semantic.contracts.common import ChunkRecord
from services.repo_semantic.contracts.search import SearchFilters
from services.repo_semantic.retrieval.types import _LegacyBm25Execution, _LexicalCache
from services.repo_semantic.lexical import sparse_terms


def build_lexical_cache(
    *,
    store,
    scope: str,
    point_to_chunk: Callable[[object], ChunkRecord],
    tokenize: Callable[[str], list[str]],
) -> _LexicalCache:
    """Build a local lexical corpus for legacy BM25 compatibility."""

    chunks = [point_to_chunk(point) for point in store.scroll_chunks(scope)]
    tokens = [sparse_terms(chunk.text) or tokenize(chunk.text) for chunk in chunks]
    return _LexicalCache(chunks=chunks, tokens=tokens)


def legacy_bm25_candidates(
    *,
    existing_candidates: dict[str, tuple[ChunkRecord, float]],
    query_tokens: list[str],
    search_scopes: list[str],
    filters: SearchFilters,
    get_lexical_cache: Callable[[str], _LexicalCache],
    matches_filters: Callable[[ChunkRecord, SearchFilters], bool],
    bm25_factory: Callable[[list[list[str]]], object] | None = None,
) -> _LegacyBm25Execution:
    """Run the bounded compatibility BM25 branch over local chunk cache."""

    bm25_factory = bm25_factory or BM25Okapi
    candidates = dict(existing_candidates)
    lexical_cache_used = False
    legacy_fallback_used = False
    if query_tokens:
        legacy_fallback_used = True
        for concrete_scope in search_scopes:
            cache = get_lexical_cache(concrete_scope)
            lexical_cache_used = True
            filtered_pairs = [
                (chunk, tokens)
                for chunk, tokens in zip(cache.chunks, cache.tokens, strict=True)
                if matches_filters(chunk, filters)
            ]
            if not filtered_pairs:
                continue
            filtered_chunks = [chunk for chunk, _ in filtered_pairs]
            filtered_tokens = [tokens for _, tokens in filtered_pairs]
            bm25 = bm25_factory(filtered_tokens)
            scores = bm25.get_scores(query_tokens)
            max_score = max(scores) if len(scores) else 0.0
            for chunk, raw_score in zip(filtered_chunks, scores, strict=True):
                normalized = float(raw_score) / float(max_score) if max_score else 0.0
                if normalized <= 0.0:
                    continue
                candidates[chunk.point_id] = (chunk, normalized)

    return _LegacyBm25Execution(
        candidates=candidates,
        lexical_cache_used=lexical_cache_used,
        legacy_fallback_used=legacy_fallback_used,
    )
