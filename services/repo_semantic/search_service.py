"""Search service that combines dense retrieval with lexical scoring."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from services.repo_semantic.config import SemanticMcpSettings
from services.repo_semantic.embeddings.base import EmbeddingProvider
from services.repo_semantic.models import (
    ChunkRecord,
    IndexCollectionStatus,
    IndexStatusResult,
    ReadChunkResult,
    SearchResult,
    SearchScope,
)
from services.repo_semantic.qdrant_store import QdrantStore

TOKEN_RE = re.compile(r"[A-Za-zА-Яа-я0-9_./:-]+", re.UNICODE)


@dataclass(slots=True)
class _LexicalCache:
    """Кэш текстов и токенов для локального BM25 по конкретной коллекции."""

    chunks: list[ChunkRecord]
    tokens: list[list[str]]


class SearchService:
    """Выполнять dense и hybrid retrieval по indexed collections."""

    def __init__(
        self,
        settings: SemanticMcpSettings,
        embedding_provider: EmbeddingProvider,
        store: QdrantStore,
        indexer,
        watcher,
    ) -> None:
        """Сохранить зависимости поиска и статуса индекса."""

        self._settings = settings
        self._embedding_provider = embedding_provider
        self._store = store
        self._indexer = indexer
        self._watcher = watcher
        self._lexical_cache: dict[str, _LexicalCache] = {}

    def invalidate_cache(self) -> None:
        """Сбросить lexical cache после reindex/rebuild."""

        self._lexical_cache.clear()

    def _tokenize(self, text: str) -> list[str]:
        """Токенизировать строку для lexical scoring."""

        return [token.lower() for token in TOKEN_RE.findall(text)]

    def _point_to_chunk(self, point) -> ChunkRecord:
        """Преобразовать payload Qdrant point в ChunkRecord."""

        payload = point.payload or {}
        return ChunkRecord(
            point_id=str(payload.get("chunk_id") or point.id),
            scope=payload["scope"],
            relative_path=payload["relative_path"],
            language=payload["language"],
            chunk_type=payload["chunk_type"],
            text=payload["text"],
            start_line=int(payload["start_line"]),
            end_line=int(payload["end_line"]),
            content_hash=payload["content_hash"],
            source_mtime=float(payload["source_mtime"]),
            symbol_path=payload.get("symbol_path"),
            heading_path=payload.get("heading_path"),
            domain_tags=list(payload.get("domain_tags") or []),
            is_generated=bool(payload.get("is_generated", False)),
            extra={},
        )

    def _scope_to_collections(self, scope: SearchScope) -> list[str]:
        """Развернуть logical scope в список concrete collections."""

        if scope == "all":
            return ["code", "docs"]
        return [scope]

    def _matches_filters(
        self,
        chunk: ChunkRecord,
        path_prefix: str | None,
        chunk_types: list[str] | None,
        domain_tags: list[str] | None,
    ) -> bool:
        """Проверить, подходит ли чанк под client-side filters."""

        if path_prefix and not chunk.relative_path.startswith(path_prefix.replace("\\", "/")):
            return False
        if chunk_types and chunk.chunk_type not in set(chunk_types):
            return False
        if domain_tags and not set(domain_tags).intersection(chunk.domain_tags):
            return False
        return True

    def _make_snippet(self, text: str, limit: int = 280) -> str:
        """Сжать чанк до короткого snippets для выдачи агенту."""

        normalized = " ".join(text.split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3].rstrip() + "..."

    def _to_search_result(
        self,
        chunk: ChunkRecord,
        score: float,
        dense_score: float | None = None,
        lexical_score: float | None = None,
    ) -> SearchResult:
        """Собрать сериализуемый результат поиска."""

        return SearchResult(
            chunk_id=chunk.point_id,
            scope=chunk.scope,
            relative_path=chunk.relative_path,
            language=chunk.language,
            chunk_type=chunk.chunk_type,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            snippet=self._make_snippet(chunk.text),
            symbol_path=chunk.symbol_path,
            heading_path=chunk.heading_path,
            domain_tags=chunk.domain_tags,
            score=score,
            dense_score=dense_score,
            lexical_score=lexical_score,
        )

    def _get_lexical_cache(self, scope: str) -> _LexicalCache:
        """Построить и закэшировать lexical corpus по коллекции."""

        if scope in self._lexical_cache:
            return self._lexical_cache[scope]
        chunks = [self._point_to_chunk(point) for point in self._store.scroll_chunks(scope)]
        tokens = [self._tokenize(chunk.text) for chunk in chunks]
        cache = _LexicalCache(chunks=chunks, tokens=tokens)
        self._lexical_cache[scope] = cache
        return cache

    def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        scope: SearchScope = "all",
        path_prefix: str | None = None,
        chunk_types: list[str] | None = None,
        domain_tags: list[str] | None = None,
    ) -> list[SearchResult]:
        """Выполнить dense semantic retrieval по code/docs коллекциям."""

        query_vector = self._embedding_provider.embed_query(query)
        results: list[SearchResult] = []
        for concrete_scope in self._scope_to_collections(scope):
            for point in self._store.search(concrete_scope, query_vector, limit=max(top_k * 6, 30)):
                chunk = self._point_to_chunk(point)
                if not self._matches_filters(chunk, path_prefix, chunk_types, domain_tags):
                    continue
                results.append(
                    self._to_search_result(
                        chunk=chunk,
                        score=float(point.score),
                        dense_score=float(point.score),
                    )
                )

        unique: dict[str, SearchResult] = {}
        for result in sorted(results, key=lambda item: item.score, reverse=True):
            unique.setdefault(result.chunk_id, result)
        return list(unique.values())[:top_k]

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        scope: SearchScope = "all",
        path_prefix: str | None = None,
        chunk_types: list[str] | None = None,
        domain_tags: list[str] | None = None,
    ) -> list[SearchResult]:
        """Выполнить hybrid retrieval: dense shortlist + BM25 lexical scoring."""

        dense_results = self.semantic_search(
            query=query,
            top_k=max(top_k * 5, 25),
            scope=scope,
            path_prefix=path_prefix,
            chunk_types=chunk_types,
            domain_tags=domain_tags,
        )
        dense_by_id = {result.chunk_id: result for result in dense_results}

        lexical_candidates: dict[str, tuple[ChunkRecord, float]] = {}
        query_tokens = self._tokenize(query)
        if query_tokens:
            for concrete_scope in self._scope_to_collections(scope):
                cache = self._get_lexical_cache(concrete_scope)
                filtered_pairs = [
                    (chunk, tokens)
                    for chunk, tokens in zip(cache.chunks, cache.tokens, strict=True)
                    if self._matches_filters(chunk, path_prefix, chunk_types, domain_tags)
                ]
                if not filtered_pairs:
                    continue
                filtered_chunks = [chunk for chunk, _ in filtered_pairs]
                filtered_tokens = [tokens for _, tokens in filtered_pairs]
                bm25 = BM25Okapi(filtered_tokens)
                scores = bm25.get_scores(query_tokens)
                max_score = max(scores) if len(scores) else 0.0
                for chunk, raw_score in zip(filtered_chunks, scores, strict=True):
                    normalized = float(raw_score) / float(max_score) if max_score else 0.0
                    lexical_candidates[chunk.point_id] = (chunk, normalized)

        combined: dict[str, tuple[ChunkRecord, float, float, float]] = {}
        for result in dense_results:
            chunk = ChunkRecord(
                point_id=result.chunk_id,
                scope=result.scope,
                relative_path=result.relative_path,
                language=result.language,
                chunk_type=result.chunk_type,
                text=result.snippet,
                start_line=result.start_line,
                end_line=result.end_line,
                content_hash="",
                source_mtime=0.0,
                symbol_path=result.symbol_path,
                heading_path=result.heading_path,
                domain_tags=result.domain_tags,
            )
            dense_score = result.dense_score or result.score
            lexical_score = lexical_candidates.get(result.chunk_id, (chunk, 0.0))[1]
            combined[result.chunk_id] = (
                chunk,
                dense_score * 0.7 + lexical_score * 0.3,
                dense_score,
                lexical_score,
            )

        for chunk_id, (chunk, lexical_score) in lexical_candidates.items():
            if chunk_id in combined:
                continue
            dense_score = dense_by_id.get(chunk_id).dense_score if chunk_id in dense_by_id else 0.0
            combined[chunk_id] = (
                chunk,
                dense_score * 0.7 + lexical_score * 0.3,
                dense_score or 0.0,
                lexical_score,
            )

        ranked = sorted(combined.values(), key=lambda item: item[1], reverse=True)
        return [
            self._to_search_result(
                chunk=chunk,
                score=score,
                dense_score=dense_score,
                lexical_score=lexical_score,
            )
            for chunk, score, dense_score, lexical_score in ranked[:top_k]
        ]

    def read_chunk(self, scope: str, chunk_id: str) -> ReadChunkResult | None:
        """Вернуть полный текст конкретного чанка."""

        point = self._store.get_chunk(scope, chunk_id)
        if point is None:
            return None
        chunk = self._point_to_chunk(point)
        return ReadChunkResult(
            chunk_id=chunk.point_id,
            scope=chunk.scope,
            relative_path=chunk.relative_path,
            language=chunk.language,
            chunk_type=chunk.chunk_type,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            symbol_path=chunk.symbol_path,
            heading_path=chunk.heading_path,
            text=chunk.text,
        )

    def find_similar_chunk(self, scope: str, chunk_id: str, top_k: int = 10) -> list[SearchResult]:
        """Найти chunks, похожие на уже известный chunk."""

        chunk = self.read_chunk(scope, chunk_id)
        if chunk is None:
            return []
        results = self.semantic_search(
            query=chunk.text,
            top_k=top_k + 1,
            scope=scope,
        )
        return [result for result in results if result.chunk_id != chunk_id][:top_k]

    def index_status(self) -> IndexStatusResult:
        """Собрать текущее состояние индекса и watcher."""

        collections = []
        for scope in ("code", "docs"):
            collections.append(
                IndexCollectionStatus(
                    scope=scope,
                    collection_name=self._store.collection_name(scope),
                    points_count=self._store.count(scope),
                    lexical_documents=len(self._get_lexical_cache(scope).chunks)
                    if self._store.collection_exists(scope)
                    else 0,
                )
            )

        return IndexStatusResult(
            repo_root=str(self._settings.repo_root),
            repo_key=self._settings.repo_key_slug,
            index_profile=self._embedding_provider.index_profile(),
            embedding_backend=self._embedding_provider.backend_name(),
            embedding_model=self._embedding_provider.model_name(),
            qdrant_url=self._settings.SEMANTIC_MCP_QDRANT_URL,
            schema_version=self._settings.SEMANTIC_MCP_INDEX_SCHEMA_VERSION,
            watch_enabled=self._settings.SEMANTIC_MCP_WATCH_ENABLED,
            watch_running=bool(self._watcher and self._watcher.is_running),
            last_full_build_ts=self._indexer.last_full_build_ts,
            collections=collections,
        )
