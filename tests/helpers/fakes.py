from __future__ import annotations

import sys
import types
from pathlib import Path

from services.repo_semantic.models import ChunkRecord


class FakeBM25Okapi:
    def __init__(self, tokens):
        self._tokens = tokens

    def get_scores(self, query_tokens):
        scores = []
        for tokens in self._tokens:
            scores.append(float(sum(1 for token in tokens if token in query_tokens)))
        return scores


def install_rank_bm25_stub() -> None:
    rank_bm25 = types.ModuleType("rank_bm25")
    rank_bm25.BM25Okapi = FakeBM25Okapi
    sys.modules.setdefault("rank_bm25", rank_bm25)


class FakeEmbeddingProvider:
    def backend_name(self) -> str:
        return "fake"

    def model_name(self) -> str:
        return "fake-model"

    def index_profile(self) -> str:
        return "test-profile"

    def embed_query(self, query: str) -> list[float]:
        return [1.0]


class FakePoint:
    def __init__(self, chunk: ChunkRecord, score: float) -> None:
        self.id = chunk.point_id
        path_parts = chunk.relative_path.replace("\\", "/").strip("/").split("/")
        self.payload = {
            "chunk_id": chunk.point_id,
            "scope": chunk.scope,
            "relative_path": chunk.relative_path,
            "file_extension": Path(chunk.relative_path).suffix.lower() or None,
            "path_prefixes": [
                "/".join(path_parts[:index])
                for index in range(1, len(path_parts) + 1)
                if path_parts[:index]
            ],
            "language": chunk.language,
            "chunk_type": chunk.chunk_type,
            "text": chunk.text,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "content_hash": chunk.content_hash,
            "source_mtime": chunk.source_mtime,
            "symbol_path": chunk.symbol_path,
            "heading_path": chunk.heading_path,
            "domain_tags": chunk.domain_tags,
            "is_generated": chunk.is_generated,
        }
        self.score = score


class FakeStore:
    def __init__(
        self,
        points_by_scope: dict[str, list[FakePoint]],
        *,
        sparse_enabled: bool = False,
        sparse_scores: dict[str, float] | None = None,
        fail_on_scroll: bool = False,
    ) -> None:
        self._points_by_scope = points_by_scope
        self._sparse_enabled = sparse_enabled
        self._sparse_scores = sparse_scores or {}
        self._fail_on_scroll = fail_on_scroll
        self.dense_query_filters: list[object] = []
        self.sparse_query_filters: list[object] = []

    def collection_name(self, scope: str) -> str:
        return f"test_{scope}"

    def collection_exists(self, scope: str) -> bool:
        return bool(self._points_by_scope.get(scope))

    def count(self, scope: str) -> int:
        return len(self._points_by_scope.get(scope, []))

    def collection_embedding_contract(self, scope: str):
        if not self.collection_exists(scope):
            return None
        return {
            "embedding_backend": "fake",
            "embedding_model": "fake-model",
            "index_schema_version": "1",
        }

    def search(self, scope: str, query_vector: list[float], limit: int, query_filter=None):
        self.dense_query_filters.append(query_filter)
        return sorted(self._points_by_scope.get(scope, []), key=lambda point: point.score, reverse=True)[:limit]

    def search_sparse(self, scope: str, indices: list[int], values: list[float], limit: int, query_filter=None):
        self.sparse_query_filters.append(query_filter)
        if not self._sparse_enabled:
            return []
        scored = [
            point
            for point in self._points_by_scope.get(scope, [])
            if self._sparse_scores.get(point.payload["chunk_id"], 0.0) > 0
        ]
        for point in scored:
            point.score = self._sparse_scores[point.payload["chunk_id"]]
        return sorted(scored, key=lambda point: point.score, reverse=True)[:limit]

    def scroll_chunks(self, scope: str):
        if self._fail_on_scroll:
            raise AssertionError("scroll_chunks should not be used")
        return list(self._points_by_scope.get(scope, []))

    def get_chunk(self, scope: str, chunk_id: str):
        for point in self._points_by_scope.get(scope, []):
            if point.payload["chunk_id"] == chunk_id:
                return point
        return None

    def sparse_vector_available(self, scope: str) -> bool:
        return self._sparse_enabled and self.collection_exists(scope)

    def dense_schema_kind(self, scope: str) -> str:
        if not self.collection_exists(scope):
            return "missing"
        return "named" if self._sparse_enabled else "unnamed_legacy"

    def read_collection_exists(self, scope: str) -> bool:
        return self.collection_exists(scope)

    def current_collection_exists(self, scope: str) -> bool:
        return self.collection_exists(scope)

    def payload_fields(self, scope: str) -> set[str]:
        points = self._points_by_scope.get(scope, [])
        return set(points[0].payload) if points else set()


class FakeIndexer:
    last_full_build_ts = "2026-05-06T00:00:00+00:00"
    last_incremental_update_ts = "2026-05-06T00:00:00+00:00"
