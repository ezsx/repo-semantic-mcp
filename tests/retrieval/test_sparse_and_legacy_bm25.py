from __future__ import annotations

import unittest

from services.repo_semantic.contracts.search import PayloadCapabilities, SearchFilters
from services.repo_semantic.retrieval.types import _FilterPlan, _LexicalCache
from services.repo_semantic.sparse import SparseManifest, build_sparse_manifest
from tests.helpers.chunks import make_chunk
from tests.helpers.fakes import FakeBM25Okapi, FakePoint, install_rank_bm25_stub

install_rank_bm25_stub()

from services.repo_semantic.retrieval.legacy_bm25 import (  # noqa: E402
    build_lexical_cache,
    legacy_bm25_candidates,
)
from services.repo_semantic.retrieval.sparse_branch import sparse_query_candidates  # noqa: E402


class ManifestRepo:
    def __init__(self, manifests: dict[str, SparseManifest]) -> None:
        self.manifests = manifests

    def load(self, scope: str) -> SparseManifest | None:
        return self.manifests.get(scope)


class ChunkPoint(FakePoint):
    def __init__(self, chunk, score: float) -> None:
        super().__init__(chunk, score)
        self.chunk = chunk


class SparseStore:
    def __init__(
        self,
        points_by_scope: dict[str, list[ChunkPoint]],
        *,
        sparse_scores: dict[str, float],
    ) -> None:
        self.points_by_scope = points_by_scope
        self.sparse_scores = sparse_scores
        self.calls: list[tuple[str, list[int], list[float], int, object]] = []

    def count(self, scope: str) -> int:
        return len(self.points_by_scope.get(scope, []))

    def search_sparse(
        self,
        scope: str,
        indices: list[int],
        values: list[float],
        limit: int,
        query_filter=None,
    ) -> list[ChunkPoint]:
        self.calls.append((scope, indices, values, limit, query_filter))
        scored = [
            point
            for point in self.points_by_scope.get(scope, [])
            if self.sparse_scores.get(point.payload["chunk_id"], 0.0) > 0
        ]
        for point in scored:
            point.score = self.sparse_scores[point.payload["chunk_id"]]
        return sorted(scored, key=lambda point: point.score, reverse=True)[:limit]


class NoSparseStore:
    pass


class CacheStore:
    def __init__(self, points_by_scope: dict[str, list[ChunkPoint]]) -> None:
        self.points_by_scope = points_by_scope
        self.scrolled_scopes: list[str] = []

    def scroll_chunks(self, scope: str) -> list[ChunkPoint]:
        self.scrolled_scopes.append(scope)
        return list(self.points_by_scope.get(scope, []))


def _plan(*, query_filter=None) -> _FilterPlan:
    return _FilterPlan(
        qdrant_filter=query_filter,
        pushed_families=["path_prefix"],
        post_filter_families=[],
        required_payload_fields=[],
        missing_payload_fields=[],
        best_effort=False,
        warning_codes=[],
        payload_capabilities=PayloadCapabilities(
            scope="code",
            collection_name="test_code",
            checked=True,
        ),
    )


class SparseBranchTests(unittest.TestCase):
    def test_sparse_branch_keeps_ready_scope_candidates_and_reports_partial_unavailable_scope(self) -> None:
        exact = make_chunk("exact", "src/settings.py", "DATABASE_URL settings")
        docs = make_chunk(
            "docs",
            "docs/settings.md",
            "DATABASE_URL docs",
            scope="docs",
            language="markdown",
            chunk_type="markdown_section",
        )
        manifest = build_sparse_manifest(scope="code", chunks=[exact], schema_version=1)
        store = SparseStore(
            {
                "code": [ChunkPoint(exact, 0.1)],
                "docs": [ChunkPoint(docs, 0.1)],
            },
            sparse_scores={"exact": 7.0},
        )
        code_filter = object()
        docs_filter = object()

        execution = sparse_query_candidates(
            store=store,
            sparse_manifests=ManifestRepo({"code": manifest}),
            query="DATABASE_URL",
            search_scopes=["code", "docs"],
            requested_scopes=["code", "docs"],
            filter_plans_by_scope={
                "code": _plan(query_filter=code_filter),
                "docs": _plan(query_filter=docs_filter),
            },
            filters=SearchFilters(),
            limit=25,
            sparse_manifest_for_scope=lambda scope: manifest if scope == "code" else None,
            sparse_manifest_unavailable_codes=lambda raw_manifest: (
                ["sparse_manifest_missing"] if raw_manifest is None else ["sparse_contract_mismatch"]
            ),
            store_sparse_available=lambda scope: scope == "code",
            point_to_chunk=lambda point: point.chunk,
            matches_filters=lambda chunk, filters: True,
        )

        self.assertEqual(list(execution.candidates), ["exact"])
        self.assertEqual(execution.candidates["exact"][1], 7.0)
        self.assertFalse(execution.branch_available)
        self.assertTrue(execution.partial_unavailable)
        self.assertTrue(execution.manifest_seen)
        self.assertEqual(execution.manifest_hash, manifest.manifest_content_hash)
        self.assertEqual(execution.unavailable_scopes, ["docs"])
        self.assertEqual(execution.unavailable_codes, ["sparse_manifest_missing"])
        self.assertEqual([(call[0], call[3], call[4]) for call in store.calls], [("code", 25, code_filter)])

    def test_sparse_branch_reports_unsupported_store_without_filter_plans(self) -> None:
        execution = sparse_query_candidates(
            store=NoSparseStore(),
            sparse_manifests=ManifestRepo({}),
            query="connect",
            search_scopes=["code"],
            requested_scopes=["code"],
            filter_plans_by_scope={"code": _plan()},
            filters=SearchFilters(),
            limit=10,
            sparse_manifest_for_scope=lambda scope: None,
            sparse_manifest_unavailable_codes=lambda raw_manifest: ["sparse_manifest_missing"],
            store_sparse_available=lambda scope: False,
            point_to_chunk=lambda point: point.chunk,
            matches_filters=lambda chunk, filters: True,
        )

        self.assertFalse(execution.branch_available)
        self.assertFalse(execution.manifest_seen)
        self.assertEqual(execution.unavailable_codes, ["sparse_search_unsupported"])
        self.assertEqual(execution.filter_plans, [])


class LegacyBm25Tests(unittest.TestCase):
    def test_build_lexical_cache_uses_sparse_terms_and_tokenizer_fallback(self) -> None:
        env_chunk = make_chunk("env", "src/settings.py", "DATABASE_URL")
        punctuation_chunk = make_chunk("fallback", "src/msg.py", "!!!")
        store = CacheStore({"code": [ChunkPoint(env_chunk, 0.1), ChunkPoint(punctuation_chunk, 0.1)]})

        cache = build_lexical_cache(
            store=store,
            scope="code",
            point_to_chunk=lambda point: point.chunk,
            tokenize=lambda text: ["fallback-token"],
        )

        self.assertEqual(store.scrolled_scopes, ["code"])
        self.assertIn("database_url", cache.tokens[0])
        self.assertIn("database", cache.tokens[0])
        self.assertIn("url", cache.tokens[0])
        self.assertEqual(cache.tokens[1], ["fallback-token"])

    def test_legacy_bm25_normalizes_scores_filters_zeroes_and_preserves_existing_candidates(self) -> None:
        existing = make_chunk("existing", "src/existing.py", "old sparse result")
        first = make_chunk("first", "src/first.py", "needle")
        second = make_chunk("second", "src/second.py", "needle needle")
        zero = make_chunk("zero", "src/zero.py", "other")
        cache = _LexicalCache(
            chunks=[first, second, zero],
            tokens=[["needle"], ["needle", "needle"], ["other"]],
        )

        execution = legacy_bm25_candidates(
            existing_candidates={"existing": (existing, 0.4)},
            query_tokens=["needle"],
            search_scopes=["code"],
            filters=SearchFilters(),
            get_lexical_cache=lambda scope: cache,
            matches_filters=lambda chunk, filters: True,
            bm25_factory=FakeBM25Okapi,
        )

        self.assertTrue(execution.lexical_cache_used)
        self.assertTrue(execution.legacy_fallback_used)
        self.assertEqual(set(execution.candidates), {"existing", "first", "second"})
        self.assertEqual(execution.candidates["second"][1], 1.0)
        self.assertEqual(execution.candidates["first"][1], 0.5)

    def test_legacy_bm25_applies_client_side_filters_before_scoring(self) -> None:
        keep = make_chunk("keep", "src/keep.py", "needle")
        reject = make_chunk("reject", "tmp/reject.py", "needle")
        cache = _LexicalCache(
            chunks=[keep, reject],
            tokens=[["needle"], ["needle"]],
        )

        execution = legacy_bm25_candidates(
            existing_candidates={},
            query_tokens=["needle"],
            search_scopes=["code"],
            filters=SearchFilters(include_paths=["src/**"]),
            get_lexical_cache=lambda scope: cache,
            matches_filters=lambda chunk, filters: chunk.relative_path.startswith("src/"),
            bm25_factory=FakeBM25Okapi,
        )

        self.assertEqual(list(execution.candidates), ["keep"])


if __name__ == "__main__":
    unittest.main()
