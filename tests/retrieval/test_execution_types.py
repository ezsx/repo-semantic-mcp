from __future__ import annotations

import dataclasses
import unittest

from services.repo_semantic.contracts.search import PayloadCapabilities, SearchDiagnostics
from services.repo_semantic.retrieval import types as retrieval_types
from tests.helpers.chunks import make_chunk
from tests.helpers.fakes import install_rank_bm25_stub

install_rank_bm25_stub()

import services.repo_semantic.search_service as search_service_module


class RetrievalExecutionTypesTests(unittest.TestCase):
    def test_search_service_keeps_internal_type_compatibility_aliases(self) -> None:
        self.assertIs(search_service_module._LexicalCache, retrieval_types._LexicalCache)
        self.assertIs(search_service_module._FilterPlan, retrieval_types._FilterPlan)
        self.assertIs(search_service_module._DenseSearchExecution, retrieval_types._DenseSearchExecution)
        self.assertIs(search_service_module._SparseSearchExecution, retrieval_types._SparseSearchExecution)
        self.assertIs(search_service_module._LegacyBm25Execution, retrieval_types._LegacyBm25Execution)
        self.assertIs(search_service_module._SearchExecution, retrieval_types._SearchExecution)

    def test_execution_types_are_slotted_dataclasses(self) -> None:
        for type_name in [
            "_LexicalCache",
            "_FilterPlan",
            "_DenseSearchExecution",
            "_SparseSearchExecution",
            "_LegacyBm25Execution",
            "_SearchExecution",
        ]:
            type_object = getattr(retrieval_types, type_name)
            self.assertTrue(dataclasses.is_dataclass(type_object), type_name)
            self.assertTrue(hasattr(type_object, "__slots__"), type_name)

    def test_execution_type_field_order_is_stable(self) -> None:
        self.assertEqual(
            [field.name for field in dataclasses.fields(retrieval_types._LexicalCache)],
            ["chunks", "tokens"],
        )
        self.assertEqual(
            [field.name for field in dataclasses.fields(retrieval_types._FilterPlan)],
            [
                "qdrant_filter",
                "pushed_families",
                "post_filter_families",
                "required_payload_fields",
                "missing_payload_fields",
                "best_effort",
                "warning_codes",
                "payload_capabilities",
            ],
        )
        self.assertEqual(
            [field.name for field in dataclasses.fields(retrieval_types._DenseSearchExecution)],
            [
                "candidates",
                "candidate_limit",
                "candidates_scanned",
                "filtered_candidates",
                "kept_candidates",
                "dense_best_effort",
                "filter_plans",
            ],
        )
        self.assertEqual(
            [field.name for field in dataclasses.fields(retrieval_types._SparseSearchExecution)],
            [
                "candidates",
                "branch_available",
                "partial_unavailable",
                "manifest_seen",
                "stats_stale",
                "manifest_hash",
                "vocabulary_hash",
                "corpus_stats_hash",
                "unavailable_scopes",
                "unavailable_codes",
                "filter_plans",
            ],
        )
        self.assertEqual(
            [field.name for field in dataclasses.fields(retrieval_types._SearchExecution)],
            ["results", "diagnostics", "warnings", "branch_ranks_by_chunk"],
        )
        self.assertEqual(
            [field.name for field in dataclasses.fields(retrieval_types._LegacyBm25Execution)],
            ["candidates", "lexical_cache_used", "legacy_fallback_used"],
        )

    def test_filter_plan_shape_accepts_payload_capabilities(self) -> None:
        plan = retrieval_types._FilterPlan(
            qdrant_filter=None,
            pushed_families=["path_prefix"],
            post_filter_families=[],
            required_payload_fields=["path_prefixes"],
            missing_payload_fields=[],
            best_effort=False,
            warning_codes=[],
            payload_capabilities=PayloadCapabilities(
                scope="code",
                collection_name="test_code",
                checked=True,
            ),
        )

        self.assertEqual(plan.pushed_families, ["path_prefix"])
        self.assertEqual(plan.payload_capabilities.scope, "code")

    def test_cache_dense_and_sparse_execution_shapes_are_constructible(self) -> None:
        chunk = make_chunk("chunk", "src/app.py", "connect flow")
        plan = retrieval_types._FilterPlan(
            qdrant_filter=None,
            pushed_families=[],
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

        cache = retrieval_types._LexicalCache(chunks=[chunk], tokens=[["connect"]])
        dense = retrieval_types._DenseSearchExecution(
            candidates={chunk.point_id: (chunk, 0.9)},
            candidate_limit=50,
            candidates_scanned=1,
            filtered_candidates=1,
            kept_candidates=1,
            dense_best_effort=False,
            filter_plans=[plan],
        )
        sparse = retrieval_types._SparseSearchExecution(
            candidates={chunk.point_id: (chunk, 1.0)},
            branch_available=True,
            partial_unavailable=False,
            manifest_seen=True,
            stats_stale=False,
            manifest_hash="manifest",
            vocabulary_hash="vocab",
            corpus_stats_hash="stats",
            unavailable_scopes=[],
            unavailable_codes=[],
            filter_plans=[plan],
        )
        legacy = retrieval_types._LegacyBm25Execution(
            candidates={chunk.point_id: (chunk, 0.7)},
            lexical_cache_used=True,
            legacy_fallback_used=True,
        )

        self.assertEqual(cache.tokens, [["connect"]])
        self.assertEqual(dense.kept_candidates, 1)
        self.assertTrue(sparse.branch_available)
        self.assertEqual(sparse.manifest_hash, "manifest")
        self.assertTrue(legacy.lexical_cache_used)

    def test_search_execution_shape_preserves_branch_rank_map(self) -> None:
        execution = retrieval_types._SearchExecution(
            results=[],
            diagnostics=SearchDiagnostics(final_results=0),
            warnings=[],
            branch_ranks_by_chunk={"chunk": {"dense": 1}},
        )

        self.assertEqual(execution.branch_ranks_by_chunk, {"chunk": {"dense": 1}})


if __name__ == "__main__":
    unittest.main()
