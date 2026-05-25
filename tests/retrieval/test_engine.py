from __future__ import annotations

import unittest

from services.repo_semantic.contracts.search import (
    ExactAnchorAnalysis,
    ExactAnchorCandidate,
    PayloadCapabilities,
    SearchResult,
)
from services.repo_semantic.retrieval.engine import (
    build_hybrid_search_execution,
    build_semantic_search_execution,
)
from services.repo_semantic.retrieval.types import _DenseSearchExecution, _SparseSearchExecution
from tests.helpers.chunks import make_chunk


def _result_factory(**kwargs) -> SearchResult:
    chunk = kwargs["chunk"]
    score = float(kwargs["score"])
    return SearchResult(
        chunk_id=chunk.point_id,
        scope=chunk.scope,
        relative_path=chunk.relative_path,
        language=chunk.language,
        chunk_type=chunk.chunk_type,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        snippet=chunk.text,
        score=score,
        final_score=score,
        match_type=kwargs.get("match_type"),
        dense_score=kwargs.get("dense_score"),
        lexical_score=kwargs.get("lexical_score"),
        matched_terms=list(kwargs.get("query_tokens") or []),
    )


def _merge_filter_plans(plans):
    return (
        ["path_prefix"] if plans else [],
        ["include_paths"] if plans else [],
        True if plans else False,
        [
            PayloadCapabilities(
                scope="code",
                collection_name="test_code",
                checked=True,
                missing_fields=["path_prefixes"],
            )
        ]
        if plans
        else [],
    )


def _exact_heavy_analysis() -> ExactAnchorAnalysis:
    return ExactAnchorAnalysis(
        anchors=[
            ExactAnchorCandidate(
                surface="DATABASE_URL",
                canonical="DATABASE_URL",
                start=0,
                end=12,
                anchor_type="env_var",
                confidence="high",
            )
        ],
        exact_anchor_heavy=True,
        exact_fallback_recommended=True,
    )


class RetrievalEngineTests(unittest.TestCase):
    def test_semantic_engine_keeps_global_order_with_per_file_cap_and_filter_diagnostics(self) -> None:
        a1 = make_chunk("a1", "src/a.py", "first")
        a2 = make_chunk("a2", "src/a.py", "second")
        b = make_chunk("b", "src/b.py", "third")
        dense_execution = _DenseSearchExecution(
            candidates={
                "a1": (a1, 0.99),
                "a2": (a2, 0.98),
                "b": (b, 0.5),
            },
            candidate_limit=30,
            candidates_scanned=3,
            filtered_candidates=3,
            kept_candidates=2,
            dense_best_effort=True,
            filter_plans=[object()],
        )

        execution = build_semantic_search_execution(
            dense_execution=dense_execution,
            top_k=2,
            max_results_per_file=1,
            query_tokens=["first"],
            exact_anchor_analysis=ExactAnchorAnalysis(),
            snippet_mode="chunk_start",
            include_explanations=True,
            to_search_result=_result_factory,
            merge_filter_plans=_merge_filter_plans,
        )

        self.assertEqual([result.chunk_id for result in execution.results], ["a1", "b"])
        self.assertEqual(execution.branch_ranks_by_chunk, {"a1": {"dense": 1}, "b": {"dense": 2}})
        self.assertEqual(
            [warning.code for warning in execution.warnings],
            [
                "filtered_dense_search_best_effort",
                "qdrant_filter_pushdown_used",
                "qdrant_filter_pushdown_partial",
                "qdrant_filter_post_filter_best_effort",
            ],
        )
        self.assertEqual(execution.diagnostics.retrieval_backend, "qdrant_dense")
        self.assertEqual(execution.diagnostics.filter_pushed_families, ["path_prefix"])
        self.assertTrue(execution.diagnostics.filter_best_effort)

    def test_hybrid_engine_uses_exact_anchor_weights_and_preserves_branch_ranks(self) -> None:
        dense_only = make_chunk("dense", "docs/overview.md", "database overview")
        exact = make_chunk("exact", "deploy/compose.yml", "DATABASE_URL")
        dense_execution = _DenseSearchExecution(
            candidates={
                "dense": (dense_only, 0.99),
                "exact": (exact, 0.01),
            },
            candidate_limit=50,
            candidates_scanned=2,
            filtered_candidates=2,
            kept_candidates=2,
            dense_best_effort=False,
            filter_plans=[],
        )
        sparse_execution = _SparseSearchExecution(
            candidates={"exact": (exact, 10.0)},
            branch_available=True,
            partial_unavailable=False,
            manifest_seen=True,
            stats_stale=False,
            manifest_hash="manifest",
            vocabulary_hash="vocab",
            corpus_stats_hash="stats",
            unavailable_scopes=[],
            unavailable_codes=[],
            filter_plans=[],
        )

        execution = build_hybrid_search_execution(
            dense_execution=dense_execution,
            sparse_execution=sparse_execution,
            sparse_candidates=dict(sparse_execution.candidates),
            lexical_cache_used=False,
            legacy_fallback_used=False,
            top_k=1,
            sparse_limit=100,
            max_results_per_file=None,
            exact_anchor_analysis=_exact_heavy_analysis(),
            result_query_tokens=["DATABASE_URL"],
            snippet_mode="chunk_start",
            include_explanations=True,
            to_search_result=_result_factory,
            merge_filter_plans=_merge_filter_plans,
        )

        self.assertEqual([result.chunk_id for result in execution.results], ["exact"])
        self.assertEqual(execution.results[0].match_type, "hybrid")
        self.assertEqual(execution.branch_ranks_by_chunk, {"exact": {"dense": 2, "sparse": 1}})
        self.assertEqual(execution.diagnostics.branch_weights, {"dense": 0.8, "sparse": 1.4})
        self.assertEqual(execution.diagnostics.retrieval_backend, "qdrant_dense_sparse_rrf")
        self.assertEqual(execution.diagnostics.sparse_manifest_hash, "manifest")
        self.assertEqual([warning.code for warning in execution.warnings], ["exact_anchor_verify_with_local_rg"])

    def test_hybrid_engine_reports_sparse_unavailable_and_legacy_fallback_diagnostics(self) -> None:
        chunk = make_chunk("needle", "src/feature.py", "needle")
        dense_execution = _DenseSearchExecution(
            candidates={"needle": (chunk, 0.8)},
            candidate_limit=50,
            candidates_scanned=1,
            filtered_candidates=1,
            kept_candidates=1,
            dense_best_effort=False,
            filter_plans=[],
        )
        sparse_execution = _SparseSearchExecution(
            candidates={},
            branch_available=False,
            partial_unavailable=True,
            manifest_seen=False,
            stats_stale=True,
            manifest_hash=None,
            vocabulary_hash=None,
            corpus_stats_hash=None,
            unavailable_scopes=["docs"],
            unavailable_codes=["sparse_manifest_missing"],
            filter_plans=[],
        )

        execution = build_hybrid_search_execution(
            dense_execution=dense_execution,
            sparse_execution=sparse_execution,
            sparse_candidates={"needle": (chunk, 1.0)},
            lexical_cache_used=True,
            legacy_fallback_used=True,
            top_k=1,
            sparse_limit=100,
            max_results_per_file=None,
            exact_anchor_analysis=ExactAnchorAnalysis(),
            result_query_tokens=["needle"],
            snippet_mode="chunk_start",
            include_explanations=True,
            to_search_result=_result_factory,
            merge_filter_plans=_merge_filter_plans,
        )

        self.assertEqual(execution.diagnostics.retrieval_backend, "legacy_dense_local_bm25")
        self.assertFalse(execution.diagnostics.sparse_available)
        self.assertTrue(execution.diagnostics.lexical_cache_used)
        self.assertEqual(
            [warning.code for warning in execution.warnings],
            [
                "sparse_branch_unavailable",
                "sparse_manifest_missing",
                "sparse_scope_partial_unavailable",
                "sparse_stats_stale",
                "legacy_lexical_fallback_used",
            ],
        )
        self.assertEqual(
            execution.diagnostics.branch_diagnostics["sparse"].warning_codes,
            [
                "sparse_stats_stale",
                "sparse_branch_unavailable",
                "sparse_manifest_missing",
            ],
        )


if __name__ == "__main__":
    unittest.main()
