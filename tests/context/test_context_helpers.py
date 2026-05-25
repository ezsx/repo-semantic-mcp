from __future__ import annotations

import unittest

from services.repo_semantic.context.actions import build_context_actions, context_read_actions, dedupe_context_actions
from services.repo_semantic.context.assembler import (
    context_file_groups,
    context_origin_branches,
    context_uncovered_terms,
    merge_context_results,
    merge_context_warnings,
)
from services.repo_semantic.context.query_plan import (
    aggregate_exact_anchors,
    context_route_used,
    normalize_context_queries,
)
from services.repo_semantic.context.diagnostics import build_context_diagnostics
from services.repo_semantic.contracts.repo_context import RecommendedAction, RepoContextResult
from services.repo_semantic.contracts.search import (
    BranchDiagnostics,
    ExactAnchorAnalysis,
    SearchDiagnostics,
    SearchResult,
    SearchWarning,
)
from services.repo_semantic.retrieval.constants import MAX_CONTEXT_SUBQUERIES
from services.repo_semantic.retrieval.types import _SearchExecution


def _result(
    chunk_id: str,
    relative_path: str,
    *,
    final_rank: int,
    final_score: float,
    line_range: str | None = None,
    matched_terms: list[str] | None = None,
    origin_queries: list[str] | None = None,
    origin_branches: list[str] | None = None,
) -> RepoContextResult:
    return RepoContextResult(
        chunk_id=chunk_id,
        scope="code",
        relative_path=relative_path,
        language="python",
        chunk_type="python_function",
        start_line=final_rank,
        end_line=final_rank + 2,
        line_range=line_range,
        snippet=f"snippet {chunk_id}",
        score=final_score,
        final_rank=final_rank,
        final_score=final_score,
        matched_terms=matched_terms or [],
        origin_queries=origin_queries or [],
        origin_branches=origin_branches or [],
    )


def _search_result(
    chunk_id: str,
    relative_path: str,
    *,
    snippet: str,
    score: float,
    start_line: int = 1,
    dense_score: float | None = None,
    lexical_score: float | None = None,
    matched_terms: list[str] | None = None,
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        scope="code",
        relative_path=relative_path,
        language="python",
        chunk_type="python_function",
        start_line=start_line,
        end_line=start_line,
        line_range=f"{start_line}-{start_line}",
        snippet=snippet,
        score=score,
        final_score=score,
        dense_score=dense_score,
        lexical_score=lexical_score,
        matched_terms=matched_terms or [],
    )


class ContextQueryPlanTests(unittest.TestCase):
    def test_normalize_context_queries_preserves_original_and_drops_duplicates_empty_and_over_cap(self) -> None:
        seen_validations: list[str] = []
        subqueries = ["alpha docs", "alpha", "", "alpha docs", *[f"extra {index}" for index in range(20)]]

        items, dropped = normalize_context_queries(
            "alpha",
            subqueries,
            validate_query=seen_validations.append,
        )

        self.assertEqual(items[0], ("alpha", "original", 1.0))
        self.assertEqual(items[1], ("alpha docs", "subquery", 0.85))
        self.assertEqual(len(items), MAX_CONTEXT_SUBQUERIES + 1)
        self.assertGreater(dropped, 0)
        self.assertIn("alpha", seen_validations)
        self.assertIn("alpha docs", seen_validations)

    def test_context_route_used_maps_auto_and_rejects_unknown(self) -> None:
        self.assertEqual(context_route_used("auto"), "hybrid")
        self.assertEqual(context_route_used("semantic"), "semantic")
        with self.assertRaises(ValueError):
            context_route_used("unknown")  # type: ignore[arg-type]

    def test_aggregate_exact_anchors_dedupes_anchor_but_keeps_query_usages(self) -> None:
        analysis, usages = aggregate_exact_anchors(
            [
                ("DATABASE_URL runtime", "original", 1.0),
                ("DATABASE_URL docs", "subquery", 0.85),
            ]
        )

        self.assertEqual([anchor.surface for anchor in analysis.anchors], ["DATABASE_URL"])
        self.assertTrue(analysis.exact_anchor_heavy)
        self.assertEqual(len(usages), 2)
        self.assertEqual(analysis.argv_hints[0], ["rg", "--fixed-strings", "--line-number", "--max-count", "20", "-e", "DATABASE_URL", "."])


class ContextAssemblerTests(unittest.TestCase):
    def test_context_origin_branches_distinguishes_sparse_from_legacy_lexical(self) -> None:
        self.assertEqual(
            context_origin_branches({"dense": 1}, SearchDiagnostics(final_results=1), "semantic"),
            ["dense"],
        )
        self.assertEqual(
            context_origin_branches(
                {"dense": 1, "sparse": 2},
                SearchDiagnostics(final_results=1, retrieval_backend="qdrant_dense_sparse_rrf"),
                "hybrid",
            ),
            ["dense", "sparse"],
        )
        self.assertEqual(
            context_origin_branches(
                {"sparse": 1},
                SearchDiagnostics(final_results=1, retrieval_backend="legacy_dense_local_bm25"),
                "hybrid",
            ),
            ["legacy_lexical"],
        )

    def test_merge_context_warnings_preserves_first_occurrence_order(self) -> None:
        warnings = [
            SearchWarning(code="a", severity="info", detail="same"),
            SearchWarning(code="b", severity="warning", detail=None),
            SearchWarning(code="a", severity="info", detail="same"),
        ]

        self.assertEqual([warning.code for warning in merge_context_warnings(warnings)], ["a", "b"])

    def test_context_uncovered_terms_respects_exact_anchor_coverage(self) -> None:
        analysis, _usages = aggregate_exact_anchors([("where is DATABASE_URL used", "original", 1.0)])

        self.assertEqual(
            context_uncovered_terms([("where is DATABASE_URL used", "original", 1.0)], analysis, ["DATABASE_URL"]),
            [],
        )
        self.assertEqual(
            context_uncovered_terms(
                [("UserRepository.findByEmail permission denied", "original", 1.0)],
                ExactAnchorAnalysis(),
                ["UserRepository.findByEmail", "UserRepository", "findByEmail"],
            ),
            ["permission", "denied"],
        )

    def test_context_file_groups_merge_by_file_and_cap_preview_fields(self) -> None:
        groups = context_file_groups(
            [
                _result(
                    "a1",
                    "src/a.py",
                    final_rank=2,
                    final_score=0.5,
                    line_range="10-12",
                    matched_terms=["alpha"],
                    origin_queries=["alpha"],
                    origin_branches=["dense"],
                ),
                _result(
                    "a2",
                    "src/a.py",
                    final_rank=1,
                    final_score=0.9,
                    line_range="20-22",
                    matched_terms=["beta"],
                    origin_queries=["beta"],
                    origin_branches=["sparse"],
                ),
                _result("b", "src/b.py", final_rank=3, final_score=0.1),
            ],
            top_k=2,
        )

        self.assertEqual([group.relative_path for group in groups], ["src/a.py", "src/b.py"])
        self.assertEqual(groups[0].best_rank, 1)
        self.assertEqual(groups[0].best_score, 0.9)
        self.assertEqual(groups[0].chunk_count, 2)
        self.assertEqual(groups[0].chunk_ids, ["a1", "a2"])
        self.assertEqual(groups[0].matched_terms, ["alpha", "beta"])

    def test_merge_context_results_dedupes_and_keeps_best_ranked_occurrence(self) -> None:
        original_other = _search_result(
            "other",
            "src/other.py",
            snippet="other",
            score=0.9,
            dense_score=0.9,
            matched_terms=["original"],
        )
        original_duplicate = _search_result(
            "dup",
            "src/dup.py",
            snippet="original lower ranked snippet",
            score=0.5,
            start_line=10,
            dense_score=0.5,
            matched_terms=["original"],
        )
        subquery_duplicate = _search_result(
            "dup",
            "src/dup.py",
            snippet="subquery highest ranked snippet",
            score=0.95,
            start_line=20,
            dense_score=0.95,
            matched_terms=["subquery"],
        )

        results, matched_terms, uncovered_terms = merge_context_results(
            executions=[
                (
                    "original",
                    "original",
                    1.0,
                    _SearchExecution(
                        results=[original_other, original_duplicate],
                        diagnostics=SearchDiagnostics(final_results=2),
                        warnings=[],
                        branch_ranks_by_chunk={"other": {"dense": 1}, "dup": {"dense": 2}},
                    ),
                ),
                (
                    "subquery",
                    "subquery",
                    0.85,
                    _SearchExecution(
                        results=[subquery_duplicate],
                        diagnostics=SearchDiagnostics(final_results=1),
                        warnings=[],
                        branch_ranks_by_chunk={"dup": {"dense": 1}},
                    ),
                ),
            ],
            query_items=[("original", "original", 1.0), ("subquery", "subquery", 0.85)],
            route_used="semantic",
            top_k=2,
            max_results_per_file=None,
            exact_anchor_analysis=ExactAnchorAnalysis(),
            stale_index=False,
            warnings=[],
        )

        duplicate = next(result for result in results if result.chunk_id == "dup")
        self.assertEqual(duplicate.snippet, "subquery highest ranked snippet")
        self.assertEqual(duplicate.line_range, "20-20")
        self.assertEqual(duplicate.origin_queries, ["original", "subquery"])
        self.assertEqual(duplicate.origin_branches, ["dense"])
        self.assertEqual(duplicate.branch_ranks, {"dense": 1})
        self.assertEqual(matched_terms, ["original", "subquery"])
        self.assertEqual(uncovered_terms, [])


class ContextActionsTests(unittest.TestCase):
    def test_context_read_actions_dedupes_file_ranges_and_caps_results(self) -> None:
        actions = context_read_actions(
            [
                _result("a1", "src/a.py", final_rank=1, final_score=0.9, line_range="1-3"),
                _result("a2", "src/a.py", final_rank=1, final_score=0.8, line_range="1-3"),
                _result("b", "src/b.py", final_rank=3, final_score=0.7),
            ]
        )

        self.assertEqual([action.relative_path for action in actions], ["src/a.py", "src/b.py"])

    def test_dedupe_context_actions_uses_code_path_range_and_argv_hint(self) -> None:
        actions = dedupe_context_actions(
            [
                RecommendedAction(code="run_local_rg", severity="info", title="rg", argv_hint=["rg", "needle"]),
                RecommendedAction(code="run_local_rg", severity="info", title="rg", argv_hint=["rg", "needle"]),
                RecommendedAction(code="run_local_rg", severity="info", title="rg", argv_hint=["rg", "other"]),
            ]
        )

        self.assertEqual([action.argv_hint for action in actions], [["rg", "needle"], ["rg", "other"]])

    def test_build_context_actions_adds_exact_sparse_and_scope_retry_hints(self) -> None:
        exact_analysis, _usages = aggregate_exact_anchors([("DATABASE_URL docs", "original", 1.0)])
        results = [_result("code", "src/settings.py", final_rank=1, final_score=0.9)]
        file_groups = context_file_groups(results, top_k=10)

        actions, sparse_codes = build_context_actions(
            results=results,
            file_groups=file_groups,
            exact_anchor_analysis=exact_analysis,
            per_query_diagnostics=[
                SearchDiagnostics(
                    final_results=1,
                    sparse_unavailable_codes=["sparse_manifest_missing"],
                )
            ],
            status=None,
            route_used="hybrid",
            scope="all",
            path_prefix=None,
            include_paths=None,
            query_items=[("DATABASE_URL docs", "original", 1.0)],
            top_k=10,
            global_matched_terms=["DATABASE_URL"],
        )

        self.assertIn("sparse_manifest_missing", sparse_codes)
        action_codes = [action.code for action in actions]
        self.assertIn("read_file_range", action_codes)
        self.assertIn("run_local_rg", action_codes)
        self.assertIn("rebuild_index", action_codes)
        self.assertIn("retry_with_docs_scope", action_codes)


class ContextDiagnosticsTests(unittest.TestCase):
    def test_build_context_diagnostics_merges_branch_and_distribution_state(self) -> None:
        exact_analysis, exact_usages = aggregate_exact_anchors([("DATABASE_URL", "original", 1.0)])
        results = [_result("code", "src/settings.py", final_rank=1, final_score=0.9)]
        file_groups = context_file_groups(results, top_k=10)

        diagnostics = build_context_diagnostics(
            route_used="hybrid",
            query_items=[("DATABASE_URL", "original", 1.0), ("database docs", "subquery", 0.85)],
            per_query_diagnostics=[
                SearchDiagnostics(
                    final_results=1,
                    branch_diagnostics={
                        "dense": BranchDiagnostics(
                            available=True,
                            candidate_limit=50,
                            candidates_returned=2,
                            warning_codes=["dense_warn"],
                        )
                    },
                    sparse_available=True,
                    filter_pushed_families=["path_prefix"],
                ),
                SearchDiagnostics(
                    final_results=1,
                    branch_diagnostics={
                        "dense": BranchDiagnostics(
                            available=False,
                            candidate_limit=80,
                            candidates_returned=3,
                            warning_codes=["other_warn"],
                        )
                    },
                    sparse_available=False,
                    filter_post_families=["include_paths"],
                    filter_best_effort=True,
                ),
            ],
            exact_anchor_analysis=exact_analysis,
            exact_anchor_usages=exact_usages,
            results=results,
            file_groups=file_groups,
            global_matched_terms=["DATABASE_URL"],
            uncovered_terms=[],
            sparse_codes={"sparse_manifest_missing"},
            include_diagnostics=False,
        )

        self.assertEqual(diagnostics.fusion_method, "multi_query_weighted_rrf")
        self.assertEqual(diagnostics.rrf_k, 60)
        self.assertEqual(diagnostics.per_query, [])
        self.assertTrue(diagnostics.branch_diagnostics["dense"].available)
        self.assertEqual(diagnostics.branch_diagnostics["dense"].candidate_limit, 80)
        self.assertEqual(diagnostics.branch_diagnostics["dense"].candidates_returned, 5)
        self.assertEqual(diagnostics.branch_diagnostics["dense"].warning_codes, ["dense_warn", "other_warn"])
        self.assertEqual(diagnostics.filter_pushed_families, ["path_prefix"])
        self.assertEqual(diagnostics.filter_post_families, ["include_paths"])
        self.assertTrue(diagnostics.filter_best_effort)
        self.assertFalse(diagnostics.sparse_available)
        self.assertEqual(diagnostics.sparse_unavailable_codes, ["sparse_manifest_missing"])
        self.assertEqual(diagnostics.scope_distribution, {"code": 1, "docs": 0})
        self.assertEqual(diagnostics.file_distribution[0].relative_path, "src/settings.py")


if __name__ == "__main__":
    unittest.main()
