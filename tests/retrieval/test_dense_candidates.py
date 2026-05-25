from __future__ import annotations

import unittest

from services.repo_semantic.contracts.search import PayloadCapabilities, SearchFilters
from services.repo_semantic.retrieval.dense import dense_candidates
from services.repo_semantic.retrieval.types import _FilterPlan

from tests.helpers.chunks import make_chunk


class DensePoint:
    def __init__(self, chunk, score: float) -> None:
        self.chunk = chunk
        self.score = score


class DenseStore:
    def __init__(self, points_by_scope: dict[str, list[DensePoint]]) -> None:
        self.points_by_scope = points_by_scope
        self.calls: list[tuple[str, int, object]] = []

    def search(self, scope: str, query_vector: list[float], limit: int, query_filter=None):
        self.calls.append((scope, limit, query_filter))
        return self.points_by_scope.get(scope, [])[:limit]


def _plan(*, best_effort: bool = False, query_filter=None) -> _FilterPlan:
    return _FilterPlan(
        qdrant_filter=query_filter,
        pushed_families=[] if best_effort else ["path_prefix"],
        post_filter_families=["include_paths"] if best_effort else [],
        required_payload_fields=[],
        missing_payload_fields=[],
        best_effort=best_effort,
        warning_codes=[],
        payload_capabilities=PayloadCapabilities(
            scope="code",
            collection_name="test_code",
            checked=True,
        ),
    )


class DenseCandidatesTests(unittest.TestCase):
    def test_dense_candidates_passes_filter_and_keeps_highest_duplicate_score(self) -> None:
        duplicate_low = make_chunk("shared", "src/low.py", "alpha")
        duplicate_high = make_chunk("shared", "src/high.py", "alpha")
        docs = make_chunk("docs", "docs/a.md", "alpha", scope="docs", language="markdown", chunk_type="markdown_section")
        store = DenseStore(
            {
                "code": [DensePoint(duplicate_low, 0.1), DensePoint(duplicate_high, 0.9)],
                "docs": [DensePoint(docs, 0.5)],
            }
        )
        code_filter = object()
        docs_filter = object()

        execution = dense_candidates(
            store=store,
            query_vector=[1.0],
            search_scopes=["code", "docs"],
            filter_plans_by_scope={"code": _plan(query_filter=code_filter), "docs": _plan(query_filter=docs_filter)},
            filters=SearchFilters(),
            target_results=10,
            max_results_per_file=None,
            point_to_chunk=lambda point: point.chunk,
            matches_filters=lambda chunk, filters: True,
        )

        self.assertEqual(execution.candidates["shared"][1], 0.9)
        self.assertEqual(execution.candidates["shared"][0].relative_path, "src/high.py")
        self.assertEqual([(call[0], call[2]) for call in store.calls], [("code", code_filter), ("docs", docs_filter)])
        self.assertEqual(execution.candidates_scanned, 3)
        self.assertEqual(execution.filtered_candidates, 3)

    def test_dense_candidates_expands_best_effort_limit_for_per_file_cap(self) -> None:
        same_file = [
            DensePoint(make_chunk(f"a-{index}", "pkg/a.py", f"alpha {index}"), 1.0 - index / 1000)
            for index in range(80)
        ]
        second_file = DensePoint(make_chunk("b", "pkg/b.py", "alpha b"), 0.1)
        store = DenseStore({"code": [*same_file, second_file]})

        execution = dense_candidates(
            store=store,
            query_vector=[1.0],
            search_scopes=["code"],
            filter_plans_by_scope={"code": _plan(best_effort=True)},
            filters=SearchFilters(),
            target_results=2,
            max_results_per_file=1,
            point_to_chunk=lambda point: point.chunk,
            matches_filters=lambda chunk, filters: True,
        )

        self.assertEqual([call[1] for call in store.calls], [80, 160])
        self.assertEqual(execution.candidate_limit, 160)
        self.assertEqual(execution.kept_candidates, 2)
        self.assertFalse(execution.dense_best_effort)

    def test_dense_candidates_expands_best_effort_limit_after_post_filter_rejection(self) -> None:
        rejected = [
            DensePoint(make_chunk(f"reject-{index}", f"tmp/reject-{index}.py", "alpha"), 1.0 - index / 1000)
            for index in range(80)
        ]
        accepted = DensePoint(make_chunk("accepted", "src/app.py", "alpha"), 0.1)
        store = DenseStore({"code": [*rejected, accepted]})

        execution = dense_candidates(
            store=store,
            query_vector=[1.0],
            search_scopes=["code"],
            filter_plans_by_scope={"code": _plan(best_effort=True)},
            filters=SearchFilters(include_paths=["src/**"]),
            target_results=1,
            max_results_per_file=None,
            point_to_chunk=lambda point: point.chunk,
            matches_filters=lambda chunk, filters: chunk.relative_path.startswith("src/"),
        )

        self.assertEqual([call[1] for call in store.calls], [80, 160])
        self.assertEqual(execution.candidate_limit, 160)
        self.assertEqual(execution.candidates_scanned, 81)
        self.assertEqual(execution.filtered_candidates, 1)
        self.assertEqual(execution.kept_candidates, 1)
        self.assertEqual(list(execution.candidates), ["accepted"])
        self.assertFalse(execution.dense_best_effort)

    def test_dense_candidates_marks_best_effort_when_cap_still_under_returns(self) -> None:
        store = DenseStore(
            {
                "code": [
                    DensePoint(make_chunk(f"a-{index}", "pkg/a.py", f"alpha {index}"), 1.0 - index / 1000)
                    for index in range(400)
                ]
            }
        )

        execution = dense_candidates(
            store=store,
            query_vector=[1.0],
            search_scopes=["code"],
            filter_plans_by_scope={"code": _plan(best_effort=True)},
            filters=SearchFilters(),
            target_results=2,
            max_results_per_file=1,
            point_to_chunk=lambda point: point.chunk,
            matches_filters=lambda chunk, filters: True,
        )

        self.assertEqual(execution.candidate_limit, 320)
        self.assertEqual(execution.kept_candidates, 1)
        self.assertTrue(execution.dense_best_effort)


if __name__ == "__main__":
    unittest.main()
