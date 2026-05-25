from __future__ import annotations

import unittest

from services.repo_semantic.retrieval.fusion import weighted_rrf
from services.repo_semantic.retrieval.snippets import (
    line_matches,
    make_snippet,
    matched_terms,
    query_centered_snippet,
)

from tests.helpers.chunks import make_chunk


class RetrievalHelperTests(unittest.TestCase):
    def test_weighted_rrf_preserves_branch_scores_and_ranks(self) -> None:
        dense_first = make_chunk("dense-first", "src/a.py", "alpha")
        shared = make_chunk("shared", "src/shared.py", "alpha beta")
        sparse_first = make_chunk("sparse-first", "src/b.py", "beta")

        rows = weighted_rrf(
            {
                "dense": [(dense_first, 0.9), (shared, 0.8)],
                "sparse": [(sparse_first, 10.0), (shared, 9.0)],
            },
            {"dense": 1.0, "sparse": 1.2},
            rrf_k=60,
        )

        by_id = {chunk.point_id: (score, dense_score, sparse_score, ranks) for chunk, score, dense_score, sparse_score, ranks in rows}
        self.assertEqual([chunk.point_id for chunk, *_ in rows], ["shared", "sparse-first", "dense-first"])
        self.assertAlmostEqual(by_id["shared"][0], (1.0 / 62) + (1.2 / 62))
        self.assertEqual(by_id["shared"][1], 0.8)
        self.assertEqual(by_id["shared"][2], 9.0)
        self.assertEqual(by_id["shared"][3], {"dense": 2, "sparse": 2})
        self.assertGreater(by_id["sparse-first"][0], by_id["dense-first"][0])

    def test_weighted_rrf_tie_prefers_better_best_rank(self) -> None:
        rank_one = make_chunk("rank-one", "src/a.py", "alpha")
        rank_two = make_chunk("rank-two", "src/b.py", "beta")

        rows = weighted_rrf(
            {
                "dense": [(rank_one, 1.0)],
                "sparse": [(make_chunk("unused", "src/unused.py", "unused"), 1.0), (rank_two, 1.0)],
            },
            {"dense": 1.0, "sparse": 12 / 11},
            rrf_k=10,
        )

        tied = [row for row in rows if row[0].point_id in {"rank-one", "rank-two"}]
        self.assertAlmostEqual(tied[0][1], tied[1][1])
        self.assertEqual([row[0].point_id for row in tied], ["rank-one", "rank-two"])

    def test_make_snippet_collapses_whitespace_and_truncates(self) -> None:
        self.assertEqual(make_snippet(" alpha\n\n beta\tgamma "), "alpha beta gamma")
        self.assertEqual(make_snippet("alpha beta gamma", limit=10), "alpha b...")

    def test_matched_terms_preserves_display_surface_for_exact_token(self) -> None:
        terms = matched_terms(
            ["DATABASE_URL", "database_url", "missing"],
            "DATABASE_URL is configured here",
        )

        self.assertEqual(terms, ["DATABASE_URL"])

    def test_line_matches_are_bounded_and_use_sparse_terms(self) -> None:
        chunk = make_chunk(
            "settings",
            "src/settings.py",
            "first line\nDATABASE_URL settings\nthird DATABASE_URL",
            start_line=10,
        )

        matches = line_matches(chunk, ["DATABASE_URL"], max_matches=1)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].line, 11)
        self.assertEqual(matches[0].matched_terms, ["DATABASE_URL"])

    def test_query_centered_snippet_returns_line_window(self) -> None:
        chunk = make_chunk(
            "settings",
            "src/settings.py",
            "preamble\nDATABASE_URL settings\ntrailing",
            start_line=20,
        )

        snippet, start_line, end_line = query_centered_snippet(chunk, ["DATABASE_URL"])

        self.assertIn("DATABASE_URL settings", snippet)
        self.assertEqual(start_line, 20)
        self.assertEqual(end_line, 22)

    def test_query_centered_snippet_fallback_uses_chunk_start_line_window(self) -> None:
        chunk = make_chunk(
            "settings",
            "src/settings.py",
            "first\nsecond\nthird\nfourth",
            start_line=30,
        )

        snippet, start_line, end_line = query_centered_snippet(chunk, [])

        self.assertEqual(snippet, "first\nsecond\nthird")
        self.assertEqual(start_line, 30)
        self.assertEqual(end_line, 32)


if __name__ == "__main__":
    unittest.main()
