from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory

from services.repo_semantic.models import ChunkRecord, SearchDiagnostics, SearchResult
from services.repo_semantic.sparse import build_sparse_manifest

from tests.helpers.chunks import make_chunk as _chunk
from tests.helpers.service_factory import build_search_service

from services.repo_semantic.search_service import SearchService
import services.repo_semantic.search_service as search_service_module


class RepoContextSearchContractTests(unittest.TestCase):
    def _build_service(
        self,
        tmp: str,
        chunks: list[tuple[ChunkRecord, float]],
        *,
        sparse_enabled: bool = False,
        sparse_scores: dict[str, float] | None = None,
        fail_on_scroll: bool = False,
    ) -> SearchService:
        return build_search_service(
            tmp,
            chunks,
            sparse_enabled=sparse_enabled,
            sparse_scores=sparse_scores,
            fail_on_scroll=fail_on_scroll,
        )

    def test_repo_context_search_returns_envelope_and_original_query_first(self) -> None:
        with TemporaryDirectory() as tmp:
            service = self._build_service(
                tmp,
                [(_chunk("a", "src/app.py", "connect flow"), 0.9)],
            )

            response = service.repo_context_search(
                "connect",
                subqueries=["connect docs", "connect", "", "connect docs"],
                top_k=1,
            )
            payload = response.model_dump()

            self.assertEqual(payload["contract_version"], "repo_context_search.v1")
            self.assertEqual(payload["route_requested"], "auto")
            self.assertEqual(payload["route_used"], "hybrid")
            self.assertEqual(payload["queries_used"][0]["query"], "connect")
            self.assertEqual(payload["queries_used"][0]["role"], "original")
            self.assertEqual([item["query"] for item in payload["queries_used"]], ["connect", "connect docs"])
            self.assertIn("multi_query_subquery_dropped", {warning["code"] for warning in payload["warnings"]})
            self.assertEqual(payload["results"][0]["final_rank"], 1)
            self.assertIn("connect", payload["results"][0]["origin_queries"])

    def test_repo_context_search_multi_query_dedupes_and_tracks_origin_queries(self) -> None:
        with TemporaryDirectory() as tmp:
            service = self._build_service(
                tmp,
                [
                    (_chunk("a", "src/app.py", "connect flow runtime"), 0.9),
                    (
                        _chunk(
                            "b",
                            "docs/connect.md",
                            "connect flow docs",
                            scope="docs",
                            language="markdown",
                            chunk_type="markdown_section",
                        ),
                        0.8,
                    ),
                ],
            )

            response = service.repo_context_search("connect", subqueries=["runtime"], top_k=5)
            payload = response.model_dump()

            self.assertEqual(len({result["chunk_id"] for result in payload["results"]}), len(payload["results"]))
            self.assertIn("connect", payload["results"][0]["origin_queries"])
            self.assertIn("runtime", payload["results"][0]["origin_queries"])
            self.assertEqual(payload["diagnostics"]["fusion_method"], "multi_query_weighted_rrf")
            self.assertEqual(payload["diagnostics"]["rrf_k"], 60)

    def test_repo_context_search_distinguishes_legacy_lexical_origin(self) -> None:
        with TemporaryDirectory() as tmp:
            service = self._build_service(
                tmp,
                [(_chunk("needle", "src/feature.py", "needle owner"), 0.01)],
            )

            payload = service.repo_context_search("needle", top_k=1).model_dump()

            self.assertIn("legacy_lexical", payload["results"][0]["origin_branches"])
            self.assertNotIn("sparse", payload["results"][0]["origin_branches"])
            self.assertTrue(payload["results"][0]["verification"]["required"])
            self.assertEqual(payload["results"][0]["verification"]["reason"], "degraded_search")
            self.assertIn("rebuild_index", {action["code"] for action in payload["recommended_next_actions"]})

    def test_repo_context_search_tracks_qdrant_sparse_origin(self) -> None:
        with TemporaryDirectory() as tmp:
            exact_chunk = _chunk("exact", "deploy/compose.yml", "DATABASE_URL is configured here")
            service = self._build_service(
                tmp,
                [(exact_chunk, 0.01)],
                sparse_enabled=True,
                sparse_scores={"exact": 10.0},
                fail_on_scroll=True,
            )
            service._sparse_manifests.save(
                build_sparse_manifest(
                    scope="code",
                    chunks=[exact_chunk],
                    schema_version=service._settings.SEMANTIC_MCP_INDEX_SCHEMA_VERSION,
                )
            )

            payload = service.repo_context_search("DATABASE_URL", top_k=1).model_dump()

            self.assertIn("sparse", payload["results"][0]["origin_branches"])
            self.assertIn("dense", payload["results"][0]["origin_branches"])
            self.assertEqual(payload["diagnostics"]["sparse_available"], True)
            self.assertIn("run_local_rg", {action["code"] for action in payload["recommended_next_actions"]})
            rg_action = next(action for action in payload["recommended_next_actions"] if action["code"] == "run_local_rg")
            self.assertEqual(rg_action["cwd_hint"], "repo_root")
            self.assertEqual(rg_action["authority"], "local_rg")

    def test_repo_context_search_does_not_recommend_rebuild_for_stale_sparse_stats(self) -> None:
        with TemporaryDirectory() as tmp:
            exact_chunk = _chunk("exact", "deploy/compose.yml", "DATABASE_URL is configured here")
            service = self._build_service(
                tmp,
                [(exact_chunk, 0.01)],
                sparse_enabled=True,
                sparse_scores={"exact": 10.0},
                fail_on_scroll=True,
            )
            manifest = build_sparse_manifest(
                scope="code",
                chunks=[exact_chunk],
                schema_version=service._settings.SEMANTIC_MCP_INDEX_SCHEMA_VERSION,
            )
            service._sparse_manifests.save(manifest.model_copy(update={"sparse_stats_stale": True}))

            payload = service.repo_context_search("DATABASE_URL", top_k=1).model_dump()

            self.assertIn("sparse_stats_stale", {warning["code"] for warning in payload["warnings"]})
            self.assertNotIn("rebuild_index", {action["code"] for action in payload["recommended_next_actions"]})

    def test_repo_context_search_exact_handoff_returns_only_actions(self) -> None:
        with TemporaryDirectory() as tmp:
            service = self._build_service(
                tmp,
                [(_chunk("a", "src/app.py", "DATABASE_URL settings"), 0.9)],
            )

            payload = service.repo_context_search(
                "DATABASE_URL",
                route="exact_handoff",
                include_diagnostics=False,
            ).model_dump()

            self.assertEqual(payload["route_used"], "exact_handoff")
            self.assertEqual(payload["results"], [])
            self.assertIsNone(payload["diagnostics"])
            rg_action = next(action for action in payload["recommended_next_actions"] if action["code"] == "run_local_rg")
            self.assertIn("-e", rg_action["argv_hint"])
            self.assertEqual(rg_action["cwd_hint"], "repo_root")
            self.assertNotIn("rebuild_index", {action["code"] for action in payload["recommended_next_actions"]})

    def test_repo_context_search_file_groups_and_per_file_cap_apply_after_merge(self) -> None:
        with TemporaryDirectory() as tmp:
            service = self._build_service(
                tmp,
                [
                    (_chunk("a1", "pkg/a.py", "alpha first"), 0.99),
                    (_chunk("a2", "pkg/a.py", "alpha second"), 0.98),
                    (_chunk("b1", "pkg/b.py", "alpha third"), 0.5),
                ],
            )

            payload = service.repo_context_search(
                "alpha",
                subqueries=["second"],
                top_k=10,
                max_results_per_file=1,
            ).model_dump()

            self.assertEqual([result["chunk_id"] for result in payload["results"]], ["a1", "b1"])
            self.assertEqual([group["relative_path"] for group in payload["file_groups"]], ["pkg/a.py", "pkg/b.py"])
            self.assertLessEqual(len(payload["file_groups"][0]["chunk_ids"]), 10)
            self.assertIn("read_file_range", {action["code"] for action in payload["recommended_next_actions"]})

    def test_repo_context_search_graph_expand_warns_without_lifecycle_action(self) -> None:
        with TemporaryDirectory() as tmp:
            service = self._build_service(
                tmp,
                [(_chunk("a", "src/app.py", "connect flow"), 0.9)],
            )

            payload = service.repo_context_search("connect", graph_mode="expand", top_k=1).model_dump()

            self.assertEqual(payload["graph_mode_requested"], "expand")
            self.assertEqual(payload["graph_mode_effective"], "off")
            self.assertIn("graph_branch_unavailable", {warning["code"] for warning in payload["warnings"]})
            self.assertNotIn("build_graph", {action["code"] for action in payload["recommended_next_actions"]})

    def test_repo_context_search_uncovered_terms_follow_golden_examples(self) -> None:
        with TemporaryDirectory() as tmp:
            database_chunk = _chunk("db", "src/settings.py", "DATABASE_URL settings")
            symbol_chunk = _chunk("symbol", "src/users.py", "UserRepository.findByEmail handles auth")
            service = self._build_service(
                tmp,
                [
                    (database_chunk, 0.9),
                    (symbol_chunk, 0.8),
                ],
            )

            database_payload = service.repo_context_search("where is DATABASE_URL used", top_k=1).model_dump()
            symbol_payload = service.repo_context_search(
                "UserRepository.findByEmail permission denied",
                top_k=1,
            ).model_dump()

            self.assertEqual(database_payload["diagnostics"]["uncovered_terms"], [])
            self.assertEqual(symbol_payload["diagnostics"]["uncovered_terms"], ["permission", "denied"])

    def test_repo_context_search_duplicate_keeps_highest_ranked_occurrence_snippet(self) -> None:
        with TemporaryDirectory() as tmp:
            service = self._build_service(
                tmp,
                [(_chunk("seed", "src/seed.py", "seed"), 0.1)],
            )
            original_other = SearchResult(
                chunk_id="other",
                repo_root="repo",
                scope="code",
                relative_path="src/other.py",
                language="python",
                chunk_type="python_function",
                start_line=1,
                end_line=1,
                line_range="1-1",
                snippet="other",
                score=0.9,
                final_score=0.9,
                dense_score=0.9,
            )
            original_duplicate = SearchResult(
                chunk_id="dup",
                repo_root="repo",
                scope="code",
                relative_path="src/dup.py",
                language="python",
                chunk_type="python_function",
                start_line=10,
                end_line=10,
                line_range="10-10",
                snippet="original lower ranked snippet",
                score=0.5,
                final_score=0.5,
                dense_score=0.5,
            )
            subquery_duplicate = SearchResult(
                chunk_id="dup",
                repo_root="repo",
                scope="code",
                relative_path="src/dup.py",
                language="python",
                chunk_type="python_function",
                start_line=20,
                end_line=20,
                line_range="20-20",
                snippet="subquery highest ranked snippet",
                score=0.95,
                final_score=0.95,
                dense_score=0.95,
            )

            def fake_context_execution(**kwargs):
                query_text = kwargs["query"]
                if query_text == "original":
                    return search_service_module._SearchExecution(
                        results=[original_other, original_duplicate],
                        diagnostics=SearchDiagnostics(final_results=2),
                        warnings=[],
                        branch_ranks_by_chunk={"other": {"dense": 1}, "dup": {"dense": 2}},
                    )
                return search_service_module._SearchExecution(
                    results=[subquery_duplicate],
                    diagnostics=SearchDiagnostics(final_results=1),
                    warnings=[],
                    branch_ranks_by_chunk={"dup": {"dense": 1}},
                )

            service._context_execution = fake_context_execution  # type: ignore[method-assign]

            payload = service.repo_context_search("original", subqueries=["subquery"], top_k=2).model_dump()
            duplicate = next(result for result in payload["results"] if result["chunk_id"] == "dup")

            self.assertEqual(duplicate["snippet"], "subquery highest ranked snippet")
            self.assertEqual(duplicate["line_range"], "20-20")


if __name__ == "__main__":
    unittest.main()
