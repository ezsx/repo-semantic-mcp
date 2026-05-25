from __future__ import annotations

import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from services.repo_semantic.config import normalize_repo_identity
from services.repo_semantic.models import ChunkRecord
from services.repo_semantic.sparse import build_sparse_manifest

from tests.helpers.chunks import make_chunk as _chunk
from tests.helpers.mcp import install_mcp_stub
from tests.helpers.service_factory import build_search_service

from services.repo_semantic.search_service import SearchService


class SearchContractTests(unittest.TestCase):
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

    def test_path_prefix_is_boundary_aware_and_exclude_wins(self) -> None:
        with TemporaryDirectory() as tmp:
            service = self._build_service(
                tmp,
                [
                    (_chunk("a", "src/app.py", "connect flow"), 0.9),
                    (_chunk("b", "src/private/secret.py", "connect secret"), 0.8),
                    (_chunk("c", "src2/app.py", "connect wrong"), 0.99),
                ],
            )

            results = service.semantic_search(
                "connect",
                top_k=10,
                path_prefix="src",
                exclude_paths=["src/private/**"],
            )

            self.assertEqual([result.relative_path for result in results], ["src/app.py"])

    def test_filters_and_max_results_per_file_are_applied_after_ranking(self) -> None:
        with TemporaryDirectory() as tmp:
            service = self._build_service(
                tmp,
                [
                    (_chunk("a1", "pkg/a.py", "alpha first", language="python"), 0.99),
                    (_chunk("a2", "pkg/a.py", "alpha second", language="python"), 0.98),
                    (_chunk("b1", "pkg/b.py", "alpha third", language="python"), 0.5),
                    (_chunk("c1", "pkg/c.md", "alpha docs", language="markdown"), 0.97),
                ],
            )

            results = service.semantic_search(
                "alpha",
                top_k=10,
                include_paths=["pkg/**"],
                file_extensions=["py"],
                languages=["python"],
                max_results_per_file=1,
            )

            self.assertEqual([result.chunk_id for result in results], ["a1", "b1"])
            self.assertEqual(results[0].file_extension, ".py")
            self.assertEqual(results[0].line_range, "1-1")
            self.assertEqual(results[0].repo_root, normalize_repo_identity(str(Path(tmp) / "repo")))

    def test_hybrid_query_centered_snippet_uses_full_chunk_text(self) -> None:
        with TemporaryDirectory() as tmp:
            long_preamble = "preamble " * 80
            text = f"{long_preamble}\nneedle owner lives here\ntrailing context"
            service = self._build_service(
                tmp,
                [(_chunk("needle", "src/feature.py", text, start_line=20), 0.9)],
            )

            result = service.hybrid_search(
                "needle",
                top_k=1,
                snippet_mode="query_centered",
            )[0]

            self.assertIn("needle owner lives here", result.snippet)
            self.assertEqual(result.snippet_start_line, 21)
            self.assertTrue(any(match.line == 21 for match in result.line_matches))
            self.assertIn("lexical_term_overlap", result.why_matched)

    def test_hybrid_uses_qdrant_sparse_branch_and_rrf_without_scrolling(self) -> None:
        with TemporaryDirectory() as tmp:
            exact_chunk = _chunk("exact", "deploy/compose.yml", "DATABASE_URL is configured here")
            dense_chunk = _chunk("dense", "docs/overview.md", "database connection overview")
            service = self._build_service(
                tmp,
                [
                    (dense_chunk, 0.99),
                    (exact_chunk, 0.01),
                ],
                sparse_enabled=True,
                sparse_scores={"exact": 10.0},
                fail_on_scroll=True,
            )
            manifest = build_sparse_manifest(
                scope="code",
                chunks=[exact_chunk, dense_chunk],
                schema_version=service._settings.SEMANTIC_MCP_INDEX_SCHEMA_VERSION,
            )
            service._sparse_manifests.save(manifest)

            response = service.search_v2("DATABASE_URL", mode="hybrid", top_k=1)
            payload = response.model_dump()

            self.assertEqual(payload["results"][0]["chunk_id"], "exact")
            self.assertIn("DATABASE_URL", payload["results"][0]["matched_terms"])
            self.assertEqual(payload["diagnostics"]["fusion_method"], "weighted_rrf")
            self.assertEqual(payload["diagnostics"]["retrieval_backend"], "qdrant_dense_sparse_rrf")
            self.assertTrue(payload["diagnostics"]["sparse_available"])
            self.assertEqual(
                payload["diagnostics"]["sparse_manifest_hash"],
                manifest.manifest_content_hash,
            )
            self.assertEqual(
                payload["diagnostics"]["sparse_vocabulary_hash"],
                manifest.vocabulary_hash,
            )
            self.assertEqual(
                payload["diagnostics"]["sparse_corpus_stats_hash"],
                manifest.corpus_stats_hash,
            )
            self.assertFalse(payload["diagnostics"]["lexical_cache_used"])
            self.assertEqual(payload["diagnostics"]["branch_weights"], {"dense": 0.8, "sparse": 1.4})
            self.assertTrue(payload["diagnostics"]["exact_anchor_analysis"]["exact_anchor_heavy"])
            self.assertEqual(
                payload["diagnostics"]["exact_anchor_analysis"]["argv_hints"][0],
                [
                    "rg",
                    "--fixed-strings",
                    "--line-number",
                    "--max-count",
                    "20",
                    "-e",
                    "DATABASE_URL",
                    ".",
                ],
            )
            self.assertNotIn(
                "legacy_lexical_fallback_used",
                {warning["code"] for warning in payload["warnings"]},
            )

    def test_search_v2_pushes_supported_payload_filters_without_best_effort_warning(self) -> None:
        with TemporaryDirectory() as tmp:
            chunk = _chunk("api", "services/user/api/connect_flow.py", "connect flow")
            service = self._build_service(
                tmp,
                [(chunk, 0.9)],
                sparse_enabled=True,
                fail_on_scroll=True,
            )
            manifest = build_sparse_manifest(
                scope="code",
                chunks=[chunk],
                schema_version=service._settings.SEMANTIC_MCP_INDEX_SCHEMA_VERSION,
            )
            service._sparse_manifests.save(manifest)

            response = service.search_v2(
                "connect",
                mode="hybrid",
                top_k=1,
                path_prefix="services/user",
                file_extensions=[".py"],
                languages=["python"],
                chunk_types=["python_function"],
                domain_tags=[],
            )
            payload = response.model_dump()
            warning_codes = {warning["code"] for warning in payload["warnings"]}

            self.assertEqual(payload["results"][0]["chunk_id"], "api")
            self.assertTrue(payload["diagnostics"]["filter_pushdown_supported"])
            self.assertIn("path_prefix", payload["diagnostics"]["filter_pushed_families"])
            self.assertIn("file_extensions", payload["diagnostics"]["filter_pushed_families"])
            self.assertFalse(payload["diagnostics"]["filter_best_effort"])
            self.assertNotIn("qdrant_filter_post_filter_best_effort", warning_codes)
            self.assertTrue(service._store.dense_query_filters[0] is not None)

    def test_search_v2_reports_post_filter_best_effort_for_globs(self) -> None:
        with TemporaryDirectory() as tmp:
            service = self._build_service(
                tmp,
                [(_chunk("a", "src/app.py", "connect flow"), 0.9)],
                sparse_enabled=True,
            )

            response = service.search_v2(
                "connect",
                mode="semantic",
                top_k=1,
                include_paths=["src/**"],
            )
            payload = response.model_dump()

            self.assertTrue(payload["diagnostics"]["filter_best_effort"])
            self.assertEqual(payload["diagnostics"]["filter_post_families"], ["include_paths"])
            self.assertIn(
                "qdrant_filter_post_filter_best_effort",
                {warning["code"] for warning in payload["warnings"]},
            )

    def test_hybrid_reports_legacy_fallback_when_sparse_branch_missing(self) -> None:
        with TemporaryDirectory() as tmp:
            service = self._build_service(
                tmp,
                [(_chunk("needle", "src/feature.py", "needle owner"), 0.01)],
            )

            response = service.search_v2("needle", mode="hybrid", top_k=1)
            payload = response.model_dump()

            self.assertEqual(payload["results"][0]["chunk_id"], "needle")
            self.assertEqual(payload["diagnostics"]["fusion_method"], "weighted_rrf")
            self.assertEqual(payload["diagnostics"]["retrieval_backend"], "legacy_dense_local_bm25")
            self.assertFalse(payload["diagnostics"]["sparse_available"])
            self.assertTrue(payload["diagnostics"]["lexical_cache_used"])
            self.assertIn(
                "legacy_lexical_fallback_used",
                {warning["code"] for warning in payload["warnings"]},
            )

    def test_hybrid_reports_sparse_analyzer_mismatch_code(self) -> None:
        with TemporaryDirectory() as tmp:
            chunk = _chunk("settings", "src/settings.py", "DATABASE_URL settings")
            service = self._build_service(
                tmp,
                [(chunk, 0.5)],
                sparse_enabled=True,
            )
            manifest = build_sparse_manifest(
                scope="code",
                chunks=[chunk],
                schema_version=service._settings.SEMANTIC_MCP_INDEX_SCHEMA_VERSION,
            )
            service._sparse_manifests.save(
                manifest.model_copy(update={"lexical_analyzer_version": "code_lexical_v1"})
            )

            response = service.search_v2("DATABASE_URL", mode="hybrid", top_k=1)
            payload = response.model_dump()
            warning_codes = {warning["code"] for warning in payload["warnings"]}

            self.assertFalse(payload["diagnostics"]["sparse_available"])
            self.assertIn(
                "sparse_analyzer_version_mismatch",
                payload["diagnostics"]["sparse_unavailable_codes"],
            )
            self.assertIn("sparse_analyzer_version_mismatch", warning_codes)
            self.assertIn(
                "sparse_analyzer_version_mismatch",
                payload["diagnostics"]["branch_diagnostics"]["sparse"]["warning_codes"],
            )

    def test_hybrid_legacy_fallback_does_not_rrf_zero_score_lexical_rows(self) -> None:
        with TemporaryDirectory() as tmp:
            lower = _chunk("lower", "src/lower.py", "alpha only")
            higher = _chunk("higher", "src/higher.py", "beta only")
            service = self._build_service(
                tmp,
                [
                    (lower, 0.8),
                    (higher, 0.99),
                ],
            )

            response = service.search_v2("missing", mode="hybrid", top_k=1)
            payload = response.model_dump()

            self.assertEqual(payload["results"][0]["chunk_id"], "higher")

    def test_hybrid_scope_all_warns_and_falls_back_when_one_non_empty_scope_lacks_sparse(self) -> None:
        with TemporaryDirectory() as tmp:
            code_chunk = _chunk("code", "src/app.py", "DATABASE_URL runtime")
            docs_chunk = _chunk(
                "docs",
                "docs/config.md",
                "DATABASE_URL documentation",
                scope="docs",
                language="markdown",
                chunk_type="markdown_section",
            )
            service = self._build_service(
                tmp,
                [
                    (code_chunk, 0.5),
                    (docs_chunk, 0.4),
                ],
                sparse_enabled=True,
                sparse_scores={"code": 10.0},
            )
            manifest = build_sparse_manifest(
                scope="code",
                chunks=[code_chunk],
                schema_version=service._settings.SEMANTIC_MCP_INDEX_SCHEMA_VERSION,
            )
            service._sparse_manifests.save(manifest)

            response = service.search_v2("DATABASE_URL", mode="hybrid", scope="all", top_k=2)
            payload = response.model_dump()

            self.assertFalse(payload["diagnostics"]["sparse_available"])
            self.assertEqual(payload["diagnostics"]["retrieval_backend"], "legacy_dense_local_bm25")
            warning_codes = {warning["code"] for warning in payload["warnings"]}
            self.assertIn("sparse_scope_partial_unavailable", warning_codes)
            self.assertIn("legacy_lexical_fallback_used", warning_codes)

    def test_hybrid_does_not_scroll_when_sparse_index_is_ready_but_query_has_no_vocab_hits(self) -> None:
        with TemporaryDirectory() as tmp:
            chunk = _chunk("dense", "src/feature.py", "known indexed terms")
            service = self._build_service(
                tmp,
                [(chunk, 0.9)],
                sparse_enabled=True,
                fail_on_scroll=True,
            )
            manifest = build_sparse_manifest(
                scope="code",
                chunks=[chunk],
                schema_version=service._settings.SEMANTIC_MCP_INDEX_SCHEMA_VERSION,
            )
            service._sparse_manifests.save(manifest)

            response = service.search_v2("totally_unseen_anchor", mode="hybrid", top_k=1)
            payload = response.model_dump()

            self.assertEqual(payload["results"][0]["chunk_id"], "dense")
            self.assertTrue(payload["diagnostics"]["sparse_available"])
            self.assertFalse(payload["diagnostics"]["lexical_cache_used"])

    def test_search_v2_returns_envelope_warnings_and_diagnostics_for_empty_results(self) -> None:
        with TemporaryDirectory() as tmp:
            service = self._build_service(
                tmp,
                [(_chunk("a", "src/app.py", "connect flow"), 0.9)],
            )

            response = service.search_v2(
                "connect",
                mode="semantic",
                top_k=10,
                include_paths=["docs/**"],
            )
            payload = response.model_dump()

            self.assertEqual(payload["contract_version"], "search.v2")
            self.assertEqual(payload["mode"], "semantic")
            self.assertEqual(payload["results"], [])
            self.assertIn("empty_result_best_effort", {warning["code"] for warning in payload["warnings"]})
            self.assertEqual(payload["diagnostics"]["final_results"], 0)
            self.assertTrue(math.isfinite(payload["diagnostics"]["candidate_limit"]))

    def test_search_v2_warns_when_per_file_cap_under_returns_at_dense_cap(self) -> None:
        with TemporaryDirectory() as tmp:
            chunks = [
                (
                    _chunk(
                        f"item-{index}",
                        "pkg/large.py",
                        f"alpha candidate {index}",
                    ),
                    1.0 - index / 1000,
                )
                for index in range(400)
            ]
            service = self._build_service(tmp, chunks)

            response = service.search_v2(
                "alpha",
                mode="semantic",
                top_k=5,
                include_paths=["pkg/**"],
                max_results_per_file=1,
            )
            payload = response.model_dump()

            self.assertEqual(len(payload["results"]), 1)
            self.assertTrue(payload["diagnostics"]["dense_best_effort"])
            self.assertIn(
                "filtered_dense_search_best_effort",
                {warning["code"] for warning in payload["warnings"]},
            )

    def test_path_filters_reject_absolute_and_parent_paths(self) -> None:
        with TemporaryDirectory() as tmp:
            service = self._build_service(
                tmp,
                [(_chunk("a", "src/app.py", "connect flow"), 0.9)],
            )

            with self.assertRaises(ValueError):
                service.semantic_search("connect", include_paths=["C:/tmp/repo/src/**"])
            with self.assertRaises(ValueError):
                service.semantic_search("connect", exclude_paths=["../outside/**"])

    def test_line_matches_use_token_matches_not_substrings(self) -> None:
        with TemporaryDirectory() as tmp:
            service = self._build_service(
                tmp,
                [(_chunk("a", "src/app.py", "candidate valid", start_line=10), 0.9)],
            )

            result = service.hybrid_search("id", top_k=1, snippet_mode="query_centered")[0]

            self.assertEqual(result.matched_terms, [])
            self.assertEqual(result.line_matches, [])
            self.assertIn("query_centered_fallback_to_chunk_start", result.why_matched)

    def test_find_similar_can_return_public_cap_after_removing_seed(self) -> None:
        with TemporaryDirectory() as tmp:
            chunks = [(_chunk("seed", "src/seed.py", "alpha seed"), 1.0)]
            chunks.extend(
                (
                    _chunk(f"neighbor-{index}", f"src/n{index}.py", f"alpha neighbor {index}"),
                    0.99 - index / 1000,
                )
                for index in range(100)
            )
            service = self._build_service(tmp, chunks)

            results = service.find_similar_chunk("code", "seed", top_k=100)

            self.assertEqual(len(results), 100)
            self.assertNotIn("seed", {result.chunk_id for result in results})

    def test_legacy_search_payload_shape_is_preserved(self) -> None:
        handle = install_mcp_stub()
        try:
            from services.repo_semantic.mcp_server import _search_payload
        finally:
            handle.restore()

        rows = [{"chunk_id": "a"}]
        self.assertIs(_search_payload(rows), rows)
        self.assertEqual(_search_payload([]), "[]")


if __name__ == "__main__":
    unittest.main()
