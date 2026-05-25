from __future__ import annotations

import unittest

from services.repo_semantic.models import ChunkRecord
from services.repo_semantic.retrieval import filters


def _chunk(relative_path: str = "services/api/connect_flow.py") -> ChunkRecord:
    return ChunkRecord(
        point_id="chunk-1",
        scope="code",
        relative_path=relative_path,
        language="python",
        chunk_type="python_function",
        text="connect flow",
        start_line=1,
        end_line=3,
        content_hash="hash",
        source_mtime=0.0,
        domain_tags=["services", "api"],
    )


class RetrievalFiltersTests(unittest.TestCase):
    def test_build_filters_normalizes_path_and_extension_inputs(self) -> None:
        search_filters = filters.build_filters(
            path_prefix="services/api/",
            include_paths=["services/**/*.py"],
            file_extensions=["PY", ".md"],
            languages=["python"],
        )

        self.assertEqual(search_filters.path_prefix, "services/api")
        self.assertEqual(search_filters.include_paths, ["services/**/*.py"])
        self.assertEqual(search_filters.file_extensions, [".py", ".md"])

    def test_matches_filters_uses_boundary_prefix_and_globs(self) -> None:
        search_filters = filters.build_filters(
            path_prefix="services/api",
            include_paths=["services/**/*.py"],
            exclude_paths=["**/generated/**"],
            domain_tags=["api"],
        )

        self.assertTrue(filters.matches_filters(_chunk(), search_filters))
        self.assertFalse(filters.matches_filters(_chunk("services/apiv2/connect_flow.py"), search_filters))
        self.assertFalse(filters.matches_filters(_chunk("services/api/generated/client.py"), search_filters))

    def test_filter_plan_reports_pushdown_and_best_effort(self) -> None:
        search_filters = filters.build_filters(
            path_prefix="services/api",
            include_paths=["services/**/*.py"],
            languages=["python"],
        )

        plan = filters.filter_plan_for_scope(
            scope="code",
            collection_name="code_collection",
            payload_fields={"path_prefixes", "language"},
            filters=search_filters,
        )

        self.assertIsNotNone(plan.qdrant_filter)
        self.assertEqual(plan.pushed_families, ["path_prefix", "languages"])
        self.assertEqual(plan.post_filter_families, ["include_paths"])
        self.assertTrue(plan.best_effort)


if __name__ == "__main__":
    unittest.main()
