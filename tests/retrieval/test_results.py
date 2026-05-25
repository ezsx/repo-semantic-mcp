from __future__ import annotations

from types import SimpleNamespace
import unittest

from services.repo_semantic.models import ChunkRecord
from services.repo_semantic.retrieval.results import (
    build_read_chunk_result,
    build_search_result,
    query_terms_for_matching,
)
from services.repo_semantic.storage.qdrant.point_projection import point_to_chunk


def _chunk() -> ChunkRecord:
    return ChunkRecord(
        point_id="src/app.py:1-4",
        scope="code",
        relative_path="src/app.py",
        language="python",
        chunk_type="python_function",
        text="def connect_flow():\n    return DATABASE_URL\n",
        start_line=1,
        end_line=2,
        content_hash="hash",
        source_mtime=0.0,
        symbol_path="app.connect_flow",
        domain_tags=["src"],
    )


class RetrievalResultsTests(unittest.TestCase):
    def test_point_projection_builds_chunk_record(self) -> None:
        point = SimpleNamespace(
            id="fallback-id",
            payload={
                "chunk_id": "chunk-id",
                "scope": "code",
                "relative_path": "src/app.py",
                "language": "python",
                "chunk_type": "python_function",
                "text": "text",
                "start_line": "1",
                "end_line": "2",
                "content_hash": "hash",
                "source_mtime": "0.5",
                "domain_tags": ["src"],
                "is_generated": False,
            },
        )

        chunk = point_to_chunk(point)

        self.assertEqual(chunk.point_id, "chunk-id")
        self.assertEqual(chunk.start_line, 1)
        self.assertEqual(chunk.source_mtime, 0.5)
        self.assertEqual(chunk.domain_tags, ["src"])

    def test_build_search_result_shapes_snippet_and_explanations(self) -> None:
        result = build_search_result(
            chunk=_chunk(),
            repo_root="C:/repo",
            score=0.7,
            dense_score=0.6,
            lexical_score=0.4,
            query_tokens=["DATABASE_URL"],
            snippet_mode="query_centered",
            include_explanations=True,
            match_type="hybrid",
        )

        self.assertEqual(result.repo_root, "C:/repo")
        self.assertEqual(result.file_extension, ".py")
        self.assertIn("DATABASE_URL", result.matched_terms)
        self.assertEqual(result.why_matched, ["dense_vector_similarity", "lexical_term_overlap"])

    def test_query_terms_for_matching_preserves_exact_surface_terms(self) -> None:
        terms = query_terms_for_matching("DATABASE_URL connectFlow", tokenize=lambda text: [text])

        self.assertIn("DATABASE_URL", terms)
        self.assertIn("database_url", terms)
        self.assertIn("connectFlow", terms)

    def test_build_read_chunk_result_projects_full_text(self) -> None:
        result = build_read_chunk_result(_chunk())

        self.assertEqual(result.chunk_id, "src/app.py:1-4")
        self.assertEqual(result.text, _chunk().text)


if __name__ == "__main__":
    unittest.main()
