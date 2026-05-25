from __future__ import annotations

import unittest

from services.repo_semantic.indexing.chunk_pipeline import normalize_chunks
from services.repo_semantic.models import ChunkRecord


def _chunk(text: str) -> ChunkRecord:
    return ChunkRecord(
        point_id="src/example.py:1-4",
        scope="code",
        relative_path="src/example.py",
        language="python",
        chunk_type="python_function",
        text=text,
        start_line=10,
        end_line=13,
        content_hash="hash",
        source_mtime=0.0,
        symbol_path="example.fn",
        domain_tags=["runtime"],
        extra={"owner": "test"},
    )


class ChunkPipelineTests(unittest.TestCase):
    def test_normalize_chunks_preserves_small_chunks(self) -> None:
        chunk = _chunk("small")

        self.assertEqual(normalize_chunks([chunk], max_chars=20), [chunk])

    def test_normalize_chunks_splits_oversized_chunks_with_metadata(self) -> None:
        parts = normalize_chunks([_chunk("alpha\nbeta\ngamma\n")], max_chars=11)

        self.assertEqual([part.point_id for part in parts], ["src/example.py:1-4:part1", "src/example.py:1-4:part2"])
        self.assertEqual([part.text for part in parts], ["alpha\nbeta\n", "gamma\n"])
        self.assertEqual([part.start_line for part in parts], [10, 13])
        self.assertTrue(all(part.chunk_type == "python_function_part" for part in parts))
        self.assertTrue(all(part.symbol_path == "example.fn" for part in parts))
        self.assertEqual([part.extra["split_part"] for part in parts], ["1", "2"])


if __name__ == "__main__":
    unittest.main()
