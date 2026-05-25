from __future__ import annotations

import unittest

from services.repo_semantic.indexing.sparse_pipeline import (
    compatible_sparse_manifest,
    encode_sparse_documents,
    should_upsert_sparse_vectors,
)
from services.repo_semantic.lexical import build_sparse_manifest
from services.repo_semantic.models import ChunkRecord


def _chunk(text: str = "DATABASE_URL connect flow") -> ChunkRecord:
    return ChunkRecord(
        point_id="src/config.py:1-3",
        scope="code",
        relative_path="src/config.py",
        language="python",
        chunk_type="python_function",
        text=text,
        start_line=1,
        end_line=3,
        content_hash="hash",
        source_mtime=0.0,
    )


class SparsePipelineTests(unittest.TestCase):
    def test_compatible_sparse_manifest_requires_schema_contract(self) -> None:
        manifest = build_sparse_manifest(scope="code", chunks=[_chunk()], schema_version=2)

        self.assertIs(compatible_sparse_manifest(manifest, schema_version=2), manifest)
        self.assertIsNone(compatible_sparse_manifest(manifest, schema_version=3))

    def test_should_upsert_sparse_vectors_requires_manifest_and_sparse_collection(self) -> None:
        manifest = build_sparse_manifest(scope="code", chunks=[_chunk()], schema_version=2)

        self.assertFalse(
            should_upsert_sparse_vectors(
                None,
                current_collection_exists=True,
                sparse_vector_available=True,
            )
        )
        self.assertFalse(
            should_upsert_sparse_vectors(
                manifest,
                current_collection_exists=True,
                sparse_vector_available=False,
            )
        )
        self.assertTrue(
            should_upsert_sparse_vectors(
                manifest,
                current_collection_exists=True,
                sparse_vector_available=True,
            )
        )
        self.assertTrue(
            should_upsert_sparse_vectors(
                manifest,
                current_collection_exists=False,
                sparse_vector_available=False,
            )
        )

    def test_encode_sparse_documents_uses_manifest_vocabulary(self) -> None:
        chunk = _chunk()
        manifest = build_sparse_manifest(scope="code", chunks=[chunk], schema_version=2)

        vectors = encode_sparse_documents([chunk], manifest)

        self.assertEqual(len(vectors), 1)
        self.assertTrue(vectors[0].indices)
        self.assertEqual(len(vectors[0].indices), len(vectors[0].values))


if __name__ == "__main__":
    unittest.main()
