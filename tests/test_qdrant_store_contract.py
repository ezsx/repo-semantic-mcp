from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from qdrant_client import QdrantClient, models

from services.repo_semantic.config import SemanticMcpSettings
from services.repo_semantic.lexical import EncodedSparseVector
from services.repo_semantic.models import ChunkRecord
from services.repo_semantic.qdrant_store import QdrantStore
from services.repo_semantic.storage.qdrant.payload_codec import (
    chunk_payload,
    estimated_point_bytes,
    path_payload_fields,
    point_vector,
)
from services.repo_semantic.storage.qdrant.collection_names import (
    collection_name_for_schema,
    current_collection_name,
)
from services.repo_semantic.storage.qdrant.payload_indexes import payload_index_specs


def _chunk() -> ChunkRecord:
    return ChunkRecord(
        point_id="legacy",
        scope="code",
        relative_path="src/legacy.py",
        language="python",
        chunk_type="python_function",
        text="legacy dense searchable",
        start_line=1,
        end_line=1,
        content_hash="legacy",
        source_mtime=0.0,
    )


class QdrantStoreContractTests(unittest.TestCase):
    def _store(self, tmp: str) -> QdrantStore:
        settings = SemanticMcpSettings(
            SEMANTIC_MCP_REPO_ROOT=tmp,
            SEMANTIC_MCP_LOGICAL_REPO_ROOT=tmp,
            SEMANTIC_MCP_QDRANT_URL="http://unused",
            SEMANTIC_MCP_REGISTRY_DB_PATH=str(Path(tmp) / "registry.sqlite3"),
        )
        store = QdrantStore(settings)
        store._client = QdrantClient(":memory:")
        return store

    def test_read_path_discovers_previous_schema_unnamed_dense_collection(self) -> None:
        with TemporaryDirectory() as tmp:
            store = self._store(tmp)
            legacy_name = store._collection_name_for_schema("code", 1)
            chunk = _chunk()
            store.client.create_collection(
                collection_name=legacy_name,
                vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE),
            )
            store.client.upsert(
                collection_name=legacy_name,
                points=[
                    models.PointStruct(
                        id=store._point_id(chunk.point_id),
                        vector=[1.0],
                        payload={
                            "chunk_id": chunk.point_id,
                            "scope": chunk.scope,
                            "relative_path": chunk.relative_path,
                            "language": chunk.language,
                            "chunk_type": chunk.chunk_type,
                            "text": chunk.text,
                            "start_line": chunk.start_line,
                            "end_line": chunk.end_line,
                            "content_hash": chunk.content_hash,
                            "source_mtime": chunk.source_mtime,
                            "domain_tags": [],
                            "is_generated": False,
                            "embedding_backend": "fake",
                            "embedding_model": "fake-model",
                            "index_schema_version": 1,
                        },
                    )
                ],
            )

            self.assertFalse(store.current_collection_exists("code"))
            self.assertTrue(store.read_collection_exists("code"))
            self.assertEqual(store.count("code"), 1)
            self.assertEqual(store.dense_schema_kind("code"), "unnamed_legacy")
            self.assertEqual(store.search("code", [1.0], 1)[0].payload["chunk_id"], "legacy")
            self.assertEqual(store.purge_stale_collections(), [])
            self.assertTrue(store.client.collection_exists(legacy_name))

    def test_collection_name_helpers_match_store_facade(self) -> None:
        with TemporaryDirectory() as tmp:
            store = self._store(tmp)

            self.assertEqual(
                current_collection_name(store._settings, scope="code"),
                store.collection_name("code"),
            )
            self.assertEqual(
                collection_name_for_schema(store._settings, scope="docs", schema_version=1),
                store._collection_name_for_schema("docs", 1),
            )

    def test_dense_search_applies_qdrant_payload_filter(self) -> None:
        with TemporaryDirectory() as tmp:
            store = self._store(tmp)
            store.recreate_collection("code", vector_size=1, sparse_enabled=True)
            code_chunk = _chunk()
            docs_chunk = ChunkRecord(
                point_id="docs",
                scope="code",
                relative_path="docs/legacy.md",
                language="markdown",
                chunk_type="markdown_section",
                text="docs dense searchable",
                start_line=1,
                end_line=1,
                content_hash="docs",
                source_mtime=0.0,
            )
            store.upsert_chunks(
                "code",
                [code_chunk, docs_chunk],
                [[1.0], [1.0]],
                embedding_backend="fake",
                embedding_model="fake-model",
                schema_version=2,
            )

            results = store.search(
                "code",
                [1.0],
                10,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="path_prefixes",
                            match=models.MatchValue(value="docs"),
                        )
                    ]
                ),
            )

            self.assertEqual([point.payload["chunk_id"] for point in results], ["docs"])
            self.assertIn("path_prefixes", store.payload_fields("code"))

    def test_payload_codec_preserves_filter_and_contract_fields(self) -> None:
        chunk = _chunk()
        file_extension, path_segments, path_prefixes = path_payload_fields(chunk.relative_path)
        payload = chunk_payload(
            chunk=chunk,
            embedding_backend="fake",
            embedding_model="fake-model",
            schema_version=2,
            query_template_hash="query-hash",
            document_prefix_hash="doc-hash",
            sparse_contract_hash="sparse-contract",
            sparse_vocabulary_hash="sparse-vocab",
            sparse_corpus_stats_hash="sparse-stats",
            lexical_analyzer_version="lex-v1",
        )

        self.assertEqual(file_extension, ".py")
        self.assertEqual(path_segments, ["src", "legacy.py"])
        self.assertEqual(path_prefixes, ["src", "src/legacy.py"])
        self.assertEqual(payload["relative_path"], "src/legacy.py")
        self.assertEqual(payload["path_prefixes"], ["src", "src/legacy.py"])
        self.assertEqual(payload["sparse_contract_hash"], "sparse-contract")
        self.assertEqual(payload["lexical_analyzer_version"], "lex-v1")

    def test_payload_codec_builds_named_dense_and_sparse_vectors(self) -> None:
        dense_vector = [0.25, 0.5]
        sparse_vector = EncodedSparseVector(indices=[1, 3], values=[1.0, 0.5])

        dense_only = point_vector(
            dense_vector=dense_vector,
            dense_schema_kind="named",
            sparse_vector=None,
        )
        hybrid = point_vector(
            dense_vector=dense_vector,
            dense_schema_kind="named",
            sparse_vector=sparse_vector,
        )

        self.assertEqual(dense_only, {"dense": dense_vector})
        self.assertEqual(hybrid["dense"], dense_vector)
        self.assertEqual(hybrid["sparse"].indices, [1, 3])
        self.assertGreater(
            estimated_point_bytes(chunk=_chunk(), dense_vector=dense_vector, sparse_vector=sparse_vector),
            len(_chunk().text),
        )

    def test_payload_index_specs_include_filter_fields(self) -> None:
        specs = payload_index_specs()

        self.assertIn("relative_path", specs)
        self.assertIn("path_prefixes", specs)
        self.assertIn("domain_tags", specs)
        self.assertIn("is_generated", specs)


if __name__ == "__main__":
    unittest.main()
