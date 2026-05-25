from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from services.repo_semantic.config import SemanticMcpSettings
from services.repo_semantic.indexer import IndexablePathScanBoundsExceeded, RepositoryIndexer
from services.repo_semantic.models import ChunkRecord
from services.repo_semantic.path_manifest import manifest_record_for_chunks
from services.repo_semantic.sparse import build_sparse_manifest
from tests.helpers.fakes import FakePoint


class _NoEmbeddingProvider:
    def embed_documents(self, texts):
        raise AssertionError("incremental reindex should not embed into a missing current collection")

    def backend_name(self) -> str:
        return "fake"

    def model_name(self) -> str:
        return "fake-model"


class _EmbeddingProvider:
    def embed_documents(self, texts):
        return [[1.0] for _ in texts]

    def backend_name(self) -> str:
        return "fake"

    def model_name(self) -> str:
        return "fake-model"


class _MissingCurrentStore:
    def delete_file_chunks(self, scope: str, relative_path: str) -> bool:
        return False

    def current_collection_exists(self, scope: str) -> bool:
        return False

    def collection_exists(self, scope: str) -> bool:
        return False

    def ensure_collection(self, *args, **kwargs) -> None:
        raise AssertionError("missing manifest must not create a partial current collection")

    def upsert_chunks(self, *args, **kwargs) -> None:
        raise AssertionError("missing manifest must not upsert a partial current collection")


class _DenseOnlyCurrentStore:
    def __init__(self) -> None:
        self.upsert_payload = None

    def delete_file_chunks(self, scope: str, relative_path: str) -> bool:
        return True

    def current_collection_exists(self, scope: str) -> bool:
        return True

    def collection_exists(self, scope: str) -> bool:
        return True

    def sparse_vector_available(self, scope: str) -> bool:
        return False

    def ensure_collection(self, *args, **kwargs) -> None:
        return None

    def upsert_chunks(self, *args, **kwargs) -> None:
        self.upsert_payload = kwargs


class _SparseCurrentStore(_DenseOnlyCurrentStore):
    def sparse_vector_available(self, scope: str) -> bool:
        return True


class _UnconfirmedDeleteCurrentStore(_DenseOnlyCurrentStore):
    def delete_file_chunks(self, scope: str, relative_path: str) -> bool:
        return False


class _ScrollOnlyStore:
    def scroll_chunks(self, scope: str):
        return []

    def delete_file_chunks(self, scope: str, relative_path: str) -> bool:
        raise AssertionError("bounds-exceeded reconcile must not mutate collection")


class _Point:
    def __init__(self, relative_path: str, source_mtime: float = 0.0) -> None:
        self.payload = {"relative_path": relative_path, "source_mtime": source_mtime}


class _ScrollPointStore(_ScrollOnlyStore):
    def __init__(self, points):
        self._points = points

    def scroll_chunks(self, scope: str):
        return self._points if scope == "code" else []


class _NoopIndexedStore(_ScrollOnlyStore):
    def __init__(self, chunks: list[ChunkRecord]) -> None:
        self._points = [FakePoint(chunk, 1.0) for chunk in chunks]

    def scroll_chunks(self, scope: str):
        return self._points if scope == "code" else []


def _manifest_chunk() -> ChunkRecord:
    return ChunkRecord(
        point_id="seed",
        scope="code",
        relative_path="src/seed.py",
        language="python",
        chunk_type="python_function",
        text="def seed():\n    return 1\n",
        start_line=1,
        end_line=2,
        content_hash="seed",
        source_mtime=0.0,
    )


def _chunk_for_path(relative_path: str) -> ChunkRecord:
    return ChunkRecord(
        point_id=f"{relative_path}:1",
        scope="code",
        relative_path=relative_path,
        language="python",
        chunk_type="python_function",
        text="def app():\n    return 1\n",
        start_line=1,
        end_line=2,
        content_hash=relative_path,
        source_mtime=0.0,
    )


class IndexerContractTests(unittest.TestCase):
    def test_reindex_paths_skips_partial_current_collection_when_sparse_manifest_missing(self) -> None:
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            (repo_root / "src").mkdir()
            (repo_root / "src" / "app.py").write_text("def app():\n    return 1\n", encoding="utf-8")
            settings = SemanticMcpSettings(
                SEMANTIC_MCP_REPO_ROOT=str(repo_root),
                SEMANTIC_MCP_LOGICAL_REPO_ROOT=str(repo_root),
                SEMANTIC_MCP_REGISTRY_DB_PATH=str(Path(tmp) / "registry.sqlite3"),
                SEMANTIC_MCP_INCLUDE_GLOBS=["**/*"],
                SEMANTIC_MCP_EXCLUDE_GLOBS=[".git/**"],
            )
            indexer = RepositoryIndexer(
                settings=settings,
                embedding_provider=_NoEmbeddingProvider(),
                store=_MissingCurrentStore(),
            )

            result = indexer.reindex_paths(["src/app.py"])

            self.assertEqual(result["code"], 0)
            self.assertEqual(result["docs"], 0)

    def test_reindex_paths_does_not_send_sparse_vectors_to_dense_only_current_collection(self) -> None:
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            (repo_root / "src").mkdir()
            (repo_root / "src" / "app.py").write_text("def app():\n    return 1\n", encoding="utf-8")
            settings = SemanticMcpSettings(
                SEMANTIC_MCP_REPO_ROOT=str(repo_root),
                SEMANTIC_MCP_LOGICAL_REPO_ROOT=str(repo_root),
                SEMANTIC_MCP_REGISTRY_DB_PATH=str(Path(tmp) / "registry.sqlite3"),
                SEMANTIC_MCP_INCLUDE_GLOBS=["**/*"],
                SEMANTIC_MCP_EXCLUDE_GLOBS=[".git/**"],
            )
            store = _DenseOnlyCurrentStore()
            indexer = RepositoryIndexer(
                settings=settings,
                embedding_provider=_EmbeddingProvider(),
                store=store,
            )
            indexer._sparse_manifests.save(
                build_sparse_manifest(
                    scope="code",
                    chunks=[_manifest_chunk()],
                    schema_version=settings.SEMANTIC_MCP_INDEX_SCHEMA_VERSION,
                )
            )

            result = indexer.reindex_paths(["src/app.py"])

            self.assertEqual(result["code"], 1)
            self.assertIsNotNone(store.upsert_payload)
            self.assertIsNone(store.upsert_payload["sparse_vectors"])
            self.assertIsNone(store.upsert_payload["sparse_contract_hash"])
            path_summary = indexer.path_manifest.summarize([repo_root / "src" / "app.py"])
            self.assertFalse(path_summary.coverage_complete)
            self.assertEqual(path_summary.coverage_error_code, "path_manifest_incomplete")
            self.assertEqual(path_summary.missing_from_index_count, 0)
            self.assertTrue(indexer.path_manifest.exists())

    def test_reindex_paths_skips_partial_current_collection_even_with_sparse_manifest(self) -> None:
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            (repo_root / "src").mkdir()
            (repo_root / "src" / "app.py").write_text("def app():\n    return 1\n", encoding="utf-8")
            settings = SemanticMcpSettings(
                SEMANTIC_MCP_REPO_ROOT=str(repo_root),
                SEMANTIC_MCP_LOGICAL_REPO_ROOT=str(repo_root),
                SEMANTIC_MCP_REGISTRY_DB_PATH=str(Path(tmp) / "registry.sqlite3"),
                SEMANTIC_MCP_INCLUDE_GLOBS=["**/*"],
                SEMANTIC_MCP_EXCLUDE_GLOBS=[".git/**"],
            )
            indexer = RepositoryIndexer(
                settings=settings,
                embedding_provider=_NoEmbeddingProvider(),
                store=_MissingCurrentStore(),
            )
            indexer._sparse_manifests.save(
                build_sparse_manifest(
                    scope="code",
                    chunks=[_manifest_chunk()],
                    schema_version=settings.SEMANTIC_MCP_INDEX_SCHEMA_VERSION,
                )
            )

            result = indexer.reindex_paths(["src/app.py"])

            self.assertEqual(result["code"], 0)
            self.assertFalse(indexer.path_manifest.exists())

    def test_reindex_paths_does_not_mix_sparse_analyzer_contracts(self) -> None:
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            (repo_root / "src").mkdir()
            (repo_root / "src" / "app.py").write_text("def app():\n    return 1\n", encoding="utf-8")
            settings = SemanticMcpSettings(
                SEMANTIC_MCP_REPO_ROOT=str(repo_root),
                SEMANTIC_MCP_LOGICAL_REPO_ROOT=str(repo_root),
                SEMANTIC_MCP_REGISTRY_DB_PATH=str(Path(tmp) / "registry.sqlite3"),
                SEMANTIC_MCP_INCLUDE_GLOBS=["**/*"],
                SEMANTIC_MCP_EXCLUDE_GLOBS=[".git/**"],
            )
            store = _SparseCurrentStore()
            indexer = RepositoryIndexer(
                settings=settings,
                embedding_provider=_EmbeddingProvider(),
                store=store,
            )
            incompatible_manifest = build_sparse_manifest(
                scope="code",
                chunks=[_manifest_chunk()],
                schema_version=settings.SEMANTIC_MCP_INDEX_SCHEMA_VERSION,
            ).model_copy(update={"lexical_analyzer_version": "code_lexical_v1"})
            indexer._sparse_manifests.save(incompatible_manifest)

            result = indexer.reindex_paths(["src/app.py"])
            stored_manifest = indexer._sparse_manifests.load("code")

            self.assertEqual(result["code"], 1)
            self.assertIsNotNone(store.upsert_payload)
            self.assertIsNone(store.upsert_payload["sparse_vectors"])
            self.assertIsNone(store.upsert_payload["sparse_contract_hash"])
            self.assertEqual(stored_manifest.lexical_analyzer_version, "code_lexical_v1")

    def test_reindex_paths_keeps_path_stale_when_old_chunks_delete_is_unconfirmed(self) -> None:
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            (repo_root / "src").mkdir()
            app_path = repo_root / "src" / "app.py"
            app_path.write_text("def app():\n    return 1\n", encoding="utf-8")
            settings = SemanticMcpSettings(
                SEMANTIC_MCP_REPO_ROOT=str(repo_root),
                SEMANTIC_MCP_LOGICAL_REPO_ROOT=str(repo_root),
                SEMANTIC_MCP_REGISTRY_DB_PATH=str(Path(tmp) / "registry.sqlite3"),
                SEMANTIC_MCP_INCLUDE_GLOBS=["**/*"],
                SEMANTIC_MCP_EXCLUDE_GLOBS=[".git/**"],
            )
            store = _UnconfirmedDeleteCurrentStore()
            indexer = RepositoryIndexer(
                settings=settings,
                embedding_provider=_EmbeddingProvider(),
                store=store,
            )
            indexer.path_manifest.replace_all(
                [
                    manifest_record_for_chunks(
                        settings=settings,
                        relative_path="src/app.py",
                        chunks=[_chunk_for_path("src/app.py")],
                        indexed_at="2026-05-22T00:00:00+00:00",
                    )
                ]
            )
            app_path.write_text("def app():\n    return 2\n", encoding="utf-8")

            result = indexer.reindex_paths(["src/app.py"])
            path_summary = indexer.path_manifest.summarize([app_path])

            self.assertEqual(result["code"], 1)
            self.assertIsNotNone(store.upsert_payload)
            self.assertTrue(path_summary.coverage_complete)
            self.assertEqual(path_summary.stale_indexed_paths_count, 1)
            self.assertIn("src/app.py", path_summary.stale_paths_preview)

    def test_reindex_paths_canonicalizes_relative_path_keys(self) -> None:
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            (repo_root / "src").mkdir()
            app_path = repo_root / "src" / "app.py"
            app_path.write_text("def app():\n    return 1\n", encoding="utf-8")
            settings = SemanticMcpSettings(
                SEMANTIC_MCP_REPO_ROOT=str(repo_root),
                SEMANTIC_MCP_LOGICAL_REPO_ROOT=str(repo_root),
                SEMANTIC_MCP_REGISTRY_DB_PATH=str(Path(tmp) / "registry.sqlite3"),
                SEMANTIC_MCP_INCLUDE_GLOBS=["**/*"],
                SEMANTIC_MCP_EXCLUDE_GLOBS=[".git/**"],
            )
            store = _DenseOnlyCurrentStore()
            indexer = RepositoryIndexer(
                settings=settings,
                embedding_provider=_EmbeddingProvider(),
                store=store,
            )

            result = indexer.reindex_paths(["./src//app.py"])
            path_summary = indexer.path_manifest.summarize([app_path])

            self.assertEqual(result["code"], 1)
            self.assertIsNotNone(store.upsert_payload)
            self.assertFalse(path_summary.coverage_complete)
            self.assertEqual(path_summary.missing_from_index_count, 0)
            self.assertEqual(path_summary.stale_indexed_paths_count, 0)

    def test_reindex_paths_does_not_index_excluded_paths(self) -> None:
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            (repo_root / ".git").mkdir()
            excluded_path = repo_root / ".git" / "config"
            excluded_path.write_text("[core]\n", encoding="utf-8")
            settings = SemanticMcpSettings(
                SEMANTIC_MCP_REPO_ROOT=str(repo_root),
                SEMANTIC_MCP_LOGICAL_REPO_ROOT=str(repo_root),
                SEMANTIC_MCP_REGISTRY_DB_PATH=str(Path(tmp) / "registry.sqlite3"),
                SEMANTIC_MCP_INCLUDE_GLOBS=["**/*"],
                SEMANTIC_MCP_EXCLUDE_GLOBS=[".git/**"],
            )
            store = _DenseOnlyCurrentStore()
            indexer = RepositoryIndexer(
                settings=settings,
                embedding_provider=_EmbeddingProvider(),
                store=store,
            )

            result = indexer.reindex_paths([".git/config"])
            path_summary = indexer.path_manifest.summarize(indexer.iter_indexable_paths())

            self.assertIsNone(store.upsert_payload)
            self.assertEqual(result["code"], 0)
            self.assertFalse(path_summary.coverage_complete)
            self.assertEqual(path_summary.missing_from_index_count, 0)
            self.assertEqual(path_summary.stale_indexed_paths_count, 0)
            self.assertEqual(path_summary.deleted_indexed_paths_count, 0)

    def test_reconcile_index_stops_before_mutation_when_path_bound_exceeded(self) -> None:
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            (repo_root / "src").mkdir()
            (repo_root / "src" / "a.py").write_text("print('a')\n", encoding="utf-8")
            (repo_root / "src" / "b.py").write_text("print('b')\n", encoding="utf-8")
            settings = SemanticMcpSettings(
                SEMANTIC_MCP_REPO_ROOT=str(repo_root),
                SEMANTIC_MCP_LOGICAL_REPO_ROOT=str(repo_root),
                SEMANTIC_MCP_REGISTRY_DB_PATH=str(Path(tmp) / "registry.sqlite3"),
                SEMANTIC_MCP_INCLUDE_GLOBS=["src/**"],
                SEMANTIC_MCP_EXCLUDE_GLOBS=[".git/**"],
                SEMANTIC_MCP_RECONCILE_MAX_PATHS=1,
            )
            indexer = RepositoryIndexer(
                settings=settings,
                embedding_provider=_EmbeddingProvider(),
                store=_ScrollOnlyStore(),
            )

            result = indexer.reconcile_index()

            self.assertEqual(result["paths"], 2)
            self.assertEqual(result["bounds_exceeded"], ["max_reconcile_paths"])
            self.assertFalse(result["mutation_started"])
            self.assertEqual(result["paths_reindexed"], 0)

    def test_reconcile_index_stops_before_mutation_when_point_delete_bound_exceeded(self) -> None:
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            (repo_root / "src").mkdir()
            app_path = repo_root / "src" / "app.py"
            app_path.write_text("print('new')\n", encoding="utf-8")
            settings = SemanticMcpSettings(
                SEMANTIC_MCP_REPO_ROOT=str(repo_root),
                SEMANTIC_MCP_LOGICAL_REPO_ROOT=str(repo_root),
                SEMANTIC_MCP_REGISTRY_DB_PATH=str(Path(tmp) / "registry.sqlite3"),
                SEMANTIC_MCP_INCLUDE_GLOBS=["src/**"],
                SEMANTIC_MCP_EXCLUDE_GLOBS=[".git/**"],
                SEMANTIC_MCP_RECONCILE_MAX_POINTS_DELETED=1,
            )
            indexer = RepositoryIndexer(
                settings=settings,
                embedding_provider=_EmbeddingProvider(),
                store=_ScrollPointStore(
                    [
                        _Point("src/app.py", source_mtime=0.0),
                        _Point("src/app.py", source_mtime=0.0),
                    ]
                ),
            )

            result = indexer.reconcile_index()

            self.assertEqual(result["paths"], 1)
            self.assertEqual(result["points_deleted"], 2)
            self.assertEqual(result["bounds_exceeded"], ["max_points_deleted"])
            self.assertFalse(result["mutation_started"])

    def test_reconcile_noop_backfills_complete_path_manifest_without_rebuild(self) -> None:
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            (repo_root / "src").mkdir()
            app_path = repo_root / "src" / "app.py"
            app_path.write_text("def app():\n    return 1\n", encoding="utf-8")
            source_mtime = app_path.stat().st_mtime
            settings = SemanticMcpSettings(
                SEMANTIC_MCP_REPO_ROOT=str(repo_root),
                SEMANTIC_MCP_LOGICAL_REPO_ROOT=str(repo_root),
                SEMANTIC_MCP_REGISTRY_DB_PATH=str(Path(tmp) / "registry.sqlite3"),
                SEMANTIC_MCP_INCLUDE_GLOBS=["src/**"],
                SEMANTIC_MCP_EXCLUDE_GLOBS=[".git/**"],
            )
            indexed_chunk = _chunk_for_path("src/app.py")
            indexed_chunk.source_mtime = source_mtime
            indexer = RepositoryIndexer(
                settings=settings,
                embedding_provider=_EmbeddingProvider(),
                store=_NoopIndexedStore([indexed_chunk]),
            )

            result = indexer.reconcile_index()
            path_summary = indexer.path_manifest.summarize([app_path])
            invalidated_count, invalidated_preview, invalidated_known = (
                indexer.path_manifest.invalidated_paths_after("2026-05-22T00:00:00+00:00")
            )

            self.assertEqual(result["paths"], 0)
            self.assertFalse(result["mutation_started"])
            self.assertTrue(result["path_manifest_backfilled"])
            self.assertTrue(result["path_manifest_coverage_complete"])
            self.assertTrue(path_summary.coverage_complete)
            self.assertEqual(path_summary.missing_from_index_count, 0)
            self.assertEqual(path_summary.stale_indexed_paths_count, 0)
            self.assertEqual(invalidated_count, 0)
            self.assertEqual(invalidated_preview, [])
            self.assertTrue(invalidated_known)

    def test_reconcile_noop_respects_existing_zero_chunk_manifest_rows(self) -> None:
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            (repo_root / "src").mkdir()
            empty_path = repo_root / "src" / "empty.py"
            empty_path.write_text("", encoding="utf-8")
            source_mtime = empty_path.stat().st_mtime
            settings = SemanticMcpSettings(
                SEMANTIC_MCP_REPO_ROOT=str(repo_root),
                SEMANTIC_MCP_LOGICAL_REPO_ROOT=str(repo_root),
                SEMANTIC_MCP_REGISTRY_DB_PATH=str(Path(tmp) / "registry.sqlite3"),
                SEMANTIC_MCP_INCLUDE_GLOBS=["src/**"],
                SEMANTIC_MCP_EXCLUDE_GLOBS=[".git/**"],
            )
            indexer = RepositoryIndexer(
                settings=settings,
                embedding_provider=_EmbeddingProvider(),
                store=_NoopIndexedStore([]),
            )
            indexer.path_manifest.replace_all(
                [
                    manifest_record_for_chunks(
                        settings=settings,
                        relative_path="src/empty.py",
                        chunks=[],
                        indexed_at="2026-05-22T00:00:00+00:00",
                        content_hash="empty-hash",
                        source_mtime=source_mtime,
                        file_size=0,
                        scope="code",
                    )
                ]
            )

            result = indexer.reconcile_index()
            path_summary = indexer.path_manifest.summarize([empty_path])

            self.assertEqual(result["paths"], 0)
            self.assertFalse(result["mutation_started"])
            self.assertTrue(result["path_manifest_backfilled"])
            self.assertTrue(path_summary.coverage_complete)
            self.assertEqual(path_summary.missing_from_index_count, 0)

    def test_iter_indexable_paths_bounds_count_excluded_filesystem_entries(self) -> None:
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            excluded_dir = repo_root / ".git"
            excluded_dir.mkdir()
            (excluded_dir / "config").write_text("[core]\n", encoding="utf-8")
            settings = SemanticMcpSettings(
                SEMANTIC_MCP_REPO_ROOT=str(repo_root),
                SEMANTIC_MCP_LOGICAL_REPO_ROOT=str(repo_root),
                SEMANTIC_MCP_REGISTRY_DB_PATH=str(Path(tmp) / "registry.sqlite3"),
                SEMANTIC_MCP_INCLUDE_GLOBS=["src/**"],
                SEMANTIC_MCP_EXCLUDE_GLOBS=[".git/**"],
            )
            indexer = RepositoryIndexer(
                settings=settings,
                embedding_provider=_EmbeddingProvider(),
                store=_ScrollOnlyStore(),
            )

            with self.assertRaises(IndexablePathScanBoundsExceeded) as raised:
                list(indexer.iter_indexable_paths(max_paths_scanned=1))

            self.assertEqual(raised.exception.summary["bounds_exceeded"], ["max_scan_paths"])


if __name__ == "__main__":
    unittest.main()
