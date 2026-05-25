from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from services.repo_semantic.config import SemanticMcpSettings
from services.repo_semantic.models import ChunkRecord
from services.repo_semantic.path_manifest import (
    PathManifestStore,
    group_chunks_by_path,
    manifest_record_for_chunks,
)


def _settings(tmp: str) -> SemanticMcpSettings:
    repo_root = Path(tmp) / "repo"
    repo_root.mkdir()
    return SemanticMcpSettings(
        SEMANTIC_MCP_REPO_ROOT=str(repo_root),
        SEMANTIC_MCP_LOGICAL_REPO_ROOT=str(repo_root),
        SEMANTIC_MCP_REGISTRY_DB_PATH=str(Path(tmp) / "registry.sqlite3"),
        SEMANTIC_MCP_INCLUDE_GLOBS=["**/*"],
        SEMANTIC_MCP_DOC_PATH_PREFIXES=["docs/"],
        SEMANTIC_MCP_EXCLUDE_GLOBS=[".git/**"],
    )


def _chunk(relative_path: str, text: str = "print('ok')\n") -> ChunkRecord:
    return ChunkRecord(
        point_id=f"{relative_path}:1",
        scope="code",
        relative_path=relative_path,
        language="python",
        chunk_type="python_function",
        text=text,
        start_line=1,
        end_line=1,
        content_hash=relative_path,
        source_mtime=0.0,
    )


class PathManifestTests(unittest.TestCase):
    def test_missing_manifest_reports_incomplete_coverage(self) -> None:
        with TemporaryDirectory() as tmp:
            settings = _settings(tmp)
            store = PathManifestStore(settings)

            summary = store.summarize([])
            invalidated_count, invalidated_preview, invalidated_known = store.invalidated_paths_after(
                "2026-05-22T00:00:00+00:00"
            )

            self.assertFalse(summary.coverage_complete)
            self.assertFalse(summary.manifest_available)
            self.assertEqual(summary.coverage_error_code, "path_manifest_missing")
            self.assertEqual(invalidated_count, 0)
            self.assertEqual(invalidated_preview, [])
            self.assertFalse(invalidated_known)

    def test_manifest_summary_detects_missing_stale_and_deleted_paths(self) -> None:
        with TemporaryDirectory() as tmp:
            settings = _settings(tmp)
            (settings.repo_root / "src").mkdir()
            ready_file = settings.repo_root / "src" / "ready.py"
            stale_file = settings.repo_root / "src" / "stale.py"
            missing_file = settings.repo_root / "src" / "missing.py"
            deleted_file = settings.repo_root / "src" / "deleted.py"
            ready_file.write_text("print('ready')\n", encoding="utf-8")
            stale_file.write_text("print('old')\n", encoding="utf-8")
            missing_file.write_text("print('missing')\n", encoding="utf-8")
            deleted_file.write_text("print('deleted')\n", encoding="utf-8")

            chunks = [_chunk("src/ready.py"), _chunk("src/stale.py"), _chunk("src/deleted.py")]
            grouped = group_chunks_by_path(chunks)
            manifest = PathManifestStore(settings)
            manifest.replace_all(
                [
                    manifest_record_for_chunks(
                        settings=settings,
                        relative_path=relative_path,
                        chunks=path_chunks,
                        indexed_at="2026-05-22T00:00:00+00:00",
                    )
                    for relative_path, path_chunks in grouped.items()
                ]
            )

            stale_file.write_text("print('new content')\n", encoding="utf-8")
            deleted_file.unlink()

            summary = manifest.summarize([ready_file, stale_file, missing_file])

            self.assertTrue(summary.coverage_complete)
            self.assertEqual(summary.missing_from_index_count, 1)
            self.assertEqual(summary.stale_indexed_paths_count, 1)
            self.assertEqual(summary.deleted_indexed_paths_count, 1)
            self.assertIn("src/missing.py", summary.stale_paths_preview)
            self.assertIn("src/stale.py", summary.stale_paths_preview)
            self.assertIn("src/deleted.py", summary.deleted_paths_preview)

    def test_manifest_summary_bounds_hashing_for_status_calls(self) -> None:
        with TemporaryDirectory() as tmp:
            settings = _settings(tmp)
            (settings.repo_root / "src").mkdir()
            stale_file = settings.repo_root / "src" / "stale.py"
            stale_file.write_text("print('old')\n", encoding="utf-8")
            manifest = PathManifestStore(settings)
            manifest.replace_all(
                [
                    manifest_record_for_chunks(
                        settings=settings,
                        relative_path="src/stale.py",
                        chunks=[_chunk("src/stale.py")],
                        indexed_at="2026-05-22T00:00:00+00:00",
                    )
                ]
            )
            stale_file.write_text("print('new')\n", encoding="utf-8")

            summary = manifest.summarize([stale_file], max_hashed_paths=0)

            self.assertFalse(summary.coverage_complete)
            self.assertEqual(summary.coverage_error_code, "path_manifest_status_bounds_exceeded")
            self.assertTrue(summary.manifest_available)

    def test_mark_deleted_clears_deleted_path_from_stale_counts(self) -> None:
        with TemporaryDirectory() as tmp:
            settings = _settings(tmp)
            (settings.repo_root / "src").mkdir()
            deleted_file = settings.repo_root / "src" / "deleted.py"
            deleted_file.write_text("print('deleted')\n", encoding="utf-8")
            manifest = PathManifestStore(settings)
            manifest.replace_all(
                [
                    manifest_record_for_chunks(
                        settings=settings,
                        relative_path="src/deleted.py",
                        chunks=[_chunk("src/deleted.py")],
                        indexed_at="2026-05-22T00:00:00+00:00",
                    )
                ]
            )
            deleted_file.unlink()

            before = manifest.summarize([])
            manifest.mark_deleted(["src/deleted.py"])
            after = manifest.summarize([])

            self.assertEqual(before.deleted_indexed_paths_count, 1)
            self.assertEqual(after.deleted_indexed_paths_count, 0)

    def test_zero_chunk_file_can_be_authoritatively_indexed(self) -> None:
        with TemporaryDirectory() as tmp:
            settings = _settings(tmp)
            (settings.repo_root / "src").mkdir()
            empty_file = settings.repo_root / "src" / "empty.py"
            empty_file.write_text("", encoding="utf-8")
            manifest = PathManifestStore(settings)

            manifest.replace_all(
                [
                    manifest_record_for_chunks(
                        settings=settings,
                        relative_path="src/empty.py",
                        chunks=[],
                        indexed_at="2026-05-22T00:00:00+00:00",
                        scope="code",
                    )
                ]
            )
            summary = manifest.summarize([empty_file])

            self.assertTrue(summary.coverage_complete)
            self.assertEqual(summary.missing_from_index_count, 0)
            self.assertEqual(summary.stale_indexed_paths_count, 0)

    def test_manifest_record_uses_snapshot_hash_when_provided(self) -> None:
        with TemporaryDirectory() as tmp:
            settings = _settings(tmp)
            (settings.repo_root / "src").mkdir()
            file_path = settings.repo_root / "src" / "app.py"
            file_path.write_text("old\n", encoding="utf-8")
            source_mtime = file_path.stat().st_mtime
            file_size = file_path.stat().st_size
            file_path.write_text("new\n", encoding="utf-8")

            record = manifest_record_for_chunks(
                settings=settings,
                relative_path="src/app.py",
                chunks=[_chunk("src/app.py")],
                indexed_at="2026-05-22T00:00:00+00:00",
                content_hash="snapshot-hash",
                source_mtime=source_mtime,
                file_size=file_size,
            )

            self.assertEqual(record.content_hash, "snapshot-hash")
            self.assertEqual(record.source_mtime, source_mtime)
            self.assertEqual(record.file_size, file_size)

    def test_incremental_write_after_identity_change_does_not_preserve_full_coverage(self) -> None:
        with TemporaryDirectory() as tmp:
            settings = _settings(tmp)
            (settings.repo_root / "src").mkdir()
            app_file = settings.repo_root / "src" / "app.py"
            app_file.write_text("print('old')\n", encoding="utf-8")
            manifest = PathManifestStore(settings)
            manifest.replace_all(
                [
                    manifest_record_for_chunks(
                        settings=settings,
                        relative_path="src/app.py",
                        chunks=[_chunk("src/app.py")],
                        indexed_at="2026-05-22T00:00:00+00:00",
                    )
                ]
            )
            changed_settings = settings.model_copy(
                update={"SEMANTIC_MCP_DOCUMENT_PREFIX": "changed-prefix"}
            )
            changed_manifest = PathManifestStore(changed_settings)

            changed_manifest.upsert_ready_records(
                [
                    manifest_record_for_chunks(
                        settings=changed_settings,
                        relative_path="src/app.py",
                        chunks=[_chunk("src/app.py")],
                        indexed_at="2026-05-22T00:00:01+00:00",
                    )
                ]
            )
            summary = changed_manifest.summarize([app_file])
            invalidated_count, invalidated_preview, invalidated_known = changed_manifest.invalidated_paths_after(
                "2026-05-22T00:00:00+00:00"
            )

            self.assertFalse(summary.coverage_complete)
            self.assertEqual(summary.coverage_error_code, "path_manifest_incomplete")
            self.assertEqual(invalidated_count, 0)
            self.assertEqual(invalidated_preview, [])
            self.assertFalse(invalidated_known)

    def test_invalidated_paths_after_is_bounded_and_requires_complete_coverage(self) -> None:
        with TemporaryDirectory() as tmp:
            settings = _settings(tmp)
            manifest = PathManifestStore(settings)
            manifest.replace_all(
                [
                    manifest_record_for_chunks(
                        settings=settings,
                        relative_path=f"src/{index}.py",
                        chunks=[_chunk(f"src/{index}.py")],
                        indexed_at="2026-05-22T00:00:01+00:00",
                        content_hash=f"hash-{index}",
                        source_mtime=1.0,
                        file_size=1,
                    )
                    for index in range(3)
                ]
            )

            count, preview, known = manifest.invalidated_paths_after(
                "2026-05-22T00:00:00+00:00",
                limit=2,
            )

            self.assertEqual(count, 3)
            self.assertEqual(preview, ["src/0.py", "src/1.py"])
            self.assertFalse(known)


if __name__ == "__main__":
    unittest.main()
