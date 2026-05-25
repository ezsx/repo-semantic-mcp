from __future__ import annotations

import sqlite3
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from services.repo_semantic.registry import LifecycleLeaseActiveError, RepoRegistry


class RepoRegistryTests(unittest.TestCase):
    def test_registry_migrates_revision_columns_and_closes_connections(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "registry.sqlite3"
            connection = sqlite3.connect(db_path)
            try:
                connection.execute(
                    """
                    CREATE TABLE repo_registry (
                        repo_root TEXT PRIMARY KEY,
                        repo_key TEXT NOT NULL,
                        display_name TEXT NOT NULL,
                        status TEXT NOT NULL,
                        active INTEGER NOT NULL DEFAULT 0,
                        index_profile TEXT NOT NULL,
                        include_globs_json TEXT NOT NULL,
                        doc_prefixes_json TEXT NOT NULL,
                        exclude_globs_json TEXT NOT NULL,
                        last_full_build_ts TEXT,
                        last_incremental_update_ts TEXT,
                        last_error TEXT,
                        watch_enabled INTEGER NOT NULL DEFAULT 0,
                        watch_running INTEGER NOT NULL DEFAULT 0,
                        code_points_count INTEGER NOT NULL DEFAULT 0,
                        docs_points_count INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                connection.commit()
            finally:
                connection.close()

            registry = RepoRegistry(db_path)
            entry = registry.upsert_repo(
                "C:/tmp/example",
                status="indexed",
                active=True,
                index_profile="cpu_e5",
                include_globs=["**/*"],
                doc_prefixes=["docs/"],
                exclude_globs=[".git/**"],
                indexed_branch="main",
                indexed_commit_hash="abc123",
                code_points_count=3,
                docs_points_count=2,
            )

            self.assertEqual(entry.indexed_branch, "main")
            self.assertEqual(entry.indexed_commit_hash, "abc123")
            self.assertEqual(registry.get_active_repo().repo_root, entry.repo_root)

            connection = sqlite3.connect(db_path)
            try:
                columns = {row[1] for row in connection.execute("PRAGMA table_info(repo_registry)")}
            finally:
                connection.close()
            self.assertIn("indexed_branch", columns)
            self.assertIn("indexed_commit_hash", columns)

            connection = sqlite3.connect(db_path)
            try:
                lease_tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }
            finally:
                connection.close()
            self.assertIn("lifecycle_lease", lease_tables)
            self.assertIn("lifecycle_audit", lease_tables)

    def test_lifecycle_lease_blocks_concurrent_mutations_and_audits_release(self) -> None:
        with TemporaryDirectory() as tmp:
            registry = RepoRegistry(Path(tmp) / "registry.sqlite3")

            lease = registry.acquire_lifecycle_lease(
                "C:/repo",
                index_profile="cpu",
                backend_id="embedding/cpu",
                operation="reindex_paths",
                owner_id="test-owner",
            )
            with self.assertRaisesRegex(RuntimeError, "already active"):
                registry.acquire_lifecycle_lease(
                    "C:/repo",
                    index_profile="cpu",
                    backend_id="embedding/cpu",
                    operation="rebuild_index",
                    owner_id="other-owner",
                )

            active = registry.get_active_lifecycle_lease(
                "C:/repo",
                index_profile="cpu",
                backend_id="embedding/cpu",
            )
            self.assertIsNotNone(active)
            self.assertEqual(active.operation, "reindex_paths")

            registry.release_lifecycle_lease(lease, status="success", detail={"paths": 2})
            self.assertIsNone(
                registry.get_active_lifecycle_lease(
                    "C:/repo",
                    index_profile="cpu",
                    backend_id="embedding/cpu",
                )
            )
            audit = registry.list_lifecycle_audit(limit=1)[0]
            self.assertEqual(audit["operation"], "reindex_paths")
            self.assertEqual(audit["status"], "success")
            self.assertEqual(audit["detail"], {"paths": 2})

    def test_same_idempotency_key_marks_active_lifecycle_retry(self) -> None:
        with TemporaryDirectory() as tmp:
            registry = RepoRegistry(Path(tmp) / "registry.sqlite3")

            lease = registry.acquire_lifecycle_lease(
                "C:/repo",
                index_profile="cpu",
                backend_id="embedding/cpu",
                operation="start_watcher",
                owner_id="test-owner",
                idempotency_key="safe-start",
            )
            try:
                with self.assertRaises(LifecycleLeaseActiveError) as raised:
                    registry.acquire_lifecycle_lease(
                        "C:/repo",
                        index_profile="cpu",
                        backend_id="embedding/cpu",
                        operation="start_watcher",
                        owner_id="same-owner",
                        idempotency_key="safe-start",
                    )
                self.assertTrue(raised.exception.same_idempotency_key)
                self.assertEqual(raised.exception.existing_operation, "start_watcher")
            finally:
                registry.release_lifecycle_lease(lease, status="success")

    def test_redacted_idempotency_keys_compare_by_fingerprint(self) -> None:
        with TemporaryDirectory() as tmp:
            registry = RepoRegistry(Path(tmp) / "registry.sqlite3")

            lease = registry.acquire_lifecycle_lease(
                "C:/repo",
                index_profile="cpu",
                backend_id="embedding/cpu",
                operation="start_watcher",
                owner_id="test-owner",
                idempotency_key="token=a",
            )
            try:
                with self.assertRaises(LifecycleLeaseActiveError) as raised:
                    registry.acquire_lifecycle_lease(
                        "C:/repo",
                        index_profile="cpu",
                        backend_id="embedding/cpu",
                        operation="start_watcher",
                        owner_id="same-owner",
                        idempotency_key="token=b",
                    )
                self.assertFalse(raised.exception.same_idempotency_key)
            finally:
                registry.release_lifecycle_lease(lease, status="success")

    def test_expired_lifecycle_lease_is_recovered_on_next_acquire(self) -> None:
        with TemporaryDirectory() as tmp:
            registry = RepoRegistry(Path(tmp) / "registry.sqlite3")
            first = registry.acquire_lifecycle_lease(
                "C:/repo",
                index_profile="cpu",
                backend_id="embedding/cpu",
                operation="reindex_paths",
                owner_id="first-owner",
            )
            expired = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
            connection = sqlite3.connect(registry.db_path)
            try:
                connection.execute(
                    "UPDATE lifecycle_lease SET expires_at = ? WHERE lease_id = ?",
                    (expired, first.lease_id),
                )
                connection.commit()
            finally:
                connection.close()

            second = registry.acquire_lifecycle_lease(
                "C:/repo",
                index_profile="cpu",
                backend_id="embedding/cpu",
                operation="rebuild_index",
                owner_id="second-owner",
            )

            self.assertTrue(second.recovered_expired)
            registry.release_lifecycle_lease(first, status="success")
            audit_statuses = {event["operation"]: event["status"] for event in registry.list_lifecycle_audit()}
            self.assertEqual(audit_statuses["reindex_paths"], "expired_recovered")
            registry.release_lifecycle_lease(second, status="success")

    def test_lifecycle_lease_heartbeat_extends_expiry(self) -> None:
        with TemporaryDirectory() as tmp:
            registry = RepoRegistry(Path(tmp) / "registry.sqlite3")
            lease = registry.acquire_lifecycle_lease(
                "C:/repo",
                index_profile="cpu",
                backend_id="embedding/cpu",
                operation="rebuild_index",
                owner_id="owner",
                ttl_sec=1,
            )
            expired = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
            connection = sqlite3.connect(registry.db_path)
            try:
                connection.execute(
                    "UPDATE lifecycle_lease SET expires_at = ? WHERE lease_id = ?",
                    (expired, lease.lease_id),
                )
                connection.commit()
            finally:
                connection.close()

            self.assertTrue(registry.heartbeat_lifecycle_lease(lease, ttl_sec=60))
            self.assertIsNotNone(
                registry.get_active_lifecycle_lease(
                    "C:/repo",
                    index_profile="cpu",
                    backend_id="embedding/cpu",
                )
            )
            registry.release_lifecycle_lease(lease, status="success")

    def test_lifecycle_audit_redacts_sensitive_detail_and_idempotency(self) -> None:
        with TemporaryDirectory() as tmp:
            registry = RepoRegistry(Path(tmp) / "registry.sqlite3")
            lease = registry.acquire_lifecycle_lease(
                "C:/repo",
                index_profile="cpu",
                backend_id="embedding/cpu",
                operation="reindex_paths",
                owner_id="owner",
                idempotency_key="token=abc123",
            )
            registry.release_lifecycle_lease(
                lease,
                status="error",
                detail={"api_key": "sk-test", "nested": {"password": "pw"}},
                error="secret=boom",
            )

            event = registry.list_lifecycle_audit(limit=1)[0]
            self.assertEqual(event["idempotency_key"], "<redacted>")
            self.assertIsInstance(event["idempotency_fingerprint"], str)
            self.assertEqual(event["detail"]["api_key"], "<redacted>")
            self.assertEqual(event["detail"]["nested"]["password"], "<redacted>")
            self.assertEqual(event["error"], "<redacted>")


if __name__ == "__main__":
    unittest.main()
