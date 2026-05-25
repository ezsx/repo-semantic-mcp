from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from tempfile import TemporaryDirectory
import time

from services.repo_semantic.registry import RepoRegistry
from services.repo_semantic.lifecycle.watcher_ops import (
    start_watcher_for_runtime,
    stop_watcher_for_runtime,
)
from services.repo_semantic.watcher import RepositoryWatcher


class FakeStatus:
    repo_state = "indexed"

    def model_dump(self) -> dict[str, object]:
        return {"repo_state": self.repo_state}


class FakeSearchService:
    def __init__(self) -> None:
        self.ensure_called = False
        self.sync_calls: list[dict[str, object]] = []

    def ensure_search_available(self) -> None:
        self.ensure_called = True

    def sync_registry_state(self, **kwargs: object) -> None:
        self.sync_calls.append(kwargs)

    def index_status(self) -> FakeStatus:
        return FakeStatus()


class FakeWatcher:
    def __init__(self) -> None:
        self.is_running = False
        self.reconcile_called = False
        self.capture_called = False
        self.start_initial_snapshot = None

    def start(self, initial_snapshot=None) -> None:
        self.start_initial_snapshot = initial_snapshot
        self.is_running = True

    def stop(self) -> None:
        self.is_running = False

    def capture_snapshot(self, **kwargs) -> dict[str, int]:
        self.capture_called = True
        return {"before.py": 1}

    def run_startup_reconcile(self) -> dict[str, int]:
        self.reconcile_called = True
        return {"paths": 2, "code": 1, "docs": 1}


class BoundsExceededWatcher(FakeWatcher):
    def run_startup_reconcile(self) -> dict[str, object]:
        self.reconcile_called = True
        return {
            "paths": 251,
            "code": 0,
            "docs": 0,
            "paths_scanned": 1000,
            "paths_reindexed": 0,
            "paths_deleted": 0,
            "max_paths_reindexed": 250,
            "bounds_exceeded": ["max_reconcile_paths"],
            "mutation_started": False,
            "warning_codes": ["startup_reconcile_bounds_exceeded"],
        }


class FakeIndexer:
    def __init__(self, repo_root: Path, paths: list[Path]) -> None:
        self._settings = SimpleNamespace(
            repo_root=repo_root,
            logical_repo_identity=str(repo_root),
            embedding_backend_id="embedding/cpu",
            SEMANTIC_MCP_RECONCILE_MAX_SCAN_PATHS=50000,
            SEMANTIC_MCP_RECONCILE_MAX_DURATION_MS=180000,
        )
        self._paths = paths

    def iter_indexable_paths(self):
        return iter(self._paths)


class MutableFakeIndexer:
    def __init__(self, repo_root: Path) -> None:
        self._settings = SimpleNamespace(repo_root=repo_root)
        self.reindexed_paths: list[str] = []

    def iter_indexable_paths(self):
        return self._settings.repo_root.glob("*.py")

    def reindex_paths(self, paths: list[str]) -> dict[str, int]:
        self.reindexed_paths.extend(paths)
        return {"code": len(paths), "docs": 0}


class BoundsReconcileIndexer:
    def reconcile_index(self) -> dict[str, object]:
        return {
            "paths": 251,
            "code": 0,
            "docs": 0,
            "bounds_exceeded": ["max_reconcile_paths"],
            "mutation_started": False,
        }


class TouchedReconcileIndexer:
    def reconcile_index(self) -> dict[str, object]:
        return {
            "paths": 2,
            "code": 1,
            "docs": 1,
            "touched_paths": ["app.py", "docs/readme.md"],
            "mutation_started": True,
        }


class CountingContext:
    def __init__(self) -> None:
        self.enters = 0

    def __call__(self):
        return self

    def __enter__(self):
        self.enters += 1

    def __exit__(self, exc_type, exc, traceback):
        return None


class FailOnceContext:
    def __init__(self) -> None:
        self.enters = 0

    def __call__(self):
        return self

    def __enter__(self):
        self.enters += 1
        if self.enters == 1:
            raise RuntimeError("Lifecycle mutation lease is already active")
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None


class ActiveContext:
    def __init__(self) -> None:
        self.active = False

    def __call__(self):
        return self

    def __enter__(self):
        self.active = True
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.active = False
        return None


class FakeEmbeddingProvider:
    def index_profile(self) -> str:
        return "cpu"


class WatcherOpsTests(unittest.TestCase):
    def test_start_watcher_for_runtime_checks_search_and_syncs_state(self) -> None:
        search_service = FakeSearchService()
        watcher = FakeWatcher()
        runtime = SimpleNamespace(watcher=watcher, search_service=search_service)

        payload = start_watcher_for_runtime(runtime)

        self.assertTrue(search_service.ensure_called)
        self.assertTrue(watcher.capture_called)
        self.assertTrue(watcher.reconcile_called)
        self.assertEqual(watcher.start_initial_snapshot, {"before.py": 1})
        self.assertTrue(payload["watch_running"])
        self.assertEqual(payload["startup_reconcile"], {"paths": 2, "code": 1, "docs": 1})
        self.assertEqual(
            search_service.sync_calls,
            [{"status": "indexed", "watch_running": True, "record_index_revision": True}],
        )

    def test_start_watcher_for_runtime_is_idempotent_when_already_running(self) -> None:
        search_service = FakeSearchService()
        watcher = FakeWatcher()
        watcher.is_running = True
        runtime = SimpleNamespace(watcher=watcher, search_service=search_service)

        payload = start_watcher_for_runtime(runtime)

        self.assertTrue(search_service.ensure_called)
        self.assertFalse(watcher.capture_called)
        self.assertFalse(watcher.reconcile_called)
        self.assertTrue(payload["watch_running"])
        self.assertEqual(payload["startup_reconcile"]["skipped"], True)
        self.assertEqual(
            search_service.sync_calls,
            [{"status": "indexed", "watch_running": True}],
        )

    def test_start_watcher_already_running_respects_active_lifecycle_lease(self) -> None:
        with TemporaryDirectory() as tmp:
            search_service = FakeSearchService()
            watcher = FakeWatcher()
            watcher.is_running = True
            repo_root = Path(tmp) / "repo"
            runtime = SimpleNamespace(
                watcher=watcher,
                search_service=search_service,
                indexer=FakeIndexer(repo_root, []),
                registry=RepoRegistry(Path(tmp) / "registry.sqlite3"),
                embedding_provider=FakeEmbeddingProvider(),
            )
            active_lease = runtime.registry.acquire_lifecycle_lease(
                str(repo_root),
                index_profile="cpu",
                backend_id="embedding/cpu",
                operation="rebuild_index",
            )

            try:
                with self.assertRaises(RuntimeError):
                    start_watcher_for_runtime(runtime)
            finally:
                runtime.registry.release_lifecycle_lease(active_lease, status="success")

            self.assertEqual(search_service.sync_calls, [])

    def test_start_watcher_same_idempotency_key_returns_retry_payload(self) -> None:
        with TemporaryDirectory() as tmp:
            search_service = FakeSearchService()
            watcher = FakeWatcher()
            repo_root = Path(tmp) / "repo"
            runtime = SimpleNamespace(
                watcher=watcher,
                search_service=search_service,
                indexer=FakeIndexer(repo_root, []),
                registry=RepoRegistry(Path(tmp) / "registry.sqlite3"),
                embedding_provider=FakeEmbeddingProvider(),
            )
            active_lease = runtime.registry.acquire_lifecycle_lease(
                str(repo_root),
                index_profile="cpu",
                backend_id="embedding/cpu",
                operation="start_watcher",
                idempotency_key="safe-start",
            )

            try:
                payload = start_watcher_for_runtime(runtime, idempotency_key="safe-start")
            finally:
                runtime.registry.release_lifecycle_lease(active_lease, status="success")

            self.assertFalse(payload["watch_running"])
            self.assertTrue(payload["idempotent_retry"])
            self.assertTrue(payload["operation_in_progress"])
            self.assertEqual(payload["startup_reconcile"]["reason"], "idempotent_retry_in_progress")
            self.assertFalse(watcher.capture_called)
            self.assertFalse(watcher.reconcile_called)
            self.assertEqual(search_service.sync_calls, [])

    def test_stop_watcher_for_runtime_handles_disabled_watcher(self) -> None:
        runtime = SimpleNamespace(watcher=None, search_service=FakeSearchService())

        payload = stop_watcher_for_runtime(runtime)

        self.assertFalse(payload["watch_running"])

    def test_stop_watcher_for_runtime_syncs_repo_state(self) -> None:
        search_service = FakeSearchService()
        watcher = FakeWatcher()
        watcher.start()
        runtime = SimpleNamespace(watcher=watcher, search_service=search_service)

        payload = stop_watcher_for_runtime(runtime)

        self.assertFalse(payload["watch_running"])
        self.assertEqual(search_service.sync_calls, [{"status": "indexed", "watch_running": False}])

    def test_repository_watcher_snapshot_tolerates_deleted_files(self) -> None:
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            existing = repo_root / "existing.py"
            missing = repo_root / "missing.py"
            existing.write_text("print('ok')\n", encoding="utf-8")
            indexer = FakeIndexer(repo_root, [existing, missing])
            watcher = RepositoryWatcher(indexer)

            snapshot = watcher.capture_snapshot()

            self.assertIn("existing.py", snapshot)
            self.assertNotIn("missing.py", snapshot)

    def test_start_watcher_uses_lifecycle_lease_for_startup_reconcile(self) -> None:
        with TemporaryDirectory() as tmp:
            search_service = FakeSearchService()
            watcher = FakeWatcher()
            indexer = FakeIndexer(Path(tmp) / "repo", [])
            runtime = SimpleNamespace(
                watcher=watcher,
                search_service=search_service,
                indexer=indexer,
                registry=RepoRegistry(Path(tmp) / "registry.sqlite3"),
                embedding_provider=FakeEmbeddingProvider(),
            )

            payload = start_watcher_for_runtime(runtime, idempotency_key="safe-watcher-start")

            self.assertTrue(payload["watch_running"])
            self.assertIsNone(
                runtime.registry.get_active_lifecycle_lease(
                    str(Path(tmp) / "repo"),
                    index_profile="cpu",
                    backend_id="embedding/cpu",
                )
            )
            audit = runtime.registry.list_lifecycle_audit(limit=1)[0]
            self.assertEqual(audit["operation"], "start_watcher")
            self.assertEqual(audit["status"], "success")
            self.assertEqual(audit["idempotency_key"], "safe-watcher-start")

    def test_start_watcher_stops_before_live_polling_when_reconcile_bounds_exceeded(self) -> None:
        with TemporaryDirectory() as tmp:
            search_service = FakeSearchService()
            watcher = BoundsExceededWatcher()
            indexer = FakeIndexer(Path(tmp) / "repo", [])
            runtime = SimpleNamespace(
                watcher=watcher,
                search_service=search_service,
                indexer=indexer,
                registry=RepoRegistry(Path(tmp) / "registry.sqlite3"),
                embedding_provider=FakeEmbeddingProvider(),
            )

            payload = start_watcher_for_runtime(runtime)

            self.assertTrue(watcher.capture_called)
            self.assertTrue(watcher.reconcile_called)
            self.assertFalse(watcher.is_running)
            self.assertFalse(payload["watch_running"])
            self.assertTrue(payload["permission_required"])
            self.assertEqual(
                payload["recommended_actions"][0]["code"],
                "startup_reconcile_bounds_exceeded",
            )
            self.assertEqual(search_service.sync_calls, [])

    def test_start_watcher_stops_before_reconcile_when_snapshot_scan_bound_exceeded(self) -> None:
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            first = repo_root / "first.py"
            second = repo_root / "second.py"
            first.write_text("print('first')\n", encoding="utf-8")
            second.write_text("print('second')\n", encoding="utf-8")
            search_service = FakeSearchService()
            indexer = FakeIndexer(repo_root, [first, second])
            indexer._settings.SEMANTIC_MCP_RECONCILE_MAX_SCAN_PATHS = 1
            runtime = SimpleNamespace(
                watcher=RepositoryWatcher(indexer),
                search_service=search_service,
                indexer=indexer,
                registry=RepoRegistry(Path(tmp) / "registry.sqlite3"),
                embedding_provider=FakeEmbeddingProvider(),
            )

            payload = start_watcher_for_runtime(runtime)

            self.assertFalse(payload["watch_running"])
            self.assertTrue(payload["permission_required"])
            self.assertEqual(payload["startup_reconcile"]["bounds_exceeded"], ["max_scan_paths"])
            self.assertEqual(
                payload["recommended_actions"][0]["code"],
                "startup_reconcile_bounds_exceeded",
            )
            self.assertFalse(runtime.watcher.is_running)
            self.assertEqual(search_service.sync_calls, [])

    def test_startup_reconcile_does_not_emit_index_changed_when_bounds_exceeded(self) -> None:
        changed_calls: list[bool] = []
        watcher = RepositoryWatcher(
            BoundsReconcileIndexer(),
            on_index_changed=lambda: changed_calls.append(True),
        )

        result = watcher.run_startup_reconcile()

        self.assertEqual(result["bounds_exceeded"], ["max_reconcile_paths"])
        self.assertEqual(changed_calls, [])

    def test_startup_reconcile_passes_touched_paths_to_index_changed_callback(self) -> None:
        changed_calls: list[tuple[list[str], bool]] = []
        watcher = RepositoryWatcher(
            TouchedReconcileIndexer(),
            on_index_changed=lambda paths, startup: changed_calls.append((list(paths), startup)),
        )

        result = watcher.run_startup_reconcile()

        self.assertEqual(result["paths"], 2)
        self.assertEqual(changed_calls, [(["app.py", "docs/readme.md"], True)])

    def test_running_watcher_uses_reindex_context_for_live_changes(self) -> None:
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            app = repo_root / "app.py"
            app.write_text("print('v1')\n", encoding="utf-8")
            indexer = MutableFakeIndexer(repo_root)
            reindex_context = CountingContext()
            watcher = RepositoryWatcher(
                indexer,
                debounce_sec=0.05,
                reindex_context=reindex_context,
            )
            watcher.capture_snapshot()

            watcher.start()
            try:
                time.sleep(0.08)
                app.write_text("print('v2')\n", encoding="utf-8")
                deadline = time.monotonic() + 2.0
                while not indexer.reindexed_paths and time.monotonic() < deadline:
                    time.sleep(0.05)
            finally:
                watcher.stop()

            self.assertIn("app.py", indexer.reindexed_paths)
            self.assertGreaterEqual(reindex_context.enters, 1)

    def test_running_watcher_updates_registry_inside_reindex_context(self) -> None:
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            app = repo_root / "app.py"
            app.write_text("print('v1')\n", encoding="utf-8")
            indexer = MutableFakeIndexer(repo_root)
            reindex_context = ActiveContext()
            callback_active_states: list[bool] = []
            watcher = RepositoryWatcher(
                indexer,
                debounce_sec=0.05,
                on_index_changed=lambda: callback_active_states.append(reindex_context.active),
                reindex_context=reindex_context,
            )
            watcher.capture_snapshot()

            watcher.start()
            try:
                time.sleep(0.08)
                app.write_text("print('v2')\n", encoding="utf-8")
                deadline = time.monotonic() + 2.0
                while not callback_active_states and time.monotonic() < deadline:
                    time.sleep(0.05)
            finally:
                watcher.stop()

            self.assertEqual(callback_active_states, [True])

    def test_running_watcher_passes_live_touched_paths_to_callback(self) -> None:
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            app = repo_root / "app.py"
            app.write_text("print('v1')\n", encoding="utf-8")
            indexer = MutableFakeIndexer(repo_root)
            changed_calls: list[tuple[list[str], bool]] = []
            watcher = RepositoryWatcher(
                indexer,
                debounce_sec=0.05,
                on_index_changed=lambda paths, startup: changed_calls.append((list(paths), startup)),
            )
            watcher.capture_snapshot()

            watcher.start()
            try:
                time.sleep(0.08)
                app.write_text("print('v2')\n", encoding="utf-8")
                deadline = time.monotonic() + 2.0
                while not changed_calls and time.monotonic() < deadline:
                    time.sleep(0.05)
            finally:
                watcher.stop()

            self.assertEqual(changed_calls, [(["app.py"], False)])

    def test_running_watcher_retries_changes_after_busy_lifecycle_lease(self) -> None:
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            app = repo_root / "app.py"
            app.write_text("print('v1')\n", encoding="utf-8")
            indexer = MutableFakeIndexer(repo_root)
            reindex_context = FailOnceContext()
            watcher = RepositoryWatcher(
                indexer,
                debounce_sec=0.05,
                reindex_context=reindex_context,
            )
            watcher.capture_snapshot()

            watcher.start()
            try:
                time.sleep(0.08)
                app.write_text("print('v2')\n", encoding="utf-8")
                deadline = time.monotonic() + 2.0
                while len(indexer.reindexed_paths) < 1 and time.monotonic() < deadline:
                    time.sleep(0.05)
            finally:
                watcher.stop()

            self.assertGreaterEqual(reindex_context.enters, 2)
            self.assertIn("app.py", indexer.reindexed_paths)


if __name__ == "__main__":
    unittest.main()
