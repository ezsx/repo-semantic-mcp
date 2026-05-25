from __future__ import annotations

from types import SimpleNamespace
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from services.repo_semantic.lifecycle.index_ops import (
    rebuild_current_index,
    reindex_current_paths,
    update_include_globs_and_rebuild,
)
from services.repo_semantic.registry import RepoRegistry


class FakeStatus:
    def model_dump(self) -> dict[str, object]:
        return {"search_available": True}


class FakeSearchService:
    def __init__(self) -> None:
        self.sync_calls: list[dict[str, object]] = []
        self.invalidated = False

    def sync_registry_state(self, **kwargs: object) -> None:
        self.sync_calls.append(kwargs)

    def invalidate_cache(self) -> None:
        self.invalidated = True

    def index_status(self) -> FakeStatus:
        return FakeStatus()


class FakeIndexer:
    def __init__(self) -> None:
        self._settings = SimpleNamespace(
            logical_repo_identity="C:/repo",
            embedding_backend_id="embedding/cpu",
            effective_include_globs=["src/**"],
            update_globs=self.update_globs,
        )
        self.reindex_input: list[str] | None = None

    def rebuild_index(self) -> dict[str, int]:
        return {"code": 2, "docs": 1}

    def reindex_paths(self, paths: list[str]) -> dict[str, int]:
        self.reindex_input = paths
        return {"code": 1, "docs": 0}

    def update_globs(self, globs: list[str]) -> None:
        self._settings.effective_include_globs = list(globs)


class FakeWatcher:
    def __init__(self) -> None:
        self.is_running = True
        self.stop_called = False

    def stop(self) -> None:
        self.stop_called = True
        self.is_running = False


class FakeEmbeddingProvider:
    def index_profile(self) -> str:
        return "cpu"


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(
        watcher=FakeWatcher(),
        indexer=FakeIndexer(),
        search_service=FakeSearchService(),
        registry=None,
        embedding_provider=FakeEmbeddingProvider(),
    )


class IndexOpsTests(unittest.TestCase):
    def test_rebuild_current_index_stops_watcher_and_records_revision(self) -> None:
        runtime = _runtime()

        payload = rebuild_current_index(runtime)

        self.assertTrue(runtime.watcher.stop_called)
        self.assertTrue(runtime.search_service.invalidated)
        self.assertEqual(payload["rebuild"], {"code": 2, "docs": 1})
        self.assertEqual(
            runtime.search_service.sync_calls,
            [
                {"status": "indexing", "watch_running": False},
                {"status": "indexed", "record_index_revision": True},
            ],
        )

    def test_reindex_current_paths_uses_current_state_callback(self) -> None:
        runtime = _runtime()

        payload = reindex_current_paths(
            runtime,
            paths=["src/app.py"],
            current_runtime_repo_state=lambda _runtime: "indexed",
        )

        self.assertEqual(payload, {"code": 1, "docs": 0})
        self.assertEqual(runtime.indexer.reindex_input, ["src/app.py"])
        self.assertEqual(
            runtime.search_service.sync_calls[-1],
            {"status": "indexed", "record_index_revision": True},
        )

    def test_update_include_globs_returns_old_and_new_globs(self) -> None:
        runtime = _runtime()

        payload = update_include_globs_and_rebuild(runtime, globs=["services/**"])

        self.assertEqual(payload["old_globs"], ["src/**"])
        self.assertEqual(payload["new_globs"], ["services/**"])
        self.assertEqual(payload["rebuild"], {"code": 2, "docs": 1})
        self.assertEqual(runtime.search_service.sync_calls[0], {"status": "indexing"})

    def test_reindex_paths_is_blocked_by_active_lifecycle_lease(self) -> None:
        with TemporaryDirectory() as tmp:
            runtime = _runtime()
            runtime.registry = RepoRegistry(Path(tmp) / "registry.sqlite3")
            runtime.registry.acquire_lifecycle_lease(
                "C:/repo",
                index_profile="cpu",
                backend_id="embedding/cpu",
                operation="rebuild_index",
                owner_id="other-owner",
            )

            with self.assertRaisesRegex(RuntimeError, "already active"):
                reindex_current_paths(
                    runtime,
                    paths=["src/app.py"],
                    current_runtime_repo_state=lambda _runtime: "indexed",
                )

            self.assertIsNone(runtime.indexer.reindex_input)

    def test_rebuild_index_releases_lifecycle_lease_and_audits_success(self) -> None:
        with TemporaryDirectory() as tmp:
            runtime = _runtime()
            runtime.registry = RepoRegistry(Path(tmp) / "registry.sqlite3")

            rebuild_current_index(runtime)

            self.assertIsNone(
                runtime.registry.get_active_lifecycle_lease(
                    "C:/repo",
                    index_profile="cpu",
                    backend_id="embedding/cpu",
                )
            )
            audit = runtime.registry.list_lifecycle_audit(limit=1)[0]
            self.assertEqual(audit["operation"], "rebuild_index")
            self.assertEqual(audit["status"], "success")


if __name__ == "__main__":
    unittest.main()
