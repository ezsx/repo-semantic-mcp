"""Polling watcher for incremental semantic reindex."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from services.repo_semantic.indexer import IndexablePathScanBoundsExceeded, RepositoryIndexer
from services.repo_semantic.logging import jlog


class StartupReconcileBoundsExceeded(RuntimeError):
    """Raised when startup watcher snapshot exceeds safe recovery bounds."""

    def __init__(self, summary: dict[str, object]) -> None:
        super().__init__("startup_reconcile_bounds_exceeded")
        self.summary = summary


class RepositoryWatcher:
    """Периодически отслеживать изменения файлов и запускать reindex."""

    def __init__(
        self,
        indexer: RepositoryIndexer,
        debounce_sec: int = 3,
        on_index_changed: Callable[..., None] | None = None,
        reindex_context: Callable[[], Any] | None = None,
    ) -> None:
        """Сохранить indexer и период опроса."""

        self._indexer = indexer
        self._debounce_sec = debounce_sec
        self._on_index_changed = on_index_changed
        self._reindex_context = reindex_context or (lambda: nullcontext(None))
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._snapshot: dict[str, int] = {}

    def _notify_index_changed(self, paths: list[str] | None = None, *, startup: bool = False) -> None:
        if self._on_index_changed is None:
            return
        try:
            self._on_index_changed(paths or [], startup)
        except TypeError:
            try:
                self._on_index_changed(paths or [])
            except TypeError:
                self._on_index_changed()

    @property
    def is_running(self) -> bool:
        """Показать, запущен ли polling thread."""

        return bool(self._thread and self._thread.is_alive())

    def _build_snapshot(
        self,
        *,
        max_paths: int | None = None,
        max_duration_ms: int | None = None,
    ) -> dict[str, int]:
        """Снять map path -> mtime_ns для индексируемых файлов."""

        started_at = time.monotonic()
        snapshot: dict[str, int] = {}
        try:
            path_iter = self._indexer.iter_indexable_paths(
                max_paths_scanned=max_paths,
                max_duration_ms=max_duration_ms,
            )
        except TypeError:
            path_iter = self._indexer.iter_indexable_paths()
        try:
            for path in path_iter:
                relative_path = path.relative_to(self._indexer._settings.repo_root).as_posix()
                try:
                    snapshot[relative_path] = path.stat().st_mtime_ns
                except FileNotFoundError:
                    # The polling watcher races normal editor/delete activity.
                    # Treat files deleted during snapshot collection as absent; the
                    # deleted-path diff against the previous snapshot will handle
                    # index cleanup on the next loop.
                    continue
                duration_ms = int((time.monotonic() - started_at) * 1000)
                bounds_exceeded: list[str] = []
                if max_paths is not None and len(snapshot) > max_paths:
                    bounds_exceeded.append("max_scan_paths")
                if max_duration_ms is not None and duration_ms > max_duration_ms:
                    bounds_exceeded.append("max_duration_ms")
                if bounds_exceeded:
                    raise StartupReconcileBoundsExceeded(
                        {
                            "paths": 0,
                            "code": 0,
                            "docs": 0,
                            "paths_scanned": len(snapshot),
                            "paths_reindexed": 0,
                            "paths_deleted": 0,
                            "points_deleted": None,
                            "max_paths_scanned": max_paths,
                            "max_duration_ms": max_duration_ms,
                            "bounds_exceeded": bounds_exceeded,
                            "mutation_started": False,
                            "duration_ms": duration_ms,
                            "warning_codes": ["startup_reconcile_bounds_exceeded"],
                        }
                    )
        except IndexablePathScanBoundsExceeded as exc:
            raise StartupReconcileBoundsExceeded(
                {
                    "paths": 0,
                    "code": 0,
                    "docs": 0,
                    "paths_scanned": exc.summary.get("paths_scanned", len(snapshot)),
                    "paths_reindexed": 0,
                    "paths_deleted": 0,
                    "points_deleted": None,
                    "max_paths_scanned": exc.summary.get("max_paths_scanned", max_paths),
                    "max_duration_ms": exc.summary.get("max_duration_ms", max_duration_ms),
                    "bounds_exceeded": list(exc.summary.get("bounds_exceeded") or []),
                    "mutation_started": False,
                    "duration_ms": exc.summary.get("duration_ms", 0),
                    "warning_codes": ["startup_reconcile_bounds_exceeded"],
                }
            ) from exc
        return snapshot

    def capture_snapshot(
        self,
        *,
        max_paths: int | None = None,
        max_duration_ms: int | None = None,
    ) -> dict[str, int]:
        """Снять snapshot и сохранить его как текущую baseline-карту watcher."""

        snapshot = self._build_snapshot(max_paths=max_paths, max_duration_ms=max_duration_ms)
        self._snapshot = snapshot
        return snapshot

    def _watch_loop(self) -> None:
        """Фоновый polling loop с debounce."""

        while not self._stop_event.wait(self._debounce_sec):
            current = self._build_snapshot()
            changed = [
                path
                for path, mtime in current.items()
                if self._snapshot.get(path) != mtime
            ]
            deleted = [path for path in self._snapshot if path not in current]
            touched = changed + deleted
            if touched:
                try:
                    with self._reindex_context():
                        self._indexer.reindex_paths(touched)
                        self._notify_index_changed(touched, startup=False)
                    jlog("info", "semantic_watcher_reindexed", paths=len(touched))
                except Exception as exc:  # noqa: BLE001
                    jlog("warning", "semantic_watcher_reindex_failed", error=str(exc))
                    continue
            self._snapshot = current

    def run_startup_reconcile(self) -> dict[str, object]:
        """Выполнить startup reconcile синхронно до объявления readiness."""

        reconcile_result = self._indexer.reconcile_index()
        if reconcile_result.get("paths", 0) > 0 and reconcile_result.get("mutation_started", True):
            paths = [
                str(path)
                for path in reconcile_result.get("touched_paths", [])
                if str(path)
            ]
            self._notify_index_changed(paths, startup=True)
            jlog(
                "info",
                "semantic_watcher_startup_reconciled",
                paths=reconcile_result.get("paths", 0),
                code_chunks=reconcile_result.get("code", 0),
                docs_chunks=reconcile_result.get("docs", 0),
            )
        return reconcile_result

    def start(self, initial_snapshot: dict[str, int] | None = None) -> None:
        """Запустить watcher, если он еще не работает."""

        if self.is_running:
            return
        self._stop_event.clear()
        self._snapshot = initial_snapshot if initial_snapshot is not None else self._build_snapshot()
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Остановить watcher и дождаться завершения thread."""

        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=max(self._debounce_sec, 1) + 1)
            self._thread = None
