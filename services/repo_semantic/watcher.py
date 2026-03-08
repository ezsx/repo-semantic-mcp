"""Polling watcher for incremental semantic reindex."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from services.repo_semantic.indexer import RepositoryIndexer
from services.repo_semantic.logging import jlog


class RepositoryWatcher:
    """Периодически отслеживать изменения файлов и запускать reindex."""

    def __init__(self, indexer: RepositoryIndexer, debounce_sec: int = 3) -> None:
        """Сохранить indexer и период опроса."""

        self._indexer = indexer
        self._debounce_sec = debounce_sec
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._snapshot: dict[str, int] = {}

    @property
    def is_running(self) -> bool:
        """Показать, запущен ли polling thread."""

        return bool(self._thread and self._thread.is_alive())

    def _build_snapshot(self) -> dict[str, int]:
        """Снять map path -> mtime_ns для индексируемых файлов."""

        snapshot: dict[str, int] = {}
        for path in self._indexer.iter_indexable_paths():
            relative_path = path.relative_to(self._indexer._settings.repo_root).as_posix()
            snapshot[relative_path] = path.stat().st_mtime_ns
        return snapshot

    def _watch_loop(self) -> None:
        """Фоновый polling loop с debounce."""

        self._snapshot = self._build_snapshot()
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
                    self._indexer.reindex_paths(touched)
                    jlog("info", "semantic_watcher_reindexed", paths=len(touched))
                except Exception as exc:  # noqa: BLE001
                    jlog("warning", "semantic_watcher_reindex_failed", error=str(exc))
            self._snapshot = current

    def start(self) -> None:
        """Запустить watcher, если он еще не работает."""

        if self.is_running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Остановить watcher и дождаться завершения thread."""

        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=max(self._debounce_sec, 1) + 1)
            self._thread = None

