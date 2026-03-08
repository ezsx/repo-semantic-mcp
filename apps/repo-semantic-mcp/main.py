import os
import sys
import time

# PYTHONPATH включает /repo и /repo/libs как у остальных entrypoint'ов
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from services.repo_semantic.config import SemanticMcpSettings
from services.repo_semantic.embeddings import build_embedding_provider
from services.repo_semantic.indexer import RepositoryIndexer
from services.repo_semantic.logging import jlog
from services.repo_semantic.mcp_server import AppRuntime, configure_runtime, mcp
from services.repo_semantic.qdrant_store import QdrantStore
from services.repo_semantic.search_service import SearchService
from services.repo_semantic.watcher import RepositoryWatcher


def _wait_for_dependencies(store, embedding_provider, attempts: int = 90, delay_sec: int = 2) -> None:
    """Подождать готовность Qdrant и embedding backend.

    После старта Docker Desktop или после ребута машины контейнеры могут подняться
    не одновременно. Явный retry делает semantic MCP устойчивым к обычной гонке
    startup order и позволяет rely на `restart: unless-stopped`.
    """

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            store.healthcheck()
            embedding_provider.healthcheck()
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            jlog(
                "info",
                "semantic_dependency_wait_retry",
                attempt=attempt,
                attempts=attempts,
                error=str(exc),
            )
            time.sleep(delay_sec)

    raise RuntimeError(
        f"Semantic MCP dependencies did not become ready after {attempts} attempts"
    ) from last_error


def main() -> None:
    """Запустить semantic MCP server и, при необходимости, индексатор."""

    settings = SemanticMcpSettings()
    embedding_provider = build_embedding_provider(settings)
    store = QdrantStore(settings)
    indexer = RepositoryIndexer(
        settings=settings,
        embedding_provider=embedding_provider,
        store=store,
    )
    watcher = (
        RepositoryWatcher(indexer=indexer, debounce_sec=settings.SEMANTIC_MCP_WATCH_DEBOUNCE_SEC)
        if settings.SEMANTIC_MCP_WATCH_ENABLED
        else None
    )
    search_service = SearchService(
        settings=settings,
        embedding_provider=embedding_provider,
        store=store,
        indexer=indexer,
        watcher=watcher,
    )

    configure_runtime(
        AppRuntime(
            search_service=search_service,
            indexer=indexer,
            watcher=watcher,
        )
    )

    _wait_for_dependencies(store=store, embedding_provider=embedding_provider)

    total_points = sum(store.count(scope) for scope in ("code", "docs"))
    if settings.SEMANTIC_MCP_AUTO_INDEX_ON_START and total_points == 0:
        jlog("info", "semantic_auto_index_start")
        indexer.rebuild_index()
        search_service.invalidate_cache()

    if watcher:
        watcher.start()
        jlog("info", "semantic_watcher_started")

    transport = settings.SEMANTIC_MCP_TRANSPORT.strip().lower()
    if transport in {"http", "streamable-http", "streamable_http"}:
        mcp.settings.host = settings.SEMANTIC_MCP_HTTP_HOST
        mcp.settings.port = settings.SEMANTIC_MCP_HTTP_PORT
    jlog(
        "info",
        "semantic_mcp_starting",
        transport=transport,
        qdrant_url=settings.SEMANTIC_MCP_QDRANT_URL,
        embedding_backend=settings.SEMANTIC_MCP_EMBEDDING_BACKEND,
        embedding_model=settings.SEMANTIC_MCP_EMBEDDING_MODEL,
    )

    try:
        if transport in {"http", "streamable-http", "streamable_http"}:
            mcp.run(transport="streamable-http")
        else:
            mcp.run()
    finally:
        if watcher:
            watcher.stop()
            jlog("info", "semantic_watcher_stopped")


if __name__ == "__main__":
    main()
