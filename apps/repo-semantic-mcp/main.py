import os
import sys
import threading
import time
from types import SimpleNamespace

# PYTHONPATH включает /repo и /repo/libs как у остальных entrypoint'ов
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from services.repo_semantic.config import SemanticMcpSettings
from services.repo_semantic.embeddings import build_embedding_provider
from services.repo_semantic.graph import GraphService
from services.repo_semantic.indexer import RepositoryIndexer
from services.repo_semantic.lifecycle.lease import lifecycle_global_mutation, lifecycle_mutation_for_identity
from services.repo_semantic.logging import jlog
from services.repo_semantic.mcp_server import (
    AppRuntime,
    configure_runtime,
    mark_bootstrap_error,
    mark_bootstrap_ready,
    mcp,
    set_bootstrap_phase,
)
from services.repo_semantic.qdrant_store import QdrantStore
from services.repo_semantic.registry import RepoRegistry
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

    qdrant_url = getattr(getattr(store, "_settings", None), "SEMANTIC_MCP_QDRANT_URL", "unknown")
    embedding_url = embedding_provider.endpoint_label()
    error_suffix = f"; last_error={last_error}" if last_error else ""
    raise RuntimeError(
        "Semantic MCP dependencies did not become ready after "
        f"{attempts} attempts (qdrant={qdrant_url}; embedding={embedding_url}{error_suffix})"
    ) from last_error


def _bootstrap_runtime(
    settings: SemanticMcpSettings,
    store: QdrantStore,
    embedding_provider,
    indexer: RepositoryIndexer,
    search_service: SearchService,
    watcher: RepositoryWatcher | None,
    registry: RepoRegistry,
) -> None:
    """Довести runtime до ready-state без скрытой индексации на connect."""

    def _repo_lifecycle(operation: str):
        return lifecycle_mutation_for_identity(
            registry=registry,
            repo_root=settings.logical_repo_identity,
            index_profile=embedding_provider.index_profile(),
            backend_id=settings.embedding_backend_id,
            operation=operation,
        )

    try:
        set_bootstrap_phase("waiting_for_dependencies")
        _wait_for_dependencies(
            store=store,
            embedding_provider=embedding_provider,
            attempts=settings.SEMANTIC_MCP_DEPENDENCY_WAIT_ATTEMPTS,
            delay_sec=settings.SEMANTIC_MCP_DEPENDENCY_WAIT_DELAY_SEC,
        )

        set_bootstrap_phase("loading_runtime_status")
        final_points = sum(store.count(scope) for scope in ("code", "docs"))
        contract_issue = search_service.current_index_contract_issue()
        with _repo_lifecycle("bootstrap_status"):
            existing_entry = registry.get_repo(settings.logical_repo_identity)
            if existing_entry is not None:
                settings.apply_repo_registry_config(
                    include_globs=existing_entry.include_globs,
                    doc_prefixes=existing_entry.doc_prefixes,
                )
            if existing_entry is None:
                repo_state = "stale" if final_points > 0 and contract_issue else ("indexed" if final_points > 0 else "registered")
            elif final_points > 0 and existing_entry.status in {"registered", "error"}:
                repo_state = "stale" if contract_issue else "indexed"
            elif final_points > 0 and existing_entry.status == "indexed" and contract_issue:
                repo_state = "stale"
            else:
                repo_state = existing_entry.status
            search_service.sync_registry_state(
                status=repo_state,
                active=existing_entry.active if existing_entry is not None else False,
                last_error=contract_issue or (existing_entry.last_error if existing_entry is not None else None),
                watch_running=False,
            )
        current_status = search_service.index_status()
        mark_bootstrap_ready(
            repo_root=settings.logical_repo_identity,
            mounted_repo_root=str(settings.repo_root),
            profile=settings.SEMANTIC_MCP_PROFILE_NAME,
            total_points=final_points,
            repo_state=current_status.repo_state,
            active=current_status.active,
            active_repo_root=current_status.active_repo_root,
            runtime_switch_required=current_status.runtime_switch_required,
            search_available=current_status.search_available,
            watch_enabled=bool(watcher),
        )
        jlog(
            "info",
            "semantic_bootstrap_ready",
            repo_root=settings.logical_repo_identity,
            mounted_repo_root=str(settings.repo_root),
            profile=settings.SEMANTIC_MCP_PROFILE_NAME,
            total_points=final_points,
            repo_state=current_status.repo_state,
            active=current_status.active,
            active_repo_root=current_status.active_repo_root,
            runtime_switch_required=current_status.runtime_switch_required,
            search_available=current_status.search_available,
            watch_enabled=bool(watcher),
        )
    except Exception as exc:  # noqa: BLE001
        try:
            with _repo_lifecycle("bootstrap_error"):
                existing_entry = registry.get_repo(settings.logical_repo_identity)
                search_service.sync_registry_state(
                    status="error",
                    last_error=str(exc),
                    active=existing_entry.active if existing_entry is not None else False,
                    watch_running=False,
                )
        except Exception:  # noqa: BLE001
            pass
        mark_bootstrap_error(
            str(exc),
            phase="bootstrap_failed",
            repo_root=settings.logical_repo_identity,
            mounted_repo_root=str(settings.repo_root),
            profile=settings.SEMANTIC_MCP_PROFILE_NAME,
        )
        jlog(
            "error",
            "semantic_bootstrap_failed",
            repo_root=settings.logical_repo_identity,
            mounted_repo_root=str(settings.repo_root),
            profile=settings.SEMANTIC_MCP_PROFILE_NAME,
            error=str(exc),
        )


def _backend_entries_equivalent(existing_entry, current_payload: dict[str, object]) -> bool:
    """Проверить, что backend entries совместимы по текущему runtime/backend contract."""

    return (
        existing_entry.role == current_payload["role"]
        and existing_entry.backend_type == current_payload["backend_type"]
        and existing_entry.transport == current_payload["transport"]
        and existing_entry.endpoint == current_payload["endpoint"]
        and existing_entry.health_path == current_payload["health_path"]
        and existing_entry.model_name == current_payload["model_name"]
    )


def _is_legacy_backend_alias(backend_id: str) -> bool:
    """Определить backend id старого auto-generated формата до backend catalog."""

    return backend_id.startswith("embedding/tei_http_") or backend_id.startswith("embedding/fastembed_local_")


def main() -> None:
    """Запустить semantic MCP server и, при необходимости, индексатор."""

    settings = SemanticMcpSettings()
    embedding_provider = build_embedding_provider(settings)
    registry = RepoRegistry(settings.registry_db_path)
    current_backend_payload = settings.current_embedding_backend_payload()
    backend_payloads = [current_backend_payload, *settings.current_optional_backend_payloads()]
    registry_runtime = SimpleNamespace(registry=registry)
    with lifecycle_global_mutation(registry_runtime, "bootstrap_backends", scope="backend_roles"):
        for payload in backend_payloads:
            registry.upsert_backend(**payload)
            selected_backend = registry.get_role_backend(payload["role"])
            if selected_backend is None:
                registry.set_role_backend(payload["role"], payload["backend_id"])
            elif (
                selected_backend.backend_id != payload["backend_id"]
                and _backend_entries_equivalent(selected_backend, payload)
            ):
                registry.set_role_backend(payload["role"], payload["backend_id"])
        for backend_entry in registry.list_backends(role="embedding"):
            if (
                backend_entry.backend_id != settings.embedding_backend_id
                and _is_legacy_backend_alias(backend_entry.backend_id)
                and _backend_entries_equivalent(backend_entry, current_backend_payload)
            ):
                registry.delete_backend(backend_entry.backend_id)
    with lifecycle_global_mutation(registry_runtime, "bootstrap_repo_registry", scope="repo_registry"):
        with lifecycle_mutation_for_identity(
            registry=registry,
            repo_root=settings.logical_repo_identity,
            index_profile=embedding_provider.index_profile(),
            backend_id=settings.embedding_backend_id,
            operation="bootstrap_repo_registry",
        ):
            existing_entry = registry.get_repo(settings.logical_repo_identity)
            if existing_entry is not None:
                settings.apply_repo_registry_config(
                    include_globs=existing_entry.include_globs,
                    doc_prefixes=existing_entry.doc_prefixes,
                )
            registry.ensure_registered(
                settings.logical_repo_identity,
                index_profile=embedding_provider.index_profile(),
                include_globs=list(settings.effective_include_globs),
                doc_prefixes=list(settings.effective_doc_prefixes),
                exclude_globs=list(settings.SEMANTIC_MCP_EXCLUDE_GLOBS),
                active=existing_entry.active if existing_entry is not None else False,
            )
            existing_entry = registry.get_repo(settings.logical_repo_identity)
    store = QdrantStore(settings)
    indexer = RepositoryIndexer(
        settings=settings,
        embedding_provider=embedding_provider,
        store=store,
    )
    graph_service = GraphService(
        settings=settings,
        chunk_store=store,
        registry=registry,
        path_manifest=indexer.path_manifest,
    )

    def _on_index_changed(paths: list[str] | None = None, startup: bool = False) -> None:
        search_service.sync_registry_state(
            status="indexed",
            watch_running=bool(watcher and watcher.is_running),
            record_index_revision=True,
        )
        graph_update_enabled = (
            settings.SEMANTIC_MCP_GRAPH_UPDATE_ON_STARTUP_RECONCILE
            if startup
            else settings.SEMANTIC_MCP_GRAPH_UPDATE_ON_WATCHER
        )
        if (
            graph_update_enabled
            and graph_service is not None
            and paths
        ):
            graph_service.update_graph(paths=paths)

    watcher = (
        RepositoryWatcher(
            indexer=indexer,
            debounce_sec=settings.SEMANTIC_MCP_WATCH_DEBOUNCE_SEC,
            on_index_changed=_on_index_changed,
            reindex_context=lambda: lifecycle_mutation_for_identity(
                registry=registry,
                repo_root=settings.logical_repo_identity,
                index_profile=embedding_provider.index_profile(),
                backend_id=settings.embedding_backend_id,
                operation="watcher_reindex",
            ),
        )
        if settings.SEMANTIC_MCP_WATCH_ENABLED
        else None
    )
    search_service = SearchService(
        settings=settings,
        embedding_provider=embedding_provider,
        store=store,
        indexer=indexer,
        watcher=watcher,
        registry=registry,
        graph_status_summary=graph_service.graph_summary,
        graph_read_provider=graph_service.read_provider(),
    )

    configure_runtime(
        AppRuntime(
            search_service=search_service,
            indexer=indexer,
            watcher=watcher,
            registry=registry,
            embedding_provider=embedding_provider,
            graph_service=graph_service,
        )
    )

    set_bootstrap_phase(
        "runtime_configured",
        repo_root=settings.logical_repo_identity,
        mounted_repo_root=str(settings.repo_root),
        profile=settings.SEMANTIC_MCP_PROFILE_NAME,
    )

    transport = settings.SEMANTIC_MCP_TRANSPORT.strip().lower()
    if transport in {"http", "streamable-http", "streamable_http"}:
        mcp.settings.host = settings.SEMANTIC_MCP_HTTP_HOST
        mcp.settings.port = settings.SEMANTIC_MCP_HTTP_PORT
    set_bootstrap_phase(
        "transport_binding",
        repo_root=settings.logical_repo_identity,
        mounted_repo_root=str(settings.repo_root),
        profile=settings.SEMANTIC_MCP_PROFILE_NAME,
        transport=transport,
    )
    threading.Thread(
        target=_bootstrap_runtime,
        name="semantic-bootstrap",
        daemon=True,
        kwargs={
            "settings": settings,
            "store": store,
            "embedding_provider": embedding_provider,
            "indexer": indexer,
            "search_service": search_service,
            "watcher": watcher,
            "registry": registry,
        },
    ).start()
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
        if watcher and watcher.is_running:
            watcher.stop()
            jlog("info", "semantic_watcher_stopped")


if __name__ == "__main__":
    main()
