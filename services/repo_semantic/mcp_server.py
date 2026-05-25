"""FastMCP facade for repo semantic search."""

from __future__ import annotations

import json
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass, field

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from services.repo_semantic.config import normalize_repo_identity
from services.repo_semantic.indexer import RepositoryIndexer
from services.repo_semantic.lifecycle.backend_ops import (
    backend_action_plan_payload,
    backend_status_payload,
    default_action_plan_repo_root,
    embedding_backend_switch_payload,
    resolve_backend_entry,
    runtime_embedding_backend_id,
)
from services.repo_semantic.lifecycle.index_ops import (
    rebuild_current_index,
    reindex_current_paths,
    update_include_globs_and_rebuild,
)
from services.repo_semantic.lifecycle.lease import lifecycle_global_mutation, lifecycle_mutation
from services.repo_semantic.lifecycle.repo_ops import (
    activate_repo_payload,
    current_runtime_repo_state,
    register_repo_payload,
    runtime_repo_root,
    runtime_switch_required_payload,
)
from services.repo_semantic.lifecycle.watcher_ops import (
    start_watcher_for_runtime,
    stop_watcher_for_runtime,
)
from services.repo_semantic.models import (
    BackendRole,
    GraphBuildResult,
    GraphMode,
    RepoContextRoute,
    SearchMode,
    SearchScope,
    SnippetMode,
)
from services.repo_semantic.registry import LifecycleLeaseActiveError, RepoRegistry
from services.repo_semantic.search_service import SearchService
from services.repo_semantic.watcher import RepositoryWatcher


@dataclass(slots=True)
class AppRuntime:
    """Runtime зависимости semantic MCP."""

    search_service: SearchService
    indexer: RepositoryIndexer
    watcher: RepositoryWatcher | None
    registry: RepoRegistry
    embedding_provider: object
    graph_service: object | None = None


@dataclass(slots=True)
class BootstrapState:
    """Host-visible bootstrap-состояние MCP транспорта."""

    phase: str = "starting"
    ready: bool = False
    error: str | None = None
    updated_at: float = field(default_factory=time.time)
    details: dict[str, object] = field(default_factory=dict)


_RUNTIME: AppRuntime | None = None
_BOOTSTRAP_LOCK = threading.Lock()
_BOOTSTRAP_STATE = BootstrapState()

mcp = FastMCP(
    name="repo-semantic-search",
    streamable_http_path="/mcp",
    stateless_http=True,
    instructions=(
        "Используй этот MCP для repo-wide semantic shortlist. "
        "Для docs-only и code-only поиска предпочитай явные tools соответствующей коллекции. "
        "Фильтруй результаты через path_prefix/include_paths/exclude_paths, file_extensions, languages, domain_tags и chunk_types. "
        "domain_tags строятся из пути: src/services/* → ['src','services'], docs/* → ['docs']. "
        "chunk_types для кода: class_summary, python_method, python_function, module_preamble; "
        "для структурных файлов: json_array_item, yaml_section, toml_section. "
        "update_include_globs(['auto']) — перестроить индекс с авто-детектом структуры репо. "
        "Индекс строится только явными lifecycle tools; подключение клиента не запускает rebuild."
    ),
)


def configure_runtime(runtime: AppRuntime) -> None:
    """Сохранить runtime singleton перед стартом MCP сервера."""

    global _RUNTIME
    _RUNTIME = runtime


def set_bootstrap_phase(phase: str, **details: object) -> None:
    """Обновить фазу bootstrap до состояния ready."""

    with _BOOTSTRAP_LOCK:
        _BOOTSTRAP_STATE.phase = phase
        _BOOTSTRAP_STATE.ready = False
        _BOOTSTRAP_STATE.error = None
        _BOOTSTRAP_STATE.updated_at = time.time()
        _BOOTSTRAP_STATE.details = dict(details)


def mark_bootstrap_ready(**details: object) -> None:
    """Пометить MCP как полностью готовый к обслуживанию клиентов."""

    with _BOOTSTRAP_LOCK:
        _BOOTSTRAP_STATE.phase = "ready"
        _BOOTSTRAP_STATE.ready = True
        _BOOTSTRAP_STATE.error = None
        _BOOTSTRAP_STATE.updated_at = time.time()
        _BOOTSTRAP_STATE.details = dict(details)


def mark_bootstrap_error(error: str, *, phase: str = "bootstrap_failed", **details: object) -> None:
    """Сохранить ошибку bootstrap и оставить HTTP transport живым для диагностики."""

    with _BOOTSTRAP_LOCK:
        _BOOTSTRAP_STATE.phase = phase
        _BOOTSTRAP_STATE.ready = False
        _BOOTSTRAP_STATE.error = error
        _BOOTSTRAP_STATE.updated_at = time.time()
        _BOOTSTRAP_STATE.details = dict(details)


def _bootstrap_payload(*, include_index_status: bool = False) -> dict[str, object]:
    """Собрать диагностический payload для health/readiness endpoints."""

    with _BOOTSTRAP_LOCK:
        payload: dict[str, object] = {
            "phase": _BOOTSTRAP_STATE.phase,
            "ready": _BOOTSTRAP_STATE.ready,
            "error": _BOOTSTRAP_STATE.error,
            "updated_at": _BOOTSTRAP_STATE.updated_at,
            "details": dict(_BOOTSTRAP_STATE.details),
        }

    payload["runtime_configured"] = _RUNTIME is not None
    payload["streamable_http_path"] = "/mcp"

    if _RUNTIME is not None:
        payload["watch_running"] = bool(_RUNTIME.watcher and _RUNTIME.watcher.is_running)
        payload["registry"] = _RUNTIME.registry.status_summary()
        if include_index_status and payload["ready"]:
            try:
                live_status = _RUNTIME.search_service.index_status().model_dump()
                payload["index_status"] = live_status
                payload["details"].update(
                    {
                        "repo_root": live_status["repo_root"],
                        "mounted_repo_root": live_status["mounted_repo_root"],
                        "repo_state": live_status["repo_state"],
                        "active": live_status["active"],
                        "active_repo_root": live_status["active_repo_root"],
                        "runtime_switch_required": live_status["runtime_switch_required"],
                        "search_available": live_status["search_available"],
                        "watch_enabled": live_status["watch_enabled"],
                    }
                )
                payload["search_ready"] = bool(payload["index_status"]["search_available"])
            except Exception as exc:  # noqa: BLE001
                payload["index_status_error"] = str(exc)

    return payload


def _runtime(*, require_ready: bool = True) -> AppRuntime:
    """Вернуть подготовленный runtime или бросить явную ошибку."""

    if _RUNTIME is None:
        raise RuntimeError("Semantic MCP runtime is not configured")

    if require_ready:
        payload = _bootstrap_payload(include_index_status=False)
        if not payload["ready"]:
            phase = payload["phase"]
            error = payload["error"]
            if error:
                raise RuntimeError(
                    f"Semantic MCP bootstrap failed during phase '{phase}': {error}"
                )
            raise RuntimeError(
                f"Semantic MCP is still bootstrapping (phase '{phase}'). Check /readyz for progress."
            )

    return _RUNTIME


def _dump(data) -> str:
    """Компактно сериализовать структуру в JSON string resource."""

    return json.dumps(data, ensure_ascii=False, indent=2)


def _normalize_repo_root(repo_root: str | None) -> str:
    """Нормализовать repo_root относительно текущего runtime repo."""

    runtime_repo_root_value = runtime_repo_root(_runtime())
    if repo_root is None:
        return runtime_repo_root_value
    candidate = repo_root.strip()
    if not candidate:
        return runtime_repo_root_value
    return normalize_repo_identity(candidate)


def _require_runtime_repo(repo_root: str | None) -> str:
    """Разрешить lifecycle операции только для текущего runtime repo."""

    normalized = _normalize_repo_root(repo_root)
    runtime_repo_root_value = runtime_repo_root(_runtime())
    if normalized != runtime_repo_root_value:
        raise RuntimeError(
            "This runtime can operate only on its current repo_root. "
            f"requested={normalized}; runtime_repo_root={runtime_repo_root_value}. "
            "Use a dedicated runtime switch flow before building another repo."
        )
    return runtime_repo_root_value


def _require_graph_service(runtime: AppRuntime):
    """Return configured graph service or a host-visible error."""

    if runtime.graph_service is None:
        raise RuntimeError("Graph service is not configured for this runtime")
    return runtime.graph_service


def _skipped_graph_build_payload(
    runtime: AppRuntime,
    *,
    rebuilt: bool,
    reason: str,
    warning_code: str,
):
    """Return the stable GraphBuildResult shape for skipped graph lifecycle calls."""

    graph_service = _require_graph_service(runtime)
    status = graph_service.graph_status(verify_source_revision=False)
    return GraphBuildResult(
        repo_root=status.repo_root,
        profile=status.profile,
        graph_path=status.graph_path,
        built=False,
        rebuilt=rebuilt,
        state=status.state,
        warning_codes=[warning_code, *status.warning_codes],
        status=status,
        reason=reason,
    ).model_dump()


def _search_payload(items: list[dict]) -> list[dict] | str:
    """Вернуть совместимый payload для search tools.

    FastMCP кодирует непустой list как набор TextContent items, но пустой list
    превращает в content=[] и structuredContent=null. Некоторые MCP-клиенты
    трактуют такой ответ как malformed "Unexpected response type".

    Для пустой выдачи возвращаем явный JSON string "[]", чтобы у клиента всегда
    был хотя бы один content item. Непустую выдачу оставляем в прежнем виде,
    чтобы не ломать текущих потребителей.
    """

    return items if items else "[]"


def _backend_action_plan_payload(
    runtime: AppRuntime,
    backend_entry,
    *,
    repo_root: str | None = None,
) -> dict[str, object]:
    """Собрать operator-facing action plan для explicit backend lifecycle."""

    target_repo_root = (
        _normalize_repo_root(repo_root)
        if repo_root is not None
        else default_action_plan_repo_root(runtime, runtime_repo_root=runtime_repo_root)
    )
    return backend_action_plan_payload(
        runtime,
        backend_entry,
        target_repo_root=target_repo_root,
    )


@mcp.custom_route("/healthz", methods=["GET"])
async def route_healthz(_request: Request) -> Response:
    """Базовый liveness probe для host-visible HTTP endpoint."""

    return JSONResponse(_bootstrap_payload(include_index_status=False), status_code=200)


@mcp.custom_route("/readyz", methods=["GET"])
async def route_readyz(_request: Request) -> Response:
    """Готовность transport/runtime слоя semantic MCP без требования search-ready."""

    payload = _bootstrap_payload(include_index_status=False)
    return JSONResponse(payload, status_code=200 if payload["ready"] else 503)


@mcp.custom_route("/statusz", methods=["GET"])
async def route_statusz(_request: Request) -> Response:
    """Диагностический статус для helper scripts и ручной проверки."""

    return JSONResponse(_bootstrap_payload(include_index_status=True), status_code=200)


@mcp.tool()
def semantic_search(
    query: str,
    top_k: int = 10,
    scope: SearchScope = "all",
    path_prefix: str | None = None,
    chunk_types: list[str] | None = None,
    domain_tags: list[str] | None = None,
    include_paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
    file_extensions: list[str] | None = None,
    languages: list[str] | None = None,
    max_results_per_file: int | None = None,
    snippet_mode: SnippetMode = "chunk_start",
    include_explanations: bool = True,
):
    """Dense semantic retrieval по всем или одной logical collection."""

    items = [
        item.model_dump()
        for item in _runtime().search_service.semantic_search(
            query=query,
            top_k=top_k,
            scope=scope,
            path_prefix=path_prefix,
            chunk_types=chunk_types,
            domain_tags=domain_tags,
            include_paths=include_paths,
            exclude_paths=exclude_paths,
            file_extensions=file_extensions,
            languages=languages,
            max_results_per_file=max_results_per_file,
            snippet_mode=snippet_mode,
            include_explanations=include_explanations,
        )
    ]
    return _search_payload(items)


@mcp.tool()
def semantic_search_code(
    query: str,
    top_k: int = 10,
    path_prefix: str | None = None,
    chunk_types: list[str] | None = None,
    domain_tags: list[str] | None = None,
    include_paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
    file_extensions: list[str] | None = None,
    languages: list[str] | None = None,
    max_results_per_file: int | None = None,
    snippet_mode: SnippetMode = "chunk_start",
    include_explanations: bool = True,
):
    """Dense semantic retrieval только по code collection."""

    return semantic_search(
        query=query,
        top_k=top_k,
        scope="code",
        path_prefix=path_prefix,
        chunk_types=chunk_types,
        domain_tags=domain_tags,
        include_paths=include_paths,
        exclude_paths=exclude_paths,
        file_extensions=file_extensions,
        languages=languages,
        max_results_per_file=max_results_per_file,
        snippet_mode=snippet_mode,
        include_explanations=include_explanations,
    )


@mcp.tool()
def semantic_search_docs(
    query: str,
    top_k: int = 10,
    path_prefix: str | None = None,
    chunk_types: list[str] | None = None,
    domain_tags: list[str] | None = None,
    include_paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
    file_extensions: list[str] | None = None,
    languages: list[str] | None = None,
    max_results_per_file: int | None = None,
    snippet_mode: SnippetMode = "chunk_start",
    include_explanations: bool = True,
):
    """Dense semantic retrieval только по docs collection."""

    return semantic_search(
        query=query,
        top_k=top_k,
        scope="docs",
        path_prefix=path_prefix,
        chunk_types=chunk_types,
        domain_tags=domain_tags,
        include_paths=include_paths,
        exclude_paths=exclude_paths,
        file_extensions=file_extensions,
        languages=languages,
        max_results_per_file=max_results_per_file,
        snippet_mode=snippet_mode,
        include_explanations=include_explanations,
    )


@mcp.tool()
def hybrid_search(
    query: str,
    top_k: int = 10,
    scope: SearchScope = "all",
    path_prefix: str | None = None,
    chunk_types: list[str] | None = None,
    domain_tags: list[str] | None = None,
    include_paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
    file_extensions: list[str] | None = None,
    languages: list[str] | None = None,
    max_results_per_file: int | None = None,
    snippet_mode: SnippetMode = "chunk_start",
    include_explanations: bool = True,
):
    """Hybrid retrieval по всем или одной logical collection."""

    items = [
        item.model_dump()
        for item in _runtime().search_service.hybrid_search(
            query=query,
            top_k=top_k,
            scope=scope,
            path_prefix=path_prefix,
            chunk_types=chunk_types,
            domain_tags=domain_tags,
            include_paths=include_paths,
            exclude_paths=exclude_paths,
            file_extensions=file_extensions,
            languages=languages,
            max_results_per_file=max_results_per_file,
            snippet_mode=snippet_mode,
            include_explanations=include_explanations,
        )
    ]
    return _search_payload(items)


@mcp.tool()
def hybrid_search_code(
    query: str,
    top_k: int = 10,
    path_prefix: str | None = None,
    chunk_types: list[str] | None = None,
    domain_tags: list[str] | None = None,
    include_paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
    file_extensions: list[str] | None = None,
    languages: list[str] | None = None,
    max_results_per_file: int | None = None,
    snippet_mode: SnippetMode = "chunk_start",
    include_explanations: bool = True,
):
    """Hybrid retrieval только по code collection."""

    return hybrid_search(
        query=query,
        top_k=top_k,
        scope="code",
        path_prefix=path_prefix,
        chunk_types=chunk_types,
        domain_tags=domain_tags,
        include_paths=include_paths,
        exclude_paths=exclude_paths,
        file_extensions=file_extensions,
        languages=languages,
        max_results_per_file=max_results_per_file,
        snippet_mode=snippet_mode,
        include_explanations=include_explanations,
    )


@mcp.tool()
def hybrid_search_docs(
    query: str,
    top_k: int = 10,
    path_prefix: str | None = None,
    chunk_types: list[str] | None = None,
    domain_tags: list[str] | None = None,
    include_paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
    file_extensions: list[str] | None = None,
    languages: list[str] | None = None,
    max_results_per_file: int | None = None,
    snippet_mode: SnippetMode = "chunk_start",
    include_explanations: bool = True,
):
    """Hybrid retrieval только по docs collection."""

    return hybrid_search(
        query=query,
        top_k=top_k,
        scope="docs",
        path_prefix=path_prefix,
        chunk_types=chunk_types,
        domain_tags=domain_tags,
        include_paths=include_paths,
        exclude_paths=exclude_paths,
        file_extensions=file_extensions,
        languages=languages,
        max_results_per_file=max_results_per_file,
        snippet_mode=snippet_mode,
        include_explanations=include_explanations,
    )


@mcp.tool()
def search_v2(
    query: str,
    mode: SearchMode = "hybrid",
    top_k: int = 10,
    scope: SearchScope = "all",
    path_prefix: str | None = None,
    chunk_types: list[str] | None = None,
    domain_tags: list[str] | None = None,
    include_paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
    file_extensions: list[str] | None = None,
    languages: list[str] | None = None,
    max_results_per_file: int | None = None,
    snippet_mode: SnippetMode = "chunk_start",
    include_explanations: bool = True,
):
    """Agent-grade search envelope with diagnostics and response-level warnings."""

    return _runtime().search_service.search_v2(
        query=query,
        mode=mode,
        top_k=top_k,
        scope=scope,
        path_prefix=path_prefix,
        chunk_types=chunk_types,
        domain_tags=domain_tags,
        include_paths=include_paths,
        exclude_paths=exclude_paths,
        file_extensions=file_extensions,
        languages=languages,
        max_results_per_file=max_results_per_file,
        snippet_mode=snippet_mode,
        include_explanations=include_explanations,
    ).model_dump()


@mcp.tool()
def repo_context_search(
    query: str,
    subqueries: list[str] | None = None,
    route: RepoContextRoute = "auto",
    graph_mode: GraphMode = "auto",
    scope: SearchScope = "all",
    path_prefix: str | None = None,
    include_paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
    file_extensions: list[str] | None = None,
    languages: list[str] | None = None,
    chunk_types: list[str] | None = None,
    domain_tags: list[str] | None = None,
    top_k: int = 20,
    max_results_per_file: int | None = 3,
    snippet_mode: SnippetMode = "query_centered",
    include_diagnostics: bool = True,
    include_explanations: bool = True,
):
    """Agent-facing context envelope over dense+sparse retrieval."""

    return _runtime().search_service.repo_context_search(
        query=query,
        subqueries=subqueries,
        route=route,
        graph_mode=graph_mode,
        scope=scope,
        path_prefix=path_prefix,
        include_paths=include_paths,
        exclude_paths=exclude_paths,
        file_extensions=file_extensions,
        languages=languages,
        chunk_types=chunk_types,
        domain_tags=domain_tags,
        top_k=top_k,
        max_results_per_file=max_results_per_file,
        snippet_mode=snippet_mode,
        include_diagnostics=include_diagnostics,
        include_explanations=include_explanations,
    ).model_dump()


@mcp.tool()
def find_similar_chunk(scope: str, chunk_id: str, top_k: int = 10):
    """Найти похожие chunks, начиная от уже известного chunk id."""

    items = [
        item.model_dump()
        for item in _runtime().search_service.find_similar_chunk(
            scope=scope,
            chunk_id=chunk_id,
            top_k=top_k,
        )
    ]
    return _search_payload(items)


@mcp.tool()
def read_chunk(scope: str, chunk_id: str):
    """Вернуть полный текст конкретного чанка из code/docs коллекции."""

    result = _runtime().search_service.read_chunk(scope=scope, chunk_id=chunk_id)
    return result.model_dump() if result else None


@mcp.tool()
def index_status():
    """Показать текущее состояние индекса и watcher."""

    return _runtime().search_service.index_status().model_dump()


@mcp.tool()
def graph_status():
    """Показать состояние SQLite graph artifact без build/update side effects."""

    runtime = _runtime()
    return _require_graph_service(runtime).graph_status().model_dump()


@mcp.tool()
def update_graph(
    paths: list[str] | None = None,
    repo_root: str | None = None,
    idempotency_key: str | None = None,
    allow_large_update: bool = False,
):
    """Безопасно обновить SQLite graph artifact для bounded набора путей."""

    _require_runtime_repo(repo_root)
    runtime = _runtime()
    status = runtime.search_service.index_status()
    if not status.search_available:
        graph_service = _require_graph_service(runtime)
        graph_status_result = graph_service.graph_status(verify_source_revision=False)
        return {
            "contract_version": "graph_update.v1",
            "repo_root": graph_status_result.repo_root,
            "profile": graph_status_result.profile,
            "graph_path": graph_service.graph_path,
            "idempotency_key": idempotency_key,
            "allow_large_update": allow_large_update,
            "updated": False,
            "skipped": True,
            "skip_reason": "graph_update_index_unavailable",
            "warning_codes": ["graph_update_index_unavailable"],
            "status": graph_status_result.model_dump(),
        }
    graph_service = _require_graph_service(runtime)
    try:
        with lifecycle_mutation(runtime, "update_graph", idempotency_key=idempotency_key):
            return graph_service.update_graph(
                paths=paths or [],
                idempotency_key=idempotency_key,
                allow_large_update=allow_large_update,
            ).model_dump()
    except LifecycleLeaseActiveError:
        graph_status_result = graph_service.graph_status(verify_source_revision=False)
        return {
            "contract_version": "graph_update.v1",
            "repo_root": graph_status_result.repo_root,
            "profile": graph_status_result.profile,
            "graph_path": graph_service.graph_path,
            "idempotency_key": idempotency_key,
            "allow_large_update": allow_large_update,
            "operation_in_progress": True,
            "updated": False,
            "skipped": True,
            "skip_reason": "lifecycle_operation_in_progress",
            "warning_codes": ["lifecycle_operation_in_progress"],
            "status": graph_status_result.model_dump(),
        }


@mcp.tool()
def build_graph(repo_root: str | None = None, force_rebuild: bool = False):
    """Явно построить graph artifact для текущего runtime repo."""

    _require_runtime_repo(repo_root)
    runtime = _runtime()
    status = runtime.search_service.index_status()
    if not status.search_available:
        return _skipped_graph_build_payload(
            runtime,
            rebuilt=force_rebuild,
            reason="Semantic index is not available for graph build.",
            warning_code="graph_build_index_unavailable",
        )
    graph_service = _require_graph_service(runtime)
    with lifecycle_mutation(runtime, "rebuild_graph" if force_rebuild else "build_graph"):
        result = (
            graph_service.rebuild_graph()
            if force_rebuild
            else graph_service.build_graph()
        )
    return result.model_dump()


@mcp.tool()
def rebuild_graph(repo_root: str | None = None):
    """Явно удалить и заново построить graph artifact для текущего runtime repo."""

    _require_runtime_repo(repo_root)
    runtime = _runtime()
    status = runtime.search_service.index_status()
    if not status.search_available:
        return _skipped_graph_build_payload(
            runtime,
            rebuilt=True,
            reason="Semantic index is not available for graph rebuild.",
            warning_code="graph_build_index_unavailable",
        )
    with lifecycle_mutation(runtime, "rebuild_graph"):
        return _require_graph_service(runtime).rebuild_graph().model_dump()


@mcp.tool()
def rebuild_index():
    """Полностью перестроить docs и code коллекции."""

    runtime = _runtime()
    pending_backend_switch = embedding_backend_switch_payload(runtime)
    if pending_backend_switch is not None:
        return pending_backend_switch
    return rebuild_current_index(runtime)


@mcp.tool()
def reindex_paths(paths: list[str]):
    """Переиндексировать конкретные файлы по относительным путям."""

    runtime = _runtime()
    pending_backend_switch = embedding_backend_switch_payload(runtime)
    if pending_backend_switch is not None:
        return pending_backend_switch
    return reindex_current_paths(
        runtime,
        paths=paths,
        current_runtime_repo_state=current_runtime_repo_state,
    )


@mcp.tool()
def update_include_globs(globs: list[str]):
    """Обновить include globs и полностью перестроить индекс.

    Принимает список glob-паттернов (например ["src/**", "docs/**", "scripts/**"])
    или ["auto"] для возврата к авто-детектированию из структуры репозитория.
    После обновления запускает полный rebuild индекса.

    Примеры:
      update_include_globs(["src/**", "docs/**"])
      update_include_globs(["auto"])
    """

    runtime = _runtime()
    pending_backend_switch = embedding_backend_switch_payload(runtime)
    if pending_backend_switch is not None:
        return pending_backend_switch
    return update_include_globs_and_rebuild(runtime, globs=globs)


@mcp.tool()
def list_repos():
    """Вернуть все зарегистрированные repos из persistent registry."""

    return [entry.model_dump() for entry in _runtime().registry.list_repos()]


@mcp.tool()
def list_backends(role: BackendRole | None = None):
    """Вернуть inference backends из persistent registry с runtime status."""

    runtime = _runtime()
    return [
        backend_status_payload(runtime, entry)
        for entry in runtime.registry.list_backends(role=role)
    ]


@mcp.tool()
def get_backend_status(role: BackendRole | None = None, backend_id: str | None = None):
    """Вернуть статус конкретного backend или backend, выбранного для role."""

    runtime = _runtime()
    entry = resolve_backend_entry(runtime, role, backend_id)
    return backend_status_payload(runtime, entry)


@mcp.tool()
def set_role_backend(role: BackendRole, backend_id: str):
    """Явно выбрать backend для inference role без скрытого hot-switch runtime."""

    runtime = _runtime()
    with lifecycle_global_mutation(runtime, "set_role_backend", scope="backend_roles"):
        repo_lease_context = lifecycle_mutation(runtime, "set_role_backend") if role == "embedding" else nullcontext(None)
        with repo_lease_context:
            selected = runtime.registry.set_role_backend(role, backend_id)
            runtime_backend_id = runtime_embedding_backend_id(runtime) if role == "embedding" else None
            watcher_stopped = False
            if (
                role == "embedding"
                and runtime_backend_id
                and selected.backend_id != runtime_backend_id
                and runtime.watcher is not None
                and runtime.watcher.is_running
            ):
                runtime.watcher.stop()
                watcher_stopped = True
                runtime.search_service.sync_registry_state(
                    status=runtime.search_service.index_status().repo_state,
                    watch_running=False,
                )
        return {
            "role": role,
            "backend_id": selected.backend_id,
            "runtime_backend_id": runtime_backend_id,
            "active_switch_applied": bool(runtime_backend_id and selected.backend_id == runtime_backend_id),
            "runtime_switch_required": bool(runtime_backend_id and selected.backend_id != runtime_backend_id),
            "watcher_stopped": watcher_stopped,
            "status": backend_status_payload(runtime, selected),
        }


@mcp.tool()
def get_backend_action_plan(
    role: BackendRole | None = None,
    backend_id: str | None = None,
    repo_root: str | None = None,
):
    """Вернуть operator-facing action plan для explicit backend lifecycle."""

    runtime = _runtime()
    entry = resolve_backend_entry(runtime, role, backend_id)
    return _backend_action_plan_payload(runtime, entry, repo_root=repo_root)


@mcp.tool()
def ensure_backend(
    role: BackendRole | None = None,
    backend_id: str | None = None,
    repo_root: str | None = None,
):
    """Проверить backend readiness без скрытого старта и вернуть explicit next action."""

    runtime = _runtime()
    entry = resolve_backend_entry(runtime, role, backend_id)
    plan = _backend_action_plan_payload(runtime, entry, repo_root=repo_root)
    return {
        "ensured": plan["operator_action_state"] == "ready",
        "operator_action_state": plan["operator_action_state"],
        "summary": plan["summary"],
        "plan": plan,
    }


@mcp.tool()
def list_legacy_repo_entries():
    """Показать legacy registry entries старого single-runtime формата."""

    runtime = _runtime()
    return [entry.model_dump() for entry in runtime.registry.list_legacy_repos()]


@mcp.tool()
def prune_legacy_repo_entries(force_remove_active: bool = False):
    """Удалить legacy registry entries старого формата безопасным explicit действием."""

    runtime = _runtime()
    return runtime.registry.prune_legacy_repos(force_remove_active=force_remove_active)


@mcp.tool()
def get_repo_status(repo_root: str | None = None):
    """Вернуть статус runtime repo или конкретного зарегистрированного repo."""

    runtime = _runtime()
    if repo_root is None:
        return runtime.search_service.index_status().model_dump()

    entry = runtime.registry.get_repo(repo_root)
    if entry is None:
        raise RuntimeError(f"Repo is not registered: {repo_root}")
    return entry.model_dump()


@mcp.tool()
def register_repo(
    repo_root: str,
    include_globs: list[str] | None = None,
    doc_prefixes: list[str] | None = None,
    activate: bool = False,
):
    """Зарегистрировать repo в registry без запуска тяжёлой индексации."""

    runtime = _runtime()
    target_repo_root = _normalize_repo_root(repo_root)
    with lifecycle_global_mutation(runtime, "register_repo", scope="repo_registry"):
        if target_repo_root == runtime_repo_root(runtime):
            with lifecycle_mutation(runtime, "register_repo"):
                return register_repo_payload(
                    runtime,
                    target_repo_root=target_repo_root,
                    include_globs=include_globs,
                    doc_prefixes=doc_prefixes,
                    activate=activate,
                )
        return register_repo_payload(
            runtime,
            target_repo_root=target_repo_root,
            include_globs=include_globs,
            doc_prefixes=doc_prefixes,
            activate=activate,
        )


@mcp.tool()
def activate_repo(repo_root: str):
    """Сделать repo active. Для другого repo пока требуется отдельный runtime switch."""

    runtime = _runtime()
    requested_repo_root = _normalize_repo_root(repo_root)
    with lifecycle_global_mutation(runtime, "activate_repo", scope="repo_registry"):
        if requested_repo_root == runtime_repo_root(runtime):
            with lifecycle_mutation(runtime, "activate_repo"):
                return activate_repo_payload(
                    runtime,
                    requested_repo_root=requested_repo_root,
                )
        return activate_repo_payload(
            runtime,
            requested_repo_root=requested_repo_root,
        )


@mcp.tool()
def build_index(repo_root: str | None = None, force_rebuild: bool = False):
    """Явно построить индекс для текущего runtime repo."""

    runtime = _runtime()
    target_repo_root = _normalize_repo_root(repo_root)
    if target_repo_root != runtime_repo_root(runtime):
        payload = runtime_switch_required_payload(runtime, target_repo_root)
        if payload["status"]["status"] == "indexed" and not force_rebuild:
            payload["rebuild_skipped"] = True
            payload["reason"] = (
                "index already available; use force_rebuild=true after switching runtime explicitly"
            )
        else:
            payload["build_deferred"] = True
            payload["reason"] = "host-side runtime switch is required before build_index can run"
        return payload
    pending_backend_switch = embedding_backend_switch_payload(runtime)
    if pending_backend_switch is not None:
        return pending_backend_switch
    current_status = runtime.search_service.index_status()
    if current_status.search_available and not force_rebuild:
        return {
            "rebuild_skipped": True,
            "reason": "index already available; use force_rebuild=true to rebuild explicitly",
            "status": current_status.model_dump(),
        }
    return rebuild_current_index(runtime)


@mcp.tool()
def start_watcher(repo_root: str | None = None, idempotency_key: str | None = None):
    """Явно запустить incremental watcher для active runtime repo."""

    _require_runtime_repo(repo_root)
    runtime = _runtime()
    pending_backend_switch = embedding_backend_switch_payload(runtime)
    if pending_backend_switch is not None:
        return pending_backend_switch
    return start_watcher_for_runtime(runtime, idempotency_key=idempotency_key)


@mcp.tool()
def stop_watcher():
    """Остановить watcher для текущего runtime repo."""

    runtime = _runtime()
    return stop_watcher_for_runtime(runtime)


@mcp.resource("index://status")
def resource_index_status() -> str:
    """Экспортировать текущий статус индекса как resource."""

    return _dump(index_status())


@mcp.resource("index://collections")
def resource_index_collections() -> str:
    """Экспортировать список logical collections."""

    status = index_status()
    return _dump(status["collections"])


@mcp.resource("index://config")
def resource_index_config() -> str:
    """Экспортировать user-facing конфигурацию semantic MCP."""

    status = index_status()
    return _dump(
        {
            "repo_root": status["repo_root"],
            "mounted_repo_root": status["mounted_repo_root"],
            "active_repo_root": status["active_repo_root"],
            "runtime_switch_required": status["runtime_switch_required"],
            "repo_key": status["repo_key"],
            "repo_state": status["repo_state"],
            "active": status["active"],
            "search_available": status["search_available"],
            "reason_if_unavailable": status["reason_if_unavailable"],
            "index_profile": status["index_profile"],
            "embedding_backend": status["embedding_backend"],
            "embedding_backend_type": status["embedding_backend_type"],
            "embedding_backend_id": status["embedding_backend_id"],
            "selected_embedding_backend_id": status["selected_embedding_backend_id"],
            "backend_switch_required": status["backend_switch_required"],
            "embedding_model": status["embedding_model"],
            "qdrant_url": status["qdrant_url"],
            "schema_version": status["schema_version"],
            "watch_enabled": status["watch_enabled"],
            "include_globs": status["include_globs"],
            "doc_prefixes": status["doc_prefixes"],
        }
    )
