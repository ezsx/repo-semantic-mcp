"""FastMCP facade for repo semantic search."""

from __future__ import annotations

import json
from dataclasses import dataclass

from mcp.server.fastmcp import FastMCP

from services.repo_semantic.indexer import RepositoryIndexer
from services.repo_semantic.models import SearchScope
from services.repo_semantic.search_service import SearchService
from services.repo_semantic.watcher import RepositoryWatcher


@dataclass(slots=True)
class AppRuntime:
    """Runtime зависимости semantic MCP."""

    search_service: SearchService
    indexer: RepositoryIndexer
    watcher: RepositoryWatcher | None


_RUNTIME: AppRuntime | None = None

mcp = FastMCP(
    name="repo-semantic-search",
    instructions=(
        "Используй этот MCP для repo-wide semantic shortlist. "
        "Для docs-only и code-only поиска предпочитай явные tools соответствующей коллекции."
    ),
)


def configure_runtime(runtime: AppRuntime) -> None:
    """Сохранить runtime singleton перед стартом MCP сервера."""

    global _RUNTIME
    _RUNTIME = runtime


def _runtime() -> AppRuntime:
    """Вернуть подготовленный runtime или бросить явную ошибку."""

    if _RUNTIME is None:
        raise RuntimeError("Semantic MCP runtime is not configured")
    return _RUNTIME


def _dump(data) -> str:
    """Компактно сериализовать структуру в JSON string resource."""

    return json.dumps(data, ensure_ascii=False, indent=2)


@mcp.tool()
def semantic_search(
    query: str,
    top_k: int = 10,
    scope: SearchScope = "all",
    path_prefix: str | None = None,
    chunk_types: list[str] | None = None,
    domain_tags: list[str] | None = None,
):
    """Dense semantic retrieval по всем или одной logical collection."""

    return [
        item.model_dump()
        for item in _runtime().search_service.semantic_search(
            query=query,
            top_k=top_k,
            scope=scope,
            path_prefix=path_prefix,
            chunk_types=chunk_types,
            domain_tags=domain_tags,
        )
    ]


@mcp.tool()
def semantic_search_code(
    query: str,
    top_k: int = 10,
    path_prefix: str | None = None,
    chunk_types: list[str] | None = None,
    domain_tags: list[str] | None = None,
):
    """Dense semantic retrieval только по code collection."""

    return semantic_search(
        query=query,
        top_k=top_k,
        scope="code",
        path_prefix=path_prefix,
        chunk_types=chunk_types,
        domain_tags=domain_tags,
    )


@mcp.tool()
def semantic_search_docs(
    query: str,
    top_k: int = 10,
    path_prefix: str | None = None,
    chunk_types: list[str] | None = None,
    domain_tags: list[str] | None = None,
):
    """Dense semantic retrieval только по docs collection."""

    return semantic_search(
        query=query,
        top_k=top_k,
        scope="docs",
        path_prefix=path_prefix,
        chunk_types=chunk_types,
        domain_tags=domain_tags,
    )


@mcp.tool()
def hybrid_search(
    query: str,
    top_k: int = 10,
    scope: SearchScope = "all",
    path_prefix: str | None = None,
    chunk_types: list[str] | None = None,
    domain_tags: list[str] | None = None,
):
    """Hybrid retrieval по всем или одной logical collection."""

    return [
        item.model_dump()
        for item in _runtime().search_service.hybrid_search(
            query=query,
            top_k=top_k,
            scope=scope,
            path_prefix=path_prefix,
            chunk_types=chunk_types,
            domain_tags=domain_tags,
        )
    ]


@mcp.tool()
def hybrid_search_code(
    query: str,
    top_k: int = 10,
    path_prefix: str | None = None,
    chunk_types: list[str] | None = None,
    domain_tags: list[str] | None = None,
):
    """Hybrid retrieval только по code collection."""

    return hybrid_search(
        query=query,
        top_k=top_k,
        scope="code",
        path_prefix=path_prefix,
        chunk_types=chunk_types,
        domain_tags=domain_tags,
    )


@mcp.tool()
def hybrid_search_docs(
    query: str,
    top_k: int = 10,
    path_prefix: str | None = None,
    chunk_types: list[str] | None = None,
    domain_tags: list[str] | None = None,
):
    """Hybrid retrieval только по docs collection."""

    return hybrid_search(
        query=query,
        top_k=top_k,
        scope="docs",
        path_prefix=path_prefix,
        chunk_types=chunk_types,
        domain_tags=domain_tags,
    )


@mcp.tool()
def find_similar_chunk(scope: str, chunk_id: str, top_k: int = 10):
    """Найти похожие chunks, начиная от уже известного chunk id."""

    return [
        item.model_dump()
        for item in _runtime().search_service.find_similar_chunk(
            scope=scope,
            chunk_id=chunk_id,
            top_k=top_k,
        )
    ]


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
def rebuild_index():
    """Полностью перестроить docs и code коллекции."""

    runtime = _runtime()
    result = runtime.indexer.rebuild_index()
    runtime.search_service.invalidate_cache()
    return {
        "rebuild": result,
        "status": runtime.search_service.index_status().model_dump(),
    }


@mcp.tool()
def reindex_paths(paths: list[str]):
    """Переиндексировать конкретные файлы по относительным путям."""

    runtime = _runtime()
    result = runtime.indexer.reindex_paths(paths)
    runtime.search_service.invalidate_cache()
    return result


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
            "embedding_backend": status["embedding_backend"],
            "embedding_model": status["embedding_model"],
            "qdrant_url": status["qdrant_url"],
            "schema_version": status["schema_version"],
            "watch_enabled": status["watch_enabled"],
        }
    )
