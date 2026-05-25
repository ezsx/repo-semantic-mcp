from __future__ import annotations

from pathlib import Path

from services.repo_semantic.config import SemanticMcpSettings
from services.repo_semantic.models import ChunkRecord
from services.repo_semantic.registry import RepoRegistry

from tests.helpers.fakes import (
    FakeBM25Okapi,
    FakeEmbeddingProvider,
    FakeIndexer,
    FakePoint,
    FakeStore,
    install_rank_bm25_stub,
)

install_rank_bm25_stub()

from services.repo_semantic.search_service import SearchService
import services.repo_semantic.retrieval.legacy_bm25 as legacy_bm25_module

legacy_bm25_module.BM25Okapi = FakeBM25Okapi


def build_search_service(
    tmp: str,
    chunks: list[tuple[ChunkRecord, float]],
    *,
    sparse_enabled: bool = False,
    sparse_scores: dict[str, float] | None = None,
    fail_on_scroll: bool = False,
) -> SearchService:
    repo_root = Path(tmp) / "repo"
    repo_root.mkdir()
    registry = RepoRegistry(Path(tmp) / "registry.sqlite3")
    settings = SemanticMcpSettings(
        SEMANTIC_MCP_REPO_ROOT=str(repo_root),
        SEMANTIC_MCP_LOGICAL_REPO_ROOT=str(repo_root),
        SEMANTIC_MCP_REGISTRY_DB_PATH=str(Path(tmp) / "registry.sqlite3"),
        SEMANTIC_MCP_INCLUDE_GLOBS=["*", "**/*"],
        SEMANTIC_MCP_DOC_PATH_PREFIXES=["docs/"],
        SEMANTIC_MCP_EXCLUDE_GLOBS=[".git/**"],
    )
    backend_payload = settings.current_embedding_backend_payload()
    registry.upsert_backend(**backend_payload)
    registry.set_role_backend("embedding", settings.embedding_backend_id)
    code_points = sum(1 for chunk, _ in chunks if chunk.scope == "code")
    docs_points = sum(1 for chunk, _ in chunks if chunk.scope == "docs")
    registry.upsert_repo(
        settings.logical_repo_identity,
        status="indexed",
        active=True,
        index_profile="test-profile",
        include_globs=["*", "**/*"],
        doc_prefixes=["docs/"],
        exclude_globs=[".git/**"],
        code_points_count=code_points,
        docs_points_count=docs_points,
    )
    points_by_scope: dict[str, list[FakePoint]] = {"code": [], "docs": []}
    for chunk, score in chunks:
        points_by_scope[chunk.scope].append(FakePoint(chunk, score))
    return SearchService(
        settings=settings,
        embedding_provider=FakeEmbeddingProvider(),
        store=FakeStore(
            points_by_scope,
            sparse_enabled=sparse_enabled,
            sparse_scores=sparse_scores,
            fail_on_scroll=fail_on_scroll,
        ),
        indexer=FakeIndexer(),
        watcher=None,
        registry=registry,
    )
