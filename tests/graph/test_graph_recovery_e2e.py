from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from services.repo_semantic.config import SemanticMcpSettings
from services.repo_semantic.graph import GraphService
from services.repo_semantic.lifecycle.watcher_ops import start_watcher_for_runtime
from services.repo_semantic.models import ChunkRecord
from services.repo_semantic.registry import RepoRegistry
from services.repo_semantic.watcher import RepositoryWatcher
from tests.helpers.fakes import (
    FakeEmbeddingProvider,
    FakePoint,
    FakeStore,
    install_rank_bm25_stub,
)

install_rank_bm25_stub()

from services.repo_semantic.search_service import SearchService


def _git(repo_root: Path, *args: str) -> str:
    if shutil.which("git") is None:
        pytest.skip("git executable is required for graph recovery fixture")
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _write_fixture(repo_root: Path, *, route: str) -> None:
    (repo_root / "services/api").mkdir(parents=True, exist_ok=True)
    (repo_root / "docs").mkdir(parents=True, exist_ok=True)
    (repo_root / "tests").mkdir(parents=True, exist_ok=True)
    (repo_root / "services/api/connect.py").write_text(
        (
            "from fastapi import APIRouter\n"
            "router = APIRouter()\n\n"
            f"@router.get('{route}')\n"
            "def connect_endpoint():\n"
            "    return {'ok': True}\n"
        ),
        encoding="utf-8",
    )
    (repo_root / "docs/checkout-flow.md").write_text(
        f"# Checkout Flow\nCall `{route}` from the Frontend.\n",
        encoding="utf-8",
    )
    (repo_root / "tests/test_checkout_flow.py").write_text(
        (
            "from services.api.connect import connect_endpoint\n\n"
            "def test_connect_endpoint():\n"
            "    assert connect_endpoint()\n"
        ),
        encoding="utf-8",
    )


def _chunk_from_file(repo_root: Path, relative_path: str) -> ChunkRecord:
    path = repo_root / relative_path
    text = path.read_text(encoding="utf-8")
    is_docs = relative_path.startswith("docs/")
    content_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()
    symbol_path = None
    heading_path = None
    if relative_path.endswith("connect.py"):
        symbol_path = "connect_endpoint"
    if relative_path.endswith("test_checkout_flow.py"):
        symbol_path = "test_connect_endpoint"
    if is_docs:
        heading_path = "Checkout Flow"
    return ChunkRecord(
        point_id=relative_path.replace("/", ":"),
        scope="docs" if is_docs else "code",  # type: ignore[arg-type]
        relative_path=relative_path,
        language="markdown" if is_docs else "python",
        chunk_type="markdown_section" if is_docs else "python_function",
        text=text,
        start_line=1,
        end_line=max(1, text.count("\n") + 1),
        content_hash=content_hash,
        source_mtime=path.stat().st_mtime,
        symbol_path=symbol_path,
        heading_path=heading_path,
        domain_tags=["docs"] if is_docs else ["services"],
    )


class FixtureIndexer:
    def __init__(
        self,
        *,
        repo_root: Path,
        settings: SemanticMcpSettings,
        store: FakeStore,
        registry: RepoRegistry,
    ) -> None:
        self._settings = settings
        self._repo_root = repo_root
        self._store = store
        self._registry = registry
        self.reindexed_paths: list[str] = []
        self.last_full_build_ts: str | None = None
        self.last_incremental_update_ts: str | None = None

    def iter_indexable_paths(self, **_: object):
        for path in sorted(self._repo_root.rglob("*")):
            if not path.is_file() or ".git" in path.parts:
                continue
            if path.suffix in {".py", ".md"}:
                yield path

    def _relative_paths(self) -> list[str]:
        return [
            path.relative_to(self._repo_root).as_posix()
            for path in self.iter_indexable_paths()
        ]

    def _upsert_registry(self) -> None:
        current_branch = _git(self._repo_root, "branch", "--show-current")
        current_commit = _git(self._repo_root, "rev-parse", "HEAD")
        self.last_incremental_update_ts = datetime.now(timezone.utc).isoformat()
        self._registry.upsert_repo(
            self._settings.logical_repo_identity,
            status="indexed",
            active=True,
            index_profile="test-profile",
            include_globs=["**/*.py", "docs/**/*.md"],
            doc_prefixes=["docs/"],
            exclude_globs=[".git/**"],
            last_full_build_ts=self.last_full_build_ts,
            last_incremental_update_ts=self.last_incremental_update_ts,
            indexed_branch=current_branch,
            indexed_commit_hash=current_commit,
            code_points_count=len(self._store._points_by_scope.get("code", [])),
            docs_points_count=len(self._store._points_by_scope.get("docs", [])),
        )

    def reindex_paths(self, paths: list[str]) -> dict[str, object]:
        normalized = sorted({str(path).replace("\\", "/").strip("/") for path in paths if str(path)})
        self.reindexed_paths.extend(normalized)
        for scope in ("code", "docs"):
            self._store._points_by_scope[scope] = [
                point
                for point in self._store._points_by_scope.get(scope, [])
                if point.payload["relative_path"] not in normalized
            ]
        code = 0
        docs = 0
        for relative_path in normalized:
            path = self._repo_root / relative_path
            if not path.exists():
                continue
            chunk = _chunk_from_file(self._repo_root, relative_path)
            score = 0.95 if relative_path.endswith("connect.py") else 0.2
            self._store._points_by_scope[chunk.scope].append(FakePoint(chunk, score))
            if chunk.scope == "docs":
                docs += 1
            else:
                code += 1
        self._upsert_registry()
        return {"paths": len(normalized), "code": code, "docs": docs}

    def reconcile_index(self) -> dict[str, object]:
        indexed_mtimes = {
            point.payload["relative_path"]: float(point.payload.get("source_mtime") or 0.0)
            for points in self._store._points_by_scope.values()
            for point in points
        }
        current_paths = self._relative_paths()
        touched: list[str] = []
        for relative_path in current_paths:
            current_mtime = (self._repo_root / relative_path).stat().st_mtime
            if abs(indexed_mtimes.get(relative_path, 0.0) - current_mtime) > 1e-6:
                touched.append(relative_path)
        for relative_path in indexed_mtimes:
            if relative_path not in current_paths:
                touched.append(relative_path)
        touched = sorted(set(touched))
        if touched:
            counts = self.reindex_paths(touched)
            return {
                **counts,
                "touched_paths": touched,
                "deleted_paths": [
                    path for path in touched if not (self._repo_root / path).exists()
                ],
                "mutation_started": True,
                "bounds_exceeded": [],
            }
        self._upsert_registry()
        return {
            "paths": 0,
            "code": 0,
            "docs": 0,
            "touched_paths": [],
            "deleted_paths": [],
            "mutation_started": False,
            "bounds_exceeded": [],
        }


def _settings(tmp_path: Path, repo_root: Path) -> SemanticMcpSettings:
    return SemanticMcpSettings(
        SEMANTIC_MCP_REPO_ROOT=str(repo_root),
        SEMANTIC_MCP_LOGICAL_REPO_ROOT=str(repo_root),
        SEMANTIC_MCP_REGISTRY_DB_PATH=str(tmp_path / "registry.sqlite3"),
        SEMANTIC_MCP_INCLUDE_GLOBS=["**/*.py", "docs/**/*.md"],
        SEMANTIC_MCP_DOC_PATH_PREFIXES=["docs/"],
        SEMANTIC_MCP_EXCLUDE_GLOBS=[".git/**"],
    )


def test_branch_switch_startup_reconcile_updates_graph_without_rebuild(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _git(repo_root, "init", "-b", "main")
    _git(repo_root, "config", "user.email", "tests@example.local")
    _git(repo_root, "config", "user.name", "Repo Semantic Tests")
    _write_fixture(repo_root, route="/api/v1/connect")
    _git(repo_root, "add", ".")
    _git(repo_root, "commit", "-m", "initial")

    settings = _settings(tmp_path, repo_root)
    registry = RepoRegistry(settings.registry_db_path)
    backend_payload = settings.current_embedding_backend_payload()
    registry.upsert_backend(**backend_payload)
    registry.set_role_backend("embedding", settings.embedding_backend_id)
    initial_chunks = [
        _chunk_from_file(repo_root, "services/api/connect.py"),
        _chunk_from_file(repo_root, "docs/checkout-flow.md"),
        _chunk_from_file(repo_root, "tests/test_checkout_flow.py"),
    ]
    store = FakeStore(
        {
            "code": [
                FakePoint(initial_chunks[0], 0.95),
                FakePoint(initial_chunks[2], 0.2),
            ],
            "docs": [FakePoint(initial_chunks[1], 0.2)],
        },
        sparse_enabled=True,
        sparse_scores={
            "services/api/connect.py": 2.0,
            "docs/checkout-flow.md": 1.0,
            "tests/test_checkout_flow.py": 1.0,
        },
    )
    indexer = FixtureIndexer(
        repo_root=repo_root,
        settings=settings,
        store=store,
        registry=registry,
    )
    indexer._upsert_registry()
    graph_service = GraphService(settings=settings, chunk_store=store, registry=registry)
    graph_service.build_graph()

    def forbidden_lifecycle(*_: object, **__: object) -> None:
        raise AssertionError("build/rebuild lifecycle must not run in recovery fixture")

    indexer.build_index = forbidden_lifecycle  # type: ignore[attr-defined]
    indexer.rebuild_index = forbidden_lifecycle  # type: ignore[attr-defined]
    graph_service.build_graph = forbidden_lifecycle  # type: ignore[method-assign]
    graph_service.rebuild_graph = forbidden_lifecycle  # type: ignore[method-assign]

    _git(repo_root, "checkout", "-b", "feature/v2-route")
    time.sleep(0.02)
    _write_fixture(repo_root, route="/api/v2/connect")
    _git(repo_root, "add", ".")
    _git(repo_root, "commit", "-m", "switch route")
    _git(repo_root, "checkout", "main")
    _git(repo_root, "checkout", "feature/v2-route")

    lifecycle_counters = {
        "build_index": 0,
        "rebuild_index": 0,
        "build_graph": 0,
        "rebuild_graph": 0,
        "update_graph": 0,
    }

    def on_index_changed(paths: list[str], startup: bool) -> None:
        assert startup
        lifecycle_counters["update_graph"] += 1
        graph_service.update_graph(paths=paths)

    watcher = RepositoryWatcher(indexer, debounce_sec=0.05, on_index_changed=on_index_changed)
    service = SearchService(
        settings=settings,
        embedding_provider=FakeEmbeddingProvider(),
        store=store,
        indexer=indexer,
        watcher=watcher,
        registry=registry,
        graph_status_summary=graph_service.graph_summary,
        graph_read_provider=graph_service.read_provider(),
    )
    runtime = SimpleNamespace(
        watcher=watcher,
        search_service=service,
        indexer=indexer,
        registry=registry,
        embedding_provider=FakeEmbeddingProvider(),
    )
    start_payload = start_watcher_for_runtime(runtime, idempotency_key="fixture-branch-switch")
    try:
        assert start_payload["watch_running"] is True
        reconcile = start_payload["startup_reconcile"]
        assert reconcile["paths"] >= 2
        assert set(reconcile["touched_paths"]) >= {
            "docs/checkout-flow.md",
            "services/api/connect.py",
        }
        assert lifecycle_counters == {
            "build_index": 0,
            "rebuild_index": 0,
            "build_graph": 0,
            "rebuild_graph": 0,
            "update_graph": 1,
        }
        status = graph_service.graph_status()
        assert status.state == "ready"
        assert status.update_required is False
        assert status.stale_path_count == 0
        assert "graph_source_revision_hash_mismatch" not in status.warning_codes

        payload = service.repo_context_search(
            "find docs and tests for /api/v2/connect handler",
            graph_mode="expand",
            top_k=10,
        ).model_dump()
    finally:
        watcher.stop()

    assert set(indexer.reindexed_paths) >= {
        "docs/checkout-flow.md",
        "services/api/connect.py",
    }
    payload_text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    assert "/api/v2/connect" in payload_text
    assert "/api/v1/connect" not in payload_text
    assert payload["diagnostics"]["graph_used"] is True
    assert payload["graph_mode_effective"] == "expand"
    assert any("graph" in result["origin_branches"] for result in payload["results"])
