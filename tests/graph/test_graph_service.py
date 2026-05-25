from __future__ import annotations

from pathlib import Path
import sqlite3

from services.repo_semantic.config import SemanticMcpSettings
from services.repo_semantic.graph import GraphService
from services.repo_semantic.models import ChunkRecord
from services.repo_semantic.registry import RepoRegistry

from tests.helpers.fakes import FakePoint, FakeStore


class _NoInvalidatedPathManifest:
    def invalidated_paths_after(self, timestamp: str, *, limit: int = 12):
        return 0, [], True


class _UnknownInvalidatedPathManifest:
    def invalidated_paths_after(self, timestamp: str, *, limit: int = 12):
        return 0, [], False


class _ManyKnownInvalidatedPathManifest:
    def invalidated_paths_after(self, timestamp: str, *, limit: int = 12):
        paths = [f"services/path_{index}.py" for index in range(30)]
        return len(paths), paths[:limit], len(paths) <= limit


def _settings(tmp_path: Path) -> SemanticMcpSettings:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    return SemanticMcpSettings(
        SEMANTIC_MCP_REPO_ROOT=str(repo_root),
        SEMANTIC_MCP_LOGICAL_REPO_ROOT=str(repo_root),
        SEMANTIC_MCP_REGISTRY_DB_PATH=str(tmp_path / "registry.sqlite3"),
        SEMANTIC_MCP_INCLUDE_GLOBS=["*", "**/*"],
        SEMANTIC_MCP_DOC_PATH_PREFIXES=["docs/"],
        SEMANTIC_MCP_EXCLUDE_GLOBS=[".git/**"],
    )


def _chunk(
    chunk_id: str,
    relative_path: str,
    text: str,
    *,
    scope: str = "code",
    language: str = "python",
    chunk_type: str = "python_function",
    symbol_path: str | None = None,
    heading_path: str | None = None,
    start_line: int = 1,
) -> ChunkRecord:
    return ChunkRecord(
        point_id=chunk_id,
        scope=scope,  # type: ignore[arg-type]
        relative_path=relative_path,
        language=language,
        chunk_type=chunk_type,
        text=text,
        start_line=start_line,
        end_line=start_line + max(1, text.count("\n") + 1) - 1,
        content_hash=f"hash-{chunk_id}",
        source_mtime=1.0,
        symbol_path=symbol_path,
        heading_path=heading_path,
        domain_tags=["services"] if scope == "code" else ["docs"],
    )


def _service_components(tmp_path: Path, chunks: list[ChunkRecord], *, path_manifest=None):
    settings = _settings(tmp_path)
    registry = RepoRegistry(settings.registry_db_path)
    registry.upsert_repo(
        settings.logical_repo_identity,
        status="indexed",
        active=True,
        index_profile="test-profile",
        include_globs=["*", "**/*"],
        doc_prefixes=["docs/"],
        exclude_globs=[".git/**"],
        code_points_count=sum(1 for chunk in chunks if chunk.scope == "code"),
        docs_points_count=sum(1 for chunk in chunks if chunk.scope == "docs"),
    )
    points_by_scope = {
        "code": [FakePoint(chunk, 1.0) for chunk in chunks if chunk.scope == "code"],
        "docs": [FakePoint(chunk, 1.0) for chunk in chunks if chunk.scope == "docs"],
    }
    store = FakeStore(points_by_scope, sparse_enabled=True)
    return (
        GraphService(
            settings=settings,
            chunk_store=store,
            registry=registry,
            path_manifest=path_manifest,
        ),
        store,
        registry,
        settings,
    )


def _service(tmp_path: Path, chunks: list[ChunkRecord]) -> GraphService:
    service, _, _, _ = _service_components(tmp_path, chunks)
    return service


def test_graph_status_reports_missing_without_creating_artifact(tmp_path: Path) -> None:
    service = _service(tmp_path, [])

    status = service.graph_status()

    assert status.state == "missing"
    assert not status.available
    assert not status.expansion_allowed
    assert "graph_missing" in status.warning_codes
    assert not Path(status.graph_path).exists()


def test_build_graph_creates_file_chunk_symbol_doc_and_env_rows(tmp_path: Path) -> None:
    service = _service(
        tmp_path,
        [
            _chunk(
                "code-1",
                "services/api/settings.py",
                "DATABASE_URL = os.environ['DATABASE_URL']\nclass Settings: pass",
                symbol_path="Settings",
            ),
            _chunk(
                "docs-1",
                "docs/checkout-flow.md",
                "# Checkout Flow\nUses DATABASE_URL during runtime.",
                scope="docs",
                language="markdown",
                chunk_type="markdown_section",
                heading_path="Checkout Flow",
            ),
        ],
    )

    result = service.build_graph()

    assert result.built
    assert result.state == "ready"
    assert result.files_count == 2
    assert result.chunks_count == 2
    assert result.nodes_count >= 6
    assert result.edges_count >= 5
    assert result.status is not None
    assert result.status.counts.node_type_counts["file"] == 2
    assert result.status.counts.node_type_counts["chunk"] == 2
    assert result.status.counts.node_type_counts["symbol"] == 1
    assert result.status.counts.node_type_counts["doc_section"] == 1
    assert result.status.counts.node_type_counts["env_var"] == 1
    assert result.status.counts.edge_type_counts["file_contains_chunk"] == 2
    assert result.status.counts.edge_type_counts["chunk_defines_symbol"] == 1
    assert result.status.counts.edge_type_counts["file_contains_doc_section"] == 1
    assert result.status.source_index_contract is not None
    assert result.status.source_index_contract["indexed_chunks_total"] == 2


def test_build_graph_creates_rich_code_reference_edges(tmp_path: Path) -> None:
    service = _service(
        tmp_path,
        [
            _chunk(
                "handler",
                "services/api/checkout_flow.py",
                (
                    "from fastapi import APIRouter\n"
                    "router = APIRouter()\n\n"
                    "@router.get('/api/v1/connect')\n"
                    "def connect_endpoint():\n"
                    "    return {'ok': True}\n"
                ),
                symbol_path="connect_endpoint",
            ),
            _chunk(
                "tests",
                "tests/test_checkout_flow.py",
                (
                    "from services.api.checkout_flow import connect_endpoint\n\n"
                    "def test_connect_endpoint():\n"
                    "    assert connect_endpoint()\n"
                ),
                symbol_path="test_connect_endpoint",
            ),
            _chunk(
                "docs",
                "docs/checkout-flow.md",
                "# Checkout Flow\nCall `/api/v1/connect` from the frontend. See [handler](../services/api/checkout_flow.py).\n",
                scope="docs",
                language="markdown",
                chunk_type="markdown_section",
                heading_path="Checkout Flow",
            ),
        ],
    )

    result = service.build_graph()

    assert result.built
    assert result.status is not None
    node_counts = result.status.counts.node_type_counts
    edge_counts = result.status.counts.edge_type_counts
    assert node_counts["module"] >= 2
    assert node_counts["route"] == 1
    assert node_counts["test_case"] == 1
    assert edge_counts["file_defines_module"] >= 2
    assert edge_counts["module_imports_module"] >= 1
    assert edge_counts["file_imports_file"] >= 1
    assert edge_counts["route_defined_in_chunk"] == 1
    assert edge_counts["route_handled_by_symbol"] == 1
    assert edge_counts["test_targets_file"] >= 1
    assert edge_counts["test_targets_symbol"] == 1
    assert edge_counts["doc_references_route"] == 1
    assert edge_counts["doc_references_path"] == 1


def test_build_graph_extracts_multimethod_call_style_and_relative_import_edges(tmp_path: Path) -> None:
    service = _service(
        tmp_path,
        [
            _chunk(
                "routes",
                "services/web/routes.py",
                (
                    "@app.route('/multi', methods=['GET', 'POST'])\n"
                    "def multi_handler():\n"
                    "    return 'ok'\n\n"
                    "def explicit_handler():\n"
                    "    return 'ok'\n\n"
                    "router.add_api_route('/explicit', explicit_handler, methods=['POST'])\n\n"
                    "def django_view(request):\n"
                    "    return request\n\n"
                    "urlpatterns = [path('django/', django_view)]\n"
                ),
            ),
            _chunk(
                "pkg-init",
                "pkg/__init__.py",
                "from . import handlers\n",
            ),
            _chunk(
                "pkg-handlers",
                "pkg/handlers.py",
                "def handle():\n    return True\n",
            ),
        ],
    )

    result = service.build_graph()

    assert result.status is not None
    edge_counts = result.status.counts.edge_type_counts
    assert result.status.counts.node_type_counts["route"] == 4
    assert edge_counts["route_defined_in_chunk"] == 4
    assert edge_counts["route_handled_by_symbol"] == 4
    assert edge_counts["module_imports_module"] >= 1
    assert edge_counts["file_imports_file"] >= 1


def test_build_graph_keeps_test_targets_per_test_body(tmp_path: Path) -> None:
    service = _service(
        tmp_path,
        [
            _chunk(
                "a",
                "services/a.py",
                "def alpha():\n    return 'a'\n",
                symbol_path="alpha",
            ),
            _chunk(
                "b",
                "services/b.py",
                "def beta():\n    return 'b'\n",
                symbol_path="beta",
            ),
            _chunk(
                "tests",
                "tests/test_ab.py",
                (
                    "from services.a import alpha\n"
                    "from services.b import beta\n\n"
                    "def test_alpha():\n"
                    "    assert alpha()\n\n"
                    "def test_beta():\n"
                    "    assert beta()\n"
                ),
            ),
        ],
    )

    result = service.build_graph()

    assert result.status is not None
    assert result.status.counts.node_type_counts["test_case"] == 2
    assert result.status.counts.edge_type_counts["test_targets_symbol"] == 2
    assert result.status.counts.edge_type_counts["test_targets_file"] == 2


def test_build_graph_surfaces_optional_extractor_warnings(tmp_path: Path) -> None:
    service = _service(
        tmp_path,
        [
            _chunk(
                "broken",
                "services/broken.py",
                "def broken(:\n    pass\n",
            )
        ],
    )

    result = service.build_graph()

    assert result.status is not None
    assert result.status.state == "ready"
    assert result.status.expansion_allowed
    assert "graph_extractor_python_ast_parse_failed" in result.status.warning_codes
    assert "graph_extractor_python_route_ast_parse_failed" in result.status.warning_codes


def test_rebuild_graph_replaces_existing_artifact(tmp_path: Path) -> None:
    service = _service(
        tmp_path,
        [
            _chunk(
                "code-1",
                "services/api/settings.py",
                "DATABASE_URL = os.environ['DATABASE_URL']",
                symbol_path="Settings",
            )
        ],
    )
    first = service.build_graph()

    second = service.rebuild_graph()

    assert first.built
    assert second.built
    assert second.rebuilt
    assert second.files_count == 1
    assert second.status is not None
    assert second.status.counts.file_state_counts == {"ready": 1}


def test_graph_status_returns_error_for_malformed_metadata(tmp_path: Path) -> None:
    service = _service(
        tmp_path,
        [
            _chunk(
                "code-1",
                "services/api/settings.py",
                "DATABASE_URL = os.environ['DATABASE_URL']",
                symbol_path="Settings",
            )
        ],
    )
    service.build_graph()
    with sqlite3.connect(service.graph_path) as connection:
        connection.execute(
            "UPDATE graph_metadata SET source_index_contract_json = ?",
            ("{not-json",),
        )

    status = service.graph_status()

    assert status.state == "error"
    assert not status.available
    assert status.error_code == "graph_status_read_failed"


def test_graph_status_marks_stale_when_chunk_source_contract_changes(tmp_path: Path) -> None:
    service, store, _, _ = _service_components(
        tmp_path,
        [
            _chunk(
                "code-1",
                "services/api/settings.py",
                "DATABASE_URL = os.environ['DATABASE_URL']",
                symbol_path="Settings",
            )
        ],
    )
    service.build_graph()
    changed = _chunk(
        "code-1",
        "services/api/settings.py",
        "REDIS_URL = os.environ['REDIS_URL']",
        symbol_path="Settings",
    )
    changed.content_hash = "changed-content"
    store._points_by_scope["code"] = [FakePoint(changed, 1.0)]

    status = service.graph_status()

    assert status.state == "stale"
    assert not status.expansion_allowed
    assert "graph_source_index_contract_mismatch" in status.warning_codes
    assert "graph_source_revision_hash_mismatch" in status.warning_codes


def test_graph_status_allows_query_aware_expansion_when_invalidated_paths_are_known(tmp_path: Path) -> None:
    service = _service(
        tmp_path,
        [
            _chunk(
                "code-1",
                "services/api/settings.py",
                "DATABASE_URL = os.environ['DATABASE_URL']",
                symbol_path="Settings",
            )
        ],
    )
    ready = service.build_graph().status
    assert ready is not None

    status = service._store.status(
        repo_root=service._settings.logical_repo_identity,
        profile=service._profile(),
        current_head_commit=ready.built_commit,
        expected_source_index_contract=ready.source_index_contract,
        expected_extractor_versions_hash=ready.extractor_versions_hash,
        invalidated_path_count=1,
        invalidated_paths_preview=["services/api/settings.py"],
    )

    assert status.state == "stale"
    assert status.expansion_allowed
    assert "graph_worktree_changes" in status.warning_codes


def test_graph_status_blocks_expansion_when_invalidated_paths_are_unknown(tmp_path: Path) -> None:
    service = _service(
        tmp_path,
        [
            _chunk(
                "code-1",
                "services/api/settings.py",
                "DATABASE_URL = os.environ['DATABASE_URL']",
                symbol_path="Settings",
            )
        ],
    )
    ready = service.build_graph().status
    assert ready is not None

    status = service._store.status(
        repo_root=service._settings.logical_repo_identity,
        profile=service._profile(),
        current_head_commit=ready.built_commit,
        expected_source_index_contract=ready.source_index_contract,
        expected_extractor_versions_hash=ready.extractor_versions_hash,
        invalidated_path_count=2,
        invalidated_paths_preview=["services/api/settings.py"],
    )

    assert status.state == "stale"
    assert not status.expansion_allowed
    assert "graph_worktree_changes" in status.warning_codes


def test_graph_summary_marks_stale_when_index_revision_changes(tmp_path: Path) -> None:
    service, _, registry, settings = _service_components(
        tmp_path,
        [
            _chunk(
                "code-1",
                "services/api/settings.py",
                "DATABASE_URL = os.environ['DATABASE_URL']",
                symbol_path="Settings",
            )
        ],
        path_manifest=_NoInvalidatedPathManifest(),
    )
    service.build_graph()
    entry = registry.get_repo(settings.logical_repo_identity)
    assert entry is not None
    registry.upsert_repo(
        settings.logical_repo_identity,
        status=entry.status,
        active=entry.active,
        index_profile=entry.index_profile,
        include_globs=entry.include_globs,
        doc_prefixes=entry.doc_prefixes,
        exclude_globs=entry.exclude_globs,
        last_full_build_ts=entry.last_full_build_ts,
        last_incremental_update_ts="2026-05-06T01:00:00+00:00",
        indexed_branch=entry.indexed_branch,
        indexed_commit_hash=entry.indexed_commit_hash,
        code_points_count=entry.code_points_count,
        docs_points_count=entry.docs_points_count,
    )

    summary = service.graph_summary()

    assert summary.state == "stale"
    assert summary.expansion_allowed
    assert "graph_source_index_contract_mismatch" in summary.warning_codes
    assert "graph_source_index_revision_mismatch" in summary.warning_codes


def test_graph_summary_blocks_revision_mismatch_when_manifest_paths_unknown(tmp_path: Path) -> None:
    service, _, registry, settings = _service_components(
        tmp_path,
        [
            _chunk(
                "code-1",
                "services/api/settings.py",
                "DATABASE_URL = os.environ['DATABASE_URL']",
                symbol_path="Settings",
            )
        ],
        path_manifest=_UnknownInvalidatedPathManifest(),
    )
    service.build_graph()
    entry = registry.get_repo(settings.logical_repo_identity)
    assert entry is not None
    registry.upsert_repo(
        settings.logical_repo_identity,
        status=entry.status,
        active=entry.active,
        index_profile=entry.index_profile,
        include_globs=entry.include_globs,
        doc_prefixes=entry.doc_prefixes,
        exclude_globs=entry.exclude_globs,
        last_full_build_ts=entry.last_full_build_ts,
        last_incremental_update_ts="2026-05-06T01:00:00+00:00",
        indexed_branch=entry.indexed_branch,
        indexed_commit_hash=entry.indexed_commit_hash,
        code_points_count=entry.code_points_count,
        docs_points_count=entry.docs_points_count,
    )

    summary = service.graph_summary()

    assert summary.state == "stale"
    assert not summary.expansion_allowed
    assert "graph_invalidated_paths_unknown" in summary.warning_codes


def test_graph_summary_uses_configured_invalidated_path_limit(tmp_path: Path) -> None:
    service, _, registry, settings = _service_components(
        tmp_path,
        [
            _chunk(
                "code-1",
                "services/api/settings.py",
                "DATABASE_URL = os.environ['DATABASE_URL']",
                symbol_path="Settings",
            )
        ],
        path_manifest=_ManyKnownInvalidatedPathManifest(),
    )
    service.build_graph()
    entry = registry.get_repo(settings.logical_repo_identity)
    assert entry is not None
    registry.upsert_repo(
        settings.logical_repo_identity,
        status=entry.status,
        active=entry.active,
        index_profile=entry.index_profile,
        include_globs=entry.include_globs,
        doc_prefixes=entry.doc_prefixes,
        exclude_globs=entry.exclude_globs,
        last_full_build_ts=entry.last_full_build_ts,
        last_incremental_update_ts="2026-05-06T01:00:00+00:00",
        indexed_branch=entry.indexed_branch,
        indexed_commit_hash=entry.indexed_commit_hash,
        code_points_count=entry.code_points_count,
        docs_points_count=entry.docs_points_count,
    )

    summary = service.graph_summary()

    assert summary.state == "stale"
    assert summary.expansion_allowed
    assert summary.stale_path_count == 30
    assert len(summary.invalidated_paths_preview) == 30
    assert "graph_invalidated_paths_unknown" not in summary.warning_codes


def test_update_graph_discovers_signature_drift_and_updates_path_rows(tmp_path: Path) -> None:
    service, store, _, _ = _service_components(
        tmp_path,
        [
            _chunk(
                "settings",
                "services/api/settings.py",
                "DATABASE_URL = os.environ['DATABASE_URL']",
                symbol_path="Settings",
            )
        ],
    )
    service.build_graph()
    changed = _chunk(
        "settings",
        "services/api/settings.py",
        "REDIS_URL = os.environ['REDIS_URL']",
        symbol_path="Settings",
    )
    changed.content_hash = "changed-settings"
    store._points_by_scope["code"] = [FakePoint(changed, 1.0)]

    result = service.update_graph(paths=[])

    assert result.updated
    assert result.paths_discovered == 1
    assert result.paths_required == 1
    assert result.status is not None
    assert result.status.state == "ready"
    assert not result.status.update_required
    with sqlite3.connect(service.graph_path) as connection:
        env_keys = {
            row[0]
            for row in connection.execute(
                "SELECT key FROM graph_nodes WHERE node_type = 'env_var'"
            ).fetchall()
        }
    assert env_keys == {"REDIS_URL"}


def test_update_graph_explicit_path_is_idempotent(tmp_path: Path) -> None:
    service, store, _, _ = _service_components(
        tmp_path,
        [
            _chunk(
                "settings",
                "services/api/settings.py",
                "DATABASE_URL = os.environ['DATABASE_URL']",
                symbol_path="Settings",
            )
        ],
    )
    service.build_graph()
    changed = _chunk(
        "settings-v2",
        "services/api/settings.py",
        "DATABASE_URL = os.environ['DATABASE_URL']\nREDIS_URL = os.environ['REDIS_URL']",
        symbol_path="Settings",
    )
    changed.content_hash = "settings-v2"
    store._points_by_scope["code"] = [FakePoint(changed, 1.0)]

    first = service.update_graph(paths=["services/api/settings.py"], idempotency_key="same-update")
    second = service.update_graph(paths=["services/api/settings.py"], idempotency_key="same-update")

    assert first.updated
    assert second.updated
    assert second.idempotent_retry
    with sqlite3.connect(service.graph_path) as connection:
        rows = connection.execute(
            "SELECT node_id, normalized_term FROM graph_node_terms ORDER BY node_id, normalized_term"
        ).fetchall()
    assert len(rows) == len(set(rows))


def test_update_graph_idempotency_key_does_not_hide_new_file_error(tmp_path: Path) -> None:
    service, store, _, _ = _service_components(
        tmp_path,
        [
            _chunk(
                "settings",
                "services/api/settings.py",
                "DATABASE_URL = os.environ['DATABASE_URL']",
                symbol_path="Settings",
            )
        ],
    )
    service.build_graph()
    changed = _chunk(
        "settings",
        "services/api/settings.py",
        "DATABASE_URL = os.environ['DATABASE_URL']\nREDIS_URL = os.environ['REDIS_URL']",
        symbol_path="Settings",
    )
    changed.content_hash = "changed-settings"
    store._points_by_scope["code"] = [FakePoint(changed, 1.0)]
    first = service.update_graph(paths=["services/api/settings.py"], idempotency_key="same-update")
    assert first.updated

    with sqlite3.connect(service.graph_path) as connection:
        connection.execute(
            """
            UPDATE graph_files
            SET graph_state = 'error', error_code = 'forced_error'
            WHERE relative_path = 'services/api/settings.py'
            """
        )

    second = service.update_graph(paths=[], idempotency_key="same-update")

    assert second.updated
    assert not second.idempotent_retry
    assert second.paths_required == 1


def test_update_graph_chunk_count_bound_fails_before_scrolling_large_index(tmp_path: Path) -> None:
    service, store, _, settings = _service_components(
        tmp_path,
        [
            _chunk(
                "settings",
                "services/api/settings.py",
                "DATABASE_URL = os.environ['DATABASE_URL']",
                symbol_path="Settings",
            )
        ],
    )
    service.build_graph()
    settings.SEMANTIC_MCP_GRAPH_UPDATE_MAX_CHUNKS_SCANNED = 100
    store._points_by_scope["code"] = [
        FakePoint(
            _chunk(f"settings-{index}", f"services/api/settings_{index}.py", "VALUE = 1"),
            1.0,
        )
        for index in range(101)
    ]
    store._fail_on_scroll = True

    result = service.update_graph(paths=["services/api/settings.py"])

    assert not result.updated
    assert result.skip_reason == "graph_update_bounds_exceeded"
    assert result.bounds_exceeded == ["max_chunks_scanned"]
    assert result.recommended_actions[0]["tool_arguments"] == {
        "paths": [],
        "allow_large_update": True,
    }


def test_update_graph_row_bounds_roll_back_without_metadata_update(tmp_path: Path) -> None:
    service, store, _, settings = _service_components(
        tmp_path,
        [
            _chunk(
                "settings",
                "services/api/settings.py",
                "DATABASE_URL = os.environ['DATABASE_URL']",
                symbol_path="Settings",
            )
        ],
    )
    service.build_graph()
    changed = _chunk(
        "settings",
        "services/api/settings.py",
        "DATABASE_URL = os.environ['DATABASE_URL']\nREDIS_URL = os.environ['REDIS_URL']",
        symbol_path="Settings",
    )
    changed.content_hash = "changed-settings"
    store._points_by_scope["code"] = [FakePoint(changed, 1.0)]
    settings.SEMANTIC_MCP_GRAPH_UPDATE_MAX_ROWS_UPSERTED = 1

    result = service.update_graph(paths=["services/api/settings.py"])

    assert not result.updated
    assert result.skip_reason == "graph_update_bounds_exceeded"
    assert result.bounds_exceeded == ["max_rows_upserted"]
    status = service.graph_status()
    assert status.state == "stale"
    assert "graph_source_revision_hash_mismatch" in status.warning_codes


def test_update_graph_allow_large_uses_manual_row_and_duration_bounds(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import services.repo_semantic.graph.service as graph_service_module
    import services.repo_semantic.graph.store as graph_store_module

    clock = {"value": 100.0}

    def fake_perf_counter() -> float:
        clock["value"] += 1.0
        return clock["value"]

    monkeypatch.setattr(graph_service_module, "perf_counter", fake_perf_counter)
    monkeypatch.setattr(graph_store_module, "perf_counter", fake_perf_counter)
    service, store, _, settings = _service_components(
        tmp_path,
        [
            _chunk(
                "settings",
                "services/api/settings.py",
                "DATABASE_URL = os.environ['DATABASE_URL']",
                symbol_path="Settings",
            )
        ],
    )
    service.build_graph()
    changed = _chunk(
        "settings",
        "services/api/settings.py",
        "DATABASE_URL = os.environ['DATABASE_URL']\nREDIS_URL = os.environ['REDIS_URL']",
        symbol_path="Settings",
    )
    changed.content_hash = "changed-settings"
    store._points_by_scope["code"] = [FakePoint(changed, 1.0)]
    settings.SEMANTIC_MCP_GRAPH_UPDATE_MAX_ROWS_UPSERTED = 1
    settings.SEMANTIC_MCP_GRAPH_UPDATE_LARGE_MAX_ROWS_UPSERTED = 1000
    settings.SEMANTIC_MCP_GRAPH_UPDATE_MAX_DURATION_MS = 1
    settings.SEMANTIC_MCP_GRAPH_UPDATE_LARGE_MAX_DURATION_MS = 100000

    result = service.update_graph(paths=["services/api/settings.py"], allow_large_update=True)

    assert result.updated
    assert result.allow_large_update
    assert result.nodes_upserted > 0


def test_update_graph_delete_row_bounds_include_replacement_deletes(tmp_path: Path) -> None:
    service, store, _, settings = _service_components(
        tmp_path,
        [
            _chunk(
                "settings",
                "services/api/settings.py",
                "DATABASE_URL = os.environ['DATABASE_URL']",
                symbol_path="Settings",
            )
        ],
    )
    service.build_graph()
    changed = _chunk(
        "settings",
        "services/api/settings.py",
        "DATABASE_URL = os.environ['DATABASE_URL']\nREDIS_URL = os.environ['REDIS_URL']",
        symbol_path="Settings",
    )
    changed.content_hash = "changed-settings"
    store._points_by_scope["code"] = [FakePoint(changed, 1.0)]
    settings.SEMANTIC_MCP_GRAPH_UPDATE_MAX_ROWS_DELETED = 0

    result = service.update_graph(paths=["services/api/settings.py"])

    assert not result.updated
    assert result.bounds_exceeded == ["max_rows_deleted"]
    status = service.graph_status()
    assert status.state == "stale"


def test_update_graph_explicit_paths_are_seeds_not_complete_plan(tmp_path: Path) -> None:
    service, store, _, _ = _service_components(
        tmp_path,
        [
            _chunk("a", "services/a.py", "A_ENV = os.environ['A_ENV']", symbol_path="A"),
            _chunk("b", "services/b.py", "B_ENV = os.environ['B_ENV']", symbol_path="B"),
        ],
    )
    service.build_graph()
    changed_b = _chunk(
        "b",
        "services/b.py",
        "B2_ENV = os.environ['B2_ENV']",
        symbol_path="B",
    )
    changed_b.content_hash = "changed-b"
    store._points_by_scope["code"] = [
        point
        for point in store._points_by_scope["code"]
        if point.payload["relative_path"] != "services/b.py"
    ]
    store._points_by_scope["code"].append(FakePoint(changed_b, 1.0))

    result = service.update_graph(paths=["services/a.py"])

    assert result.updated
    assert result.paths_required == 2
    assert result.status is not None
    assert result.status.state == "ready"
    with sqlite3.connect(service.graph_path) as connection:
        env_keys = {
            row[0]
            for row in connection.execute(
                "SELECT key FROM graph_nodes WHERE node_type = 'env_var'"
            ).fetchall()
        }
    assert "B2_ENV" in env_keys
    assert "B_ENV" not in env_keys


def test_update_graph_includes_shared_route_reverse_dependents(tmp_path: Path) -> None:
    service, store, _, _ = _service_components(
        tmp_path,
        [
            _chunk(
                "handler",
                "services/api/checkout_flow.py",
                (
                    "from fastapi import APIRouter\n"
                    "router = APIRouter()\n\n"
                    "@router.get('/api/v1/connect')\n"
                    "def connect_endpoint():\n"
                    "    return {'ok': True}\n"
                ),
                symbol_path="connect_endpoint",
            ),
            _chunk(
                "docs",
                "docs/checkout-flow.md",
                "# Checkout Flow\nCall `/api/v1/connect` from the frontend.\n",
                scope="docs",
                language="markdown",
                chunk_type="markdown_section",
                heading_path="Checkout Flow",
            ),
        ],
    )
    service.build_graph()
    changed_handler = _chunk(
        "handler",
        "services/api/checkout_flow.py",
        (
            "from fastapi import APIRouter\n"
            "router = APIRouter()\n\n"
            "@router.get('/api/v1/connect')\n"
            "def connect_endpoint():\n"
            "    return {'changed': True}\n"
        ),
        symbol_path="connect_endpoint",
    )
    changed_handler.content_hash = "changed-handler"
    store._points_by_scope["code"] = [FakePoint(changed_handler, 1.0)]

    result = service.update_graph(paths=["services/api/checkout_flow.py"])

    assert result.updated
    assert result.reverse_dependent_paths_added >= 1
    with sqlite3.connect(service.graph_path) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM graph_edges WHERE edge_type = 'doc_references_route'"
        ).fetchone()[0]
    assert count == 1


def test_update_graph_discovers_new_shared_route_reverse_dependents(tmp_path: Path) -> None:
    service, store, _, _ = _service_components(
        tmp_path,
        [
            _chunk(
                "handler",
                "services/api/checkout_flow.py",
                "def connect_endpoint():\n    return {'ok': True}\n",
                symbol_path="connect_endpoint",
            ),
            _chunk(
                "docs",
                "docs/checkout-flow.md",
                "# Checkout Flow\nCall `/api/v1/new` from the frontend.\n",
                scope="docs",
                language="markdown",
                chunk_type="markdown_section",
                heading_path="Checkout Flow",
            ),
        ],
    )
    service.build_graph()
    changed_handler = _chunk(
        "handler",
        "services/api/checkout_flow.py",
        (
            "from fastapi import APIRouter\n"
            "router = APIRouter()\n\n"
            "@router.get('/api/v1/new')\n"
            "def connect_endpoint():\n"
            "    return {'ok': True}\n"
        ),
        symbol_path="connect_endpoint",
    )
    changed_handler.content_hash = "new-route"
    store._points_by_scope["code"] = [FakePoint(changed_handler, 1.0)]

    result = service.update_graph(paths=["services/api/checkout_flow.py"])

    assert result.updated
    assert result.reverse_dependent_paths_added >= 1
    with sqlite3.connect(service.graph_path) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM graph_edges WHERE edge_type = 'doc_references_route'"
        ).fetchone()[0]
    assert count == 1


def test_update_graph_reverse_dependency_zero_bound_fails_closed(tmp_path: Path) -> None:
    service, store, _, settings = _service_components(
        tmp_path,
        [
            _chunk(
                "handler",
                "services/api/checkout_flow.py",
                (
                    "from fastapi import APIRouter\n"
                    "router = APIRouter()\n\n"
                    "@router.get('/api/v1/connect')\n"
                    "def connect_endpoint():\n"
                    "    return {'ok': True}\n"
                ),
                symbol_path="connect_endpoint",
            ),
            _chunk(
                "docs",
                "docs/checkout-flow.md",
                "# Checkout Flow\nCall `/api/v1/connect` from the frontend.\n",
                scope="docs",
                language="markdown",
                chunk_type="markdown_section",
                heading_path="Checkout Flow",
            ),
        ],
    )
    service.build_graph()
    settings.SEMANTIC_MCP_GRAPH_UPDATE_MAX_REVERSE_DEPENDENT_PATHS = 0
    changed_handler = _chunk(
        "handler",
        "services/api/checkout_flow.py",
        (
            "from fastapi import APIRouter\n"
            "router = APIRouter()\n\n"
            "@router.get('/api/v1/connect')\n"
            "def connect_endpoint():\n"
            "    return {'changed': True}\n"
        ),
        symbol_path="connect_endpoint",
    )
    changed_handler.content_hash = "changed-handler"
    store._points_by_scope["code"] = [FakePoint(changed_handler, 1.0)]

    result = service.update_graph(paths=["services/api/checkout_flow.py"])

    assert not result.updated
    assert result.skip_reason == "reverse_dependencies_exceed_auto_bounds"
    assert "graph_reverse_dependencies_not_repaired" in result.warning_codes
    assert result.recommended_actions[0]["code"] == "update_graph"
    assert result.recommended_actions[0]["tool_arguments"] == {
        "paths": [],
        "allow_large_update": True,
    }

    manual_result = service.update_graph(
        paths=["services/api/checkout_flow.py"],
        allow_large_update=True,
    )

    assert manual_result.updated
    assert manual_result.allow_large_update
    assert manual_result.reverse_dependent_paths_added == 1


def test_update_graph_metadata_only_repairs_revision_drift(tmp_path: Path) -> None:
    service, _, registry, settings = _service_components(
        tmp_path,
        [
            _chunk(
                "settings",
                "services/api/settings.py",
                "DATABASE_URL = os.environ['DATABASE_URL']",
                symbol_path="Settings",
            )
        ],
    )
    service.build_graph()
    entry = registry.get_repo(settings.logical_repo_identity)
    assert entry is not None
    registry.upsert_repo(
        settings.logical_repo_identity,
        status=entry.status,
        active=entry.active,
        index_profile=entry.index_profile,
        include_globs=entry.include_globs,
        doc_prefixes=entry.doc_prefixes,
        exclude_globs=entry.exclude_globs,
        last_full_build_ts=entry.last_full_build_ts,
        last_incremental_update_ts="2026-05-07T00:00:00+00:00",
        indexed_branch=entry.indexed_branch,
        indexed_commit_hash=entry.indexed_commit_hash,
        code_points_count=entry.code_points_count,
        docs_points_count=entry.docs_points_count,
    )
    stale = service.graph_status()
    assert stale.update_required

    result = service.update_graph(paths=[])

    assert result.updated
    assert result.metadata_only_update
    assert result.paths_required == 0
    assert result.status is not None
    assert result.status.state == "ready"
    assert not result.status.update_required


def test_update_graph_backfills_legacy_missing_signatures_without_full_path_update(tmp_path: Path) -> None:
    service, _, registry, settings = _service_components(
        tmp_path,
        [
            _chunk("a", "services/a.py", "A_ENV = os.environ['A_ENV']", symbol_path="A"),
            _chunk("b", "services/b.py", "B_ENV = os.environ['B_ENV']", symbol_path="B"),
        ],
    )
    service.build_graph()
    with sqlite3.connect(service.graph_path) as connection:
        connection.execute("UPDATE graph_files SET graph_input_signature = NULL")

    entry = registry.get_repo(settings.logical_repo_identity)
    assert entry is not None
    registry.upsert_repo(
        settings.logical_repo_identity,
        status=entry.status,
        active=entry.active,
        index_profile=entry.index_profile,
        include_globs=entry.include_globs,
        doc_prefixes=entry.doc_prefixes,
        exclude_globs=entry.exclude_globs,
        last_full_build_ts=entry.last_full_build_ts,
        last_incremental_update_ts="2026-05-07T01:00:00+00:00",
        indexed_branch=entry.indexed_branch,
        indexed_commit_hash=entry.indexed_commit_hash,
        code_points_count=entry.code_points_count,
        docs_points_count=entry.docs_points_count,
    )
    settings.SEMANTIC_MCP_GRAPH_UPDATE_MAX_PATHS = 1

    result = service.update_graph(paths=[])

    assert result.updated
    assert result.metadata_only_update
    assert result.paths_discovered == 0
    with sqlite3.connect(service.graph_path) as connection:
        missing = connection.execute(
            "SELECT COUNT(*) FROM graph_files WHERE graph_input_signature IS NULL"
        ).fetchone()[0]
    assert missing == 0


def test_update_graph_does_not_backfill_legacy_signature_when_graph_rows_differ(
    tmp_path: Path,
) -> None:
    original = _chunk(
        "settings",
        "services/api/settings.py",
        "DATABASE_URL = os.environ['DATABASE_URL']",
        symbol_path="Settings",
    )
    service, store, registry, settings = _service_components(tmp_path, [original])
    service.build_graph()
    with sqlite3.connect(service.graph_path) as connection:
        connection.execute("UPDATE graph_files SET graph_input_signature = NULL")

    changed = _chunk(
        "settings",
        "services/api/settings.py",
        "DATABASE_URL = os.environ['DATABASE_URL']",
        symbol_path="RenamedSettings",
    )
    changed.content_hash = original.content_hash
    store._points_by_scope["code"] = [FakePoint(changed, 1.0)]
    entry = registry.get_repo(settings.logical_repo_identity)
    assert entry is not None
    registry.upsert_repo(
        settings.logical_repo_identity,
        status=entry.status,
        active=entry.active,
        index_profile=entry.index_profile,
        include_globs=entry.include_globs,
        doc_prefixes=entry.doc_prefixes,
        exclude_globs=entry.exclude_globs,
        last_full_build_ts=entry.last_full_build_ts,
        last_incremental_update_ts="2026-05-07T01:00:00+00:00",
        indexed_branch=entry.indexed_branch,
        indexed_commit_hash=entry.indexed_commit_hash,
        code_points_count=entry.code_points_count,
        docs_points_count=entry.docs_points_count,
    )

    result = service.update_graph(paths=[])

    assert result.updated
    assert not result.metadata_only_update
    assert result.paths_discovered == 1
    assert result.nodes_upserted > 0
    with sqlite3.connect(service.graph_path) as connection:
        symbols = {
            row[0]
            for row in connection.execute(
                "SELECT key FROM graph_nodes WHERE node_type = 'symbol'"
            ).fetchall()
        }
    assert "RenamedSettings" in symbols
    assert "Settings" not in symbols


def test_update_graph_deletes_removed_path_rows(tmp_path: Path) -> None:
    service, store, _, _ = _service_components(
        tmp_path,
        [
            _chunk("settings", "services/api/settings.py", "DATABASE_URL = 'x'", symbol_path="Settings"),
            _chunk("old", "services/old.py", "OLD_ENV = 'x'", symbol_path="old"),
        ],
    )
    service.build_graph()
    store._points_by_scope["code"] = [
        point
        for point in store._points_by_scope["code"]
        if point.payload["relative_path"] != "services/old.py"
    ]

    result = service.update_graph(paths=[])

    assert result.updated
    assert result.paths_deleted == 1
    with sqlite3.connect(service.graph_path) as connection:
        rows = connection.execute(
            "SELECT COUNT(*) FROM graph_nodes WHERE relative_path = 'services/old.py'"
        ).fetchone()
    assert rows is not None
    assert rows[0] == 0
