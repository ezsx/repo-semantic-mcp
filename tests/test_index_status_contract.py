from __future__ import annotations

import subprocess
import hashlib
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from services.repo_semantic.config import SemanticMcpSettings
from services.repo_semantic import models
from services.repo_semantic.models import ChunkRecord
from services.repo_semantic.path_manifest import PathManifestStore
from services.repo_semantic.registry import RepoRegistry
from services.repo_semantic.sparse import (
    LEXICAL_ANALYZER_VERSION,
    SPARSE_ENCODER_KIND,
    build_sparse_manifest,
)

rank_bm25 = types.ModuleType("rank_bm25")


class _BM25Okapi:
    def __init__(self, tokens):
        self._tokens = tokens

    def get_scores(self, query_tokens):
        return [0.0 for _ in self._tokens]


rank_bm25.BM25Okapi = _BM25Okapi
sys.modules.setdefault("rank_bm25", rank_bm25)

from services.repo_semantic.search_service import SearchService


class _FakeEmbeddingProvider:
    def backend_name(self) -> str:
        return "fake"

    def model_name(self) -> str:
        return "fake-model"

    def index_profile(self) -> str:
        return "test-profile"

    def embed_query(self, query: str) -> list[float]:
        return [0.0]


class _FakeStore:
    def __init__(
        self,
        *,
        code_points: int = 0,
        docs_points: int = 0,
        contracts=None,
        sparse_enabled: bool = False,
        payload_fields: set[str] | None = None,
        payload_index_fields: set[str] | None = None,
    ) -> None:
        self._counts = {"code": code_points, "docs": docs_points}
        self._contracts = contracts or {}
        self._sparse_enabled = sparse_enabled
        self._payload_fields = payload_fields
        self._payload_index_fields = payload_index_fields

    def collection_name(self, scope: str) -> str:
        return f"test_{scope}"

    def collection_exists(self, scope: str) -> bool:
        return self._counts.get(scope, 0) > 0

    def count(self, scope: str) -> int:
        return self._counts.get(scope, 0)

    def collection_embedding_contract(self, scope: str):
        return self._contracts.get(scope)

    def read_collection_exists(self, scope: str) -> bool:
        return self.collection_exists(scope)

    def current_collection_exists(self, scope: str) -> bool:
        return self.collection_exists(scope)

    def dense_schema_kind(self, scope: str) -> str:
        if not self.collection_exists(scope):
            return "missing"
        return "named" if self._sparse_enabled else "unnamed_legacy"

    def sparse_vector_available(self, scope: str) -> bool:
        return self._sparse_enabled and self.collection_exists(scope)

    def payload_fields(self, scope: str) -> set[str]:
        if not self.collection_exists(scope):
            return set()
        if self._payload_fields is not None:
            return set(self._payload_fields)
        return {
            "scope",
            "relative_path",
            "file_extension",
            "language",
            "chunk_type",
            "domain_tags",
            "is_generated",
            "content_hash",
            "path_prefixes",
        } if self._sparse_enabled else {
            "scope",
            "relative_path",
            "language",
            "chunk_type",
            "domain_tags",
            "content_hash",
        }

    def payload_index_fields(self, scope: str) -> set[str] | None:
        if not self.collection_exists(scope):
            return set()
        return None if self._payload_index_fields is None else set(self._payload_index_fields)


class _FakeIndexer:
    last_full_build_ts = "2026-05-06T00:00:00+00:00"
    last_incremental_update_ts = "2026-05-06T00:00:00+00:00"

    def __init__(self, settings: SemanticMcpSettings | None = None) -> None:
        self._settings = settings
        if settings is not None:
            self.path_manifest = PathManifestStore(settings)

    def iter_indexable_paths(self):
        return []


class _FakeWatcher:
    def __init__(self, *, running: bool = False) -> None:
        self.is_running = running


def _chunk(chunk_id: str, text: str = "DATABASE_URL settings") -> ChunkRecord:
    return ChunkRecord(
        point_id=chunk_id,
        scope="code",
        relative_path=f"src/{chunk_id}.py",
        language="python",
        chunk_type="python_function",
        text=text,
        start_line=1,
        end_line=1,
        content_hash=chunk_id,
        source_mtime=0.0,
    )


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


class IndexStatusContractTests(unittest.TestCase):
    def _build_service(
        self,
        tmp: str,
        repo_root: Path,
        *,
        repo_status: str = "indexed",
        active: bool = True,
        selected_backend_id: str | None = None,
        code_points: int = 1,
        docs_points: int = 0,
        contracts=None,
        indexed_branch: str | None = None,
        indexed_commit_hash: str | None = None,
        sparse_enabled: bool = False,
        payload_fields: set[str] | None = None,
        payload_index_fields: set[str] | None = None,
        graph_status_summary=None,
        watch_enabled: bool = False,
        watcher=None,
    ) -> SearchService:
        registry = RepoRegistry(Path(tmp) / "registry.sqlite3")
        settings = SemanticMcpSettings(
            SEMANTIC_MCP_REPO_ROOT=str(repo_root),
            SEMANTIC_MCP_LOGICAL_REPO_ROOT=str(repo_root),
            SEMANTIC_MCP_REGISTRY_DB_PATH=str(Path(tmp) / "registry.sqlite3"),
            SEMANTIC_MCP_INCLUDE_GLOBS=["*", "**/*"],
            SEMANTIC_MCP_DOC_PATH_PREFIXES=["docs/"],
            SEMANTIC_MCP_EXCLUDE_GLOBS=[".git/**"],
            SEMANTIC_MCP_WATCH_ENABLED=watch_enabled,
        )
        backend_payload = settings.current_embedding_backend_payload()
        registry.upsert_backend(**backend_payload)
        if selected_backend_id and selected_backend_id != settings.embedding_backend_id:
            other_payload = dict(backend_payload)
            other_payload["backend_id"] = selected_backend_id
            registry.upsert_backend(**other_payload)
        registry.set_role_backend("embedding", selected_backend_id or settings.embedding_backend_id)
        registry.upsert_repo(
            settings.logical_repo_identity,
            status=repo_status,
            active=active,
            index_profile="test-profile",
            include_globs=["*", "**/*"],
            doc_prefixes=["docs/"],
            exclude_globs=[".git/**"],
            indexed_branch=indexed_branch,
            indexed_commit_hash=indexed_commit_hash,
            code_points_count=code_points,
            docs_points_count=docs_points,
        )
        return SearchService(
            settings=settings,
            embedding_provider=_FakeEmbeddingProvider(),
            store=_FakeStore(
                code_points=code_points,
                docs_points=docs_points,
                contracts=contracts,
                sparse_enabled=sparse_enabled,
                payload_fields=payload_fields,
                payload_index_fields=payload_index_fields,
            ),
            indexer=_FakeIndexer(settings),
            watcher=watcher,
            registry=registry,
            graph_status_summary=graph_status_summary,
        )

    def test_index_status_exposes_v2_freshness_without_blocking_search_availability(self) -> None:
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            _git(repo_root, "init")
            _git(repo_root, "config", "user.email", "tests@example.local")
            _git(repo_root, "config", "user.name", "Tests")
            (repo_root / "service.py").write_text("print('v1')\n", encoding="utf-8")
            _git(repo_root, "add", "service.py")
            _git(repo_root, "commit", "-m", "initial")
            head_commit = _git(repo_root, "rev-parse", "HEAD")
            branch = _git(repo_root, "branch", "--show-current")
            service = self._build_service(
                tmp,
                repo_root,
                indexed_branch=branch,
                indexed_commit_hash=head_commit,
            )

            (repo_root / "service.py").write_text("print('v2')\n", encoding="utf-8")
            (repo_root / "notes.md").write_text("# Notes\n", encoding="utf-8")
            status = service.index_status()
            payload = status.model_dump()

            self.assertTrue(payload["search_available"])
            self.assertEqual(payload["head_commit_hash"], head_commit)
            self.assertEqual(payload["indexed_commit_hash"], head_commit)
            self.assertTrue(payload["worktree_dirty"])
            self.assertTrue(payload["index_stale"])
            self.assertIn("git", payload)
            self.assertIn("freshness", payload)
            self.assertIn("path_freshness", payload)
            self.assertFalse(payload["path_freshness"]["coverage_complete"])
            self.assertEqual(payload["path_freshness"]["coverage_error_code"], "path_manifest_missing")
            self.assertIn("indexable_tracked_files_changed", payload["freshness"]["stale_reasons"])
            self.assertIn("indexable_untracked_files_present", payload["freshness"]["stale_reasons"])
            self.assertEqual(payload["contract_version"], "index_status.v2")
            self.assertEqual(payload["availability"]["search_unavailable_codes"], [])
            self.assertTrue(payload["availability"]["search_available"])
            self.assertEqual(payload["availability"]["next_actions"][0]["code"], "use_exact_search_fallback")
            self.assertEqual(payload["freshness"]["policy"]["stale_severity"], "warning")
            self.assertTrue(payload["freshness"]["policy"]["exact_fallback_recommended"])
            self.assertEqual(payload["counts"]["code_points"], 1)
            self.assertEqual(payload["counts"]["total_points"], 1)
            self.assertIsNone(payload["counts"]["code_lexical_documents"])
            self.assertEqual(payload["lifecycle"]["state"], "indexed")
            self.assertFalse(payload["lifecycle"]["action_in_progress"])
            self.assertEqual(payload["watcher"]["policy"], "single_active_repo")
            self.assertEqual(payload["collections"][0]["lexical_documents"], 1)

    def test_index_status_reports_missing_path_manifest_as_unknown_coverage(self) -> None:
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            _git(repo_root, "init")
            service = self._build_service(tmp, repo_root)

            payload = service.index_status().model_dump()

            self.assertIsNotNone(payload["path_freshness"])
            self.assertFalse(payload["path_freshness"]["coverage_complete"])
            self.assertFalse(payload["path_freshness"]["manifest_available"])
            self.assertEqual(payload["path_freshness"]["coverage_error_code"], "path_manifest_missing")
            self.assertEqual(payload["freshness"]["primary_state"], "unknown")
            self.assertIn("path_manifest_missing", payload["freshness"]["reason_codes"])
            self.assertIn("repo_semantic_ensure_ready.py", payload["freshness"]["host_side_hint"])
            self.assertIn("--mode safe-recover", payload["freshness"]["host_side_hint"])
            self.assertNotIn("-Build", payload["freshness"]["host_side_hint"])

    def test_index_status_cache_invalidates_on_git_head_change(self) -> None:
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            _git(repo_root, "init")
            _git(repo_root, "config", "user.email", "tests@example.local")
            _git(repo_root, "config", "user.name", "Tests")
            (repo_root / "service.py").write_text("print('v1')\n", encoding="utf-8")
            _git(repo_root, "add", "service.py")
            _git(repo_root, "commit", "-m", "initial")
            first_head = _git(repo_root, "rev-parse", "HEAD")
            branch = _git(repo_root, "branch", "--show-current")
            service = self._build_service(
                tmp,
                repo_root,
                indexed_branch=branch,
                indexed_commit_hash=first_head,
            )

            first_status = service.index_status()
            (repo_root / "service.py").write_text("print('v2')\n", encoding="utf-8")
            _git(repo_root, "add", "service.py")
            _git(repo_root, "commit", "-m", "second")
            second_head = _git(repo_root, "rev-parse", "HEAD")
            second_status = service.index_status()

            self.assertEqual(first_status.head_commit_hash, first_head)
            self.assertEqual(second_status.head_commit_hash, second_head)
            self.assertTrue(second_status.index_stale)

    def test_index_status_cache_invalidates_on_worktree_change(self) -> None:
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            _git(repo_root, "init")
            _git(repo_root, "config", "user.email", "tests@example.local")
            _git(repo_root, "config", "user.name", "Tests")
            (repo_root / "service.py").write_text("print('v1')\n", encoding="utf-8")
            _git(repo_root, "add", "service.py")
            _git(repo_root, "commit", "-m", "initial")
            head = _git(repo_root, "rev-parse", "HEAD")
            branch = _git(repo_root, "branch", "--show-current")
            service = self._build_service(
                tmp,
                repo_root,
                indexed_branch=branch,
                indexed_commit_hash=head,
            )

            first_status = service.index_status()
            (repo_root / "service.py").write_text("print('dirty')\n", encoding="utf-8")
            second_status = service.index_status()

            self.assertFalse(first_status.worktree_dirty)
            self.assertTrue(second_status.worktree_dirty)
            self.assertEqual(second_status.git.changed_indexable_files_count, 1)

    def test_index_status_warns_when_enabled_watcher_is_not_running(self) -> None:
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            _git(repo_root, "init")
            service = self._build_service(
                tmp,
                repo_root,
                watch_enabled=True,
                watcher=_FakeWatcher(running=False),
            )

            payload = service.index_status().model_dump()

            self.assertTrue(payload["watch_enabled"])
            self.assertFalse(payload["watch_running"])
            self.assertTrue(payload["watcher"]["enabled"])
            self.assertFalse(payload["watcher"]["running"])
            self.assertTrue(payload["watcher"]["runtime_available"])
            self.assertEqual(payload["watcher"]["autostart_policy"], "manual")
            self.assertTrue(payload["watcher"]["start_required"])
            self.assertIsNone(payload["watcher"]["start_blocked_reason"])
            self.assertIn(
                "watcher_enabled_but_not_running",
                {warning["code"] for warning in payload["warnings"]},
            )
            self.assertIn(
                "start_watcher",
                {action["code"] for action in payload["availability"]["next_actions"]},
            )

    def test_index_status_reports_active_lifecycle_lease(self) -> None:
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            _git(repo_root, "init")
            service = self._build_service(tmp, repo_root, repo_status="indexed")
            lease = service._registry.acquire_lifecycle_lease(
                str(repo_root),
                index_profile="test-profile",
                backend_id=service._settings.embedding_backend_id,
                operation="reindex_paths",
                owner_id="test-owner",
            )

            payload = service.index_status().model_dump()

            self.assertTrue(payload["lifecycle"]["action_in_progress"])
            self.assertEqual(payload["lifecycle"]["phase"], "reindex_paths")
            self.assertIsNotNone(payload["lifecycle"]["operation_id"])
            self.assertIsNotNone(payload["lifecycle"]["started_at"])
            service._registry.release_lifecycle_lease(lease, status="success")

    def test_index_status_cache_invalidates_on_lifecycle_lease_changes(self) -> None:
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            _git(repo_root, "init")
            service = self._build_service(tmp, repo_root, repo_status="indexed")

            first = service.index_status().model_dump()
            lease = service._registry.acquire_lifecycle_lease(
                str(repo_root),
                index_profile="test-profile",
                backend_id=service._settings.embedding_backend_id,
                operation="rebuild_index",
                owner_id="test-owner",
            )
            second = service.index_status().model_dump()
            service._registry.release_lifecycle_lease(lease, status="success")
            third = service.index_status().model_dump()

            self.assertFalse(first["lifecycle"]["action_in_progress"])
            self.assertTrue(second["lifecycle"]["action_in_progress"])
            self.assertEqual(second["lifecycle"]["phase"], "rebuild_index")
            self.assertFalse(third["lifecycle"]["action_in_progress"])

    def test_index_status_lifecycle_lease_uses_runtime_profile_when_registry_profile_is_stale(self) -> None:
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            _git(repo_root, "init")
            service = self._build_service(tmp, repo_root, repo_status="indexed")
            service._registry.upsert_repo(
                str(repo_root),
                status="indexed",
                active=True,
                index_profile="stale-profile",
                include_globs=["*", "**/*"],
                doc_prefixes=["docs/"],
                exclude_globs=[".git/**"],
                code_points_count=1,
                docs_points_count=0,
            )
            service.index_status()
            lease = service._registry.acquire_lifecycle_lease(
                str(repo_root),
                index_profile=service._embedding_provider.index_profile(),
                backend_id=service._settings.embedding_backend_id,
                operation="rebuild_index",
                owner_id="test-owner",
            )
            try:
                payload = service.index_status().model_dump()
            finally:
                service._registry.release_lifecycle_lease(lease, status="success")

            self.assertTrue(payload["lifecycle"]["action_in_progress"])
            self.assertEqual(payload["lifecycle"]["phase"], "rebuild_index")

    def test_sync_registry_state_updates_incremental_timestamp_for_watcher(self) -> None:
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            _git(repo_root, "init")
            indexer = _FakeIndexer()
            service = self._build_service(
                tmp,
                repo_root,
                watch_enabled=True,
                watcher=_FakeWatcher(running=True),
            )
            service._indexer = indexer

            indexer.last_incremental_update_ts = "2026-05-17T17:35:00+00:00"
            service.sync_registry_state(status="indexed", watch_running=True)
            after = service.index_status().model_dump()

            self.assertTrue(after["watcher"]["running"])
            self.assertEqual(after["watcher"]["last_incremental_update_ts"], "2026-05-17T17:35:00+00:00")

    def test_index_status_v2_reports_missing_index_code(self) -> None:
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            _git(repo_root, "init")
            service = self._build_service(
                tmp,
                repo_root,
                repo_status="registered",
                code_points=0,
            )

            payload = service.index_status().model_dump()

            self.assertFalse(payload["search_available"])
            self.assertEqual(payload["availability"]["search_unavailable_codes"], ["index_missing"])
            self.assertTrue(payload["availability"]["host_action_required"])
            self.assertEqual(payload["availability"]["next_actions"][0]["code"], "build_index_required")
            self.assertEqual(payload["reason_if_unavailable"], "index has not been built yet")

    def test_index_status_v2_reports_backend_switch_code(self) -> None:
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            _git(repo_root, "init")
            settings = SemanticMcpSettings(SEMANTIC_MCP_REPO_ROOT=str(repo_root))
            switched_backend_id = settings.embedding_backend_id + "_other"

            service = self._build_service(
                tmp,
                repo_root,
                selected_backend_id=switched_backend_id,
            )

            payload = service.index_status().model_dump()

            self.assertFalse(payload["search_available"])
            self.assertEqual(payload["availability"]["search_unavailable_codes"], ["backend_switch_required"])
            self.assertTrue(payload["backend"]["switch_required"])
            self.assertFalse(payload["backend"]["compatible"])

    def test_index_status_v2_blocks_schema_mismatch_without_mutating_lifecycle_state(self) -> None:
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            _git(repo_root, "init")
            contracts = {
                "code": {
                    "embedding_backend": "fake",
                    "embedding_model": "fake-model",
                    "index_schema_version": "999",
                }
            }
            service = self._build_service(
                tmp,
                repo_root,
                contracts=contracts,
            )

            payload = service.index_status().model_dump()

            self.assertFalse(payload["search_available"])
            self.assertEqual(payload["availability"]["search_unavailable_codes"], ["embedding_contract_mismatch"])
            self.assertTrue(payload["index_contract"]["blocking"])
            self.assertEqual(payload["index_contract"]["compatibility"], "incompatible")
            self.assertIn("schema_version_mismatch", payload["index_contract"]["incompatibility_codes"])
            self.assertEqual(payload["lifecycle"]["state"], "indexed")

    def test_index_status_v2_blocks_template_hash_mismatch(self) -> None:
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            _git(repo_root, "init")
            document_prefix_hash = hashlib.sha256("passage: ".encode("utf-8")).hexdigest()
            contracts = {
                "code": {
                    "embedding_backend": "fake",
                    "embedding_model": "fake-model",
                    "index_schema_version": "1",
                    "query_template_hash": "bad",
                    "document_prefix_hash": document_prefix_hash,
                }
            }
            service = self._build_service(
                tmp,
                repo_root,
                contracts=contracts,
            )

            payload = service.index_status().model_dump()

            self.assertFalse(payload["search_available"])
            self.assertEqual(payload["availability"]["search_unavailable_codes"], ["embedding_contract_mismatch"])
            self.assertTrue(payload["index_contract"]["blocking"])
            self.assertIn("query_template_hash_mismatch", payload["index_contract"]["incompatibility_codes"])

    def test_index_status_reports_sparse_retrieval_contract_without_blocking_dense(self) -> None:
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            _git(repo_root, "init")
            service = self._build_service(tmp, repo_root)

            payload = service.index_status().model_dump()

            self.assertTrue(payload["search_available"])
            self.assertTrue(payload["retrieval"]["dense_available"])
            self.assertFalse(payload["retrieval"]["sparse_available"])
            self.assertIn(
                "sparse_manifest_missing",
                payload["retrieval"]["sparse_unavailable_codes"],
            )
            self.assertEqual(
                payload["retrieval"]["expected_lexical_analyzer_version"],
                LEXICAL_ANALYZER_VERSION,
            )
            self.assertEqual(
                payload["retrieval"]["expected_sparse_encoder_kind"],
                SPARSE_ENCODER_KIND,
            )
            self.assertIn(
                "explicit_sparse_rebuild_recommended",
                {action["code"] for action in payload["availability"]["next_actions"]},
            )

    def test_index_status_maps_sparse_stats_stale_to_freshness_state(self) -> None:
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            _git(repo_root, "init")
            service = self._build_service(
                tmp,
                repo_root,
                sparse_enabled=True,
                indexed_commit_hash="synthetic-head",
            )
            service._indexer.path_manifest.replace_all([])
            manifest = build_sparse_manifest(
                scope="code",
                chunks=[_chunk("settings")],
                schema_version=service._settings.SEMANTIC_MCP_INDEX_SCHEMA_VERSION,
            )
            service._sparse_manifests.save(manifest.model_copy(update={"sparse_stats_stale": True}))

            payload = service.index_status().model_dump()

            self.assertTrue(payload["search_available"])
            self.assertFalse(payload["index_stale"])
            self.assertTrue(payload["retrieval"]["sparse_stats_stale"])
            self.assertEqual(payload["freshness"]["primary_state"], "sparse_stats_stale")
            self.assertIn("sparse_stats_stale", payload["freshness"]["states"])
            self.assertIn("sparse_stats_stale", payload["freshness"]["reason_codes"])
            self.assertEqual(payload["freshness"]["policy"]["stale_severity"], "warning")
            self.assertFalse(payload["freshness"]["policy"]["exact_fallback_recommended"])

    def test_index_status_distinguishes_sparse_analyzer_mismatch(self) -> None:
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            _git(repo_root, "init")
            service = self._build_service(
                tmp,
                repo_root,
                sparse_enabled=True,
            )
            manifest = build_sparse_manifest(
                scope="code",
                chunks=[_chunk("settings")],
                schema_version=service._settings.SEMANTIC_MCP_INDEX_SCHEMA_VERSION,
            )
            service._sparse_manifests.save(
                manifest.model_copy(update={"lexical_analyzer_version": "code_lexical_v1"})
            )

            payload = service.index_status().model_dump()

            self.assertTrue(payload["search_available"])
            self.assertFalse(payload["retrieval"]["sparse_available"])
            self.assertEqual(
                payload["retrieval"]["stored_lexical_analyzer_version"],
                "code_lexical_v1",
            )
            self.assertIn(
                "sparse_analyzer_version_mismatch",
                payload["retrieval"]["sparse_unavailable_codes"],
            )

    def test_index_status_reports_payload_index_readiness(self) -> None:
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            _git(repo_root, "init")
            payload_fields = {
                "scope",
                "relative_path",
                "file_extension",
                "language",
                "chunk_type",
                "domain_tags",
                "is_generated",
                "content_hash",
                "path_prefixes",
            }
            service = self._build_service(
                tmp,
                repo_root,
                sparse_enabled=True,
                payload_fields=payload_fields,
                payload_index_fields=payload_fields - {"path_prefixes"},
            )

            payload = service.index_status().model_dump()
            code_scope = next(
                scope_status
                for scope_status in payload["retrieval"]["scope_statuses"]
                if scope_status["scope"] == "code"
            )
            path_prefix_status = next(
                item
                for item in code_scope["payload_indexes"]
                if item["field_name"] == "path_prefixes"
            )

            self.assertTrue(payload["retrieval"]["payload_filter_pushdown_supported"])
            self.assertFalse(path_prefix_status["present"])
            self.assertEqual(path_prefix_status["warning_code"], "payload_index_missing")

    def test_index_status_reports_graph_summary_and_explicit_build_action(self) -> None:
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            _git(repo_root, "init")
            service = self._build_service(
                tmp,
                repo_root,
                graph_status_summary=lambda: models.GraphStatusSummary(
                    available=False,
                    state="missing",
                    expansion_allowed=False,
                    warning_codes=["graph_missing"],
                ),
            )

            payload = service.index_status().model_dump()

            self.assertEqual(payload["graph"]["state"], "missing")
            self.assertFalse(payload["graph"]["expansion_allowed"])
            self.assertIn("graph_missing", {warning["code"] for warning in payload["warnings"]})
            self.assertIn(
                "build_graph",
                {action["code"] for action in payload["availability"]["next_actions"]},
            )

    def test_index_status_suppresses_graph_lifecycle_actions_during_active_lease(self) -> None:
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            _git(repo_root, "init")
            service = self._build_service(
                tmp,
                repo_root,
                graph_status_summary=lambda: models.GraphStatusSummary(
                    available=False,
                    state="missing",
                    expansion_allowed=False,
                    warning_codes=["graph_missing"],
                ),
            )
            lease = service._registry.acquire_lifecycle_lease(
                str(repo_root),
                index_profile="test-profile",
                backend_id=service._settings.embedding_backend_id,
                operation="start_watcher",
                owner_id="test-owner",
            )

            payload = service.index_status().model_dump()

            self.assertTrue(payload["lifecycle"]["action_in_progress"])
            self.assertNotIn(
                "build_graph",
                {action["code"] for action in payload["availability"]["next_actions"]},
            )
            service._registry.release_lifecycle_lease(lease, status="success")

    def test_index_status_does_not_recommend_rebuild_for_degraded_allowed_graph(self) -> None:
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            _git(repo_root, "init")
            service = self._build_service(
                tmp,
                repo_root,
                graph_status_summary=lambda: models.GraphStatusSummary(
                    available=True,
                    state="stale",
                    expansion_allowed=True,
                    warning_codes=["graph_source_index_revision_mismatch"],
                ),
            )

            payload = service.index_status().model_dump()

            self.assertEqual(payload["graph"]["state"], "stale")
            self.assertTrue(payload["graph"]["expansion_allowed"])
            self.assertIn("graph_stale", {warning["code"] for warning in payload["warnings"]})
            self.assertIn("graph_degraded", payload["freshness"]["states"])
            self.assertIn("graph_stale", payload["freshness"]["reason_codes"])
            self.assertNotIn(
                "rebuild_graph",
                {action["code"] for action in payload["availability"]["next_actions"]},
            )

    def test_index_status_does_not_recommend_rebuild_for_unknown_graph_invalidations(self) -> None:
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / "repo"
            repo_root.mkdir()
            _git(repo_root, "init")
            service = self._build_service(
                tmp,
                repo_root,
                graph_status_summary=lambda: models.GraphStatusSummary(
                    available=True,
                    state="stale",
                    expansion_allowed=False,
                    warning_codes=["graph_invalidated_paths_unknown"],
                ),
            )

            payload = service.index_status().model_dump()

            self.assertEqual(payload["graph"]["state"], "stale")
            self.assertFalse(payload["graph"]["expansion_allowed"])
            self.assertIn("graph_stale", {warning["code"] for warning in payload["warnings"]})
            self.assertNotIn(
                "rebuild_graph",
                {action["code"] for action in payload["availability"]["next_actions"]},
            )


if __name__ == "__main__":
    unittest.main()
