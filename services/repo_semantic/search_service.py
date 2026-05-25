"""Search service that combines dense retrieval with lexical scoring."""

from __future__ import annotations

from collections.abc import Callable
import time
from typing import Literal

from services.repo_semantic.config import SemanticMcpSettings
from services.repo_semantic.context import query_plan as context_query_plan
from services.repo_semantic.context import search as context_search
from services.repo_semantic.embeddings.base import EmbeddingProvider
from services.repo_semantic.git_status import inspect_git_worktree
from services.repo_semantic.models import (
    ChunkRecord,
    ExactAnchorAnalysis,
    ExactAnchorUsage,
    GraphMode,
    IndexCollectionContract,
    IndexStatusResult,
    PayloadCapabilities,
    PayloadIndexStatus,
    ReadChunkResult,
    RepoRegistryEntry,
    RepoContextRoute,
    RepoContextSearchResponse,
    RetrievalContractStatus,
    RetrievalScopeStatus,
    SearchDiagnostics,
    SearchFilters,
    SearchMode,
    SearchResponse,
    SearchResult,
    SearchScope,
    SearchWarning,
    SnippetMode,
)
from services.repo_semantic.qdrant_store import QdrantStore
from services.repo_semantic.registry import RepoRegistry
from services.repo_semantic.graph.provider import expand_graph_context
from services.repo_semantic.retrieval.constants import (
    DENSE_FILTERED_CANDIDATE_CAP,
    MAX_EXPLANATIONS,
    MAX_QUERY_CHARS,
    MAX_TOP_K,
    RRF_K,
    SPARSE_CANDIDATE_CAP,
    TOKEN_RE,
)
from services.repo_semantic.retrieval.dense import dense_candidates
from services.repo_semantic.retrieval.diversity import apply_max_results_per_file
from services.repo_semantic.retrieval.engine import (
    build_hybrid_search_execution,
    build_semantic_search_execution,
)
from services.repo_semantic.retrieval.fusion import weighted_rrf
from services.repo_semantic.retrieval import filters as retrieval_filters
from services.repo_semantic.retrieval.legacy_bm25 import (
    build_lexical_cache,
    legacy_bm25_candidates,
)
from services.repo_semantic.retrieval import results as retrieval_results
from services.repo_semantic.retrieval.sparse_branch import sparse_query_candidates
from services.repo_semantic.retrieval.types import (
    _DenseSearchExecution,
    _FilterPlan,
    _LegacyBm25Execution,
    _LexicalCache,
    _SearchExecution,
    _SparseSearchExecution,
)
from services.repo_semantic.lexical import (
    LEXICAL_ANALYZER_VERSION,
    SPARSE_ENCODER_KIND,
    SparseManifest,
    SparseManifestStore,
    analyze_exact_anchors,
    sparse_contract_hash,
    sparse_terms,
)
from services.repo_semantic.storage.qdrant.point_projection import point_to_chunk
from services.repo_semantic.status import index_contract as status_index_contract
from services.repo_semantic.status import index_status as status_index_status
from services.repo_semantic.status import readiness as status_readiness
from services.repo_semantic.status import retrieval_status as status_retrieval


class SearchService:
    """Выполнять dense и hybrid retrieval по indexed collections."""

    def __init__(
        self,
        settings: SemanticMcpSettings,
        embedding_provider: EmbeddingProvider,
        store: QdrantStore,
        indexer,
        watcher,
        registry: RepoRegistry,
        graph_status_summary: Callable[[], object] | None = None,
        graph_read_provider: object | None = None,
    ) -> None:
        """Сохранить зависимости поиска и статуса индекса."""

        self._settings = settings
        self._embedding_provider = embedding_provider
        self._store = store
        self._indexer = indexer
        self._watcher = watcher
        self._registry = registry
        self._graph_status_summary = graph_status_summary
        self._graph_read_provider = graph_read_provider
        self._lexical_cache: dict[str, _LexicalCache] = {}
        self._status_cache: IndexStatusResult | None = None
        self._status_cache_ts: float = 0.0
        self._status_cache_ttl_sec: float = 10.0
        self._sparse_manifests = SparseManifestStore(settings)

    def invalidate_status_cache(self) -> None:
        """Сбросить cached index_status diagnostics после state changes."""

        self._status_cache = None
        self._status_cache_ts = 0.0

    def _status_cache_git_snapshot_matches(self, cached: IndexStatusResult) -> bool:
        """Return whether the cheap git guard still matches cached status."""

        cached_git = cached.git
        if cached_git is None:
            return False
        repo_entry = self._registry.get_repo(self._settings.logical_repo_identity)
        snapshot = inspect_git_worktree(
            self._settings.repo_root,
            include_globs=list(repo_entry.include_globs)
            if repo_entry
            else list(self._settings.effective_include_globs),
            exclude_globs=list(repo_entry.exclude_globs)
            if repo_entry
            else list(self._settings.SEMANTIC_MCP_EXCLUDE_GLOBS),
        )
        return (
            cached_git.is_git_repo == snapshot.is_git_repo
            and cached_git.branch == snapshot.branch
            and cached_git.head_commit == snapshot.head_commit
            and cached_git.worktree_dirty == snapshot.worktree_dirty
            and cached_git.changed_files_count == snapshot.changed_files_count
            and cached_git.untracked_files_count == snapshot.untracked_files_count
            and cached_git.changed_indexable_files_count == snapshot.changed_indexable_files_count
            and cached_git.untracked_indexable_files_count == snapshot.untracked_indexable_files_count
            and cached_git.changed_indexable_files_sample == snapshot.changed_indexable_files_sample
            and cached_git.untracked_indexable_files_sample == snapshot.untracked_indexable_files_sample
            and cached_git.error == snapshot.error
        )

    def _status_cache_lifecycle_matches(self, cached: IndexStatusResult) -> bool:
        """Return whether lifecycle lease state is still safe to serve from cache."""

        if cached.lifecycle.action_in_progress:
            return False
        try:
            repo_entry = self._registry.get_repo(self._settings.logical_repo_identity)
            if repo_entry is None:
                return True
            active_lease = self._registry.get_active_lifecycle_lease(
                self._settings.logical_repo_identity,
                index_profile=self._embedding_provider.index_profile(),
                backend_id=self._settings.embedding_backend_id,
            )
        except Exception:  # noqa: BLE001
            return False
        return active_lease is None

    def invalidate_cache(self) -> None:
        """Сбросить lexical cache после reindex/rebuild."""

        self._lexical_cache.clear()
        self.invalidate_status_cache()

    def _tokenize(self, text: str) -> list[str]:
        """Токенизировать строку для lexical scoring."""

        return [token.lower() for token in TOKEN_RE.findall(text)]

    def _point_to_chunk(self, point) -> ChunkRecord:
        """Преобразовать payload Qdrant point в ChunkRecord."""

        return point_to_chunk(point)

    def _scope_to_collections(self, scope: SearchScope) -> list[str]:
        """Развернуть logical scope в список concrete collections."""

        if scope == "all":
            return ["code", "docs"]
        return [scope]

    def _search_scopes(self, scope: SearchScope) -> list[str]:
        """Return requested scopes that can affect retrieval results."""

        requested = self._scope_to_collections(scope)
        non_empty = [
            concrete_scope
            for concrete_scope in requested
            if self._store.count(concrete_scope) > 0
        ]
        return non_empty or requested

    def _validate_query(self, query: str) -> None:
        """Apply query safety caps before calling embedding or lexical backends."""

        if len(query) > MAX_QUERY_CHARS:
            raise ValueError(f"query is too long; max {MAX_QUERY_CHARS} characters")

    def _normalize_top_k(self, top_k: int, *, max_top_k: int = MAX_TOP_K) -> int:
        """Clamp top_k to the public contract cap."""

        return min(max(int(top_k), 0), max_top_k)

    def _normalize_filter_values(
        self,
        values: list[str] | None,
        *,
        family: str,
        lower: bool = False,
    ) -> list[str]:
        """Normalize generic non-path filter values with a bounded cardinality."""

        return retrieval_filters.normalize_filter_values(
            values,
            family=family,
            lower=lower,
        )

    def _normalize_file_extensions(self, values: list[str] | None) -> list[str]:
        """Normalize extension filters to lower-case values with a leading dot."""

        return retrieval_filters.normalize_file_extensions(values)

    def _normalize_filter_path(self, value: str, *, family: str, allow_glob: bool) -> str:
        """Normalize repo-relative POSIX paths and reject unsafe path shapes."""

        return retrieval_filters.normalize_filter_path(
            value,
            family=family,
            allow_glob=allow_glob,
        )

    def _normalize_path_filters(
        self,
        values: list[str] | None,
        *,
        family: str,
    ) -> list[str]:
        """Normalize include/exclude glob-like repo-relative path filters."""

        return retrieval_filters.normalize_path_filters(values, family=family)

    def _build_filters(
        self,
        *,
        path_prefix: str | None = None,
        include_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
        file_extensions: list[str] | None = None,
        languages: list[str] | None = None,
        chunk_types: list[str] | None = None,
        domain_tags: list[str] | None = None,
    ) -> SearchFilters:
        """Build the normalized internal filter model shared by search modes."""

        return retrieval_filters.build_filters(
            path_prefix=path_prefix,
            include_paths=include_paths,
            exclude_paths=exclude_paths,
            file_extensions=file_extensions,
            languages=languages,
            chunk_types=chunk_types,
            domain_tags=domain_tags,
        )

    def _filters_active(self, filters: SearchFilters) -> bool:
        """Return whether client-side filters can reduce candidate recall."""

        return retrieval_filters.filters_active(filters)

    def _payload_fields_for_scope(self, scope: str) -> set[str]:
        reader = getattr(self._store, "payload_fields", None)
        if reader is not None:
            try:
                fields = set(str(field) for field in reader(scope))
                if fields:
                    return fields
            except Exception:  # noqa: BLE001
                pass
        dense_schema_kind = self._store_dense_schema_kind(scope)
        current_collection_exists = self._store_current_collection_exists(scope)
        if dense_schema_kind == "named" and current_collection_exists:
            return {
                "scope",
                "relative_path",
                "file_extension",
                "language",
                "chunk_type",
                "domain_tags",
                "is_generated",
                "content_hash",
                "path_segments",
                "path_prefixes",
            }
        if dense_schema_kind == "unnamed_legacy":
            return {
                "scope",
                "relative_path",
                "language",
                "chunk_type",
                "domain_tags",
                "content_hash",
            }
        return set()

    def _payload_index_statuses(self, scope: str) -> list[PayloadIndexStatus]:
        """Return best-effort payload index presence for status diagnostics."""

        collection_name = self._store.collection_name(scope)
        index_fields: set[str] | None = None
        reader = getattr(self._store, "payload_index_fields", None)
        if reader is not None:
            try:
                raw_fields = reader(scope)
                if raw_fields is not None:
                    index_fields = {str(field) for field in raw_fields}
            except Exception:  # noqa: BLE001
                index_fields = None

        return retrieval_filters.payload_index_statuses(
            scope=scope,
            collection_name=collection_name,
            index_fields=index_fields,
        )

    def _filter_plan_for_scope(self, scope: str, filters: SearchFilters) -> _FilterPlan:
        """Build Qdrant payload filter plan while preserving legacy fallback semantics."""

        collection_name = self._store.collection_name(scope)
        payload_fields = self._payload_fields_for_scope(scope)
        return retrieval_filters.filter_plan_for_scope(
            scope=scope,
            collection_name=collection_name,
            payload_fields=payload_fields,
            filters=filters,
        )

    def _merge_filter_plans(self, plans: list[_FilterPlan]) -> tuple[list[str], list[str], bool, list[PayloadCapabilities]]:
        return retrieval_filters.merge_filter_plans(plans)

    def _normalize_chunk_path(self, relative_path: str) -> str:
        """Normalize indexed paths for matching without changing stored payloads."""

        return retrieval_filters.normalize_chunk_path(relative_path)

    def _prefix_matches(self, relative_path: str, path_prefix: str) -> bool:
        """Boundary-aware prefix predicate for the legacy path_prefix alias."""

        return retrieval_filters.prefix_matches(relative_path, path_prefix)

    def _path_glob_matches(self, relative_path: str, pattern: str) -> bool:
        """Match repo-relative paths with fnmatchcase plus directory exactness."""

        return retrieval_filters.path_glob_matches(relative_path, pattern)

    def _matches_filters(self, chunk: ChunkRecord, filters: SearchFilters) -> bool:
        """Проверить, подходит ли чанк под client-side filters."""

        return retrieval_filters.matches_filters(chunk, filters)

    def _to_search_result(
        self,
        chunk: ChunkRecord,
        score: float,
        dense_score: float | None = None,
        lexical_score: float | None = None,
        *,
        query_tokens: list[str] | None = None,
        snippet_mode: SnippetMode = "chunk_start",
        include_explanations: bool = True,
        match_type: str | None = None,
    ) -> SearchResult:
        """Собрать сериализуемый результат поиска."""

        return retrieval_results.build_search_result(
            chunk=chunk,
            repo_root=self._settings.logical_repo_identity,
            score=score,
            dense_score=dense_score,
            lexical_score=lexical_score,
            query_tokens=query_tokens,
            snippet_mode=snippet_mode,
            include_explanations=include_explanations,
            match_type=match_type,
        )

    def _query_terms_for_matching(self, query: str) -> list[str]:
        """Return display-oriented query terms while preserving canonical matching."""

        return retrieval_results.query_terms_for_matching(
            query,
            tokenize=self._tokenize,
        )

    def _get_lexical_cache(self, scope: str) -> _LexicalCache:
        """Построить и закэшировать lexical corpus по коллекции."""

        if scope in self._lexical_cache:
            return self._lexical_cache[scope]
        cache = build_lexical_cache(
            store=self._store,
            scope=scope,
            point_to_chunk=self._point_to_chunk,
            tokenize=self._tokenize,
        )
        self._lexical_cache[scope] = cache
        return cache

    def _sparse_manifest_for_scope(self, scope: str) -> SparseManifest | None:
        """Load a compatible sparse manifest for the requested scope."""

        manifest = self._sparse_manifests.load(scope)
        if manifest is None:
            return None
        if manifest.schema_version != self._settings.SEMANTIC_MCP_INDEX_SCHEMA_VERSION:
            return None
        if manifest.lexical_analyzer_version != LEXICAL_ANALYZER_VERSION:
            return None
        if manifest.sparse_encoder_kind != SPARSE_ENCODER_KIND:
            return None
        expected_contract_hash = sparse_contract_hash(
            schema_version=self._settings.SEMANTIC_MCP_INDEX_SCHEMA_VERSION,
        )
        if manifest.immutable_sparse_contract_hash != expected_contract_hash:
            return None
        return manifest

    def _sparse_manifest_unavailable_codes(self, manifest: SparseManifest | None) -> list[str]:
        """Return precise sparse-manifest incompatibility codes for status."""

        if manifest is None:
            return ["sparse_manifest_missing"]
        codes: list[str] = []
        if manifest.schema_version != self._settings.SEMANTIC_MCP_INDEX_SCHEMA_VERSION:
            codes.append("sparse_contract_mismatch")
        if manifest.lexical_analyzer_version != LEXICAL_ANALYZER_VERSION:
            codes.append("sparse_analyzer_version_mismatch")
        if manifest.sparse_encoder_kind != SPARSE_ENCODER_KIND:
            codes.append("sparse_encoder_kind_mismatch")
        expected_contract_hash = sparse_contract_hash(
            schema_version=self._settings.SEMANTIC_MCP_INDEX_SCHEMA_VERSION,
        )
        if manifest.immutable_sparse_contract_hash != expected_contract_hash:
            codes.append("sparse_contract_mismatch")
        return sorted(set(codes))

    def _store_sparse_available(self, scope: str) -> bool:
        """Return whether the store exposes a Qdrant sparse vector for scope."""

        checker = getattr(self._store, "sparse_vector_available", None)
        if checker is None:
            return False
        try:
            return bool(checker(scope))
        except Exception:  # noqa: BLE001
            return False

    def _store_dense_schema_kind(self, scope: str) -> str:
        checker = getattr(self._store, "dense_schema_kind", None)
        if checker is None:
            return "unknown"
        try:
            return str(checker(scope))
        except Exception:  # noqa: BLE001
            return "unknown"

    def _store_read_collection_exists(self, scope: str) -> bool:
        checker = getattr(self._store, "read_collection_exists", None)
        if checker is None:
            return self._store.collection_exists(scope)
        try:
            return bool(checker(scope))
        except Exception:  # noqa: BLE001
            return False

    def _store_current_collection_exists(self, scope: str) -> bool:
        checker = getattr(self._store, "current_collection_exists", None)
        if checker is None:
            return self._store.collection_exists(scope)
        try:
            return bool(checker(scope))
        except Exception:  # noqa: BLE001
            return False

    def _sparse_scope_ready(self, scope: str) -> bool:
        manifest = self._sparse_manifest_for_scope(scope)
        if manifest is None:
            return False
        if manifest.intentionally_empty:
            return False
        return self._store_sparse_available(scope)

    def _sparse_query_candidates(
        self,
        *,
        query: str,
        scope: SearchScope,
        filters: SearchFilters,
        limit: int,
    ) -> _SparseSearchExecution:
        """Fetch sparse branch candidates from Qdrant sparse vectors."""

        search_scopes = self._search_scopes(scope)
        filter_plans_by_scope = {
            concrete_scope: self._filter_plan_for_scope(concrete_scope, filters)
            for concrete_scope in search_scopes
        }
        return sparse_query_candidates(
            store=self._store,
            sparse_manifests=self._sparse_manifests,
            query=query,
            search_scopes=search_scopes,
            requested_scopes=self._scope_to_collections(scope),
            filter_plans_by_scope=filter_plans_by_scope,
            filters=filters,
            limit=limit,
            sparse_manifest_for_scope=self._sparse_manifest_for_scope,
            sparse_manifest_unavailable_codes=self._sparse_manifest_unavailable_codes,
            store_sparse_available=self._store_sparse_available,
            point_to_chunk=self._point_to_chunk,
            matches_filters=self._matches_filters,
        )

    def _legacy_bm25_candidates(
        self,
        *,
        existing_candidates: dict[str, tuple[ChunkRecord, float]],
        query_tokens: list[str],
        scope: SearchScope,
        filters: SearchFilters,
    ) -> _LegacyBm25Execution:
        """Fetch compatibility local BM25 candidates for legacy indexes."""

        return legacy_bm25_candidates(
            existing_candidates=existing_candidates,
            query_tokens=query_tokens,
            search_scopes=self._search_scopes(scope),
            filters=filters,
            get_lexical_cache=self._get_lexical_cache,
            matches_filters=self._matches_filters,
        )

    def _weighted_rrf(
        self,
        branches: dict[str, list[tuple[ChunkRecord, float]]],
        weights: dict[str, float],
    ) -> list[tuple[ChunkRecord, float, float, float, dict[str, int]]]:
        """Fuse branch ranks with weighted reciprocal rank fusion."""

        return weighted_rrf(branches, weights, rrf_k=RRF_K)

    def _current_repo_entry(self) -> RepoRegistryEntry | None:
        """Вернуть registry entry для runtime repo."""

        return self._registry.get_repo(self._settings.logical_repo_identity)

    def _runtime_repo_is_active(self, repo_entry: RepoRegistryEntry | None) -> bool:
        """Проверить, что runtime repo совпадает с текущим active repo."""

        if repo_entry is None:
            return False
        active_repo = self._registry.get_active_repo()
        return bool(active_repo and active_repo.repo_root == repo_entry.repo_root)

    def _runtime_embedding_backend_id(self) -> str:
        """Вернуть backend id текущего embedding runtime."""

        return self._settings.embedding_backend_id

    def _selected_embedding_backend_id(self) -> str | None:
        """Вернуть backend id, выбранный для роли embedding в registry."""

        selected = self._registry.get_role_backend("embedding")
        return selected.backend_id if selected is not None else None

    def _index_contract_issue(self) -> str | None:
        """Проверить, что текущий runtime совместим с уже построенным индексом."""

        return status_index_contract.index_contract_issue(
            store=self._store,
            settings=self._settings,
            embedding_backend=self._embedding_provider.backend_name(),
            embedding_model=self._embedding_provider.model_name(),
        )

    def _sha256_text(self, value: str) -> str:
        """Return lowercase SHA-256 for contract string values."""

        return status_index_contract.sha256_text(value)

    def _search_unavailable_codes(
        self,
        repo_entry: RepoRegistryEntry | None,
        *,
        contract_issue: str | None = None,
    ) -> list[str]:
        """Return stable machine-readable reasons why search is unavailable."""

        return status_readiness.search_unavailable_codes(
            repo_entry,
            selected_backend_id=self._selected_embedding_backend_id(),
            runtime_backend_id=self._runtime_embedding_backend_id(),
            runtime_repo_is_active=self._runtime_repo_is_active(repo_entry),
            contract_issue=contract_issue,
        )

    def _reason_for_unavailable_codes(
        self,
        codes: list[str],
        repo_entry: RepoRegistryEntry | None,
        *,
        contract_issue: str | None = None,
    ) -> str | None:
        """Return the legacy prose reason for existing consumers."""

        return status_readiness.reason_for_unavailable_codes(
            codes,
            repo_entry,
            contract_issue=contract_issue,
        )

    def current_index_contract_issue(self) -> str | None:
        """Вернуть user-facing проблему совместимости runtime и текущего индекса."""

        return self._index_contract_issue()

    def _search_unavailable_reason(self, repo_entry: RepoRegistryEntry | None) -> str | None:
        """Определить, почему search-path сейчас недоступен."""

        contract_issue = self._index_contract_issue()
        codes = self._search_unavailable_codes(repo_entry, contract_issue=contract_issue)
        return self._reason_for_unavailable_codes(codes, repo_entry, contract_issue=contract_issue)

    def ensure_search_available(self) -> None:
        """Явно запретить search-path для repo без готового индекса."""

        repo_entry = self._current_repo_entry()
        reason = self._search_unavailable_reason(repo_entry)
        if reason:
            raise RuntimeError(
                "Semantic search is unavailable for the active repo: "
                f"{reason}. Use build_index() first."
            )

    def sync_registry_state(
        self,
        *,
        status: str,
        last_error: str | None = None,
        active: bool | None = None,
        watch_running: bool | None = None,
        record_index_revision: bool = False,
    ) -> RepoRegistryEntry:
        """Синхронизировать runtime state в persistent registry."""

        existing = self._current_repo_entry()
        git_status = (
            inspect_git_worktree(
                self._settings.repo_root,
                include_globs=list(self._settings.effective_include_globs),
                exclude_globs=list(self._settings.SEMANTIC_MCP_EXCLUDE_GLOBS),
            )
            if record_index_revision
            else None
        )
        effective_active = self._runtime_repo_is_active(existing) if active is None else active
        code_points = self._store.count("code") if self._store_read_collection_exists("code") else 0
        docs_points = self._store.count("docs") if self._store_read_collection_exists("docs") else 0
        effective_watch_running = (
            bool(self._watcher and self._watcher.is_running)
            if watch_running is None
            else watch_running
        )
        self.invalidate_status_cache()
        return self._registry.upsert_repo(
            self._settings.logical_repo_identity,
            repo_key=self._settings.repo_key_slug,
            status=status,
            active=effective_active,
            index_profile=self._embedding_provider.index_profile(),
            include_globs=list(self._settings.effective_include_globs),
            doc_prefixes=list(self._settings.effective_doc_prefixes),
            exclude_globs=list(self._settings.SEMANTIC_MCP_EXCLUDE_GLOBS),
            last_full_build_ts=self._indexer.last_full_build_ts or (existing.last_full_build_ts if existing else None),
            last_incremental_update_ts=self._indexer.last_incremental_update_ts
            or (existing.last_incremental_update_ts if existing else None),
            indexed_branch=git_status.branch
            if git_status is not None
            else (existing.indexed_branch if existing else None),
            indexed_commit_hash=git_status.head_commit
            if git_status is not None
            else (existing.indexed_commit_hash if existing else None),
            last_error=last_error,
            watch_enabled=self._settings.SEMANTIC_MCP_WATCH_ENABLED,
            watch_running=effective_watch_running,
            code_points_count=code_points,
            docs_points_count=docs_points,
        )

    def semantic_search(
        self,
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
    ) -> list[SearchResult]:
        """Выполнить dense semantic retrieval по code/docs коллекциям."""

        return self._semantic_search_execution(
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
        ).results

    def _dense_candidates(
        self,
        *,
        query_vector: list[float],
        scope: SearchScope,
        filters: SearchFilters,
        target_results: int,
        max_results_per_file: int | None,
    ) -> _DenseSearchExecution:
        """Fetch dense candidates, expanding the unfiltered shortlist under filters."""

        search_scopes = self._search_scopes(scope)
        filter_plans_by_scope = {
            concrete_scope: self._filter_plan_for_scope(concrete_scope, filters)
            for concrete_scope in search_scopes
        }
        return dense_candidates(
            store=self._store,
            query_vector=query_vector,
            search_scopes=search_scopes,
            filter_plans_by_scope=filter_plans_by_scope,
            filters=filters,
            target_results=target_results,
            max_results_per_file=max_results_per_file,
            point_to_chunk=self._point_to_chunk,
            matches_filters=self._matches_filters,
        )

    def _apply_max_results_per_file(
        self,
        ranked: list,
        *,
        top_k: int,
        max_results_per_file: int | None,
        relative_path,
    ) -> list:
        """Keep global score ordering while limiting chunks per file."""

        return apply_max_results_per_file(
            ranked,
            top_k=top_k,
            max_results_per_file=max_results_per_file,
            relative_path=relative_path,
        )

    def _semantic_search_execution(
        self,
        *,
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
        max_top_k: int = MAX_TOP_K,
    ) -> _SearchExecution:
        """Dense semantic retrieval with v1.1 filters and diagnostics."""

        self.ensure_search_available()
        self._validate_query(query)
        if snippet_mode not in {"chunk_start", "query_centered"}:
            raise ValueError("snippet_mode must be 'chunk_start' or 'query_centered'")
        top_k = self._normalize_top_k(top_k, max_top_k=max_top_k)
        if top_k == 0:
            diagnostics = SearchDiagnostics(final_results=0)
            return _SearchExecution(
                results=[],
                diagnostics=diagnostics,
                warnings=[],
                branch_ranks_by_chunk={},
            )
        filters = self._build_filters(
            path_prefix=path_prefix,
            include_paths=include_paths,
            exclude_paths=exclude_paths,
            file_extensions=file_extensions,
            languages=languages,
            chunk_types=chunk_types,
            domain_tags=domain_tags,
        )
        query_vector = self._embedding_provider.embed_query(query)
        dense_execution = self._dense_candidates(
            query_vector=query_vector,
            scope=scope,
            filters=filters,
            target_results=top_k,
            max_results_per_file=max_results_per_file,
        )
        query_tokens = self._query_terms_for_matching(query)
        exact_anchor_analysis = analyze_exact_anchors(query)
        return build_semantic_search_execution(
            dense_execution=dense_execution,
            top_k=top_k,
            max_results_per_file=max_results_per_file,
            query_tokens=query_tokens,
            exact_anchor_analysis=exact_anchor_analysis,
            snippet_mode=snippet_mode,
            include_explanations=include_explanations,
            to_search_result=self._to_search_result,
            merge_filter_plans=self._merge_filter_plans,
        )

    def hybrid_search(
        self,
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
    ) -> list[SearchResult]:
        """Выполнить hybrid retrieval: dense shortlist + BM25 lexical scoring."""

        return self._hybrid_search_execution(
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
        ).results

    def _hybrid_search_execution(
        self,
        *,
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
    ) -> _SearchExecution:
        """Hybrid retrieval with full chunk text preserved for final shaping."""

        self.ensure_search_available()
        self._validate_query(query)
        if snippet_mode not in {"chunk_start", "query_centered"}:
            raise ValueError("snippet_mode must be 'chunk_start' or 'query_centered'")
        top_k = self._normalize_top_k(top_k)
        if top_k == 0:
            diagnostics = SearchDiagnostics(final_results=0)
            return _SearchExecution(
                results=[],
                diagnostics=diagnostics,
                warnings=[],
                branch_ranks_by_chunk={},
            )
        filters = self._build_filters(
            path_prefix=path_prefix,
            include_paths=include_paths,
            exclude_paths=exclude_paths,
            file_extensions=file_extensions,
            languages=languages,
            chunk_types=chunk_types,
            domain_tags=domain_tags,
        )
        exact_anchor_analysis = analyze_exact_anchors(query)
        query_vector = self._embedding_provider.embed_query(query)
        dense_limit = min(DENSE_FILTERED_CANDIDATE_CAP, max(50, top_k * 8))
        sparse_limit = min(SPARSE_CANDIDATE_CAP, max(100, top_k * 12))
        dense_execution = self._dense_candidates(
            query_vector=query_vector,
            scope=scope,
            filters=filters,
            target_results=dense_limit,
            max_results_per_file=None,
        )

        sparse_execution = self._sparse_query_candidates(
            query=query,
            scope=scope,
            filters=filters,
            limit=sparse_limit,
        )
        sparse_candidates = sparse_execution.candidates
        lexical_cache_used = False
        legacy_fallback_used = False
        query_tokens = sparse_terms(query) or self._tokenize(query)
        result_query_tokens = self._query_terms_for_matching(query)

        if not sparse_execution.branch_available and query_tokens:
            legacy_execution = self._legacy_bm25_candidates(
                existing_candidates=sparse_candidates,
                query_tokens=query_tokens,
                scope=scope,
                filters=filters,
            )
            sparse_candidates = legacy_execution.candidates
            lexical_cache_used = legacy_execution.lexical_cache_used
            legacy_fallback_used = legacy_execution.legacy_fallback_used

        return build_hybrid_search_execution(
            dense_execution=dense_execution,
            sparse_execution=sparse_execution,
            sparse_candidates=sparse_candidates,
            lexical_cache_used=lexical_cache_used,
            legacy_fallback_used=legacy_fallback_used,
            top_k=top_k,
            sparse_limit=sparse_limit,
            max_results_per_file=max_results_per_file,
            exact_anchor_analysis=exact_anchor_analysis,
            result_query_tokens=result_query_tokens,
            snippet_mode=snippet_mode,
            include_explanations=include_explanations,
            to_search_result=self._to_search_result,
            merge_filter_plans=self._merge_filter_plans,
        )

    def search_v2(
        self,
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
    ) -> SearchResponse:
        """Agent-grade search response with diagnostics and warnings."""

        if mode not in {"semantic", "hybrid"}:
            raise ValueError("mode must be 'semantic' or 'hybrid'")
        execution = (
            self._semantic_search_execution
            if mode == "semantic"
            else self._hybrid_search_execution
        )(
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
        warnings = list(execution.warnings)
        if not execution.results:
            warnings.append(
                SearchWarning(
                    code="empty_result_best_effort",
                    severity="info",
                    detail="No indexed chunks matched the query and filters.",
                )
            )
        try:
            status = self.index_status()
            if status.freshness and status.freshness.policy and status.freshness.policy.exact_fallback_recommended:
                warnings.append(
                    SearchWarning(
                        code="stale_index_verify_with_exact_search",
                        severity="warning",
                        detail="Index freshness is stale; verify candidates with exact search before editing.",
                    )
                )
        except Exception:  # noqa: BLE001
            pass
        return SearchResponse(
            query=query,
            mode=mode,
            results=execution.results,
            warnings=warnings,
            diagnostics=execution.diagnostics,
        )

    def _normalize_context_queries(
        self,
        query: str,
        subqueries: list[str] | None,
    ) -> tuple[list[tuple[str, str, float]], int]:
        """Normalize context-search queries while preserving original first."""

        return context_query_plan.normalize_context_queries(
            query,
            subqueries,
            validate_query=self._validate_query,
        )

    def _context_route_used(self, route: RepoContextRoute) -> Literal["semantic", "hybrid", "exact_handoff"]:
        return context_query_plan.context_route_used(route)

    def _aggregate_exact_anchors(
        self,
        query_items: list[tuple[str, str, float]],
    ) -> tuple[ExactAnchorAnalysis, list[ExactAnchorUsage]]:
        return context_query_plan.aggregate_exact_anchors(query_items)

    def _context_execution(
        self,
        *,
        query: str,
        route_used: Literal["semantic", "hybrid"],
        scope: SearchScope,
        path_prefix: str | None,
        include_paths: list[str] | None,
        exclude_paths: list[str] | None,
        file_extensions: list[str] | None,
        languages: list[str] | None,
        chunk_types: list[str] | None,
        domain_tags: list[str] | None,
        top_k: int,
        snippet_mode: SnippetMode,
        include_explanations: bool,
    ) -> _SearchExecution:
        runner = self._semantic_search_execution if route_used == "semantic" else self._hybrid_search_execution
        return runner(
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
            max_results_per_file=None,
            snippet_mode=snippet_mode,
            include_explanations=include_explanations,
        )

    def _read_graph_chunk(self, scope: str, chunk_id: str) -> ChunkRecord | None:
        point = self._store.get_chunk(scope, chunk_id)
        if point is None:
            return None
        return self._point_to_chunk(point)

    def _graph_context_expansion(self, **kwargs):
        return expand_graph_context(
            graph_read_provider=self._graph_read_provider,
            graph_auto_enabled=self._settings.SEMANTIC_MCP_REPO_CONTEXT_GRAPH_AUTO_ENABLED,
            read_chunk=self._read_graph_chunk,
            matches_filters=self._matches_filters,
            **kwargs,
        )

    def _graph_search_result(
        self,
        *,
        chunk: ChunkRecord,
        score: float,
        matched_terms: list[str],
        snippet_mode: SnippetMode,
        include_explanations: bool,
    ) -> SearchResult:
        result = self._to_search_result(
            chunk=chunk,
            score=score,
            query_tokens=matched_terms,
            snippet_mode=snippet_mode,
            include_explanations=include_explanations,
            match_type=None,
        )
        if matched_terms and not result.matched_terms:
            result.matched_terms = matched_terms[:MAX_EXPLANATIONS]
        if include_explanations:
            result.why_matched = list(dict.fromkeys([*result.why_matched, "graph_evidence_path"]))[:MAX_EXPLANATIONS]
        return result

    def repo_context_search(
        self,
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
    ) -> RepoContextSearchResponse:
        """Agent-facing context envelope over semantic/hybrid retrieval."""

        return context_search.repo_context_search_response(
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
            normalize_top_k=self._normalize_top_k,
            normalize_context_queries=self._normalize_context_queries,
            context_route_used=self._context_route_used,
            aggregate_exact_anchors=self._aggregate_exact_anchors,
            context_execution=self._context_execution,
            graph_context_expansion=self._graph_context_expansion,
            graph_result_builder=self._graph_search_result,
            build_filters=self._build_filters,
            index_status=self.index_status,
        )

    def read_chunk(self, scope: str, chunk_id: str) -> ReadChunkResult | None:
        """Вернуть полный текст конкретного чанка."""

        self.ensure_search_available()
        point = self._store.get_chunk(scope, chunk_id)
        if point is None:
            return None
        chunk = self._point_to_chunk(point)
        return retrieval_results.build_read_chunk_result(chunk)

    def find_similar_chunk(self, scope: str, chunk_id: str, top_k: int = 10) -> list[SearchResult]:
        """Найти chunks, похожие на уже известный chunk."""

        self.ensure_search_available()
        chunk = self.read_chunk(scope, chunk_id)
        if chunk is None:
            return []
        requested_top_k = self._normalize_top_k(top_k)
        results = self._semantic_search_execution(
            query=chunk.text,
            top_k=requested_top_k + 1,
            scope=scope,
            max_top_k=MAX_TOP_K + 1,
        ).results
        return [result for result in results if result.chunk_id != chunk_id][:requested_top_k]

    def _collection_contract(
        self,
        scope: str,
        *,
        points_count: int,
        lexical_documents: int | None,
        runtime_embedding_backend: str,
        runtime_embedding_model: str,
        runtime_query_template_hash: str,
        runtime_document_prefix_hash: str,
    ) -> IndexCollectionContract:
        """Build per-collection runtime-vs-stored compatibility status."""

        return status_index_contract.collection_contract(
            store=self._store,
            settings=self._settings,
            scope=scope,
            points_count=points_count,
            lexical_documents=lexical_documents,
            runtime_embedding_backend=runtime_embedding_backend,
            runtime_embedding_model=runtime_embedding_model,
            runtime_query_template_hash=runtime_query_template_hash,
            runtime_document_prefix_hash=runtime_document_prefix_hash,
            read_collection_exists=self._store_read_collection_exists,
        )

    def _retrieval_scope_status(
        self,
        scope: str,
        *,
        points_count: int,
    ) -> RetrievalScopeStatus:
        """Build dense/sparse retrieval status for one logical scope."""

        return status_retrieval.retrieval_scope_status(
            scope=scope,
            points_count=points_count,
            store=self._store,
            read_collection_exists=self._store_read_collection_exists,
            current_collection_exists=self._store_current_collection_exists,
            dense_schema_kind_for_scope=self._store_dense_schema_kind,
            load_sparse_manifest=self._sparse_manifests.load,
            sparse_manifest_for_scope=self._sparse_manifest_for_scope,
            sparse_manifest_unavailable_codes=self._sparse_manifest_unavailable_codes,
            store_sparse_available=self._store_sparse_available,
            payload_index_statuses=self._payload_index_statuses,
        )

    def _retrieval_status(self, points_by_scope: dict[str, int]) -> RetrievalContractStatus:
        """Build global dense/sparse retrieval status."""

        return status_retrieval.retrieval_status(
            points_by_scope=points_by_scope,
            scope_status_for_scope=lambda scope, points_count: self._retrieval_scope_status(
                scope,
                points_count=points_count,
            ),
            payload_fields_for_scope=self._payload_fields_for_scope,
        )

    def index_status(self) -> IndexStatusResult:
        """Собрать текущее состояние индекса и watcher."""

        now = time.monotonic()
        if (
            self._status_cache is not None
            and now - self._status_cache_ts <= self._status_cache_ttl_sec
            and self._status_cache_git_snapshot_matches(self._status_cache)
            and self._status_cache_lifecycle_matches(self._status_cache)
        ):
            return self._status_cache

        status = status_index_status.build_index_status(
            settings=self._settings,
            embedding_provider=self._embedding_provider,
            store=self._store,
            indexer=self._indexer,
            watcher=self._watcher,
            registry=self._registry,
            read_collection_exists=self._store_read_collection_exists,
            collection_contract=self._collection_contract,
            retrieval_status_for_points=self._retrieval_status,
            graph_status_summary=self._graph_status_summary,
            sha256_text=self._sha256_text,
        )
        self._status_cache = status
        self._status_cache_ts = now
        return status
