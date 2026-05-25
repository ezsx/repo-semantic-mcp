"""Graph lifecycle and status service."""

from __future__ import annotations

from time import perf_counter

from services.repo_semantic.config import SemanticMcpSettings
from services.repo_semantic.contracts.graph import (
    GraphBuildResult,
    GraphStatusResult,
    GraphStatusSummary,
    GraphUpdateResult,
)
from services.repo_semantic.graph.store import GraphUpdateBoundsExceeded, PATH_OWNED_NODE_TYPES
from services.repo_semantic.graph.types import GraphChunkLink, GraphEdge, GraphNode
from services.repo_semantic.git_status import (
    inspect_git_revision,
    inspect_git_worktree,
    tracked_indexable_changes_after,
    untracked_indexable_changes_after,
)
from services.repo_semantic.graph.artifacts import graph_artifact_path, stable_json_hash
from services.repo_semantic.graph.builder import (
    GRAPH_EXTRACTOR_VERSIONS,
    build_graph_snapshot,
    graph_input_signature_for_chunks,
    source_contract_from_chunks,
)
from services.repo_semantic.graph.store import GraphStore, graph_summary_from_status
from services.repo_semantic.registry import RepoRegistry
from services.repo_semantic.storage.qdrant.point_projection import point_to_chunk


def _dedupe_paths(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for path in paths:
        normalized = str(path).replace("\\", "/").strip("/")
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


class GraphService:
    """Explicit graph artifact lifecycle over already-indexed chunks."""

    def __init__(
        self,
        *,
        settings: SemanticMcpSettings,
        chunk_store,
        registry: RepoRegistry,
        path_manifest=None,
    ) -> None:
        self._settings = settings
        self._chunk_store = chunk_store
        self._registry = registry
        self._path_manifest = path_manifest
        self._store = GraphStore(graph_artifact_path(settings))
        self._update_idempotency_results: dict[str, GraphUpdateResult] = {}
        self._update_idempotency_fingerprints: dict[str, str] = {}

    @property
    def graph_path(self) -> str:
        return str(self._store.db_path)

    def _profile(self) -> str:
        repo_entry = self._registry.get_repo(self._settings.logical_repo_identity)
        return (
            repo_entry.index_profile
            if repo_entry is not None
            else self._settings.SEMANTIC_MCP_PROFILE_NAME
        )

    def _git_snapshot(self):
        repo_entry = self._registry.get_repo(self._settings.logical_repo_identity)
        include_globs = (
            list(repo_entry.include_globs)
            if repo_entry is not None
            else list(self._settings.effective_include_globs)
        )
        exclude_globs = (
            list(repo_entry.exclude_globs)
            if repo_entry is not None
            else list(self._settings.SEMANTIC_MCP_EXCLUDE_GLOBS)
        )
        return inspect_git_worktree(
            self._settings.repo_root,
            include_globs=include_globs,
            exclude_globs=exclude_globs,
        )

    def _collections_contract(self) -> dict[str, dict[str, object]]:
        return {
            scope: {
                "collection_name": self._chunk_store.collection_name(scope),
                "points_count": self._chunk_store.count(scope),
                "readable": bool(self._chunk_store.read_collection_exists(scope)),
            }
            for scope in ("code", "docs")
        }

    def _index_revision(self) -> dict[str, object]:
        repo_entry = self._registry.get_repo(self._settings.logical_repo_identity)
        if repo_entry is None:
            return {}
        return {
            "last_full_build_ts": repo_entry.last_full_build_ts,
            "last_incremental_update_ts": repo_entry.last_incremental_update_ts,
            "indexed_branch": repo_entry.indexed_branch,
            "indexed_commit_hash": repo_entry.indexed_commit_hash,
            "code_points_count": repo_entry.code_points_count,
            "docs_points_count": repo_entry.docs_points_count,
        }

    def _current_source_contract(self, *, chunks: list | None = None) -> dict[str, object]:
        if chunks is not None:
            return source_contract_from_chunks(
                schema_version=self._settings.SEMANTIC_MCP_INDEX_SCHEMA_VERSION,
                profile=self._profile(),
                collections=self._collections_contract(),
                chunks=chunks,
                index_revision=self._index_revision(),
            )
        return {
            "schema_version": self._settings.SEMANTIC_MCP_INDEX_SCHEMA_VERSION,
            "profile": self._profile(),
            "collections": self._collections_contract(),
            "index_revision": self._index_revision(),
        }

    def graph_status(self, *, verify_source_revision: bool = True) -> GraphStatusResult:
        """Return detailed graph status without building or updating graph state."""

        repo_entry = self._registry.get_repo(self._settings.logical_repo_identity)
        git_snapshot = (
            self._git_snapshot()
            if verify_source_revision
            else inspect_git_revision(self._settings.repo_root)
        )
        chunks = self._indexed_chunks() if verify_source_revision else None
        expected_contract = self._current_source_contract(chunks=chunks)
        indexed_at = (
            repo_entry.last_incremental_update_ts or repo_entry.last_full_build_ts
            if repo_entry is not None
            else None
        )
        tracked_stale = (
            verify_source_revision
            and tracked_indexable_changes_after(git_snapshot, indexed_at)
        )
        untracked_stale = (
            verify_source_revision
            and untracked_indexable_changes_after(git_snapshot, indexed_at)
        )
        invalidated_preview = []
        if tracked_stale:
            invalidated_preview.extend(git_snapshot.changed_indexable_files_sample)
        if untracked_stale:
            invalidated_preview.extend(git_snapshot.untracked_indexable_files_sample)
        invalidated_count = (
            (git_snapshot.changed_indexable_files_count if tracked_stale else 0)
            + (git_snapshot.untracked_indexable_files_count if untracked_stale else 0)
        )
        status = self._store.status(
            repo_root=self._settings.logical_repo_identity,
            profile=self._profile(),
            current_head_commit=git_snapshot.head_commit,
            expected_source_index_contract=expected_contract,
            expected_extractor_versions_hash=stable_json_hash(GRAPH_EXTRACTOR_VERSIONS),
            invalidated_path_count=invalidated_count,
            invalidated_paths_preview=invalidated_preview,
        )
        needs_manifest_invalidations = (
            status.state == "stale"
            and "graph_source_index_revision_mismatch" in status.warning_codes
        )
        manifest_paths_known = self._path_manifest is None
        if self._path_manifest is not None and status.built_at and needs_manifest_invalidations:
            invalidated_path_limit = getattr(
                self._settings,
                "SEMANTIC_MCP_GRAPH_INVALIDATED_PATH_LIMIT",
                200,
            )
            try:
                manifest_count, manifest_preview, manifest_paths_known = self._path_manifest.invalidated_paths_after(
                    status.built_at,
                    limit=invalidated_path_limit,
                )
            except Exception:  # noqa: BLE001
                manifest_count = 0
                manifest_preview = []
                manifest_paths_known = False
            if manifest_count > 0 or not manifest_paths_known:
                merged_preview = _dedupe_paths([*invalidated_preview, *manifest_preview])[:invalidated_path_limit]
                status = self._store.status(
                    repo_root=self._settings.logical_repo_identity,
                    profile=self._profile(),
                    current_head_commit=git_snapshot.head_commit,
                    expected_source_index_contract=expected_contract,
                    expected_extractor_versions_hash=stable_json_hash(GRAPH_EXTRACTOR_VERSIONS),
                    invalidated_path_count=invalidated_count + manifest_count,
                    invalidated_paths_preview=merged_preview,
                )
        if (
            status.state == "stale"
            and "graph_source_index_revision_mismatch" in status.warning_codes
            and not manifest_paths_known
        ):
            status = status.model_copy(
                update={
                    "expansion_allowed": False,
                    "warning_codes": _dedupe_paths(
                        [*status.warning_codes, "graph_invalidated_paths_unknown"]
                    ),
                }
            )
        elif (
            status.state == "stale"
            and status.expansion_allowed
            and "graph_source_index_revision_mismatch" in status.warning_codes
            and self._path_manifest is None
        ):
            status = status.model_copy(
                update={
                    "expansion_allowed": False,
                    "warning_codes": _dedupe_paths(
                        [*status.warning_codes, "graph_invalidated_paths_unknown"]
                    ),
                }
            )
        return status

    def graph_summary(self) -> GraphStatusSummary:
        """Return compact graph status for index_status.v2."""

        return graph_summary_from_status(self.graph_status(verify_source_revision=False))

    def _indexed_chunks(self) -> list:
        chunks = []
        for scope in ("code", "docs"):
            for point in self._chunk_store.scroll_chunks(scope):
                chunks.append(point_to_chunk(point))
        return chunks

    def _indexed_chunk_count_hint(self) -> int | None:
        store_counts: list[int] = []
        for scope in ("code", "docs"):
            try:
                store_counts.append(int(self._chunk_store.count(scope)))
            except Exception:  # noqa: BLE001
                store_counts = []
                break
        if store_counts and sum(store_counts) > 0:
            return sum(store_counts)
        repo_entry = self._registry.get_repo(self._settings.logical_repo_identity)
        if repo_entry is None:
            return None
        counts = [
            repo_entry.code_points_count,
            repo_entry.docs_points_count,
        ]
        if any(count is None for count in counts):
            return None
        total = sum(int(count or 0) for count in counts)
        return total if total > 0 else None

    def _update_max_duration_ms(self, *, allow_large_update: bool) -> int | None:
        max_duration_ms = self._settings.SEMANTIC_MCP_GRAPH_UPDATE_MAX_DURATION_MS
        if allow_large_update:
            max_duration_ms = max(
                max_duration_ms,
                self._settings.SEMANTIC_MCP_GRAPH_UPDATE_LARGE_MAX_DURATION_MS,
            )
        return max_duration_ms

    def _duration_exceeded(self, started_at: float, *, max_duration_ms: int | None) -> bool:
        if max_duration_ms is None or max_duration_ms < 0:
            return False
        return int((perf_counter() - started_at) * 1000) > max_duration_ms

    def update_graph(
        self,
        *,
        paths: list[str] | None = None,
        idempotency_key: str | None = None,
        allow_large_update: bool = False,
    ) -> GraphUpdateResult:
        """Incrementally update the graph artifact from current indexed chunks."""

        started_at = perf_counter()
        requested_paths = paths or []
        normalized_requested_paths = _dedupe_paths(requested_paths)
        profile = self._profile()
        max_duration_ms = self._update_max_duration_ms(allow_large_update=allow_large_update)
        status_before = self.graph_status(verify_source_revision=False)
        if status_before.state == "missing":
            return self._skipped_update_result(
                profile=profile,
                status=status_before,
                idempotency_key=idempotency_key,
                paths_requested=len(requested_paths),
                paths_normalized=len(normalized_requested_paths),
                skip_reason="graph_missing",
                warning_codes=["graph_missing"],
                action_code="build_graph",
                allow_large_update=allow_large_update,
            )
        if status_before.state in {"incompatible", "building"}:
            return self._skipped_update_result(
                profile=profile,
                status=status_before,
                idempotency_key=idempotency_key,
                paths_requested=len(requested_paths),
                paths_normalized=len(normalized_requested_paths),
                skip_reason=f"graph_{status_before.state}",
                warning_codes=[f"graph_{status_before.state}"],
                action_code="rebuild_graph",
                allow_large_update=allow_large_update,
            )
        if status_before.state == "error" and "graph_file_errors" not in status_before.warning_codes:
            return self._skipped_update_result(
                profile=profile,
                status=status_before,
                idempotency_key=idempotency_key,
                paths_requested=len(requested_paths),
                paths_normalized=len(normalized_requested_paths),
                skip_reason="graph_artifact_error",
                warning_codes=["graph_artifact_error"],
                action_code="rebuild_graph",
                allow_large_update=allow_large_update,
            )

        max_chunks_scanned = self._settings.SEMANTIC_MCP_GRAPH_UPDATE_MAX_CHUNKS_SCANNED
        if allow_large_update:
            max_chunks_scanned = max(
                max_chunks_scanned,
                self._settings.SEMANTIC_MCP_GRAPH_UPDATE_DISCOVERY_MAX_CHUNKS,
            )
        chunk_count_hint = self._indexed_chunk_count_hint()
        if chunk_count_hint is not None and chunk_count_hint > max_chunks_scanned:
            return self._skipped_update_result(
                profile=profile,
                status=status_before,
                idempotency_key=idempotency_key,
                paths_requested=len(requested_paths),
                paths_normalized=len(normalized_requested_paths),
                skip_reason="graph_update_bounds_exceeded",
                warning_codes=["graph_update_bounds_exceeded"],
                bounds_exceeded=["max_chunks_scanned"],
                action_code="rebuild_graph" if allow_large_update else "update_graph",
                allow_large_update=allow_large_update,
                large_retry_available=not allow_large_update,
            )

        chunks = self._indexed_chunks()
        if len(chunks) > max_chunks_scanned:
            return self._skipped_update_result(
                profile=profile,
                status=status_before,
                idempotency_key=idempotency_key,
                paths_requested=len(requested_paths),
                paths_normalized=len(normalized_requested_paths),
                skip_reason="graph_update_bounds_exceeded",
                warning_codes=["graph_update_bounds_exceeded"],
                bounds_exceeded=["max_chunks_scanned"],
                action_code="rebuild_graph" if allow_large_update else "update_graph",
                allow_large_update=allow_large_update,
                large_retry_available=not allow_large_update,
            )
        if self._duration_exceeded(started_at, max_duration_ms=max_duration_ms):
            return self._skipped_update_result(
                profile=profile,
                status=status_before,
                idempotency_key=idempotency_key,
                paths_requested=len(requested_paths),
                paths_normalized=len(normalized_requested_paths),
                skip_reason="graph_update_bounds_exceeded",
                warning_codes=["graph_update_bounds_exceeded"],
                bounds_exceeded=["max_duration_ms"],
                action_code="rebuild_graph" if allow_large_update else "update_graph",
                allow_large_update=allow_large_update,
                large_retry_available=not allow_large_update,
            )

        chunks_by_path = _chunks_by_path(chunks)
        current_signatures = {
            relative_path: graph_input_signature_for_chunks(path_chunks)
            for relative_path, path_chunks in chunks_by_path.items()
        }
        current_file_hashes = _file_content_hashes_by_path(chunks_by_path)
        graph_file_state = self._store.graph_file_state()
        required_paths, signature_backfill_candidates = self._discover_update_paths(
            requested_paths=[],
            current_signatures=current_signatures,
            current_file_hashes=current_file_hashes,
            graph_file_state=graph_file_state,
        )
        source_index_contract = None
        snapshot = None
        if signature_backfill_candidates:
            source_index_contract = self._current_source_contract(chunks=chunks)
            snapshot = self._build_snapshot_for_update(
                chunks=chunks,
                source_index_contract=source_index_contract,
            )
            safe_paths = self._store.safe_legacy_signature_backfill_paths(
                snapshot=snapshot,
                signatures_by_path=signature_backfill_candidates,
            )
            signatures_to_backfill = {
                path: signature
                for path, signature in signature_backfill_candidates.items()
                if path in safe_paths
            }
            required_paths.extend(
                path
                for path in signature_backfill_candidates
                if path not in safe_paths
            )
        else:
            signatures_to_backfill = {}
        if signatures_to_backfill:
            self._store.backfill_graph_input_signatures(signatures_to_backfill)
            for relative_path, signature in signatures_to_backfill.items():
                if relative_path in graph_file_state:
                    graph_file_state[relative_path]["graph_input_signature"] = signature
        discovered_paths = _dedupe_paths([*normalized_requested_paths, *required_paths])
        if self._duration_exceeded(started_at, max_duration_ms=max_duration_ms):
            return self._skipped_update_result(
                profile=profile,
                status=status_before,
                idempotency_key=idempotency_key,
                paths_requested=len(requested_paths),
                paths_normalized=len(normalized_requested_paths),
                skip_reason="graph_update_bounds_exceeded",
                warning_codes=["graph_update_bounds_exceeded"],
                bounds_exceeded=["max_duration_ms"],
                action_code="rebuild_graph" if allow_large_update else "update_graph",
                allow_large_update=allow_large_update,
                large_retry_available=not allow_large_update,
            )
        metadata_only_update = (
            not discovered_paths
            and status_before.update_required
            and status_before.update_blocked_reason is None
        )
        if not discovered_paths and not metadata_only_update:
            return self._skipped_update_result(
                profile=profile,
                status=status_before,
                idempotency_key=idempotency_key,
                paths_requested=len(requested_paths),
                paths_discovered=0,
                paths_normalized=len(normalized_requested_paths),
                skip_reason="graph_update_not_required",
                warning_codes=[],
                allow_large_update=allow_large_update,
            )
        max_paths = self._settings.SEMANTIC_MCP_GRAPH_UPDATE_MAX_PATHS
        if allow_large_update:
            max_paths = max(
                max_paths,
                self._settings.SEMANTIC_MCP_GRAPH_UPDATE_DISCOVERY_MAX_PATHS,
            )
        if len(discovered_paths) > max_paths:
            return self._skipped_update_result(
                profile=profile,
                status=status_before,
                idempotency_key=idempotency_key,
                paths_requested=len(requested_paths),
                paths_discovered=len(discovered_paths),
                paths_normalized=len(normalized_requested_paths),
                skip_reason="graph_update_bounds_exceeded",
                warning_codes=["graph_update_bounds_exceeded"],
                bounds_exceeded=["max_paths"],
                action_code="rebuild_graph" if allow_large_update else "update_graph",
                allow_large_update=allow_large_update,
                large_retry_available=not allow_large_update,
            )

        max_reverse_dependent_paths = self._settings.SEMANTIC_MCP_GRAPH_UPDATE_MAX_REVERSE_DEPENDENT_PATHS
        if allow_large_update:
            max_reverse_dependent_paths = max(
                max_reverse_dependent_paths,
                self._settings.SEMANTIC_MCP_GRAPH_UPDATE_DISCOVERY_MAX_PATHS,
            )
        reverse_paths, reverse_complete = self._store.reverse_dependent_paths(
            discovered_paths,
            limit=max_reverse_dependent_paths,
        )
        if not reverse_complete:
            return self._skipped_update_result(
                profile=profile,
                status=status_before,
                idempotency_key=idempotency_key,
                paths_requested=len(requested_paths),
                paths_discovered=len(discovered_paths),
                paths_normalized=len(normalized_requested_paths),
                skip_reason="reverse_dependencies_exceed_auto_bounds",
                warning_codes=["graph_reverse_dependencies_not_repaired"],
                action_code="rebuild_graph" if allow_large_update else "update_graph",
                allow_large_update=allow_large_update,
                large_retry_available=not allow_large_update,
            )
        if self._duration_exceeded(started_at, max_duration_ms=max_duration_ms):
            return self._skipped_update_result(
                profile=profile,
                status=status_before,
                idempotency_key=idempotency_key,
                paths_requested=len(requested_paths),
                paths_discovered=len(discovered_paths),
                paths_normalized=len(normalized_requested_paths),
                skip_reason="graph_update_bounds_exceeded",
                warning_codes=["graph_update_bounds_exceeded"],
                bounds_exceeded=["max_duration_ms"],
                action_code="rebuild_graph" if allow_large_update else "update_graph",
                allow_large_update=allow_large_update,
                large_retry_available=not allow_large_update,
            )

        affected_paths = _dedupe_paths([*discovered_paths, *reverse_paths])
        if source_index_contract is None:
            source_index_contract = self._current_source_contract(chunks=chunks)
        source_contract_revision = stable_json_hash(source_index_contract)
        if snapshot is None:
            snapshot = self._build_snapshot_for_update(
                chunks=chunks,
                source_index_contract=source_index_contract,
            )
        new_reverse_paths = _new_reverse_dependent_paths(snapshot, affected_paths)
        if len(new_reverse_paths) > max_reverse_dependent_paths:
            return self._skipped_update_result(
                profile=profile,
                status=status_before,
                idempotency_key=idempotency_key,
                paths_requested=len(requested_paths),
                paths_discovered=len(discovered_paths),
                paths_normalized=len(normalized_requested_paths),
                skip_reason="reverse_dependencies_exceed_auto_bounds",
                warning_codes=["graph_reverse_dependencies_not_repaired"],
                action_code="rebuild_graph" if allow_large_update else "update_graph",
                allow_large_update=allow_large_update,
                large_retry_available=not allow_large_update,
            )
        affected_paths = _dedupe_paths([*affected_paths, *new_reverse_paths])
        if self._duration_exceeded(started_at, max_duration_ms=max_duration_ms):
            return self._skipped_update_result(
                profile=profile,
                status=status_before,
                idempotency_key=idempotency_key,
                paths_requested=len(requested_paths),
                paths_discovered=len(discovered_paths),
                paths_normalized=len(normalized_requested_paths),
                skip_reason="graph_update_bounds_exceeded",
                warning_codes=["graph_update_bounds_exceeded"],
                bounds_exceeded=["max_duration_ms"],
                action_code="rebuild_graph" if allow_large_update else "update_graph",
                allow_large_update=allow_large_update,
                large_retry_available=not allow_large_update,
            )
        if len(affected_paths) > max_paths:
            return self._skipped_update_result(
                profile=profile,
                status=status_before,
                idempotency_key=idempotency_key,
                paths_requested=len(requested_paths),
                paths_discovered=len(discovered_paths),
                paths_normalized=len(normalized_requested_paths),
                skip_reason="graph_update_bounds_exceeded",
                warning_codes=["graph_update_bounds_exceeded"],
                bounds_exceeded=["max_paths"],
                action_code="rebuild_graph" if allow_large_update else "update_graph",
                allow_large_update=allow_large_update,
                large_retry_available=not allow_large_update,
            )
        chunks_for_updated_paths = sum(len(chunks_by_path.get(path, [])) for path in affected_paths)
        max_changed_chunks = self._settings.SEMANTIC_MCP_GRAPH_UPDATE_MAX_CHANGED_CHUNKS
        if allow_large_update:
            max_changed_chunks = max(
                max_changed_chunks,
                self._settings.SEMANTIC_MCP_GRAPH_UPDATE_DISCOVERY_MAX_CHUNKS,
            )
        if chunks_for_updated_paths > max_changed_chunks:
            return self._skipped_update_result(
                profile=profile,
                status=status_before,
                idempotency_key=idempotency_key,
                paths_requested=len(requested_paths),
                paths_discovered=len(discovered_paths),
                paths_normalized=len(normalized_requested_paths),
                skip_reason="graph_update_bounds_exceeded",
                warning_codes=["graph_update_bounds_exceeded"],
                bounds_exceeded=["max_changed_chunks"],
                action_code="rebuild_graph" if allow_large_update else "update_graph",
                allow_large_update=allow_large_update,
                large_retry_available=not allow_large_update,
            )

        update_plan_hash = stable_json_hash(
            {
                "paths": affected_paths,
                "source_index_contract": source_index_contract,
                "allow_large_update": allow_large_update,
                "current_signatures": {
                    path: current_signatures.get(path)
                    for path in affected_paths
                    if path in current_signatures
                },
                "graph_file_state": {
                    path: {
                        "graph_state": graph_file_state.get(path, {}).get("graph_state"),
                        "error_code": graph_file_state.get(path, {}).get("error_code"),
                        "graph_input_signature": graph_file_state.get(path, {}).get("graph_input_signature"),
                    }
                    for path in affected_paths
                    if path in graph_file_state
                },
            }
        )
        idempotency_fingerprint = stable_json_hash(
            {
                "requested_paths": normalized_requested_paths,
                "source_contract_revision": source_contract_revision,
                "allow_large_update": allow_large_update,
            }
        )
        if idempotency_key:
            cached = self._update_idempotency_results.get(idempotency_key)
            if (
                cached is not None
                and cached.source_contract_revision_after == source_contract_revision
                and not required_paths
                and self._update_idempotency_fingerprints.get(idempotency_key) == idempotency_fingerprint
                and cached.allow_large_update == allow_large_update
            ):
                return cached.model_copy(
                    update={
                        "idempotent_retry": True,
                        "operation_in_progress": False,
                    },
                    deep=True,
                )
        try:
            max_rows_deleted = self._settings.SEMANTIC_MCP_GRAPH_UPDATE_MAX_ROWS_DELETED
            max_rows_upserted = self._settings.SEMANTIC_MCP_GRAPH_UPDATE_MAX_ROWS_UPSERTED
            max_duration_ms = self._settings.SEMANTIC_MCP_GRAPH_UPDATE_MAX_DURATION_MS
            if allow_large_update:
                max_rows_deleted = max(
                    max_rows_deleted,
                    self._settings.SEMANTIC_MCP_GRAPH_UPDATE_LARGE_MAX_ROWS_DELETED,
                )
                max_rows_upserted = max(
                    max_rows_upserted,
                    self._settings.SEMANTIC_MCP_GRAPH_UPDATE_LARGE_MAX_ROWS_UPSERTED,
                )
                max_duration_ms = max(
                    max_duration_ms,
                    self._settings.SEMANTIC_MCP_GRAPH_UPDATE_LARGE_MAX_DURATION_MS,
                )
            store_result = self._store.update_paths(
                snapshot=snapshot,
                affected_paths=affected_paths,
                max_rows_deleted=max_rows_deleted,
                max_rows_upserted=max_rows_upserted,
                max_duration_ms=max_duration_ms,
                started_at=started_at,
            )
        except GraphUpdateBoundsExceeded as exc:
            return self._skipped_update_result(
                profile=profile,
                status=status_before,
                idempotency_key=idempotency_key,
                paths_requested=len(requested_paths),
                paths_discovered=len(discovered_paths),
                paths_normalized=len(normalized_requested_paths),
                skip_reason="graph_update_bounds_exceeded",
                warning_codes=["graph_update_bounds_exceeded"],
                bounds_exceeded=list(exc.bounds_exceeded),
                action_code="rebuild_graph" if allow_large_update else "update_graph",
                allow_large_update=allow_large_update,
                large_retry_available=not allow_large_update,
            )
        status_after = self.graph_status()
        update_result = GraphUpdateResult(
            repo_root=self._settings.logical_repo_identity,
            profile=profile,
            graph_path=self.graph_path,
            idempotency_key=idempotency_key,
            allow_large_update=allow_large_update,
            updated=True,
            paths_requested=len(requested_paths),
            paths_discovered=len(discovered_paths),
            paths_normalized=len(normalized_requested_paths),
            paths_required=len(affected_paths),
            reverse_dependent_paths_added=len(_dedupe_paths([*reverse_paths, *new_reverse_paths])),
            paths_updated=len([path for path in affected_paths if path in current_signatures]),
            paths_deleted=len(store_result.deleted_paths),
            zero_chunk_paths=0,
            paths_missing_from_index=len(store_result.deleted_paths),
            chunks_scanned_total=len(chunks),
            chunks_for_updated_paths=chunks_for_updated_paths,
            nodes_deleted=store_result.nodes_deleted,
            edges_deleted=store_result.edges_deleted,
            nodes_upserted=store_result.nodes_upserted,
            edges_upserted=store_result.edges_upserted,
            source_contract_updated=True,
            source_contract_revision_before=stable_json_hash(status_before.source_index_contract or {}),
            source_contract_revision_after=source_contract_revision,
            graph_state_after=status_after.state,
            expansion_allowed_after=status_after.expansion_allowed,
            update_plan_hash=update_plan_hash,
            metadata_only_update=metadata_only_update,
            warning_codes=list(status_after.warning_codes),
            status=status_after,
        )
        if idempotency_key:
            self._update_idempotency_results[idempotency_key] = update_result
            self._update_idempotency_fingerprints[idempotency_key] = idempotency_fingerprint
        return update_result

    def build_graph(self, *, rebuild: bool = False) -> GraphBuildResult:
        """Build or rebuild the graph artifact from the current indexed chunks."""

        chunks = self._indexed_chunks()

        profile = self._profile()
        if not chunks:
            status = self.graph_status()
            return GraphBuildResult(
                repo_root=self._settings.logical_repo_identity,
                profile=profile,
                graph_path=self.graph_path,
                built=False,
                rebuilt=rebuild,
                state=status.state,
                warning_codes=["graph_build_no_indexed_chunks"],
                status=status,
                reason="No indexed chunks are available for graph build.",
            )

        git_snapshot = self._git_snapshot()
        repo_entry = self._registry.get_repo(self._settings.logical_repo_identity)
        built_commit = (
            repo_entry.indexed_commit_hash
            if repo_entry is not None and repo_entry.indexed_commit_hash
            else git_snapshot.head_commit
        )
        source_index_contract = self._current_source_contract(chunks=chunks)
        snapshot = build_graph_snapshot(
            repo_root=self._settings.logical_repo_identity,
            profile=profile,
            chunks=chunks,
            built_commit=built_commit,
            include_globs=list(self._settings.effective_include_globs),
            exclude_globs=list(self._settings.SEMANTIC_MCP_EXCLUDE_GLOBS),
            source_index_contract=source_index_contract,
        )
        if rebuild:
            self._store.reset()
        self._store.replace_graph(snapshot)
        status = self.graph_status()
        return GraphBuildResult(
            repo_root=self._settings.logical_repo_identity,
            profile=profile,
            graph_path=self.graph_path,
            built=True,
            rebuilt=rebuild,
            state=status.state,
            built_at=status.built_at,
            built_commit=status.built_commit,
            files_count=len(snapshot.files),
            chunks_count=snapshot.chunks_count,
            nodes_count=len(snapshot.nodes),
            edges_count=len(snapshot.edges),
            warning_codes=list(status.warning_codes),
            status=status,
        )

    def _build_snapshot_for_update(self, *, chunks: list, source_index_contract: dict[str, object]):
        git_snapshot = self._git_snapshot()
        repo_entry = self._registry.get_repo(self._settings.logical_repo_identity)
        built_commit = (
            repo_entry.indexed_commit_hash
            if repo_entry is not None and repo_entry.indexed_commit_hash
            else git_snapshot.head_commit
        )
        return build_graph_snapshot(
            repo_root=self._settings.logical_repo_identity,
            profile=self._profile(),
            chunks=chunks,
            built_commit=built_commit,
            include_globs=list(self._settings.effective_include_globs),
            exclude_globs=list(self._settings.SEMANTIC_MCP_EXCLUDE_GLOBS),
            source_index_contract=source_index_contract,
        )

    def _discover_update_paths(
        self,
        *,
        requested_paths: list[str],
        current_signatures: dict[str, str],
        current_file_hashes: dict[str, str],
        graph_file_state: dict[str, dict[str, object]],
    ) -> tuple[list[str], dict[str, str]]:
        discovered: list[str] = list(requested_paths)
        backfill_signatures: dict[str, str] = {}
        for relative_path, signature in current_signatures.items():
            row = graph_file_state.get(relative_path)
            if row is None:
                discovered.append(relative_path)
                continue
            stored_signature = row.get("graph_input_signature")
            if stored_signature == signature:
                continue
            if (
                not stored_signature
                and row.get("graph_state") == "ready"
                and not row.get("error_code")
                and row.get("content_hash") == current_file_hashes.get(relative_path)
            ):
                backfill_signatures[relative_path] = signature
                continue
            discovered.append(relative_path)
        for relative_path, row in graph_file_state.items():
            if relative_path not in current_signatures:
                discovered.append(relative_path)
                continue
            if row.get("graph_state") != "ready" or row.get("error_code"):
                discovered.append(relative_path)
        return _dedupe_paths(discovered), backfill_signatures

    def _skipped_update_result(
        self,
        *,
        profile: str,
        status: GraphStatusResult,
        idempotency_key: str | None,
        paths_requested: int = 0,
        paths_discovered: int = 0,
        paths_normalized: int = 0,
        skip_reason: str,
        warning_codes: list[str],
        bounds_exceeded: list[str] | None = None,
        action_code: str | None = None,
        allow_large_update: bool = False,
        large_retry_available: bool = False,
    ) -> GraphUpdateResult:
        actions = []
        if action_code:
            action: dict[str, object] = {
                "code": action_code,
                "requires_user_permission": True,
                "safe_auto_run": False,
                "destructive": action_code in {"rebuild_graph", "rebuild_index"},
                "expensive": action_code in {"rebuild_graph", "rebuild_index", "build_graph"},
                "reason_codes": warning_codes,
            }
            if large_retry_available and action_code == "update_graph":
                action["tool_arguments"] = {
                    "paths": [],
                    "allow_large_update": True,
                }
                action["detail"] = (
                    "The update exceeded safe automatic bounds; retry update_graph "
                    "with allow_large_update=true to use higher bounded manual limits."
                )
            actions.append(
                action
            )
        return GraphUpdateResult(
            repo_root=self._settings.logical_repo_identity,
            profile=profile,
            graph_path=self.graph_path,
            idempotency_key=idempotency_key,
            allow_large_update=allow_large_update,
            updated=False,
            skipped=True,
            skip_reason=skip_reason,
            paths_requested=paths_requested,
            paths_discovered=paths_discovered,
            paths_normalized=paths_normalized,
            graph_state_after=status.state,
            expansion_allowed_after=status.expansion_allowed,
            bounds_exceeded=bounds_exceeded or [],
            warning_codes=warning_codes,
            recommended_actions=actions,
            status=status,
        )

    def rebuild_graph(self) -> GraphBuildResult:
        """Drop and rebuild the graph artifact from indexed chunks."""

        return self.build_graph(rebuild=True)

    def read_provider(self) -> "GraphReadProvider":
        """Return a lifecycle-free read provider for retrieval code."""

        return GraphReadProvider(self)


class GraphReadProvider:
    """Read-only graph interface used by repo_context_search."""

    def __init__(self, service: GraphService) -> None:
        self._service = service

    def graph_status(self, *, verify_source_revision: bool = True) -> GraphStatusResult:
        return self._service.graph_status(verify_source_revision=verify_source_revision)

    def find_nodes_for_chunks(self, chunk_ids: list[str], *, limit: int = 100) -> list[GraphNode]:
        return self._service._store.find_nodes_for_chunks(chunk_ids, limit=limit)

    def find_file_nodes(self, relative_paths: list[str], *, limit: int = 100) -> list[GraphNode]:
        return self._service._store.find_file_nodes(relative_paths, limit=limit)

    def find_symbol_nodes(self, bindings: list[tuple[str, str]], *, limit: int = 100) -> list[GraphNode]:
        return self._service._store.find_symbol_nodes(bindings, limit=limit)

    def find_doc_section_nodes(self, bindings: list[tuple[str, str]], *, limit: int = 100) -> list[GraphNode]:
        return self._service._store.find_doc_section_nodes(bindings, limit=limit)

    def find_term_nodes(self, terms: list[tuple[str, str | None]], *, limit: int = 100) -> list[GraphNode]:
        return self._service._store.find_term_nodes(terms, limit=limit)

    def neighbors(
        self,
        node_ids: list[str],
        *,
        edge_types: list[str],
        max_neighbors_per_node: int = 30,
        max_total: int = 200,
    ) -> list[GraphEdge]:
        return self._service._store.neighbors(
            node_ids,
            edge_types=edge_types,
            max_neighbors_per_node=max_neighbors_per_node,
            max_total=max_total,
        )

    def chunks_for_nodes(self, node_ids: list[str], *, limit: int = 300) -> list[GraphChunkLink]:
        return self._service._store.chunks_for_nodes(node_ids, limit=limit)


def _chunks_by_path(chunks: list) -> dict[str, list]:
    result: dict[str, list] = {}
    for chunk in chunks:
        relative_path = str(chunk.relative_path).replace("\\", "/").strip("/")
        result.setdefault(relative_path, []).append(chunk)
    return result


def _file_content_hashes_by_path(chunks_by_path: dict[str, list]) -> dict[str, str]:
    return {
        relative_path: stable_json_hash(
            sorted(str(chunk.content_hash) for chunk in path_chunks)
        )
        for relative_path, path_chunks in chunks_by_path.items()
    }


def _new_reverse_dependent_paths(snapshot, affected_paths: list[str]) -> list[str]:
    affected = set(_dedupe_paths(affected_paths))
    if not affected:
        return []
    nodes_by_id = {str(row["node_id"]): row for row in snapshot.nodes}
    path_owned_node_ids = {
        str(row["node_id"])
        for row in snapshot.nodes
        if row.get("relative_path")
        and str(row["relative_path"]).replace("\\", "/").strip("/") in affected
        and str(row.get("node_type")) in PATH_OWNED_NODE_TYPES
    }
    affected_chunk_ids = {
        str(row["chunk_id"])
        for row in snapshot.nodes
        if row.get("relative_path")
        and str(row["relative_path"]).replace("\\", "/").strip("/") in affected
        and row.get("chunk_id")
    }
    shared_node_ids = {
        str(row["node_id"])
        for row in snapshot.nodes
        if str(row.get("node_type")) in {"env_var", "route", "module"}
        and any(
            str(link["node_id"]) == str(row["node_id"]) and str(link["chunk_id"]) in affected_chunk_ids
            for link in snapshot.node_chunks
        )
    }
    for edge in snapshot.edges:
        relative_path = str(edge.get("relative_path") or "").replace("\\", "/").strip("/")
        if relative_path not in affected:
            continue
        for endpoint in (str(edge.get("source_node_id")), str(edge.get("target_node_id"))):
            node = nodes_by_id.get(endpoint)
            if node is not None and str(node.get("node_type")) in {"env_var", "route", "module"}:
                shared_node_ids.add(endpoint)
    reverse_target_node_ids = path_owned_node_ids | shared_node_ids
    reverse_paths: list[str] = []
    for edge in snapshot.edges:
        relative_path = str(edge.get("relative_path") or "").replace("\\", "/").strip("/")
        if not relative_path or relative_path in affected:
            continue
        if str(edge.get("source_node_id")) in reverse_target_node_ids or str(edge.get("target_node_id")) in reverse_target_node_ids:
            reverse_paths.append(relative_path)
    return _dedupe_paths(reverse_paths)
