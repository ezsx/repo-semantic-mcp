"""SQLite graph artifact storage."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
import sqlite3
from time import perf_counter

from services.repo_semantic.contracts.graph import (
    GraphCounts,
    GraphStatusResult,
    GraphStatusSummary,
)
from services.repo_semantic.graph.types import GraphChunkLink, GraphEdge, GraphNode
from services.repo_semantic.graph.artifacts import stable_json_hash
from services.repo_semantic.graph.builder import GraphSnapshot
from services.repo_semantic.graph.schema import GRAPH_SCHEMA_VERSION, ensure_graph_schema
from services.repo_semantic.storage.sqlite.connections import connect_row_factory


PATH_OWNED_NODE_TYPES = {
    "file",
    "chunk",
    "symbol",
    "doc_section",
    "config_item",
    "test_case",
}
SHARED_NODE_TYPES = {"env_var", "route", "module"}


@dataclass(slots=True)
class GraphUpdateStoreResult:
    nodes_deleted: int = 0
    edges_deleted: int = 0
    nodes_upserted: int = 0
    edges_upserted: int = 0
    files_upserted: int = 0
    files_deleted: int = 0
    shared_nodes_pruned: int = 0
    deleted_paths: list[str] = field(default_factory=list)


class GraphUpdateBoundsExceeded(Exception):
    """Raised when a planned graph update exceeds configured mutation bounds."""

    def __init__(self, *, bounds_exceeded: list[str], summary: dict[str, object]) -> None:
        super().__init__("graph_update_bounds_exceeded")
        self.bounds_exceeded = bounds_exceeded
        self.summary = summary


class GraphStore:
    """Persist and inspect one per-repo/profile graph SQLite artifact."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = connect_row_factory(self._db_path)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _read_connect(self) -> sqlite3.Connection | None:
        if not self._db_path.exists():
            return None
        connection = connect_row_factory(self._db_path)
        connection.execute("PRAGMA query_only = ON")
        return connection

    def reset(self) -> None:
        """Remove the graph artifact file before an explicit rebuild."""

        if self._db_path.exists():
            self._db_path.unlink()

    def replace_graph(self, snapshot: GraphSnapshot) -> None:
        """Replace all graph rows in a single transaction."""

        with self._connect() as connection:
            ensure_graph_schema(connection)
            connection.execute("DELETE FROM graph_edges")
            connection.execute("DELETE FROM graph_node_chunks")
            connection.execute("DELETE FROM graph_node_terms")
            connection.execute("DELETE FROM graph_nodes")
            connection.execute("DELETE FROM graph_files")
            connection.execute("DELETE FROM graph_metadata")
            connection.execute(
                """
                INSERT INTO graph_metadata (
                    repo_id,
                    repo_root,
                    profile,
                    schema_version,
                    built_commit,
                    built_at,
                    include_rules_hash,
                    exclude_rules_hash,
                    extractor_versions_json,
                    source_index_contract_json,
                    graph_contract_hash
                ) VALUES (
                    :repo_id,
                    :repo_root,
                    :profile,
                    :schema_version,
                    :built_commit,
                    :built_at,
                    :include_rules_hash,
                    :exclude_rules_hash,
                    :extractor_versions_json,
                    :source_index_contract_json,
                    :graph_contract_hash
                )
                """,
                snapshot.metadata,
            )
            connection.executemany(
                """
                INSERT INTO graph_files (
                    relative_path,
                    content_hash,
                    graph_input_signature,
                    mtime,
                    graph_state,
                    indexed_at,
                    last_extracted_at,
                    error_code,
                    error_detail
                ) VALUES (
                    :relative_path,
                    :content_hash,
                    :graph_input_signature,
                    :mtime,
                    :graph_state,
                    :indexed_at,
                    :last_extracted_at,
                    :error_code,
                    :error_detail
                )
                """,
                snapshot.files,
            )
            connection.executemany(
                """
                INSERT INTO graph_nodes (
                    node_id,
                    node_type,
                    key,
                    relative_path,
                    chunk_id,
                    start_line,
                    end_line,
                    payload_json
                ) VALUES (
                    :node_id,
                    :node_type,
                    :key,
                    :relative_path,
                    :chunk_id,
                    :start_line,
                    :end_line,
                    :payload_json
                )
                """,
                snapshot.nodes,
            )
            connection.executemany(
                """
                INSERT OR IGNORE INTO graph_node_terms (
                    node_id,
                    term,
                    normalized_term,
                    term_type,
                    case_sensitive,
                    source
                ) VALUES (
                    :node_id,
                    :term,
                    :normalized_term,
                    :term_type,
                    :case_sensitive,
                    :source
                )
                """,
                [
                    {
                        **row,
                        "case_sensitive": 1 if row.get("case_sensitive") else 0,
                    }
                    for row in snapshot.node_terms
                ],
            )
            connection.executemany(
                """
                INSERT OR IGNORE INTO graph_node_chunks (
                    node_id,
                    chunk_id,
                    relation,
                    weight
                ) VALUES (
                    :node_id,
                    :chunk_id,
                    :relation,
                    :weight
                )
                """,
                snapshot.node_chunks,
            )
            connection.executemany(
                """
                INSERT INTO graph_edges (
                    edge_id,
                    source_node_id,
                    target_node_id,
                    edge_type,
                    confidence,
                    extractor,
                    relative_path,
                    payload_json
                ) VALUES (
                    :edge_id,
                    :source_node_id,
                    :target_node_id,
                    :edge_type,
                    :confidence,
                    :extractor,
                    :relative_path,
                    :payload_json
                )
                """,
                snapshot.edges,
            )

    def graph_file_state(self) -> dict[str, dict[str, object]]:
        """Return graph_files state keyed by normalized relative path."""

        connection = self._read_connect()
        if connection is None:
            return {}
        with connection:
            columns = _table_columns(connection, "graph_files")
            signature_expr = (
                "graph_input_signature"
                if "graph_input_signature" in columns
                else "NULL AS graph_input_signature"
            )
            rows = connection.execute(
                f"""
                SELECT
                    relative_path,
                    content_hash,
                    {signature_expr},
                    graph_state,
                    error_code
                FROM graph_files
                """
            ).fetchall()
        return {
            _normalize_status_path(str(row["relative_path"])): {
                "content_hash": row["content_hash"],
                "graph_input_signature": row["graph_input_signature"],
                "graph_state": row["graph_state"],
                "error_code": row["error_code"],
            }
            for row in rows
        }

    def backfill_graph_input_signatures(self, signatures_by_path: dict[str, str]) -> int:
        """Fill missing per-file graph input signatures for legacy graph rows."""

        rows = [
            (signature, _normalize_status_path(relative_path))
            for relative_path, signature in signatures_by_path.items()
            if relative_path and signature
        ]
        if not rows:
            return 0
        with self._connect() as connection:
            ensure_graph_schema(connection)
            cursor = connection.executemany(
                """
                UPDATE graph_files
                SET graph_input_signature = ?
                WHERE relative_path = ?
                  AND (graph_input_signature IS NULL OR graph_input_signature = '')
                """,
                rows,
            )
            return cursor.rowcount

    def safe_legacy_signature_backfill_paths(
        self,
        *,
        snapshot: GraphSnapshot,
        signatures_by_path: dict[str, str],
    ) -> set[str]:
        """Return legacy paths whose stored graph rows match the current snapshot."""

        candidate_paths = _normalize_path_set(signatures_by_path)
        if not candidate_paths:
            return set()
        expected = {
            path: _snapshot_path_fingerprint(snapshot, path)
            for path in candidate_paths
        }
        safe_paths: set[str] = set()
        with self._read_connect() as connection:
            if connection is None:
                return safe_paths
            for path in candidate_paths:
                if expected.get(path) == _stored_path_fingerprint(connection, path):
                    safe_paths.add(path)
        return safe_paths

    def reverse_dependent_paths(
        self,
        affected_paths: list[str],
        *,
        limit: int,
    ) -> tuple[list[str], bool]:
        """Return existing graph origin paths that reference affected path-owned nodes."""

        normalized_paths = _normalize_path_set(affected_paths)
        if not normalized_paths:
            return [], True
        connection = self._read_connect()
        if connection is None:
            return [], True
        placeholders = ",".join("?" for _ in normalized_paths)
        node_type_placeholders = ",".join("?" for _ in PATH_OWNED_NODE_TYPES)
        with connection:
            affected_chunk_ids = _select_chunk_ids_for_paths(connection, normalized_paths)
            shared_candidate_node_ids = _select_shared_candidate_node_ids(
                connection,
                affected_chunk_ids=affected_chunk_ids,
                affected_paths=normalized_paths,
            )
            rows = connection.execute(
                f"""
                SELECT DISTINCT e.relative_path AS relative_path
                FROM graph_edges e
                JOIN graph_nodes n
                  ON n.node_id = e.source_node_id OR n.node_id = e.target_node_id
                WHERE (
                    (n.relative_path IN ({placeholders})
                     AND n.node_type IN ({node_type_placeholders}))
                    {_shared_node_clause(shared_candidate_node_ids)}
                  )
                  AND e.relative_path IS NOT NULL
                  AND e.relative_path NOT IN ({placeholders})
                ORDER BY e.relative_path ASC
                LIMIT ?
                """,
                (
                    *normalized_paths,
                    *sorted(PATH_OWNED_NODE_TYPES),
                    *sorted(shared_candidate_node_ids),
                    *normalized_paths,
                    max(0, limit) + 1,
                ),
            ).fetchall()
        paths = [
            _normalize_status_path(str(row["relative_path"]))
            for row in rows
        ]
        complete = len(paths) <= max(0, limit)
        return paths[: max(0, limit)], complete

    def update_paths(
        self,
        *,
        snapshot: GraphSnapshot,
        affected_paths: list[str],
        max_rows_deleted: int | None = None,
        max_rows_upserted: int | None = None,
        max_duration_ms: int | None = None,
        started_at: float | None = None,
    ) -> GraphUpdateStoreResult:
        """Delete and upsert graph rows for affected paths in one transaction."""

        started_at = perf_counter() if started_at is None else started_at
        normalized_paths = _normalize_path_set(affected_paths)
        result = GraphUpdateStoreResult()
        if not normalized_paths:
            with self._connect() as connection:
                ensure_graph_schema(connection)
                _replace_metadata(connection, snapshot.metadata)
            return result

        snapshot_files = {
            _normalize_status_path(str(row["relative_path"])): row
            for row in snapshot.files
        }
        snapshot_nodes = {str(row["node_id"]): row for row in snapshot.nodes}
        snapshot_edges = [
            row
            for row in snapshot.edges
            if _normalize_status_path(str(row.get("relative_path") or "")) in normalized_paths
        ]
        node_ids_needed: set[str] = set()
        for row in snapshot_edges:
            node_ids_needed.add(str(row["source_node_id"]))
            node_ids_needed.add(str(row["target_node_id"]))
        for row in snapshot.nodes:
            relative_path = row.get("relative_path")
            if relative_path and _normalize_status_path(str(relative_path)) in normalized_paths:
                node_ids_needed.add(str(row["node_id"]))

        with self._connect() as connection:
            ensure_graph_schema(connection)
            path_owned_node_ids = _select_path_owned_node_ids(connection, normalized_paths)
            affected_chunk_ids = _select_chunk_ids_for_paths(connection, normalized_paths)
            shared_candidate_node_ids = _select_shared_candidate_node_ids(
                connection,
                affected_chunk_ids=affected_chunk_ids,
                affected_paths=normalized_paths,
            )
            for node_id in shared_candidate_node_ids:
                if node_id in snapshot_nodes:
                    node_ids_needed.add(node_id)
            for row in snapshot.node_chunks:
                if str(row["chunk_id"]) in affected_chunk_ids:
                    node_ids_needed.add(str(row["node_id"]))
            for node_id in list(node_ids_needed):
                row = snapshot_nodes.get(node_id)
                if row is None:
                    continue
                if str(row.get("node_type")) in SHARED_NODE_TYPES:
                    for link in snapshot.node_chunks:
                        if str(link["node_id"]) == node_id:
                            node_ids_needed.add(str(link["node_id"]))

            node_rows = [snapshot_nodes[node_id] for node_id in sorted(node_ids_needed) if node_id in snapshot_nodes]
            node_terms = [
                row
                for row in snapshot.node_terms
                if str(row["node_id"]) in node_ids_needed
            ]
            node_chunks = [
                row
                for row in snapshot.node_chunks
                if str(row["node_id"]) in node_ids_needed
            ]
            file_rows = [
                snapshot_files[path]
                for path in sorted(normalized_paths)
                if path in snapshot_files
            ]
            planned_rows_upserted = (
                len(file_rows)
                + len(node_rows)
                + len(node_terms)
                + len(node_chunks)
                + len(snapshot_edges)
            )
            _raise_update_bounds_if_exceeded(
                bounds_exceeded=_graph_update_bounds(
                    rows_upserted=planned_rows_upserted,
                    max_rows_upserted=max_rows_upserted,
                    started_at=started_at,
                    max_duration_ms=max_duration_ms,
                ),
                summary={
                    "rows_upserted": planned_rows_upserted,
                    "max_rows_upserted": max_rows_upserted,
                    "max_duration_ms": max_duration_ms,
                },
            )

            path_placeholders = ",".join("?" for _ in normalized_paths)
            result.edges_deleted += connection.execute(
                f"DELETE FROM graph_edges WHERE relative_path IN ({path_placeholders})",
                tuple(normalized_paths),
            ).rowcount
            result.edges_deleted += _delete_edges_for_node_ids(connection, path_owned_node_ids)
            rows_deleted = result.edges_deleted
            rows_deleted += _delete_node_terms(connection, path_owned_node_ids)
            rows_deleted += _delete_node_chunks(connection, path_owned_node_ids)
            if affected_chunk_ids:
                rows_deleted += _delete_node_chunks_for_chunk_ids(connection, affected_chunk_ids)

            result.nodes_deleted += _delete_nodes(connection, path_owned_node_ids)
            rows_deleted += result.nodes_deleted
            result.files_deleted += connection.execute(
                f"DELETE FROM graph_files WHERE relative_path IN ({path_placeholders})",
                tuple(normalized_paths),
            ).rowcount
            rows_deleted += result.files_deleted
            _raise_update_bounds_if_exceeded(
                bounds_exceeded=_graph_update_bounds(
                    rows_deleted=rows_deleted,
                    max_rows_deleted=max_rows_deleted,
                    started_at=started_at,
                    max_duration_ms=max_duration_ms,
                ),
                summary={
                    "rows_deleted": rows_deleted,
                    "max_rows_deleted": max_rows_deleted,
                    "max_duration_ms": max_duration_ms,
                },
            )

            _upsert_files(connection, file_rows)
            _upsert_nodes(connection, node_rows)
            rows_deleted += _replace_node_terms(connection, node_ids_needed, node_terms)
            rows_deleted += _replace_node_chunks(connection, node_ids_needed, node_chunks)
            _upsert_edges(connection, snapshot_edges)

            pruned = _prune_orphan_shared_nodes(connection)
            result.shared_nodes_pruned = pruned
            _raise_update_bounds_if_exceeded(
                bounds_exceeded=_graph_update_bounds(
                    rows_deleted=rows_deleted + pruned,
                    max_rows_deleted=max_rows_deleted,
                    started_at=started_at,
                    max_duration_ms=max_duration_ms,
                ),
                summary={
                    "rows_deleted": rows_deleted + pruned,
                    "rows_upserted": planned_rows_upserted,
                    "max_rows_deleted": max_rows_deleted,
                    "max_rows_upserted": max_rows_upserted,
                    "max_duration_ms": max_duration_ms,
                },
            )
            _replace_metadata(connection, snapshot.metadata)

        result.nodes_upserted = len(node_rows)
        result.edges_upserted = len(snapshot_edges)
        result.files_upserted = len(file_rows)
        result.deleted_paths = [path for path in normalized_paths if path not in snapshot_files]
        return result

    def _missing_status(self, *, repo_root: str, profile: str) -> GraphStatusResult:
        return GraphStatusResult(
            repo_root=repo_root,
            profile=profile,
            graph_path=str(self._db_path),
            available=False,
            state="missing",
            expansion_allowed=False,
            warning_codes=["graph_missing"],
        )

    def status(
        self,
        *,
        repo_root: str,
        profile: str,
        current_head_commit: str | None,
        expected_source_index_contract: dict[str, object] | None = None,
        expected_extractor_versions_hash: str | None = None,
        invalidated_path_count: int = 0,
        invalidated_paths_preview: list[str] | None = None,
    ) -> GraphStatusResult:
        """Return detailed graph status without mutating the artifact."""

        if not self._db_path.exists():
            return self._missing_status(repo_root=repo_root, profile=profile)

        invalidated_preview = invalidated_paths_preview or []
        try:
            with connect_row_factory(self._db_path) as connection:
                user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
                if user_version != GRAPH_SCHEMA_VERSION:
                    return GraphStatusResult(
                        repo_root=repo_root,
                        profile=profile,
                        graph_path=str(self._db_path),
                        available=False,
                        state="incompatible",
                        expansion_allowed=False,
                        schema_version=user_version,
                        warning_codes=["graph_schema_incompatible"],
                    )

                metadata = connection.execute("SELECT * FROM graph_metadata LIMIT 1").fetchone()
                if metadata is None:
                    return GraphStatusResult(
                        repo_root=repo_root,
                        profile=profile,
                        graph_path=str(self._db_path),
                        available=False,
                        state="error",
                        expansion_allowed=False,
                        schema_version=user_version,
                        warning_codes=["graph_metadata_missing"],
                        error_code="graph_metadata_missing",
                    )

                source_index_contract = json.loads(metadata["source_index_contract_json"])
                extractor_versions_hash = stable_json_hash(json.loads(metadata["extractor_versions_json"]))
                counts = self._read_counts(connection)
        except Exception as exc:  # noqa: BLE001
            return GraphStatusResult(
                repo_root=repo_root,
                profile=profile,
                graph_path=str(self._db_path),
                available=False,
                state="error",
                expansion_allowed=False,
                error_code="graph_status_read_failed",
                error_detail=str(exc),
                warning_codes=["graph_status_read_failed"],
            )

        indexed_files_total = int(source_index_contract.get("indexed_files_total") or 0)
        coverage = (
            min(1.0, counts.files / indexed_files_total)
            if indexed_files_total > 0
            else None
        )
        extractor_warning_codes = [
            str(code)
            for code in source_index_contract.get("extractor_warning_codes") or []
            if str(code)
        ]
        warning_codes: list[str] = [
            f"graph_extractor_{code}"
            for code in extractor_warning_codes
        ]
        state = "ready"
        mismatch_codes = _source_contract_mismatch_codes(
            source_index_contract,
            expected_source_index_contract,
        )
        if (
            expected_extractor_versions_hash
            and extractor_versions_hash != expected_extractor_versions_hash
        ):
            state = "incompatible"
            warning_codes.append("graph_extractor_contract_mismatch")
        elif mismatch_codes:
            state = "stale"
            warning_codes.append("graph_source_index_contract_mismatch")
            warning_codes.extend(mismatch_codes)
        elif counts.file_state_counts.get("error", 0) > 0:
            state = "error"
            warning_codes.append("graph_file_errors")
        elif counts.file_state_counts.get("partial", 0) > 0:
            state = "partial"
            warning_codes.append("graph_partial_files")

        built_commit = metadata["built_commit"]
        if (
            state == "ready"
            and built_commit
            and current_head_commit
            and str(built_commit) != str(current_head_commit)
        ):
            state = "stale"
            warning_codes.append("graph_built_commit_mismatch")
        if state in {"ready", "stale"} and invalidated_path_count > 0:
            state = "stale"
            warning_codes.append("graph_worktree_changes")

        invalidated_paths_known = (
            invalidated_path_count == 0
            or invalidated_path_count <= len({_normalize_status_path(path) for path in invalidated_preview if path})
        )
        expansion_allowed = _expansion_allowed_for_status(
            state=state,
            warning_codes=warning_codes,
            invalidated_path_count=invalidated_path_count,
            invalidated_paths_known=invalidated_paths_known,
        )
        update_fields = _status_update_fields(
            state=state,
            warning_codes=warning_codes,
            invalidated_path_count=invalidated_path_count,
            invalidated_paths_preview=invalidated_preview,
        )
        return GraphStatusResult(
            repo_root=repo_root,
            profile=profile,
            graph_path=str(self._db_path),
            available=state in {"ready", "partial", "stale"},
            state=state,  # type: ignore[arg-type]
            expansion_allowed=expansion_allowed,
            schema_version=int(metadata["schema_version"]),
            built_commit=metadata["built_commit"],
            built_at=metadata["built_at"],
            graph_contract_hash=metadata["graph_contract_hash"],
            extractor_versions_hash=extractor_versions_hash,
            source_index_contract=source_index_contract,
            file_coverage_ratio=coverage,
            stale_path_count=invalidated_path_count,
            invalidated_paths_preview=invalidated_preview,
            **update_fields,
            counts=counts,
            warning_codes=warning_codes,
        )

    def _read_counts(self, connection: sqlite3.Connection) -> GraphCounts:
        return GraphCounts(
            files=_table_count(connection, "graph_files"),
            nodes=_table_count(connection, "graph_nodes"),
            edges=_table_count(connection, "graph_edges"),
            node_terms=_table_count(connection, "graph_node_terms"),
            node_chunks=_table_count(connection, "graph_node_chunks"),
            node_type_counts=_group_counts(connection, "graph_nodes", "node_type"),
            edge_type_counts=_group_counts(connection, "graph_edges", "edge_type"),
            file_state_counts=_group_counts(connection, "graph_files", "graph_state"),
        )

    def find_nodes_for_chunks(self, chunk_ids: list[str], *, limit: int = 100) -> list[GraphNode]:
        """Return chunk-backed graph nodes preserving seed chunk order."""

        if not chunk_ids or limit <= 0:
            return []
        seen: set[str] = set()
        ordered_chunk_ids = []
        for chunk_id in chunk_ids:
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            ordered_chunk_ids.append(chunk_id)
        result: list[GraphNode] = []
        connection = self._read_connect()
        if connection is None:
            return []
        with connection:
            for chunk_id in ordered_chunk_ids:
                rows = connection.execute(
                    """
                    SELECT *
                    FROM graph_nodes
                    WHERE chunk_id = ?
                    ORDER BY
                        CASE node_type
                            WHEN 'chunk' THEN 0
                            WHEN 'symbol' THEN 1
                            WHEN 'route' THEN 2
                            WHEN 'test_case' THEN 3
                            WHEN 'doc_section' THEN 4
                            WHEN 'config_item' THEN 5
                            WHEN 'module' THEN 6
                            ELSE 99
                        END,
                        relative_path ASC,
                        start_line ASC,
                        node_id ASC
                    LIMIT ?
                    """,
                    (chunk_id, max(1, limit - len(result))),
                ).fetchall()
                result.extend(_node_from_row(row) for row in rows)
                if len(result) >= limit:
                    break
        return result[:limit]

    def find_file_nodes(self, relative_paths: list[str], *, limit: int = 100) -> list[GraphNode]:
        return self._find_nodes_by_pairs(
            "file",
            [(path.replace("\\", "/").strip("/"), None) for path in relative_paths],
            limit=limit,
        )

    def find_symbol_nodes(self, bindings: list[tuple[str, str]], *, limit: int = 100) -> list[GraphNode]:
        return self._find_nodes_by_pairs("symbol", bindings, limit=limit)

    def find_doc_section_nodes(self, bindings: list[tuple[str, str]], *, limit: int = 100) -> list[GraphNode]:
        return self._find_nodes_by_pairs("doc_section", bindings, limit=limit)

    def find_term_nodes(
        self,
        terms: list[tuple[str, str | None]],
        *,
        limit: int = 100,
    ) -> list[GraphNode]:
        """Return graph nodes bound through normalized graph_node_terms."""

        if not terms or limit <= 0:
            return []
        result: list[GraphNode] = []
        connection = self._read_connect()
        if connection is None:
            return []
        with connection:
            for normalized_term, term_type in terms:
                normalized = normalized_term.strip().lower()
                if not normalized:
                    continue
                if term_type:
                    rows = connection.execute(
                        """
                        SELECT n.*
                        FROM graph_node_terms t
                        JOIN graph_nodes n ON n.node_id = t.node_id
                        WHERE t.normalized_term = ? AND t.term_type = ?
                        ORDER BY
                            CASE t.term_type
                                WHEN 'env_var' THEN 0
                                WHEN 'route' THEN 1
                                WHEN 'path' THEN 2
                                WHEN 'cli_flag' THEN 3
                                WHEN 'sql_identifier' THEN 4
                                WHEN 'quoted_literal' THEN 5
                                ELSE 9
                            END,
                            CASE n.node_type
                                WHEN 'env_var' THEN 0
                                WHEN 'route' THEN 1
                                WHEN 'config_item' THEN 2
                                WHEN 'symbol' THEN 3
                                WHEN 'test_case' THEN 4
                                WHEN 'module' THEN 5
                                WHEN 'chunk' THEN 6
                                ELSE 9
                            END,
                            n.key ASC,
                            n.node_id ASC
                        LIMIT ?
                        """,
                        (normalized, term_type, max(1, limit - len(result))),
                    ).fetchall()
                else:
                    rows = connection.execute(
                        """
                        SELECT n.*
                        FROM graph_node_terms t
                        JOIN graph_nodes n ON n.node_id = t.node_id
                        WHERE t.normalized_term = ?
                        ORDER BY
                            CASE t.term_type
                                WHEN 'env_var' THEN 0
                                WHEN 'route' THEN 1
                                WHEN 'path' THEN 2
                                WHEN 'cli_flag' THEN 3
                                WHEN 'sql_identifier' THEN 4
                                WHEN 'quoted_literal' THEN 5
                                ELSE 9
                            END,
                            CASE n.node_type
                                WHEN 'env_var' THEN 0
                                WHEN 'route' THEN 1
                                WHEN 'config_item' THEN 2
                                WHEN 'symbol' THEN 3
                                WHEN 'test_case' THEN 4
                                WHEN 'module' THEN 5
                                WHEN 'chunk' THEN 6
                                ELSE 9
                            END,
                            n.key ASC,
                            n.node_id ASC
                        LIMIT ?
                        """,
                        (normalized, max(1, limit - len(result))),
                    ).fetchall()
                result.extend(_node_from_row(row) for row in rows)
                if len(result) >= limit:
                    break
        return _dedupe_nodes(result)[:limit]

    def neighbors(
        self,
        node_ids: list[str],
        *,
        edge_types: list[str],
        max_neighbors_per_node: int = 30,
        max_total: int = 200,
    ) -> list[GraphEdge]:
        """Return bounded incoming/outgoing edges with deterministic ordering."""

        if not node_ids or not edge_types or max_total <= 0 or max_neighbors_per_node <= 0:
            return []
        result: list[GraphEdge] = []
        placeholders = ",".join("?" for _ in edge_types)
        connection = self._read_connect()
        if connection is None:
            return []
        with connection:
            for node_id in dict.fromkeys(node_ids):
                rows = connection.execute(
                    f"""
                    SELECT
                        e.*,
                        other.node_id AS neighbor_node_id,
                        other.node_type AS neighbor_node_type,
                        other.key AS neighbor_key,
                        other.relative_path AS neighbor_relative_path,
                        other.chunk_id AS neighbor_chunk_id,
                        other.start_line AS neighbor_start_line,
                        other.end_line AS neighbor_end_line
                    FROM graph_edges e
                    JOIN graph_nodes other
                        ON other.node_id = CASE
                            WHEN e.source_node_id = ? THEN e.target_node_id
                            ELSE e.source_node_id
                        END
                    WHERE (e.source_node_id = ? OR e.target_node_id = ?)
                      AND e.edge_type IN ({placeholders})
                    ORDER BY
                        CASE e.edge_type
                            WHEN 'code_reads_env' THEN 0
                            WHEN 'config_defines_env' THEN 1
                            WHEN 'route_handled_by_symbol' THEN 2
                            WHEN 'route_defined_in_chunk' THEN 3
                            WHEN 'test_targets_symbol' THEN 4
                            WHEN 'test_targets_file' THEN 5
                            WHEN 'doc_references_route' THEN 6
                            WHEN 'doc_references_symbol' THEN 7
                            WHEN 'doc_references_path' THEN 8
                            WHEN 'doc_references_env' THEN 9
                            WHEN 'doc_references_config' THEN 10
                            WHEN 'chunk_defines_symbol' THEN 11
                            WHEN 'file_imports_file' THEN 12
                            WHEN 'module_imports_module' THEN 13
                            WHEN 'file_defines_module' THEN 14
                            WHEN 'file_contains_doc_section' THEN 15
                            WHEN 'chunk_mentions_path_or_symbol' THEN 16
                            WHEN 'policy_applies_to_path' THEN 17
                            WHEN 'file_contains_chunk' THEN 18
                            ELSE 99
                        END,
                        e.confidence DESC,
                        CASE other.node_type
                            WHEN 'env_var' THEN 0
                            WHEN 'route' THEN 1
                            WHEN 'config_item' THEN 2
                            WHEN 'symbol' THEN 3
                            WHEN 'test_case' THEN 4
                            WHEN 'module' THEN 5
                            WHEN 'chunk' THEN 6
                            WHEN 'file' THEN 7
                            ELSE 9
                        END,
                        other.relative_path ASC,
                        other.start_line ASC,
                        other.node_id ASC
                    LIMIT ?
                    """,
                    (node_id, node_id, node_id, *edge_types, max_neighbors_per_node),
                ).fetchall()
                result.extend(_edge_from_row(row) for row in rows)
                if len(result) >= max_total:
                    break
        return result[:max_total]

    def chunks_for_nodes(self, node_ids: list[str], *, limit: int = 300) -> list[GraphChunkLink]:
        """Resolve graph nodes to chunk ids through direct chunk fields and node_chunks."""

        if not node_ids or limit <= 0:
            return []
        result: list[GraphChunkLink] = []
        connection = self._read_connect()
        if connection is None:
            return []
        with connection:
            for node_id in dict.fromkeys(node_ids):
                rows = connection.execute(
                    """
                    SELECT
                        src.node_id AS node_id,
                        COALESCE(nc.chunk_id, src.chunk_id) AS chunk_id,
                        COALESCE(nc.relation, 'primary') AS relation,
                        COALESCE(nc.weight, 1.0) AS weight,
                        chunk_node.payload_json AS payload_json,
                        chunk_node.relative_path AS relative_path,
                        chunk_node.start_line AS start_line,
                        chunk_node.end_line AS end_line
                    FROM graph_nodes src
                    LEFT JOIN graph_node_chunks nc ON nc.node_id = src.node_id
                    JOIN graph_nodes chunk_node
                        ON chunk_node.chunk_id = COALESCE(nc.chunk_id, src.chunk_id)
                       AND chunk_node.node_type = 'chunk'
                    WHERE src.node_id = ?
                      AND COALESCE(nc.chunk_id, src.chunk_id) IS NOT NULL
                    ORDER BY
                        CASE COALESCE(nc.relation, 'primary')
                            WHEN 'primary' THEN 0
                            WHEN 'handler' THEN 1
                            WHEN 'test' THEN 2
                            WHEN 'containing' THEN 3
                            WHEN 'mention' THEN 4
                            WHEN 'nearest' THEN 5
                            ELSE 9
                        END,
                        COALESCE(nc.weight, 1.0) DESC,
                        chunk_node.relative_path ASC,
                        chunk_node.start_line ASC,
                        chunk_node.chunk_id ASC
                    LIMIT ?
                    """,
                    (node_id, max(1, limit - len(result))),
                ).fetchall()
                result.extend(_chunk_link_from_row(row) for row in rows)
                if len(result) >= limit:
                    break
        return _dedupe_chunk_links(result)[:limit]

    def _find_nodes_by_pairs(
        self,
        node_type: str,
        bindings: list[tuple[str, str | None]],
        *,
        limit: int,
    ) -> list[GraphNode]:
        if not bindings or limit <= 0:
            return []
        result: list[GraphNode] = []
        connection = self._read_connect()
        if connection is None:
            return []
        with connection:
            for relative_path, key in bindings:
                if key is None:
                    rows = connection.execute(
                        """
                        SELECT *
                        FROM graph_nodes
                        WHERE node_type = ? AND key = ?
                        ORDER BY relative_path ASC, start_line ASC, node_id ASC
                        LIMIT ?
                        """,
                        (node_type, relative_path, max(1, limit - len(result))),
                    ).fetchall()
                else:
                    rows = connection.execute(
                        """
                        SELECT *
                        FROM graph_nodes
                        WHERE node_type = ? AND relative_path = ? AND key = ?
                        ORDER BY start_line ASC, node_id ASC
                        LIMIT ?
                        """,
                        (node_type, relative_path.replace("\\", "/").strip("/"), key, max(1, limit - len(result))),
                    ).fetchall()
                result.extend(_node_from_row(row) for row in rows)
                if len(result) >= limit:
                    break
        return _dedupe_nodes(result)[:limit]


def graph_summary_from_status(status: GraphStatusResult) -> GraphStatusSummary:
    """Project detailed graph status into the compact index_status.v2 shape."""

    return GraphStatusSummary(
        available=status.available,
        state=status.state,
        expansion_allowed=status.expansion_allowed,
        schema_version=status.schema_version,
        built_commit=status.built_commit,
        built_at=status.built_at,
        extractor_versions_hash=status.extractor_versions_hash,
        file_coverage_ratio=status.file_coverage_ratio,
        stale_path_count=status.stale_path_count,
        invalidated_paths_preview=list(status.invalidated_paths_preview),
        update_available=status.update_available,
        update_required=status.update_required,
        update_safe_auto_run=status.update_safe_auto_run,
        update_blocked_reason=status.update_blocked_reason,
        update_path_count=status.update_path_count,
        update_paths_preview=list(status.update_paths_preview),
        rebuild_required=status.rebuild_required,
        warning_codes=list(status.warning_codes),
    )


def _source_contract_mismatch_codes(
    stored_contract: dict[str, object],
    expected_contract: dict[str, object] | None,
) -> list[str]:
    if expected_contract is None:
        return []
    mismatch_codes: list[str] = []
    if expected_contract.get("schema_version") != stored_contract.get("schema_version"):
        mismatch_codes.append("graph_source_index_hard_contract_mismatch")
    elif expected_contract.get("profile") != stored_contract.get("profile"):
        mismatch_codes.append("graph_source_index_hard_contract_mismatch")
    elif _collection_hard_contract(expected_contract) != _collection_hard_contract(stored_contract):
        mismatch_codes.append("graph_source_index_hard_contract_mismatch")
    soft_keys = [
        "index_revision",
    ]
    for key in soft_keys:
        if key in expected_contract and stored_contract.get(key) != expected_contract.get(key):
            mismatch_codes.append("graph_source_index_revision_mismatch")
            break
    count_keys = [
        "indexed_files_total",
        "indexed_chunks_total",
    ]
    for key in count_keys:
        if key in expected_contract and stored_contract.get(key) != expected_contract.get(key):
            mismatch_codes.append("graph_source_index_counts_mismatch")
            break
    expected_source_hash = expected_contract.get("source_revision_hash")
    if expected_source_hash and stored_contract.get("source_revision_hash") != expected_source_hash:
        mismatch_codes.append("graph_source_revision_hash_mismatch")
    return mismatch_codes


def _collection_hard_contract(contract: dict[str, object]) -> dict[str, dict[str, object]]:
    collections = contract.get("collections")
    if not isinstance(collections, dict):
        return {}
    result: dict[str, dict[str, object]] = {}
    for scope, raw in collections.items():
        if not isinstance(raw, dict):
            continue
        result[str(scope)] = {
            "collection_name": raw.get("collection_name"),
            "readable": bool(raw.get("readable")),
        }
    return result


def _expansion_allowed_for_status(
    *,
    state: str,
    warning_codes: list[str],
    invalidated_path_count: int,
    invalidated_paths_known: bool,
) -> bool:
    """Allow degraded graph use only when stale state is non-blocking.

    A watcher-driven incremental index update changes source-index counters and
    revision timestamps before the graph artifact is rebuilt. That makes the
    graph technically stale, but it is still useful for unchanged seeds. Keep
    expansion blocked for commit drift and unindexed worktree changes, where the
    graph could silently point at the wrong branch or edited files.
    """

    if state == "ready":
        return True
    if state != "stale":
        return False
    if invalidated_path_count > 0 and not invalidated_paths_known:
        return False
    blocking_codes = {
        "graph_built_commit_mismatch",
        "graph_source_index_hard_contract_mismatch",
        "graph_source_revision_hash_mismatch",
    }
    return not any(code in blocking_codes for code in warning_codes)


def _status_update_fields(
    *,
    state: str,
    warning_codes: list[str],
    invalidated_path_count: int,
    invalidated_paths_preview: list[str],
) -> dict[str, object]:
    hard_mismatch = "graph_source_index_hard_contract_mismatch" in warning_codes
    extractor_mismatch = "graph_extractor_contract_mismatch" in warning_codes
    missing_or_incompatible = state in {"missing", "incompatible"}
    artifact_error = state == "error" and "graph_file_errors" not in warning_codes
    source_drift = any(
        code in warning_codes
        for code in (
            "graph_source_index_contract_mismatch",
            "graph_source_index_revision_mismatch",
            "graph_source_index_counts_mismatch",
            "graph_source_revision_hash_mismatch",
            "graph_worktree_changes",
        )
    )
    update_required = (
        source_drift
        or "graph_file_errors" in warning_codes
        or invalidated_path_count > 0
    ) and not hard_mismatch and not extractor_mismatch and not missing_or_incompatible and not artifact_error
    update_blocked_reason = None
    rebuild_required = False
    if hard_mismatch:
        update_blocked_reason = "hard_contract_mismatch"
        rebuild_required = True
    elif extractor_mismatch:
        update_blocked_reason = "extractor_contract_mismatch"
        rebuild_required = True
    elif missing_or_incompatible:
        update_blocked_reason = f"graph_{state}"
        rebuild_required = state == "incompatible"
    elif artifact_error:
        update_blocked_reason = "graph_artifact_error"
        rebuild_required = True
    elif "graph_invalidated_paths_unknown" in warning_codes:
        update_blocked_reason = "graph_invalidated_paths_unknown"
    return {
        "update_available": state in {"ready", "stale", "partial", "error"} and not rebuild_required,
        "update_required": update_required,
        "update_safe_auto_run": update_required and update_blocked_reason is None,
        "update_blocked_reason": update_blocked_reason,
        "update_path_count": invalidated_path_count if invalidated_path_count else None,
        "update_paths_preview": list(invalidated_paths_preview),
        "rebuild_required": rebuild_required,
    }


def _normalize_status_path(path: str) -> str:
    return str(path).replace("\\", "/").strip("/")


def _node_from_row(row: sqlite3.Row) -> GraphNode:
    return GraphNode(
        node_id=str(row["node_id"]),
        node_type=row["node_type"],
        key=str(row["key"]),
        relative_path=row["relative_path"],
        chunk_id=row["chunk_id"],
        start_line=int(row["start_line"]) if row["start_line"] is not None else None,
        end_line=int(row["end_line"]) if row["end_line"] is not None else None,
    )


def _edge_from_row(row: sqlite3.Row) -> GraphEdge:
    return GraphEdge(
        edge_id=str(row["edge_id"]),
        source_node_id=str(row["source_node_id"]),
        target_node_id=str(row["target_node_id"]),
        edge_type=row["edge_type"],
        confidence=float(row["confidence"]),
        extractor=str(row["extractor"]),
        relative_path=row["relative_path"],
        payload_json=str(row["payload_json"] or "{}"),
        neighbor_node_id=str(row["neighbor_node_id"]),
        neighbor_node_type=row["neighbor_node_type"],
        neighbor_key=str(row["neighbor_key"]),
        neighbor_relative_path=row["neighbor_relative_path"],
        neighbor_chunk_id=row["neighbor_chunk_id"],
        neighbor_start_line=int(row["neighbor_start_line"]) if row["neighbor_start_line"] is not None else None,
        neighbor_end_line=int(row["neighbor_end_line"]) if row["neighbor_end_line"] is not None else None,
    )


def _chunk_link_from_row(row: sqlite3.Row) -> GraphChunkLink:
    payload = json.loads(row["payload_json"] or "{}")
    return GraphChunkLink(
        node_id=str(row["node_id"]),
        chunk_id=str(row["chunk_id"]),
        relation=str(row["relation"]),
        weight=float(row["weight"]),
        scope=str(payload.get("scope") or "code"),
        relative_path=str(row["relative_path"]),
        start_line=int(row["start_line"]),
        end_line=int(row["end_line"]),
    )


def _dedupe_nodes(nodes: list[GraphNode]) -> list[GraphNode]:
    seen: set[str] = set()
    result: list[GraphNode] = []
    for node in nodes:
        if node.node_id in seen:
            continue
        seen.add(node.node_id)
        result.append(node)
    return result


def _dedupe_chunk_links(links: list[GraphChunkLink]) -> list[GraphChunkLink]:
    seen: set[tuple[str, str]] = set()
    result: list[GraphChunkLink] = []
    for link in links:
        key = (link.node_id, link.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        result.append(link)
    return result


def _table_count(connection: sqlite3.Connection, table_name: str) -> int:
    row = connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
    return int(row[0]) if row else 0


def _group_counts(connection: sqlite3.Connection, table_name: str, column_name: str) -> dict[str, int]:
    rows = connection.execute(
        f"SELECT {column_name}, COUNT(*) AS count FROM {table_name} GROUP BY {column_name}"
    ).fetchall()
    return {str(row[0]): int(row[1]) for row in rows}


def _table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()}


def _normalize_path_set(paths: list[str]) -> list[str]:
    return sorted(
        {
            normalized
            for path in paths
            if (normalized := _normalize_status_path(str(path)))
        }
    )


def _canonical_graph_value(value: object) -> object:
    if isinstance(value, bool):
        return 1 if value else 0
    return value


def _canonical_rows(rows: list[dict[str, object]], fields: list[str]) -> list[dict[str, object]]:
    return sorted(
        [
            {
                field: _canonical_graph_value(row.get(field))
                for field in fields
            }
            for row in rows
        ],
        key=lambda row: json.dumps(row, sort_keys=True, ensure_ascii=False),
    )


def _snapshot_path_fingerprint(snapshot: GraphSnapshot, relative_path: str) -> str:
    normalized_path = _normalize_status_path(relative_path)
    file_rows = [
        row
        for row in snapshot.files
        if _normalize_status_path(str(row.get("relative_path") or "")) == normalized_path
    ]
    path_nodes = [
        row
        for row in snapshot.nodes
        if _normalize_status_path(str(row.get("relative_path") or "")) == normalized_path
    ]
    node_ids = {str(row["node_id"]) for row in path_nodes}
    return stable_json_hash(
        {
            "files": _canonical_rows(
                file_rows,
                ["relative_path", "content_hash", "graph_state", "error_code"],
            ),
            "nodes": _canonical_rows(
                path_nodes,
                [
                    "node_id",
                    "node_type",
                    "key",
                    "relative_path",
                    "chunk_id",
                    "start_line",
                    "end_line",
                    "payload_json",
                ],
            ),
            "node_terms": _canonical_rows(
                [row for row in snapshot.node_terms if str(row.get("node_id")) in node_ids],
                ["node_id", "term", "normalized_term", "term_type", "case_sensitive", "source"],
            ),
            "node_chunks": _canonical_rows(
                [row for row in snapshot.node_chunks if str(row.get("node_id")) in node_ids],
                ["node_id", "chunk_id", "relation", "weight"],
            ),
            "edges": _canonical_rows(
                [
                    row
                    for row in snapshot.edges
                    if _normalize_status_path(str(row.get("relative_path") or "")) == normalized_path
                ],
                [
                    "edge_id",
                    "source_node_id",
                    "target_node_id",
                    "edge_type",
                    "confidence",
                    "extractor",
                    "relative_path",
                    "payload_json",
                ],
            ),
        }
    )


def _stored_path_fingerprint(connection: sqlite3.Connection, relative_path: str) -> str:
    normalized_path = _normalize_status_path(relative_path)
    file_rows = [
        dict(row)
        for row in connection.execute(
            """
            SELECT relative_path, content_hash, graph_state, error_code
            FROM graph_files
            WHERE relative_path = ?
            """,
            (normalized_path,),
        ).fetchall()
    ]
    node_rows = [
        dict(row)
        for row in connection.execute(
            """
            SELECT node_id, node_type, key, relative_path, chunk_id,
                   start_line, end_line, payload_json
            FROM graph_nodes
            WHERE relative_path = ?
            """,
            (normalized_path,),
        ).fetchall()
    ]
    node_ids = {str(row["node_id"]) for row in node_rows}
    if node_ids:
        placeholders = ",".join("?" for _ in node_ids)
        node_terms = [
            dict(row)
            for row in connection.execute(
                f"""
                SELECT node_id, term, normalized_term, term_type, case_sensitive, source
                FROM graph_node_terms
                WHERE node_id IN ({placeholders})
                """,
                tuple(sorted(node_ids)),
            ).fetchall()
        ]
        node_chunks = [
            dict(row)
            for row in connection.execute(
                f"""
                SELECT node_id, chunk_id, relation, weight
                FROM graph_node_chunks
                WHERE node_id IN ({placeholders})
                """,
                tuple(sorted(node_ids)),
            ).fetchall()
        ]
    else:
        node_terms = []
        node_chunks = []
    edge_rows = [
        dict(row)
        for row in connection.execute(
            """
            SELECT edge_id, source_node_id, target_node_id, edge_type,
                   confidence, extractor, relative_path, payload_json
            FROM graph_edges
            WHERE relative_path = ?
            """,
            (normalized_path,),
        ).fetchall()
    ]
    return stable_json_hash(
        {
            "files": _canonical_rows(
                file_rows,
                ["relative_path", "content_hash", "graph_state", "error_code"],
            ),
            "nodes": _canonical_rows(
                node_rows,
                [
                    "node_id",
                    "node_type",
                    "key",
                    "relative_path",
                    "chunk_id",
                    "start_line",
                    "end_line",
                    "payload_json",
                ],
            ),
            "node_terms": _canonical_rows(
                node_terms,
                ["node_id", "term", "normalized_term", "term_type", "case_sensitive", "source"],
            ),
            "node_chunks": _canonical_rows(
                node_chunks,
                ["node_id", "chunk_id", "relation", "weight"],
            ),
            "edges": _canonical_rows(
                edge_rows,
                [
                    "edge_id",
                    "source_node_id",
                    "target_node_id",
                    "edge_type",
                    "confidence",
                    "extractor",
                    "relative_path",
                    "payload_json",
                ],
            ),
        }
    )


def _graph_update_bounds(
    *,
    rows_deleted: int | None = None,
    rows_upserted: int | None = None,
    max_rows_deleted: int | None = None,
    max_rows_upserted: int | None = None,
    started_at: float,
    max_duration_ms: int | None = None,
) -> list[str]:
    exceeded: list[str] = []
    if max_rows_deleted is not None and rows_deleted is not None and rows_deleted > max_rows_deleted:
        exceeded.append("max_rows_deleted")
    if max_rows_upserted is not None and rows_upserted is not None and rows_upserted > max_rows_upserted:
        exceeded.append("max_rows_upserted")
    if max_duration_ms is not None and max_duration_ms >= 0:
        elapsed_ms = int((perf_counter() - started_at) * 1000)
        if elapsed_ms > max_duration_ms:
            exceeded.append("max_duration_ms")
    return exceeded


def _raise_update_bounds_if_exceeded(*, bounds_exceeded: list[str], summary: dict[str, object]) -> None:
    if bounds_exceeded:
        raise GraphUpdateBoundsExceeded(
            bounds_exceeded=bounds_exceeded,
            summary={
                **summary,
                "bounds_exceeded": bounds_exceeded,
            },
        )


def _replace_metadata(connection: sqlite3.Connection, metadata: dict[str, object]) -> None:
    connection.execute("DELETE FROM graph_metadata")
    connection.execute(
        """
        INSERT INTO graph_metadata (
            repo_id,
            repo_root,
            profile,
            schema_version,
            built_commit,
            built_at,
            include_rules_hash,
            exclude_rules_hash,
            extractor_versions_json,
            source_index_contract_json,
            graph_contract_hash
        ) VALUES (
            :repo_id,
            :repo_root,
            :profile,
            :schema_version,
            :built_commit,
            :built_at,
            :include_rules_hash,
            :exclude_rules_hash,
            :extractor_versions_json,
            :source_index_contract_json,
            :graph_contract_hash
        )
        """,
        metadata,
    )


def _select_path_owned_node_ids(connection: sqlite3.Connection, paths: list[str]) -> set[str]:
    if not paths:
        return set()
    path_placeholders = ",".join("?" for _ in paths)
    type_placeholders = ",".join("?" for _ in PATH_OWNED_NODE_TYPES)
    rows = connection.execute(
        f"""
        SELECT node_id
        FROM graph_nodes
        WHERE relative_path IN ({path_placeholders})
          AND node_type IN ({type_placeholders})
        """,
        (*paths, *sorted(PATH_OWNED_NODE_TYPES)),
    ).fetchall()
    return {str(row["node_id"]) for row in rows}


def _select_chunk_ids_for_paths(connection: sqlite3.Connection, paths: list[str]) -> set[str]:
    if not paths:
        return set()
    rows = connection.execute(
        f"""
        SELECT DISTINCT chunk_id
        FROM graph_nodes
        WHERE relative_path IN ({",".join("?" for _ in paths)})
          AND chunk_id IS NOT NULL
        """,
        tuple(paths),
    ).fetchall()
    return {str(row["chunk_id"]) for row in rows}


def _select_shared_candidate_node_ids(
    connection: sqlite3.Connection,
    *,
    affected_chunk_ids: set[str],
    affected_paths: list[str],
) -> set[str]:
    node_ids: set[str] = set()
    shared_types = sorted(SHARED_NODE_TYPES)
    if affected_chunk_ids:
        rows = connection.execute(
            f"""
            SELECT DISTINCT n.node_id
            FROM graph_node_chunks nc
            JOIN graph_nodes n ON n.node_id = nc.node_id
            WHERE nc.chunk_id IN ({",".join("?" for _ in affected_chunk_ids)})
              AND n.node_type IN ({",".join("?" for _ in shared_types)})
            """,
            (*affected_chunk_ids, *shared_types),
        ).fetchall()
        node_ids.update(str(row["node_id"]) for row in rows)
    if affected_paths:
        rows = connection.execute(
            f"""
            SELECT DISTINCT n.node_id
            FROM graph_edges e
            JOIN graph_nodes n
              ON n.node_id = e.source_node_id OR n.node_id = e.target_node_id
            WHERE e.relative_path IN ({",".join("?" for _ in affected_paths)})
              AND n.node_type IN ({",".join("?" for _ in shared_types)})
            """,
            (*affected_paths, *shared_types),
        ).fetchall()
        node_ids.update(str(row["node_id"]) for row in rows)
    return node_ids


def _shared_node_clause(node_ids: set[str]) -> str:
    if not node_ids:
        return ""
    return f"OR n.node_id IN ({','.join('?' for _ in node_ids)})"


def _delete_edges_for_node_ids(connection: sqlite3.Connection, node_ids: set[str]) -> int:
    if not node_ids:
        return 0
    placeholders = ",".join("?" for _ in node_ids)
    return connection.execute(
        f"DELETE FROM graph_edges WHERE source_node_id IN ({placeholders}) OR target_node_id IN ({placeholders})",
        (*node_ids, *node_ids),
    ).rowcount


def _delete_node_terms(connection: sqlite3.Connection, node_ids: set[str]) -> int:
    if not node_ids:
        return 0
    return connection.execute(
        f"DELETE FROM graph_node_terms WHERE node_id IN ({','.join('?' for _ in node_ids)})",
        tuple(node_ids),
    ).rowcount


def _delete_node_chunks(connection: sqlite3.Connection, node_ids: set[str]) -> int:
    if not node_ids:
        return 0
    return connection.execute(
        f"DELETE FROM graph_node_chunks WHERE node_id IN ({','.join('?' for _ in node_ids)})",
        tuple(node_ids),
    ).rowcount


def _delete_node_chunks_for_chunk_ids(connection: sqlite3.Connection, chunk_ids: set[str]) -> int:
    if not chunk_ids:
        return 0
    return connection.execute(
        f"DELETE FROM graph_node_chunks WHERE chunk_id IN ({','.join('?' for _ in chunk_ids)})",
        tuple(chunk_ids),
    ).rowcount


def _delete_nodes(connection: sqlite3.Connection, node_ids: set[str]) -> int:
    if not node_ids:
        return 0
    return connection.execute(
        f"DELETE FROM graph_nodes WHERE node_id IN ({','.join('?' for _ in node_ids)})",
        tuple(node_ids),
    ).rowcount


def _upsert_files(connection: sqlite3.Connection, rows: list[dict[str, object]]) -> None:
    connection.executemany(
        """
        INSERT INTO graph_files (
            relative_path,
            content_hash,
            graph_input_signature,
            mtime,
            graph_state,
            indexed_at,
            last_extracted_at,
            error_code,
            error_detail
        ) VALUES (
            :relative_path,
            :content_hash,
            :graph_input_signature,
            :mtime,
            :graph_state,
            :indexed_at,
            :last_extracted_at,
            :error_code,
            :error_detail
        )
        ON CONFLICT(relative_path) DO UPDATE SET
            content_hash=excluded.content_hash,
            graph_input_signature=excluded.graph_input_signature,
            mtime=excluded.mtime,
            graph_state=excluded.graph_state,
            indexed_at=excluded.indexed_at,
            last_extracted_at=excluded.last_extracted_at,
            error_code=excluded.error_code,
            error_detail=excluded.error_detail
        """,
        rows,
    )


def _upsert_nodes(connection: sqlite3.Connection, rows: list[dict[str, object]]) -> None:
    connection.executemany(
        """
        INSERT INTO graph_nodes (
            node_id,
            node_type,
            key,
            relative_path,
            chunk_id,
            start_line,
            end_line,
            payload_json
        ) VALUES (
            :node_id,
            :node_type,
            :key,
            :relative_path,
            :chunk_id,
            :start_line,
            :end_line,
            :payload_json
        )
        ON CONFLICT(node_id) DO UPDATE SET
            node_type=excluded.node_type,
            key=excluded.key,
            relative_path=excluded.relative_path,
            chunk_id=excluded.chunk_id,
            start_line=excluded.start_line,
            end_line=excluded.end_line,
            payload_json=excluded.payload_json
        """,
        rows,
    )


def _replace_node_terms(
    connection: sqlite3.Connection,
    node_ids: set[str],
    rows: list[dict[str, object]],
) -> int:
    deleted = _delete_node_terms(connection, node_ids)
    connection.executemany(
        """
        INSERT OR IGNORE INTO graph_node_terms (
            node_id,
            term,
            normalized_term,
            term_type,
            case_sensitive,
            source
        ) VALUES (
            :node_id,
            :term,
            :normalized_term,
            :term_type,
            :case_sensitive,
            :source
        )
        """,
        [
            {
                **row,
                "case_sensitive": 1 if row.get("case_sensitive") else 0,
            }
            for row in rows
        ],
    )
    return deleted


def _replace_node_chunks(
    connection: sqlite3.Connection,
    node_ids: set[str],
    rows: list[dict[str, object]],
) -> int:
    deleted = _delete_node_chunks(connection, node_ids)
    connection.executemany(
        """
        INSERT OR IGNORE INTO graph_node_chunks (
            node_id,
            chunk_id,
            relation,
            weight
        ) VALUES (
            :node_id,
            :chunk_id,
            :relation,
            :weight
        )
        """,
        rows,
    )
    return deleted


def _upsert_edges(connection: sqlite3.Connection, rows: list[dict[str, object]]) -> None:
    connection.executemany(
        """
        INSERT INTO graph_edges (
            edge_id,
            source_node_id,
            target_node_id,
            edge_type,
            confidence,
            extractor,
            relative_path,
            payload_json
        ) VALUES (
            :edge_id,
            :source_node_id,
            :target_node_id,
            :edge_type,
            :confidence,
            :extractor,
            :relative_path,
            :payload_json
        )
        ON CONFLICT(edge_id) DO UPDATE SET
            source_node_id=excluded.source_node_id,
            target_node_id=excluded.target_node_id,
            edge_type=excluded.edge_type,
            confidence=excluded.confidence,
            extractor=excluded.extractor,
            relative_path=excluded.relative_path,
            payload_json=excluded.payload_json
        """,
        rows,
    )


def _prune_orphan_shared_nodes(connection: sqlite3.Connection) -> int:
    rows = connection.execute(
        f"""
        SELECT n.node_id
        FROM graph_nodes n
        LEFT JOIN graph_node_chunks nc ON nc.node_id = n.node_id
        LEFT JOIN graph_edges es ON es.source_node_id = n.node_id
        LEFT JOIN graph_edges et ON et.target_node_id = n.node_id
        WHERE n.node_type IN ({",".join("?" for _ in SHARED_NODE_TYPES)})
        GROUP BY n.node_id
        HAVING COUNT(nc.chunk_id) = 0
           AND COUNT(es.edge_id) = 0
           AND COUNT(et.edge_id) = 0
        """,
        tuple(sorted(SHARED_NODE_TYPES)),
    ).fetchall()
    node_ids = {str(row["node_id"]) for row in rows}
    _delete_node_terms(connection, node_ids)
    _delete_node_chunks(connection, node_ids)
    return _delete_nodes(connection, node_ids)
