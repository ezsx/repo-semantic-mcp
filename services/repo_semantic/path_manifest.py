"""Per-index path manifest for precise freshness diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import posixpath
from pathlib import Path
from pathlib import PurePosixPath, PureWindowsPath
import sqlite3
import time
from typing import Iterable

from services.repo_semantic.config import SemanticMcpSettings
from services.repo_semantic.contracts.status import PathFreshnessSummary
from services.repo_semantic.graph.artifacts import stable_json_hash
from services.repo_semantic.models import ChunkRecord


PATH_MANIFEST_SCHEMA_VERSION = 1
HASH_ALGORITHM = "sha256"


def normalize_manifest_relative_path(raw_path: str) -> str | None:
    """Normalize a repo-relative manifest path, rejecting absolute/escaping paths."""

    raw = raw_path.replace("\\", "/").strip()
    if not raw:
        return None
    if PurePosixPath(raw).is_absolute() or PureWindowsPath(raw).is_absolute():
        return None
    normalized = posixpath.normpath(raw)
    if normalized in {"", "."} or normalized == ".." or normalized.startswith("../"):
        return None
    return normalized.strip("/")


@dataclass(frozen=True, slots=True)
class PathManifestRecord:
    """One indexed file entry persisted in the path manifest."""

    relative_path: str
    scope: str
    content_hash: str
    source_mtime: float
    file_size: int
    indexed_at: str
    chunk_count: int
    point_ids: list[str]
    index_state: str = "ready"
    last_error_code: str | None = None
    last_error_detail: str | None = None


def path_manifest_path(settings: SemanticMcpSettings) -> Path:
    """Return the deterministic per-repo/profile path manifest artifact path."""

    return (
        settings.registry_db_path.parent
        / "path_manifests"
        / settings.repo_key_slug
        / settings.profile_slug
        / f"path_manifest_v{PATH_MANIFEST_SCHEMA_VERSION}.sqlite3"
    )


def expected_manifest_identity(settings: SemanticMcpSettings) -> dict[str, object]:
    """Build the manifest identity that must match for authoritative reads."""

    include_rules_hash = stable_json_hash(list(settings.effective_include_globs))
    exclude_rules_hash = stable_json_hash(list(settings.SEMANTIC_MCP_EXCLUDE_GLOBS))
    doc_prefixes_hash = stable_json_hash(list(settings.effective_doc_prefixes))
    query_template_hash = hashlib.sha256(settings.SEMANTIC_MCP_QUERY_TEMPLATE.encode("utf-8")).hexdigest()
    document_prefix_hash = hashlib.sha256(settings.SEMANTIC_MCP_DOCUMENT_PREFIX.encode("utf-8")).hexdigest()
    return {
        "manifest_schema_version": PATH_MANIFEST_SCHEMA_VERSION,
        "hash_algorithm": HASH_ALGORITHM,
        "repo_key": settings.repo_key_slug,
        "normalized_repo_root": settings.logical_repo_identity,
        "index_profile": settings.SEMANTIC_MCP_PROFILE_NAME,
        "embedding_backend_id": settings.embedding_backend_id,
        "embedding_model": settings.SEMANTIC_MCP_EMBEDDING_MODEL,
        "index_schema_version": settings.SEMANTIC_MCP_INDEX_SCHEMA_VERSION,
        "query_template_hash": query_template_hash,
        "document_prefix_hash": document_prefix_hash,
        "include_rules_hash": include_rules_hash,
        "exclude_rules_hash": exclude_rules_hash,
        "doc_prefixes_hash": doc_prefixes_hash,
        "collections": {
            "code": settings.collection_code,
            "docs": settings.collection_docs,
        },
    }


def file_content_hash(path: Path) -> str:
    """Hash one file without loading large files fully into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_record_for_chunks(
    *,
    settings: SemanticMcpSettings,
    relative_path: str,
    chunks: list[ChunkRecord],
    indexed_at: str,
    content_hash: str | None = None,
    source_mtime: float | None = None,
    file_size: int | None = None,
    scope: str | None = None,
) -> PathManifestRecord:
    """Build a ready manifest record after chunks were successfully indexed."""

    normalized_path = normalize_manifest_relative_path(relative_path)
    if normalized_path is None:
        raise ValueError(f"invalid repo-relative manifest path: {relative_path!r}")
    if content_hash is None or source_mtime is None or file_size is None:
        file_path = settings.repo_root / normalized_path
        stat = file_path.stat()
        content_hash = content_hash or file_content_hash(file_path)
        source_mtime = float(source_mtime if source_mtime is not None else stat.st_mtime)
        file_size = int(file_size if file_size is not None else stat.st_size)
    effective_scope = scope or (chunks[0].scope if chunks else "code")
    return PathManifestRecord(
        relative_path=normalized_path,
        scope=effective_scope,
        content_hash=content_hash,
        source_mtime=float(source_mtime),
        file_size=int(file_size),
        indexed_at=indexed_at,
        chunk_count=len(chunks),
        point_ids=[chunk.point_id for chunk in chunks],
        index_state="ready",
    )


def group_chunks_by_path(chunks: Iterable[ChunkRecord]) -> dict[str, list[ChunkRecord]]:
    """Group chunk records by normalized repo-relative path."""

    grouped: dict[str, list[ChunkRecord]] = {}
    for chunk in chunks:
        relative_path = normalize_manifest_relative_path(chunk.relative_path)
        if relative_path is None:
            continue
        grouped.setdefault(relative_path, []).append(chunk)
    return grouped


class PathManifestStore:
    """SQLite-backed path manifest artifact."""

    def __init__(self, settings: SemanticMcpSettings) -> None:
        self._settings = settings
        self.path = path_manifest_path(settings)

    def exists(self) -> bool:
        """Return whether the manifest artifact exists."""

        return self.path.exists()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA busy_timeout=3000")
            self._ensure_schema(connection)
            return connection
        except Exception:
            connection.close()
            raise

    def _ensure_schema(self, connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS manifest_metadata(
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS path_manifest(
                relative_path TEXT PRIMARY KEY,
                scope TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                source_mtime REAL NOT NULL,
                file_size INTEGER NOT NULL,
                indexed_at TEXT NOT NULL,
                chunk_count INTEGER NOT NULL,
                point_ids_json TEXT NOT NULL,
                index_state TEXT NOT NULL,
                last_verified_at TEXT,
                delete_confirmed_at TEXT,
                last_error_code TEXT,
                last_error_detail TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_path_manifest_state
                ON path_manifest(index_state);
            CREATE INDEX IF NOT EXISTS idx_path_manifest_scope
                ON path_manifest(scope);
            CREATE INDEX IF NOT EXISTS idx_path_manifest_indexed_at
                ON path_manifest(indexed_at);
            CREATE INDEX IF NOT EXISTS idx_path_manifest_delete_confirmed_at
                ON path_manifest(delete_confirmed_at);
            CREATE INDEX IF NOT EXISTS idx_path_manifest_indexed_at_path
                ON path_manifest(indexed_at, relative_path);
            CREATE INDEX IF NOT EXISTS idx_path_manifest_delete_confirmed_at_path
                ON path_manifest(delete_confirmed_at, relative_path);
            """
        )
        connection.execute(f"PRAGMA user_version = {PATH_MANIFEST_SCHEMA_VERSION}")

    def _write_metadata(
        self,
        connection: sqlite3.Connection,
        identity: dict[str, object],
        *,
        coverage_complete: bool,
    ) -> None:
        payload = {
            "identity": identity,
            "identity_hash": stable_json_hash(identity),
            "coverage_complete": coverage_complete,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        for key, value in payload.items():
            connection.execute(
                """
                INSERT INTO manifest_metadata(key, value)
                VALUES(?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, json.dumps(value, ensure_ascii=False, sort_keys=True)),
            )

    def _read_bool_metadata(self, connection: sqlite3.Connection, key: str) -> bool | None:
        row = connection.execute(
            "SELECT value FROM manifest_metadata WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        try:
            return bool(json.loads(row["value"]))
        except json.JSONDecodeError:
            return None

    def _read_identity_hash(self, connection: sqlite3.Connection) -> str | None:
        row = connection.execute(
            "SELECT value FROM manifest_metadata WHERE key = 'identity_hash'"
        ).fetchone()
        if row is None:
            return None
        try:
            return str(json.loads(row["value"]))
        except json.JSONDecodeError:
            return None

    def _compatible_existing_coverage_complete(
        self,
        connection: sqlite3.Connection,
        identity: dict[str, object],
    ) -> bool:
        existing_complete = self._read_bool_metadata(connection, "coverage_complete")
        if not existing_complete:
            return False
        return self._read_identity_hash(connection) == stable_json_hash(identity)

    def replace_all(self, records: Iterable[PathManifestRecord]) -> None:
        """Replace the manifest with a full build snapshot."""

        identity = expected_manifest_identity(self._settings)
        connection = self._connect()
        try:
            with connection:
                self._write_metadata(connection, identity, coverage_complete=True)
                connection.execute("DELETE FROM path_manifest")
                self._upsert_records(connection, records)
        finally:
            connection.close()

    def upsert_ready_records(self, records: Iterable[PathManifestRecord]) -> None:
        """Upsert ready path entries after successful incremental indexing."""

        identity = expected_manifest_identity(self._settings)
        connection = self._connect()
        try:
            with connection:
                self._write_metadata(
                    connection,
                    identity,
                    coverage_complete=self._compatible_existing_coverage_complete(
                        connection,
                        identity,
                    ),
                )
                self._upsert_records(connection, records)
        finally:
            connection.close()

    def mark_deleted(self, relative_paths: Iterable[str]) -> None:
        """Persist deleted path state after Qdrant points were removed."""

        identity = expected_manifest_identity(self._settings)
        deleted_at = datetime.now(timezone.utc).isoformat()
        connection = self._connect()
        try:
            with connection:
                self._write_metadata(
                    connection,
                    identity,
                    coverage_complete=self._compatible_existing_coverage_complete(
                        connection,
                        identity,
                    ),
                )
                normalized_paths = {
                    normalized
                    for path in relative_paths
                    if (normalized := normalize_manifest_relative_path(path)) is not None
                }
                for raw_path in sorted(normalized_paths):
                    if not raw_path:
                        continue
                    connection.execute(
                        """
                        INSERT INTO path_manifest(
                            relative_path, scope, content_hash, source_mtime,
                            file_size, indexed_at, chunk_count, point_ids_json,
                            index_state, delete_confirmed_at
                        )
                        VALUES(?, '', '', 0, 0, ?, 0, '[]', 'deleted', ?)
                        ON CONFLICT(relative_path) DO UPDATE SET
                            index_state = 'deleted',
                            delete_confirmed_at = excluded.delete_confirmed_at,
                            point_ids_json = '[]',
                            chunk_count = 0
                        """,
                        (raw_path, deleted_at, deleted_at),
                    )
        finally:
            connection.close()

    def ready_zero_chunk_records(self) -> dict[str, PathManifestRecord]:
        """Return authoritative ready records for files that produced no chunks."""

        if not self.path.exists():
            return {}
        identity = expected_manifest_identity(self._settings)
        connection = self._connect()
        try:
            if self._read_identity_hash(connection) != stable_json_hash(identity):
                return {}
            rows = connection.execute(
                """
                SELECT *
                FROM path_manifest
                WHERE index_state = 'ready'
                  AND chunk_count = 0
                """
            ).fetchall()
            records: dict[str, PathManifestRecord] = {}
            for row in rows:
                relative_path = normalize_manifest_relative_path(str(row["relative_path"]))
                if relative_path is None:
                    continue
                try:
                    point_ids = list(json.loads(row["point_ids_json"]))
                except json.JSONDecodeError:
                    point_ids = []
                records[relative_path] = PathManifestRecord(
                    relative_path=relative_path,
                    scope=str(row["scope"] or "code"),
                    content_hash=str(row["content_hash"] or ""),
                    source_mtime=float(row["source_mtime"] or 0.0),
                    file_size=int(row["file_size"] or 0),
                    indexed_at=str(row["indexed_at"]),
                    chunk_count=0,
                    point_ids=[str(point_id) for point_id in point_ids],
                    index_state="ready",
                    last_error_code=row["last_error_code"],
                    last_error_detail=row["last_error_detail"],
                )
            return records
        finally:
            connection.close()

    def invalidated_paths_after(self, timestamp: str, *, limit: int = 12) -> tuple[int, list[str], bool]:
        """Return bounded manifest paths changed after a graph build timestamp.

        The boolean indicates whether the returned path set is complete enough
        for query-aware graph freshness decisions.
        """

        if not timestamp or not self.path.exists():
            return 0, [], False
        connection = self._connect()
        try:
            self._ensure_schema(connection)
            identity = expected_manifest_identity(self._settings)
            if self._read_identity_hash(connection) != stable_json_hash(identity):
                return 0, [], False
            if not self._read_bool_metadata(connection, "coverage_complete"):
                return 0, [], False
            active_rows = connection.execute(
                """
                SELECT relative_path
                FROM path_manifest
                WHERE indexed_at > ?
                ORDER BY indexed_at ASC, relative_path ASC
                LIMIT ?
                """,
                (timestamp, max(0, limit) + 1),
            ).fetchall()
            deleted_rows = connection.execute(
                """
                SELECT relative_path
                FROM path_manifest
                WHERE delete_confirmed_at IS NOT NULL
                  AND delete_confirmed_at > ?
                ORDER BY delete_confirmed_at ASC, relative_path ASC
                LIMIT ?
                """,
                (timestamp, max(0, limit) + 1),
            ).fetchall()
            paths = sorted(
                {
                    str(row["relative_path"])
                    for row in [*active_rows, *deleted_rows]
                }
            )
            preview = paths[:limit]
            complete = len(active_rows) <= limit and len(deleted_rows) <= limit and len(paths) <= limit
            count = len(preview) if complete else len(preview) + 1
            return count, preview, complete
        finally:
            connection.close()

    def _upsert_records(
        self,
        connection: sqlite3.Connection,
        records: Iterable[PathManifestRecord],
    ) -> None:
        for record in records:
            connection.execute(
                """
                INSERT INTO path_manifest(
                    relative_path, scope, content_hash, source_mtime,
                    file_size, indexed_at, chunk_count, point_ids_json,
                    index_state, last_error_code, last_error_detail
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(relative_path) DO UPDATE SET
                    scope = excluded.scope,
                    content_hash = excluded.content_hash,
                    source_mtime = excluded.source_mtime,
                    file_size = excluded.file_size,
                    indexed_at = excluded.indexed_at,
                    chunk_count = excluded.chunk_count,
                    point_ids_json = excluded.point_ids_json,
                    index_state = excluded.index_state,
                    last_verified_at = NULL,
                    delete_confirmed_at = NULL,
                    last_error_code = excluded.last_error_code,
                    last_error_detail = excluded.last_error_detail
                """,
                (
                    record.relative_path,
                    record.scope,
                    record.content_hash,
                    record.source_mtime,
                    record.file_size,
                    record.indexed_at,
                    record.chunk_count,
                    json.dumps(record.point_ids, ensure_ascii=False),
                    record.index_state,
                    record.last_error_code,
                    record.last_error_detail,
                ),
            )

    def summarize(
        self,
        indexable_paths: Iterable[Path],
        *,
        max_duration_ms: int | None = None,
        max_hashed_paths: int | None = None,
    ) -> PathFreshnessSummary:
        """Compare current indexable paths with the persisted manifest."""

        started_at = time.monotonic()

        def _bounds_summary(reason: str, error_paths: list[str] | None = None) -> PathFreshnessSummary:
            return PathFreshnessSummary(
                coverage_complete=False,
                coverage_error_code=reason,
                manifest_available=True,
                manifest_path=str(self.path),
                manifest_schema_version=PATH_MANIFEST_SCHEMA_VERSION,
                identity_compatible=True,
                error_paths_preview=(error_paths or [])[:12],
            )

        def _duration_exceeded() -> bool:
            if max_duration_ms is None:
                return False
            return int((time.monotonic() - started_at) * 1000) > max_duration_ms

        if not self.path.exists():
            return PathFreshnessSummary(
                coverage_complete=False,
                coverage_error_code="path_manifest_missing",
                manifest_available=False,
                manifest_path=str(self.path),
                manifest_schema_version=PATH_MANIFEST_SCHEMA_VERSION,
                identity_compatible=False,
            )

        current_paths = {}
        for path in indexable_paths:
            if _duration_exceeded():
                return _bounds_summary("path_manifest_status_bounds_exceeded")
            if path.exists() and path.is_file():
                current_paths[path.relative_to(self._settings.repo_root).as_posix()] = path
        try:
            connection = self._connect()
            try:
                stored_identity_hash = self._read_identity_hash(connection)
                manifest_coverage_complete = self._read_bool_metadata(connection, "coverage_complete")
                expected_identity_hash = stable_json_hash(expected_manifest_identity(self._settings))
                rows = connection.execute("SELECT * FROM path_manifest").fetchall()
            finally:
                connection.close()
        except Exception as exc:  # noqa: BLE001
            return PathFreshnessSummary(
                coverage_complete=False,
                coverage_error_code="path_manifest_read_failed",
                manifest_available=True,
                manifest_path=str(self.path),
                manifest_schema_version=PATH_MANIFEST_SCHEMA_VERSION,
                identity_compatible=False,
                error_paths_preview=[str(exc)[:240]],
            )

        identity_compatible = stored_identity_hash == expected_identity_hash
        if not identity_compatible:
            return PathFreshnessSummary(
                coverage_complete=False,
                coverage_error_code="path_manifest_identity_mismatch",
                manifest_available=True,
                manifest_path=str(self.path),
                manifest_schema_version=PATH_MANIFEST_SCHEMA_VERSION,
                identity_compatible=False,
            )

        manifest_by_path = {str(row["relative_path"]): row for row in rows}
        missing_from_index: list[str] = []
        stale_indexed: list[str] = []
        deleted_indexed: list[str] = []
        error_paths: list[str] = []
        hashed_paths = 0

        for relative_path, file_path in current_paths.items():
            if _duration_exceeded():
                return _bounds_summary("path_manifest_status_bounds_exceeded", error_paths)
            row = manifest_by_path.get(relative_path)
            if row is None or row["index_state"] == "deleted":
                missing_from_index.append(relative_path)
                continue
            if row["index_state"] == "error":
                error_paths.append(relative_path)
                continue
            try:
                stat = file_path.stat()
                file_size_changed = int(row["file_size"]) != int(stat.st_size)
                source_mtime_changed = abs(float(row["source_mtime"]) - float(stat.st_mtime)) > 1e-6
                if file_size_changed or source_mtime_changed:
                    if max_hashed_paths is not None and hashed_paths >= max_hashed_paths:
                        return _bounds_summary("path_manifest_status_bounds_exceeded", error_paths)
                    hashed_paths += 1
                if (file_size_changed or source_mtime_changed) and file_content_hash(file_path) != row["content_hash"]:
                    stale_indexed.append(relative_path)
            except Exception:  # noqa: BLE001
                error_paths.append(relative_path)

        for relative_path, row in manifest_by_path.items():
            if _duration_exceeded():
                return _bounds_summary("path_manifest_status_bounds_exceeded", error_paths)
            if row["index_state"] == "deleted":
                continue
            if relative_path not in current_paths:
                deleted_indexed.append(relative_path)

        coverage_complete = bool(manifest_coverage_complete) and not error_paths
        coverage_error_code = None
        if not manifest_coverage_complete:
            coverage_error_code = "path_manifest_incomplete"
        elif error_paths:
            coverage_error_code = "path_manifest_scan_error"
        return PathFreshnessSummary(
            coverage_complete=coverage_complete,
            coverage_error_code=coverage_error_code,
            manifest_available=True,
            manifest_path=str(self.path),
            manifest_schema_version=PATH_MANIFEST_SCHEMA_VERSION,
            identity_compatible=True,
            missing_from_index_count=len(missing_from_index),
            stale_indexed_paths_count=len(stale_indexed),
            deleted_indexed_paths_count=len(deleted_indexed),
            path_error_count=len(error_paths),
            stale_paths_preview=sorted((missing_from_index + stale_indexed))[:12],
            deleted_paths_preview=sorted(deleted_indexed)[:12],
            error_paths_preview=sorted(error_paths)[:12],
        )
