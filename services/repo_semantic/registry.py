"""SQLite registry для repo lifecycle metadata semantic MCP."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from services.repo_semantic.config import build_repo_key, normalize_repo_identity
from services.repo_semantic.models import (
    BackendRegistryEntry,
    BackendRole,
    RepoRegistryEntry,
    RepoStatus,
)
from services.repo_semantic.storage.sqlite.connections import connect_row_factory
from services.repo_semantic.storage.sqlite.schema import (
    ensure_backend_tables,
    ensure_lifecycle_tables,
    ensure_registry_schema,
    ensure_repo_registry_columns,
    table_columns,
)

LEGACY_REPO_ROOTS: frozenset[str] = frozenset({"/repo", "/target_repo"})


def _utc_now() -> str:
    """Вернуть текущее UTC-время в ISO формате."""

    return datetime.now(timezone.utc).isoformat()


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _lifecycle_lease_key(repo_root: str, index_profile: str, backend_id: str) -> str:
    normalized_root = normalize_repo_identity(repo_root)
    return f"{normalized_root}|{index_profile}|{backend_id}"


def _redact_lifecycle_text(value: str | None) -> str | None:
    if value is None:
        return None
    lowered = value.lower()
    if any(marker in lowered for marker in ("secret", "token", "password", "api_key", "authorization")):
        return "<redacted>"
    return value


def _lifecycle_idempotency_fingerprint(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _redact_lifecycle_value(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_lifecycle_text(value)
    if isinstance(value, list):
        return [_redact_lifecycle_value(item) for item in value]
    if isinstance(value, dict):
        return {
            key: "<redacted>"
            if isinstance(key, str)
            and any(marker in key.lower() for marker in ("secret", "token", "password", "api_key", "authorization"))
            else _redact_lifecycle_value(item)
            for key, item in value.items()
        }
    return value


@dataclass(frozen=True, slots=True)
class LifecycleLease:
    """Persistent lifecycle mutation lease."""

    lease_key: str
    lease_id: str
    audit_id: str
    repo_root: str
    index_profile: str
    backend_id: str
    operation: str
    owner_id: str
    idempotency_key: str | None
    acquired_at: str
    heartbeat_at: str
    expires_at: str
    recovered_expired: bool = False
    idempotency_fingerprint: str | None = None


class LifecycleLeaseActiveError(RuntimeError):
    """Raised when a non-expired lifecycle mutation lease already exists."""

    def __init__(
        self,
        message: str,
        *,
        existing_operation: str,
        same_idempotency_key: bool,
    ) -> None:
        super().__init__(message)
        self.existing_operation = existing_operation
        self.same_idempotency_key = same_idempotency_key


class RepoRegistry:
    """Хранить persistent repo registry и active repo state."""

    def __init__(self, db_path: Path) -> None:
        """Инициализировать registry по пути к SQLite-файлу."""

        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    @property
    def db_path(self) -> Path:
        """Вернуть путь к registry SQLite."""

        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        """Открыть соединение с SQLite в row-factory режиме."""

        return connect_row_factory(self._db_path)

    def _table_columns(self, connection: sqlite3.Connection, table_name: str) -> set[str]:
        """Вернуть множество колонок существующей таблицы."""

        return table_columns(connection, table_name)

    def _ensure_repo_registry_columns(self, connection: sqlite3.Connection) -> None:
        """Добавить отсутствующие колонки repo_registry в уже существующую БД."""

        ensure_repo_registry_columns(connection)

    def _ensure_backend_tables(self, connection: sqlite3.Connection) -> None:
        """Создать и/или дополнить backend registry таблицы."""

        ensure_backend_tables(connection)

    def _ensure_lifecycle_tables(self, connection: sqlite3.Connection) -> None:
        """Создать lifecycle lease/audit таблицы."""

        ensure_lifecycle_tables(connection)

    def _ensure_schema(self) -> None:
        """Создать или мигрировать schema при первом запуске."""

        with self._connect() as connection:
            ensure_registry_schema(connection)

    def _row_to_lifecycle_lease(
        self,
        row: sqlite3.Row,
        *,
        recovered_expired: bool = False,
    ) -> LifecycleLease:
        return LifecycleLease(
            lease_key=row["lease_key"],
            lease_id=row["lease_id"],
            audit_id=row["audit_id"],
            repo_root=row["repo_root"],
            index_profile=row["index_profile"],
            backend_id=row["backend_id"],
            operation=row["operation"],
            owner_id=row["owner_id"],
            idempotency_key=row["idempotency_key"],
            acquired_at=row["acquired_at"],
            heartbeat_at=row["heartbeat_at"],
            expires_at=row["expires_at"],
            idempotency_fingerprint=row["idempotency_fingerprint"],
            recovered_expired=recovered_expired,
        )

    def acquire_lifecycle_lease(
        self,
        repo_root: str,
        *,
        index_profile: str,
        backend_id: str,
        operation: str,
        ttl_sec: int = 900,
        owner_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> LifecycleLease:
        """Acquire a repo/profile/backend mutation lease or raise when busy."""

        normalized_root = normalize_repo_identity(repo_root)
        lease_key = _lifecycle_lease_key(normalized_root, index_profile, backend_id)
        safe_idempotency_key = _redact_lifecycle_text(idempotency_key)
        idempotency_fingerprint = _lifecycle_idempotency_fingerprint(idempotency_key)
        lease_id = str(uuid4())
        audit_id = str(uuid4())
        effective_owner_id = owner_id or f"pid:{os.getpid()}:{uuid4()}"
        now_dt = datetime.now(timezone.utc)
        now = now_dt.isoformat()
        expires_at = (now_dt + timedelta(seconds=max(1, ttl_sec))).isoformat()
        recovered_expired = False
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM lifecycle_lease WHERE lease_key = ?",
                (lease_key,),
            ).fetchone()
            if existing is not None:
                existing_expires_at = _parse_timestamp(existing["expires_at"])
                if existing_expires_at is not None and existing_expires_at > now_dt:
                    same_idempotency_key = bool(
                        idempotency_fingerprint
                        and existing["operation"] == operation
                        and existing["idempotency_fingerprint"] == idempotency_fingerprint
                    )
                    connection.rollback()
                    raise LifecycleLeaseActiveError(
                        "Lifecycle mutation lease is already active: "
                        f"operation={existing['operation']} repo_root={normalized_root} "
                        f"profile={index_profile} backend_id={backend_id}",
                        existing_operation=existing["operation"],
                        same_idempotency_key=same_idempotency_key,
                    )
                recovered_expired = True
                connection.execute(
                    "DELETE FROM lifecycle_lease WHERE lease_key = ?",
                    (lease_key,),
                )
                connection.execute(
                    """
                    UPDATE lifecycle_audit
                    SET status = 'expired_recovered',
                        finished_at = ?,
                        error = COALESCE(error, 'lease expired before release')
                    WHERE audit_id = ?
                    """,
                    (now, existing["audit_id"]),
                )

            connection.execute(
                """
                INSERT INTO lifecycle_lease(
                    lease_key, lease_id, audit_id, repo_root, index_profile,
                    backend_id, operation, owner_id, idempotency_key,
                    idempotency_fingerprint,
                    acquired_at, heartbeat_at, expires_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    lease_key,
                    lease_id,
                    audit_id,
                    normalized_root,
                    index_profile,
                    backend_id,
                    operation,
                    effective_owner_id,
                    safe_idempotency_key,
                    idempotency_fingerprint,
                    now,
                    now,
                    expires_at,
                ),
            )
            connection.execute(
                """
                INSERT INTO lifecycle_audit(
                    audit_id, lease_id, lease_key, repo_root, index_profile,
                    backend_id, operation, owner_id, idempotency_key,
                    idempotency_fingerprint, status,
                    started_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?)
                """,
                (
                    audit_id,
                    lease_id,
                    lease_key,
                    normalized_root,
                    index_profile,
                    backend_id,
                    operation,
                    effective_owner_id,
                    safe_idempotency_key,
                    idempotency_fingerprint,
                    now,
                ),
            )
            connection.commit()
        finally:
            connection.close()

        row = {
            "lease_key": lease_key,
            "lease_id": lease_id,
            "audit_id": audit_id,
            "repo_root": normalized_root,
            "index_profile": index_profile,
            "backend_id": backend_id,
            "operation": operation,
            "owner_id": effective_owner_id,
            "idempotency_key": safe_idempotency_key,
            "acquired_at": now,
            "heartbeat_at": now,
            "expires_at": expires_at,
            "idempotency_fingerprint": idempotency_fingerprint,
        }
        return LifecycleLease(**row, recovered_expired=recovered_expired)

    def heartbeat_lifecycle_lease(
        self,
        lease: LifecycleLease,
        *,
        ttl_sec: int = 900,
    ) -> bool:
        """Extend an active lifecycle lease owned by the current operation."""

        now_dt = datetime.now(timezone.utc)
        now = now_dt.isoformat()
        expires_at = (now_dt + timedelta(seconds=max(1, ttl_sec))).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE lifecycle_lease
                SET heartbeat_at = ?,
                    expires_at = ?
                WHERE lease_key = ? AND lease_id = ?
                """,
                (now, expires_at, lease.lease_key, lease.lease_id),
            )
        return bool(cursor.rowcount)

    def release_lifecycle_lease(
        self,
        lease: LifecycleLease,
        *,
        status: str,
        detail: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Release a lifecycle mutation lease and finish its audit event."""

        finished_at = _utc_now()
        safe_detail = _redact_lifecycle_value(detail or {})
        safe_error = _redact_lifecycle_text(error)
        with self._connect() as connection:
            connection.execute(
                """
                DELETE FROM lifecycle_lease
                WHERE lease_key = ? AND lease_id = ?
                """,
                (lease.lease_key, lease.lease_id),
            )
            connection.execute(
                """
                UPDATE lifecycle_audit
                SET status = ?,
                    finished_at = ?,
                    detail_json = ?,
                    error = ?
                WHERE audit_id = ? AND status = 'running'
                """,
                (
                    status,
                    finished_at,
                    json.dumps(safe_detail, ensure_ascii=False, sort_keys=True),
                    safe_error,
                    lease.audit_id,
                ),
            )

    def get_active_lifecycle_lease(
        self,
        repo_root: str,
        *,
        index_profile: str,
        backend_id: str,
    ) -> LifecycleLease | None:
        """Return the active non-expired lifecycle lease for repo/profile/backend."""

        lease_key = _lifecycle_lease_key(repo_root, index_profile, backend_id)
        now_dt = datetime.now(timezone.utc)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM lifecycle_lease WHERE lease_key = ?",
                (lease_key,),
            ).fetchone()
        if row is None:
            return None
        expires_at = _parse_timestamp(row["expires_at"])
        if expires_at is not None and expires_at <= now_dt:
            return None
        return self._row_to_lifecycle_lease(row)

    def list_lifecycle_audit(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent lifecycle audit events for diagnostics/tests."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM lifecycle_audit
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (max(1, limit),),
            ).fetchall()
        return [
            {
                "audit_id": row["audit_id"],
                "lease_id": row["lease_id"],
                "lease_key": row["lease_key"],
                "repo_root": row["repo_root"],
                "index_profile": row["index_profile"],
                "backend_id": row["backend_id"],
                "operation": row["operation"],
                "owner_id": row["owner_id"],
                "idempotency_key": _redact_lifecycle_text(row["idempotency_key"]),
                "idempotency_fingerprint": row["idempotency_fingerprint"],
                "status": row["status"],
                "started_at": row["started_at"],
                "finished_at": row["finished_at"],
                "detail": _redact_lifecycle_value(json.loads(row["detail_json"] or "{}")),
                "error": _redact_lifecycle_text(row["error"]),
            }
            for row in rows
        ]

    def _row_to_entry(self, row: sqlite3.Row) -> RepoRegistryEntry:
        """Преобразовать SQLite row в typed registry entry."""

        return RepoRegistryEntry(
            repo_root=row["repo_root"],
            repo_key=row["repo_key"],
            display_name=row["display_name"],
            status=row["status"],
            active=bool(row["active"]),
            index_profile=row["index_profile"],
            include_globs=json.loads(row["include_globs_json"]),
            doc_prefixes=json.loads(row["doc_prefixes_json"]),
            exclude_globs=json.loads(row["exclude_globs_json"]),
            last_full_build_ts=row["last_full_build_ts"],
            last_incremental_update_ts=row["last_incremental_update_ts"],
            indexed_branch=row["indexed_branch"],
            indexed_commit_hash=row["indexed_commit_hash"],
            last_error=row["last_error"],
            watch_enabled=bool(row["watch_enabled"]),
            watch_running=bool(row["watch_running"]),
            code_points_count=int(row["code_points_count"]),
            docs_points_count=int(row["docs_points_count"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _row_to_backend_entry(self, row: sqlite3.Row) -> BackendRegistryEntry:
        """Преобразовать SQLite row в typed backend registry entry."""

        return BackendRegistryEntry(
            backend_id=row["backend_id"],
            role=row["role"],
            backend_type=row["backend_type"],
            transport=row["transport"],
            endpoint=row["endpoint"],
            managed_by_service=bool(row["managed_by_service"]),
            autostart_policy=row["autostart_policy"],
            health_path=row["health_path"],
            location_type=row["location_type"],
            device_type=row["device_type"],
            model_name=row["model_name"],
            config_blob=json.loads(row["config_blob_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def list_repos(self) -> list[RepoRegistryEntry]:
        """Вернуть список всех зарегистрированных repos."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM repo_registry
                ORDER BY active DESC, updated_at DESC, repo_root ASC
                """
            ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def list_backends(self, *, role: BackendRole | None = None) -> list[BackendRegistryEntry]:
        """Вернуть backend registry entries, при необходимости отфильтрованные по role."""

        with self._connect() as connection:
            if role is None:
                rows = connection.execute(
                    """
                    SELECT *
                    FROM backend_registry
                    ORDER BY role ASC, updated_at DESC, backend_id ASC
                    """
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT *
                    FROM backend_registry
                    WHERE role = ?
                    ORDER BY updated_at DESC, backend_id ASC
                    """,
                    (role,),
                ).fetchall()
        return [self._row_to_backend_entry(row) for row in rows]

    def list_legacy_repos(self) -> list[RepoRegistryEntry]:
        """Вернуть legacy registry entries старого single-runtime формата."""

        return [entry for entry in self.list_repos() if entry.repo_root in LEGACY_REPO_ROOTS]

    def get_repo(self, repo_root: str) -> RepoRegistryEntry | None:
        """Вернуть entry конкретного repo или None."""

        normalized = normalize_repo_identity(repo_root)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM repo_registry WHERE repo_root = ?",
                (normalized,),
            ).fetchone()
        return self._row_to_entry(row) if row else None

    def get_active_repo(self) -> RepoRegistryEntry | None:
        """Вернуть текущий active repo."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM repo_registry
                WHERE active = 1
                ORDER BY updated_at DESC
                LIMIT 1
                """
            ).fetchone()
        return self._row_to_entry(row) if row else None

    def get_backend(self, backend_id: str) -> BackendRegistryEntry | None:
        """Вернуть backend entry по backend_id или None."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM backend_registry WHERE backend_id = ?",
                (backend_id,),
            ).fetchone()
        return self._row_to_backend_entry(row) if row else None

    def get_role_backend(self, role: BackendRole) -> BackendRegistryEntry | None:
        """Вернуть backend, выбранный для конкретной inference role."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT b.*
                FROM role_backend_binding rb
                JOIN backend_registry b ON b.backend_id = rb.backend_id
                WHERE rb.role = ?
                LIMIT 1
                """,
                (role,),
            ).fetchone()
        return self._row_to_backend_entry(row) if row else None

    def upsert_repo(
        self,
        repo_root: str,
        *,
        repo_key: str | None = None,
        display_name: str | None = None,
        status: RepoStatus = "registered",
        active: bool = False,
        index_profile: str,
        include_globs: list[str],
        doc_prefixes: list[str],
        exclude_globs: list[str],
        last_full_build_ts: str | None = None,
        last_incremental_update_ts: str | None = None,
        indexed_branch: str | None = None,
        indexed_commit_hash: str | None = None,
        last_error: str | None = None,
        watch_enabled: bool = False,
        watch_running: bool = False,
        code_points_count: int = 0,
        docs_points_count: int = 0,
    ) -> RepoRegistryEntry:
        """Создать или обновить repo registry entry."""

        normalized_root = normalize_repo_identity(repo_root)
        actual_repo_key = repo_key or build_repo_key(normalized_root)
        normalized_path_text = normalized_root.rstrip("/").replace("\\", "/")
        actual_display_name = display_name or normalized_path_text.split("/")[-1] or actual_repo_key
        updated_at = _utc_now()

        with self._connect() as connection:
            existing = connection.execute(
                "SELECT created_at FROM repo_registry WHERE repo_root = ?",
                (normalized_root,),
            ).fetchone()
            created_at = existing["created_at"] if existing else updated_at

            if active:
                connection.execute("UPDATE repo_registry SET active = 0")

            connection.execute(
                """
                INSERT INTO repo_registry (
                    repo_root,
                    repo_key,
                    display_name,
                    status,
                    active,
                    index_profile,
                    include_globs_json,
                    doc_prefixes_json,
                    exclude_globs_json,
                    last_full_build_ts,
                    last_incremental_update_ts,
                    indexed_branch,
                    indexed_commit_hash,
                    last_error,
                    watch_enabled,
                    watch_running,
                    code_points_count,
                    docs_points_count,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(repo_root) DO UPDATE SET
                    repo_key = excluded.repo_key,
                    display_name = excluded.display_name,
                    status = excluded.status,
                    active = excluded.active,
                    index_profile = excluded.index_profile,
                    include_globs_json = excluded.include_globs_json,
                    doc_prefixes_json = excluded.doc_prefixes_json,
                    exclude_globs_json = excluded.exclude_globs_json,
                    last_full_build_ts = excluded.last_full_build_ts,
                    last_incremental_update_ts = excluded.last_incremental_update_ts,
                    indexed_branch = excluded.indexed_branch,
                    indexed_commit_hash = excluded.indexed_commit_hash,
                    last_error = excluded.last_error,
                    watch_enabled = excluded.watch_enabled,
                    watch_running = excluded.watch_running,
                    code_points_count = excluded.code_points_count,
                    docs_points_count = excluded.docs_points_count,
                    updated_at = excluded.updated_at
                """,
                (
                    normalized_root,
                    actual_repo_key,
                    actual_display_name,
                    status,
                    1 if active else 0,
                    index_profile,
                    json.dumps(include_globs, ensure_ascii=False),
                    json.dumps(doc_prefixes, ensure_ascii=False),
                    json.dumps(exclude_globs, ensure_ascii=False),
                    last_full_build_ts,
                    last_incremental_update_ts,
                    indexed_branch,
                    indexed_commit_hash,
                    last_error,
                    1 if watch_enabled else 0,
                    1 if watch_running else 0,
                    code_points_count,
                    docs_points_count,
                    created_at,
                    updated_at,
                ),
            )

        entry = self.get_repo(normalized_root)
        if entry is None:
            raise RuntimeError(f"Failed to upsert repo registry entry for {normalized_root}")
        return entry

    def upsert_backend(
        self,
        *,
        backend_id: str,
        role: BackendRole,
        backend_type: str,
        transport: str,
        endpoint: str | None = None,
        managed_by_service: bool = False,
        autostart_policy: str = "manual",
        health_path: str | None = None,
        location_type: str = "unknown",
        device_type: str = "unknown",
        model_name: str | None = None,
        config_blob: dict[str, Any] | None = None,
    ) -> BackendRegistryEntry:
        """Создать или обновить backend registry entry."""

        updated_at = _utc_now()
        payload = config_blob or {}

        with self._connect() as connection:
            existing = connection.execute(
                "SELECT created_at FROM backend_registry WHERE backend_id = ?",
                (backend_id,),
            ).fetchone()
            created_at = existing["created_at"] if existing else updated_at
            connection.execute(
                """
                INSERT INTO backend_registry (
                    backend_id,
                    role,
                    backend_type,
                    transport,
                    endpoint,
                    managed_by_service,
                    autostart_policy,
                    health_path,
                    location_type,
                    device_type,
                    model_name,
                    config_blob_json,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(backend_id) DO UPDATE SET
                    role = excluded.role,
                    backend_type = excluded.backend_type,
                    transport = excluded.transport,
                    endpoint = excluded.endpoint,
                    managed_by_service = excluded.managed_by_service,
                    autostart_policy = excluded.autostart_policy,
                    health_path = excluded.health_path,
                    location_type = excluded.location_type,
                    device_type = excluded.device_type,
                    model_name = excluded.model_name,
                    config_blob_json = excluded.config_blob_json,
                    updated_at = excluded.updated_at
                """,
                (
                    backend_id,
                    role,
                    backend_type,
                    transport,
                    endpoint,
                    1 if managed_by_service else 0,
                    autostart_policy,
                    health_path,
                    location_type,
                    device_type,
                    model_name,
                    json.dumps(payload, ensure_ascii=False),
                    created_at,
                    updated_at,
                ),
            )

        entry = self.get_backend(backend_id)
        if entry is None:
            raise RuntimeError(f"Failed to upsert backend registry entry for {backend_id}")
        return entry

    def delete_backend(self, backend_id: str) -> bool:
        """Удалить backend entry, если он не выбран ни для одной роли."""

        with self._connect() as connection:
            bound = connection.execute(
                "SELECT 1 FROM role_backend_binding WHERE backend_id = ? LIMIT 1",
                (backend_id,),
            ).fetchone()
            if bound is not None:
                return False
            cursor = connection.execute(
                "DELETE FROM backend_registry WHERE backend_id = ?",
                (backend_id,),
            )
        return bool(cursor.rowcount)

    def set_role_backend(self, role: BackendRole, backend_id: str) -> BackendRegistryEntry:
        """Назначить backend для конкретной inference role."""

        entry = self.get_backend(backend_id)
        if entry is None:
            raise RuntimeError(f"Backend is not registered: {backend_id}")
        if entry.role != role:
            raise RuntimeError(
                f"Backend role mismatch: backend_id={backend_id} stores role={entry.role}, requested={role}"
            )

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO role_backend_binding (role, backend_id, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(role) DO UPDATE SET
                    backend_id = excluded.backend_id,
                    updated_at = excluded.updated_at
                """,
                (role, backend_id, _utc_now()),
            )

        selected = self.get_role_backend(role)
        if selected is None:
            raise RuntimeError(f"Failed to bind backend '{backend_id}' to role '{role}'")
        return selected

    def set_active_repo(self, repo_root: str) -> RepoRegistryEntry:
        """Сделать repo активным, не меняя остальную metadata."""

        normalized_root = normalize_repo_identity(repo_root)
        entry = self.get_repo(normalized_root)
        if entry is None:
            raise RuntimeError(f"Repo is not registered: {normalized_root}")
        return self.upsert_repo(
            normalized_root,
            repo_key=entry.repo_key,
            display_name=entry.display_name,
            status=entry.status,
            active=True,
            index_profile=entry.index_profile,
            include_globs=entry.include_globs,
            doc_prefixes=entry.doc_prefixes,
            exclude_globs=entry.exclude_globs,
            last_full_build_ts=entry.last_full_build_ts,
            last_incremental_update_ts=entry.last_incremental_update_ts,
            indexed_branch=entry.indexed_branch,
            indexed_commit_hash=entry.indexed_commit_hash,
            last_error=entry.last_error,
            watch_enabled=entry.watch_enabled,
            watch_running=entry.watch_running,
            code_points_count=entry.code_points_count,
            docs_points_count=entry.docs_points_count,
        )

    def ensure_registered(
        self,
        repo_root: str,
        *,
        index_profile: str,
        include_globs: list[str],
        doc_prefixes: list[str],
        exclude_globs: list[str],
        active: bool = False,
    ) -> RepoRegistryEntry:
        """Гарантировать наличие repo entry без потери existing metadata."""

        existing = self.get_repo(repo_root)
        if existing is not None:
            if active and not existing.active:
                return self.set_active_repo(repo_root)
            return existing
        return self.upsert_repo(
            repo_root,
            status="registered",
            active=active,
            index_profile=index_profile,
            include_globs=include_globs,
            doc_prefixes=doc_prefixes,
            exclude_globs=exclude_globs,
        )

    def prune_legacy_repos(self, *, force_remove_active: bool = False) -> dict[str, Any]:
        """Удалить legacy repo entries безопасно, не задевая активный repo по умолчанию."""

        legacy_entries = self.list_legacy_repos()
        removed: list[str] = []
        skipped_active: list[str] = []

        with self._connect() as connection:
            for entry in legacy_entries:
                if entry.active and not force_remove_active:
                    skipped_active.append(entry.repo_root)
                    continue
                connection.execute(
                    "DELETE FROM repo_registry WHERE repo_root = ?",
                    (entry.repo_root,),
                )
                removed.append(entry.repo_root)

        return {
            "removed": removed,
            "skipped_active": skipped_active,
            "remaining_legacy": [entry.repo_root for entry in self.list_legacy_repos()],
        }

    def status_summary(self) -> dict[str, Any]:
        """Вернуть краткий registry summary для diagnostic endpoints."""

        repos = self.list_repos()
        legacy_repos = [repo.repo_root for repo in repos if repo.repo_root in LEGACY_REPO_ROOTS]
        active = next((repo for repo in repos if repo.active), None)
        selected_backends: dict[str, str] = {}
        for role in ("embedding", "reranker", "colbert", "llm_helper"):
            selected = self.get_role_backend(role)  # type: ignore[arg-type]
            if selected is not None:
                selected_backends[role] = selected.backend_id
        return {
            "db_path": str(self._db_path),
            "repos_count": len(repos),
            "active_repo_root": active.repo_root if active else None,
            "legacy_repos_count": len(legacy_repos),
            "legacy_repo_roots": legacy_repos,
            "backends_count": len(self.list_backends()),
            "selected_backends": selected_backends,
        }
