"""SQLite schema helpers for the repo registry artifact."""

from __future__ import annotations

from datetime import datetime, timezone
import sqlite3

from services.repo_semantic.config import build_repo_key, normalize_repo_identity


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    """Return columns for an existing SQLite table."""

    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row[1]) for row in rows}


def ensure_repo_registry_columns(connection: sqlite3.Connection) -> None:
    """Add missing columns to an existing repo_registry table."""

    columns = table_columns(connection, "repo_registry")
    required_columns = {
        "repo_root": "TEXT",
        "repo_key": "TEXT NOT NULL DEFAULT ''",
        "display_name": "TEXT NOT NULL DEFAULT ''",
        "status": "TEXT NOT NULL DEFAULT 'registered'",
        "active": "INTEGER NOT NULL DEFAULT 0",
        "index_profile": "TEXT NOT NULL DEFAULT ''",
        "include_globs_json": "TEXT NOT NULL DEFAULT '[]'",
        "doc_prefixes_json": "TEXT NOT NULL DEFAULT '[]'",
        "exclude_globs_json": "TEXT NOT NULL DEFAULT '[]'",
        "last_full_build_ts": "TEXT",
        "last_incremental_update_ts": "TEXT",
        "indexed_branch": "TEXT",
        "indexed_commit_hash": "TEXT",
        "last_error": "TEXT",
        "watch_enabled": "INTEGER NOT NULL DEFAULT 0",
        "watch_running": "INTEGER NOT NULL DEFAULT 0",
        "code_points_count": "INTEGER NOT NULL DEFAULT 0",
        "docs_points_count": "INTEGER NOT NULL DEFAULT 0",
        "created_at": "TEXT NOT NULL DEFAULT ''",
        "updated_at": "TEXT NOT NULL DEFAULT ''",
    }
    for column, ddl in required_columns.items():
        if column not in columns:
            connection.execute(f"ALTER TABLE repo_registry ADD COLUMN {column} {ddl}")

    now = _utc_now()
    rows = connection.execute("SELECT repo_root, repo_key, display_name, created_at, updated_at FROM repo_registry").fetchall()
    for row in rows:
        repo_root = row["repo_root"]
        repo_key = row["repo_key"] or build_repo_key(repo_root)
        display_name = row["display_name"] or normalize_repo_identity(repo_root).rstrip("/").split("/")[-1]
        created_at = row["created_at"] or now
        updated_at = row["updated_at"] or now
        connection.execute(
            """
            UPDATE repo_registry
            SET repo_key = ?,
                display_name = ?,
                created_at = ?,
                updated_at = ?
            WHERE repo_root = ?
            """,
            (repo_key, display_name, created_at, updated_at, repo_root),
        )


def ensure_backend_tables(connection: sqlite3.Connection) -> None:
    """Create and migrate backend registry tables."""

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS backend_registry (
            backend_id TEXT PRIMARY KEY,
            role TEXT NOT NULL,
            backend_type TEXT NOT NULL,
            transport TEXT NOT NULL,
            endpoint TEXT,
            managed_by_service INTEGER NOT NULL DEFAULT 0,
            autostart_policy TEXT NOT NULL DEFAULT 'manual',
            health_path TEXT,
            location_type TEXT NOT NULL DEFAULT 'unknown',
            device_type TEXT NOT NULL DEFAULT 'unknown',
            model_name TEXT,
            config_blob_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_backend_registry_role
            ON backend_registry(role);
        CREATE TABLE IF NOT EXISTS role_backend_binding (
            role TEXT PRIMARY KEY,
            backend_id TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(backend_id) REFERENCES backend_registry(backend_id)
        );
        """
    )

    backend_columns = table_columns(connection, "backend_registry")
    required_backend_columns = {
        "backend_id": "TEXT",
        "role": "TEXT NOT NULL DEFAULT 'embedding'",
        "backend_type": "TEXT NOT NULL DEFAULT ''",
        "transport": "TEXT NOT NULL DEFAULT ''",
        "endpoint": "TEXT",
        "managed_by_service": "INTEGER NOT NULL DEFAULT 0",
        "autostart_policy": "TEXT NOT NULL DEFAULT 'manual'",
        "health_path": "TEXT",
        "location_type": "TEXT NOT NULL DEFAULT 'unknown'",
        "device_type": "TEXT NOT NULL DEFAULT 'unknown'",
        "model_name": "TEXT",
        "config_blob_json": "TEXT NOT NULL DEFAULT '{}'",
        "created_at": "TEXT NOT NULL DEFAULT ''",
        "updated_at": "TEXT NOT NULL DEFAULT ''",
    }
    for column, ddl in required_backend_columns.items():
        if column not in backend_columns:
            connection.execute(f"ALTER TABLE backend_registry ADD COLUMN {column} {ddl}")

    binding_columns = table_columns(connection, "role_backend_binding")
    required_binding_columns = {
        "role": "TEXT",
        "backend_id": "TEXT NOT NULL DEFAULT ''",
        "updated_at": "TEXT NOT NULL DEFAULT ''",
    }
    for column, ddl in required_binding_columns.items():
        if column not in binding_columns:
            connection.execute(f"ALTER TABLE role_backend_binding ADD COLUMN {column} {ddl}")


def ensure_lifecycle_tables(connection: sqlite3.Connection) -> None:
    """Create lifecycle lease and audit tables."""

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS lifecycle_lease (
            lease_key TEXT PRIMARY KEY,
            lease_id TEXT NOT NULL,
            audit_id TEXT NOT NULL,
            repo_root TEXT NOT NULL,
            index_profile TEXT NOT NULL,
            backend_id TEXT NOT NULL,
            operation TEXT NOT NULL,
            owner_id TEXT NOT NULL,
            idempotency_key TEXT,
            idempotency_fingerprint TEXT,
            acquired_at TEXT NOT NULL,
            heartbeat_at TEXT NOT NULL,
            expires_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_lifecycle_lease_repo
            ON lifecycle_lease(repo_root, index_profile, backend_id);

        CREATE TABLE IF NOT EXISTS lifecycle_audit (
            audit_id TEXT PRIMARY KEY,
            lease_id TEXT NOT NULL,
            lease_key TEXT NOT NULL,
            repo_root TEXT NOT NULL,
            index_profile TEXT NOT NULL,
            backend_id TEXT NOT NULL,
            operation TEXT NOT NULL,
            owner_id TEXT NOT NULL,
            idempotency_key TEXT,
            idempotency_fingerprint TEXT,
            status TEXT NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            detail_json TEXT NOT NULL DEFAULT '{}',
            error TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_lifecycle_audit_repo
            ON lifecycle_audit(repo_root, index_profile, backend_id, started_at);
        """
    )
    lease_columns = table_columns(connection, "lifecycle_lease")
    if "idempotency_fingerprint" not in lease_columns:
        connection.execute("ALTER TABLE lifecycle_lease ADD COLUMN idempotency_fingerprint TEXT")
    audit_columns = table_columns(connection, "lifecycle_audit")
    if "idempotency_fingerprint" not in audit_columns:
        connection.execute("ALTER TABLE lifecycle_audit ADD COLUMN idempotency_fingerprint TEXT")


def ensure_registry_schema(connection: sqlite3.Connection) -> None:
    """Create or migrate repo registry schema."""

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS repo_registry (
            repo_root TEXT PRIMARY KEY,
            repo_key TEXT NOT NULL,
            display_name TEXT NOT NULL,
            status TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 0,
            index_profile TEXT NOT NULL,
            include_globs_json TEXT NOT NULL,
            doc_prefixes_json TEXT NOT NULL,
            exclude_globs_json TEXT NOT NULL,
            last_full_build_ts TEXT,
            last_incremental_update_ts TEXT,
            indexed_branch TEXT,
            indexed_commit_hash TEXT,
            last_error TEXT,
            watch_enabled INTEGER NOT NULL DEFAULT 0,
            watch_running INTEGER NOT NULL DEFAULT 0,
            code_points_count INTEGER NOT NULL DEFAULT 0,
            docs_points_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_repo_registry_active
            ON repo_registry(active);
        """
    )
    ensure_repo_registry_columns(connection)
    ensure_backend_tables(connection)
    ensure_lifecycle_tables(connection)
    connection.execute("PRAGMA user_version = 3")
