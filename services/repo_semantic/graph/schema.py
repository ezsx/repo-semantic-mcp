"""SQLite schema for the repo graph artifact."""

from __future__ import annotations

import sqlite3

GRAPH_SCHEMA_VERSION = 1


def ensure_graph_schema(connection: sqlite3.Connection) -> None:
    """Create or migrate the graph artifact schema."""

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS graph_metadata (
            repo_id TEXT PRIMARY KEY,
            repo_root TEXT NOT NULL,
            profile TEXT NOT NULL,
            schema_version INTEGER NOT NULL,
            built_commit TEXT,
            built_at TEXT NOT NULL,
            include_rules_hash TEXT,
            exclude_rules_hash TEXT,
            extractor_versions_json TEXT NOT NULL,
            source_index_contract_json TEXT NOT NULL,
            graph_contract_hash TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS graph_files (
            relative_path TEXT PRIMARY KEY,
            content_hash TEXT NOT NULL,
            graph_input_signature TEXT,
            mtime REAL,
            graph_state TEXT NOT NULL,
            indexed_at TEXT,
            last_extracted_at TEXT,
            error_code TEXT,
            error_detail TEXT
        );

        CREATE TABLE IF NOT EXISTS graph_nodes (
            node_id TEXT PRIMARY KEY,
            node_type TEXT NOT NULL,
            key TEXT NOT NULL,
            relative_path TEXT,
            chunk_id TEXT,
            start_line INTEGER,
            end_line INTEGER,
            payload_json TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS graph_node_terms (
            node_id TEXT NOT NULL,
            term TEXT NOT NULL,
            normalized_term TEXT NOT NULL,
            term_type TEXT NOT NULL,
            case_sensitive INTEGER NOT NULL DEFAULT 0,
            source TEXT NOT NULL,
            PRIMARY KEY (node_id, normalized_term, term_type, source),
            FOREIGN KEY(node_id) REFERENCES graph_nodes(node_id)
        );

        CREATE TABLE IF NOT EXISTS graph_node_chunks (
            node_id TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            relation TEXT NOT NULL,
            weight REAL NOT NULL DEFAULT 1.0,
            PRIMARY KEY (node_id, chunk_id, relation),
            FOREIGN KEY(node_id) REFERENCES graph_nodes(node_id)
        );

        CREATE TABLE IF NOT EXISTS graph_edges (
            edge_id TEXT PRIMARY KEY,
            source_node_id TEXT NOT NULL,
            target_node_id TEXT NOT NULL,
            edge_type TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 1.0,
            extractor TEXT NOT NULL,
            relative_path TEXT,
            payload_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY(source_node_id) REFERENCES graph_nodes(node_id),
            FOREIGN KEY(target_node_id) REFERENCES graph_nodes(node_id)
        );

        CREATE INDEX IF NOT EXISTS idx_graph_nodes_type_key
            ON graph_nodes(node_type, key);
        CREATE INDEX IF NOT EXISTS idx_graph_nodes_relative_path
            ON graph_nodes(relative_path);
        CREATE INDEX IF NOT EXISTS idx_graph_nodes_chunk_id
            ON graph_nodes(chunk_id);
        CREATE INDEX IF NOT EXISTS idx_graph_node_terms_normalized
            ON graph_node_terms(normalized_term, term_type);
        CREATE INDEX IF NOT EXISTS idx_graph_node_chunks_chunk_id
            ON graph_node_chunks(chunk_id);
        CREATE INDEX IF NOT EXISTS idx_graph_edges_source_type
            ON graph_edges(source_node_id, edge_type);
        CREATE INDEX IF NOT EXISTS idx_graph_edges_target_type
            ON graph_edges(target_node_id, edge_type);
        CREATE INDEX IF NOT EXISTS idx_graph_edges_relative_path
            ON graph_edges(relative_path);
        CREATE INDEX IF NOT EXISTS idx_graph_files_path_hash
            ON graph_files(relative_path, content_hash);
        """
    )
    _ensure_column(connection, "graph_files", "graph_input_signature", "TEXT")
    connection.execute(f"PRAGMA user_version = {GRAPH_SCHEMA_VERSION}")


def _ensure_column(
    connection: sqlite3.Connection,
    table_name: str,
    column_name: str,
    definition: str,
) -> None:
    columns = {
        str(row[1])
        for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    if column_name not in columns:
        connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")
