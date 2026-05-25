"""Build the first SQLite graph artifact from indexed chunks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json

from services.repo_semantic.contracts.common import ChunkRecord
from services.repo_semantic.graph.artifacts import stable_json_hash
from services.repo_semantic.graph.schema import GRAPH_SCHEMA_VERSION
from services.repo_semantic.graph.extractors import markdown_refs, python_imports, python_routes, test_targets
from services.repo_semantic.graph.extractors.types import build_context
from services.repo_semantic.lexical import analyze_sparse_text, sparse_unique_terms

GRAPH_EXTRACTOR_VERSIONS = {
    "artifact_schema": str(GRAPH_SCHEMA_VERSION),
    "chunk_projection": "v1",
    "symbol_extractor": "chunk_metadata.v1",
    "doc_section_extractor": "chunk_metadata.v1",
    "config_extractor": "chunk_metadata.v1",
    "env_var_extractor": "lexical_analyzer.v1",
    "python_imports": python_imports.EXTRACTOR_VERSION,
    "python_routes": python_routes.EXTRACTOR_VERSION,
    "test_targets": test_targets.EXTRACTOR_VERSION,
    "markdown_refs": markdown_refs.EXTRACTOR_VERSION,
}
MAX_CHUNK_TERMS = 64


@dataclass(slots=True)
class GraphSnapshot:
    """In-memory rows ready for GraphStore.replace_graph."""

    metadata: dict[str, object]
    files: list[dict[str, object]] = field(default_factory=list)
    nodes: list[dict[str, object]] = field(default_factory=list)
    node_terms: list[dict[str, object]] = field(default_factory=list)
    node_chunks: list[dict[str, object]] = field(default_factory=list)
    edges: list[dict[str, object]] = field(default_factory=list)
    chunks_count: int = 0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _node_id(node_type: str, *parts: object) -> str:
    key = "|".join(str(part) for part in parts if part is not None)
    return f"{node_type}:{stable_json_hash([node_type, key])[:24]}"


def _edge_id(source_node_id: str, target_node_id: str, edge_type: str) -> str:
    return f"edge:{stable_json_hash([source_node_id, target_node_id, edge_type])[:32]}"


def _normalize_path(relative_path: str) -> str:
    return relative_path.replace("\\", "/").strip("/")


def _file_node_id(relative_path: str) -> str:
    return _node_id("file", _normalize_path(relative_path))


def _chunk_node_id(chunk_id: str) -> str:
    return _node_id("chunk", chunk_id)


def _term_rows(
    *,
    node_id: str,
    text: str,
    source: str,
    limit: int = MAX_CHUNK_TERMS,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for token in analyze_sparse_text(text):
        key = (token.canonical, token.token_type, source)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "node_id": node_id,
                "term": token.surface,
                "normalized_term": token.canonical,
                "term_type": token.token_type,
                "case_sensitive": token.token_type in {"env_var", "quoted_literal"},
                "source": source,
            }
        )
        if len(rows) >= limit:
            break
    return rows


def _path_term_rows(*, node_id: str, relative_path: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    normalized = _normalize_path(relative_path)
    for term in sparse_unique_terms(normalized):
        rows.append(
            {
                "node_id": node_id,
                "term": term,
                "normalized_term": term.lower(),
                "term_type": "path",
                "case_sensitive": False,
                "source": "path",
            }
        )
    return rows


def _env_var_terms(chunk: ChunkRecord) -> list[str]:
    env_vars: list[str] = []
    seen: set[str] = set()
    for token in analyze_sparse_text(chunk.text):
        if token.token_type != "env_var":
            continue
        if token.surface in seen:
            continue
        seen.add(token.surface)
        env_vars.append(token.surface)
    return env_vars


def _is_config_chunk(chunk: ChunkRecord) -> bool:
    path = _normalize_path(chunk.relative_path).lower()
    if chunk.language in {"yaml", "toml", "json"}:
        return True
    return any(part in path for part in ("compose", ".env", "config", "settings"))


def _payload_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def chunk_graph_input_signature(chunk: ChunkRecord) -> dict[str, object]:
    """Return chunk fields that affect graph rows/evidence."""

    return {
        "scope": chunk.scope,
        "relative_path": _normalize_path(chunk.relative_path),
        "chunk_id": chunk.point_id,
        "content_hash": chunk.content_hash,
        "language": chunk.language,
        "chunk_type": chunk.chunk_type,
        "symbol_path": chunk.symbol_path,
        "heading_path": chunk.heading_path,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "domain_tags": sorted(chunk.domain_tags),
        "is_generated": chunk.is_generated,
        "extra": dict(sorted(chunk.extra.items())),
    }


def graph_input_signature_for_chunks(chunks: list[ChunkRecord]) -> str:
    """Stable per-path graph input signature."""

    return stable_json_hash(
        sorted(
            [chunk_graph_input_signature(chunk) for chunk in chunks],
            key=lambda item: (
                str(item["relative_path"]),
                str(item["chunk_id"]),
                str(item["start_line"]),
                str(item["end_line"]),
            ),
        )
    )


def build_graph_snapshot(
    *,
    repo_root: str,
    profile: str,
    chunks: list[ChunkRecord],
    built_commit: str | None,
    include_globs: list[str],
    exclude_globs: list[str],
    source_index_contract: dict[str, object],
) -> GraphSnapshot:
    """Build deterministic graph rows from the indexed chunk payloads."""

    built_at = _utc_now()
    include_rules_hash = stable_json_hash(include_globs)
    exclude_rules_hash = stable_json_hash(exclude_globs)
    graph_contract_hash = stable_json_hash(
        {
            "schema_version": GRAPH_SCHEMA_VERSION,
            "extractor_versions": GRAPH_EXTRACTOR_VERSIONS,
            "source_index_contract": source_index_contract,
            "include_rules_hash": include_rules_hash,
            "exclude_rules_hash": exclude_rules_hash,
        }
    )
    snapshot = GraphSnapshot(
        metadata={
            "repo_id": stable_json_hash([repo_root, profile]),
            "repo_root": repo_root,
            "profile": profile,
            "schema_version": GRAPH_SCHEMA_VERSION,
            "built_commit": built_commit,
            "built_at": built_at,
            "include_rules_hash": include_rules_hash,
            "exclude_rules_hash": exclude_rules_hash,
            "extractor_versions_json": _payload_json(GRAPH_EXTRACTOR_VERSIONS),
            "source_index_contract_json": _payload_json(source_index_contract),
            "graph_contract_hash": graph_contract_hash,
        },
        chunks_count=len(chunks),
    )

    file_chunk_hashes: dict[str, list[str]] = {}
    file_mtimes: dict[str, float] = {}
    nodes_by_id: dict[str, dict[str, object]] = {}
    edges_by_id: dict[str, dict[str, object]] = {}
    extractor_warning_codes: set[str] = set()

    def add_node(row: dict[str, object]) -> None:
        nodes_by_id[str(row["node_id"])] = row

    def add_edge(row: dict[str, object]) -> None:
        edges_by_id[str(row["edge_id"])] = row

    def merge_extractor_rows(rows) -> None:
        for row in rows.nodes:
            add_node(row)
        snapshot.node_terms.extend(rows.node_terms)
        snapshot.node_chunks.extend(rows.node_chunks)
        for row in rows.edges:
            add_edge(row)
        extractor_warning_codes.update(str(code) for code in rows.warning_codes)

    for chunk in chunks:
        relative_path = _normalize_path(chunk.relative_path)
        file_chunk_hashes.setdefault(relative_path, []).append(chunk.content_hash)
        file_mtimes[relative_path] = max(file_mtimes.get(relative_path, 0.0), chunk.source_mtime)

        file_node = _file_node_id(relative_path)
        chunk_node = _chunk_node_id(chunk.point_id)
        add_node(
            {
                "node_id": file_node,
                "node_type": "file",
                "key": relative_path,
                "relative_path": relative_path,
                "chunk_id": None,
                "start_line": None,
                "end_line": None,
                "payload_json": _payload_json({"scope": chunk.scope, "language": chunk.language}),
            }
        )
        snapshot.node_terms.extend(_path_term_rows(node_id=file_node, relative_path=relative_path))
        snapshot.node_chunks.append(
            {
                "node_id": file_node,
                "chunk_id": chunk.point_id,
                "relation": "containing",
                "weight": 0.6,
            }
        )

        add_node(
            {
                "node_id": chunk_node,
                "node_type": "chunk",
                "key": chunk.point_id,
                "relative_path": relative_path,
                "chunk_id": chunk.point_id,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "payload_json": _payload_json({"scope": chunk.scope, "chunk_type": chunk.chunk_type}),
            }
        )
        snapshot.node_terms.extend(
            _term_rows(node_id=chunk_node, text=chunk.text, source="chunk_text")
        )
        snapshot.node_chunks.append(
            {
                "node_id": chunk_node,
                "chunk_id": chunk.point_id,
                "relation": "primary",
                "weight": 1.0,
            }
        )
        add_edge(
            {
                "edge_id": _edge_id(file_node, chunk_node, "file_contains_chunk"),
                "source_node_id": file_node,
                "target_node_id": chunk_node,
                "edge_type": "file_contains_chunk",
                "confidence": 1.0,
                "extractor": "chunk_projection",
                "relative_path": relative_path,
                "payload_json": "{}",
            }
        )

        if chunk.symbol_path:
            symbol_node = _node_id("symbol", relative_path, chunk.symbol_path)
            add_node(
                {
                    "node_id": symbol_node,
                    "node_type": "symbol",
                    "key": chunk.symbol_path,
                    "relative_path": relative_path,
                    "chunk_id": chunk.point_id,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "payload_json": _payload_json({"language": chunk.language}),
                }
            )
            snapshot.node_terms.extend(
                _term_rows(node_id=symbol_node, text=chunk.symbol_path, source="symbol")
            )
            snapshot.node_chunks.append(
                {
                    "node_id": symbol_node,
                    "chunk_id": chunk.point_id,
                    "relation": "primary",
                    "weight": 1.0,
                }
            )
            add_edge(
                {
                    "edge_id": _edge_id(chunk_node, symbol_node, "chunk_defines_symbol"),
                    "source_node_id": chunk_node,
                    "target_node_id": symbol_node,
                    "edge_type": "chunk_defines_symbol",
                    "confidence": 1.0,
                    "extractor": "symbol_extractor",
                    "relative_path": relative_path,
                    "payload_json": "{}",
                }
            )

        if chunk.heading_path:
            section_node = _node_id("doc_section", relative_path, chunk.heading_path)
            add_node(
                {
                    "node_id": section_node,
                    "node_type": "doc_section",
                    "key": chunk.heading_path,
                    "relative_path": relative_path,
                    "chunk_id": chunk.point_id,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "payload_json": _payload_json({"language": chunk.language}),
                }
            )
            snapshot.node_terms.extend(
                _term_rows(node_id=section_node, text=chunk.heading_path, source="heading")
            )
            snapshot.node_chunks.append(
                {
                    "node_id": section_node,
                    "chunk_id": chunk.point_id,
                    "relation": "primary",
                    "weight": 1.0,
                }
            )
            add_edge(
                {
                    "edge_id": _edge_id(file_node, section_node, "file_contains_doc_section"),
                    "source_node_id": file_node,
                    "target_node_id": section_node,
                    "edge_type": "file_contains_doc_section",
                    "confidence": 1.0,
                    "extractor": "doc_section_extractor",
                    "relative_path": relative_path,
                    "payload_json": "{}",
                }
            )

        if _is_config_chunk(chunk) and chunk.symbol_path:
            config_node = _node_id("config_item", relative_path, chunk.symbol_path)
            add_node(
                {
                    "node_id": config_node,
                    "node_type": "config_item",
                    "key": chunk.symbol_path,
                    "relative_path": relative_path,
                    "chunk_id": chunk.point_id,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "payload_json": _payload_json({"language": chunk.language}),
                }
            )
            snapshot.node_terms.extend(
                _term_rows(node_id=config_node, text=chunk.symbol_path, source="config_key")
            )
            snapshot.node_chunks.append(
                {
                    "node_id": config_node,
                    "chunk_id": chunk.point_id,
                    "relation": "primary",
                    "weight": 1.0,
                }
            )
            add_edge(
                {
                    "edge_id": _edge_id(chunk_node, config_node, "chunk_mentions_path_or_symbol"),
                    "source_node_id": chunk_node,
                    "target_node_id": config_node,
                    "edge_type": "chunk_mentions_path_or_symbol",
                    "confidence": 0.8,
                    "extractor": "config_extractor",
                    "relative_path": relative_path,
                    "payload_json": "{}",
                }
            )

        for env_var in _env_var_terms(chunk):
            env_node = _node_id("env_var", env_var)
            add_node(
                {
                    "node_id": env_node,
                    "node_type": "env_var",
                    "key": env_var,
                    "relative_path": None,
                    "chunk_id": None,
                    "start_line": None,
                    "end_line": None,
                    "payload_json": "{}",
                }
            )
            snapshot.node_terms.append(
                {
                    "node_id": env_node,
                    "term": env_var,
                    "normalized_term": env_var.lower(),
                    "term_type": "env_var",
                    "case_sensitive": True,
                    "source": "env_var",
                }
            )
            snapshot.node_chunks.append(
                {
                    "node_id": env_node,
                    "chunk_id": chunk.point_id,
                    "relation": "mention",
                    "weight": 0.8,
                }
            )
            edge_type = "config_defines_env" if _is_config_chunk(chunk) else "code_reads_env"
            add_edge(
                {
                    "edge_id": _edge_id(chunk_node, env_node, edge_type),
                    "source_node_id": chunk_node,
                    "target_node_id": env_node,
                    "edge_type": edge_type,
                    "confidence": 0.8,
                    "extractor": "env_var_extractor",
                    "relative_path": relative_path,
                    "payload_json": "{}",
                }
            )

    extraction_context = build_context(chunks)
    merge_extractor_rows(python_imports.extract(extraction_context))
    merge_extractor_rows(python_routes.extract(extraction_context))
    merge_extractor_rows(test_targets.extract(extraction_context))
    merge_extractor_rows(markdown_refs.extract(extraction_context))
    if extractor_warning_codes:
        final_source_index_contract = {
            **source_index_contract,
            "extractor_warning_codes": sorted(extractor_warning_codes),
        }
        snapshot.metadata["source_index_contract_json"] = _payload_json(
            final_source_index_contract
        )
        snapshot.metadata["graph_contract_hash"] = stable_json_hash(
            {
                "schema_version": GRAPH_SCHEMA_VERSION,
                "extractor_versions": GRAPH_EXTRACTOR_VERSIONS,
                "source_index_contract": final_source_index_contract,
                "include_rules_hash": include_rules_hash,
                "exclude_rules_hash": exclude_rules_hash,
            }
        )

    for relative_path in sorted(file_chunk_hashes):
        snapshot.files.append(
            {
                "relative_path": relative_path,
                "content_hash": stable_json_hash(sorted(file_chunk_hashes[relative_path])),
                "graph_input_signature": graph_input_signature_for_chunks(
                    [
                        chunk
                        for chunk in chunks
                        if _normalize_path(chunk.relative_path) == relative_path
                    ]
                ),
                "mtime": file_mtimes.get(relative_path),
                "graph_state": "ready",
                "indexed_at": built_at,
                "last_extracted_at": built_at,
                "error_code": None,
                "error_detail": None,
            }
        )

    snapshot.nodes = list(nodes_by_id.values())
    snapshot.edges = list(edges_by_id.values())
    return snapshot


def source_contract_from_chunks(
    *,
    schema_version: int,
    profile: str,
    collections: dict[str, dict[str, object]],
    chunks: list[ChunkRecord],
    index_revision: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build the source-index contract persisted in graph metadata."""

    chunk_signature = [
        chunk_graph_input_signature(chunk)
        for chunk in chunks
    ]
    return {
        "schema_version": schema_version,
        "profile": profile,
        "collections": collections,
        "index_revision": index_revision or {},
        "indexed_files_total": len({_normalize_path(chunk.relative_path) for chunk in chunks}),
        "indexed_chunks_total": len(chunks),
        "source_revision_hash": stable_json_hash(sorted(chunk_signature, key=lambda item: (
            str(item["scope"]),
            str(item["relative_path"]),
            str(item["chunk_id"]),
            str(item["content_hash"]),
        ))),
    }
