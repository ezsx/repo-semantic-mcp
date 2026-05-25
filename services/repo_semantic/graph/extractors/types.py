"""Shared types and helpers for deterministic graph extractors."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re

from services.repo_semantic.contracts.common import ChunkRecord
from services.repo_semantic.graph.artifacts import stable_json_hash
from services.repo_semantic.lexical import analyze_sparse_text, sparse_unique_terms

MAX_CHUNK_TERMS = 64


@dataclass(slots=True)
class GraphExtractionRows:
    nodes: list[dict[str, object]] = field(default_factory=list)
    node_terms: list[dict[str, object]] = field(default_factory=list)
    node_chunks: list[dict[str, object]] = field(default_factory=list)
    edges: list[dict[str, object]] = field(default_factory=list)
    warning_codes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GraphExtractionContext:
    chunks: list[ChunkRecord]
    chunks_by_path: dict[str, list[ChunkRecord]]
    known_files: set[str]
    known_symbols: dict[tuple[str, str], ChunkRecord]
    module_by_path: dict[str, str]
    path_by_module: dict[str, str]
    route_nodes_by_route: dict[str, list[str]] = field(default_factory=dict)

    def first_chunk_for_path(self, relative_path: str) -> ChunkRecord | None:
        chunks = self.chunks_by_path.get(normalize_path(relative_path), [])
        return chunks[0] if chunks else None

    def chunk_for_line(self, relative_path: str, line: int | None) -> ChunkRecord | None:
        chunks = self.chunks_by_path.get(normalize_path(relative_path), [])
        if not chunks:
            return None
        if line is None:
            return chunks[0]
        for chunk in chunks:
            if chunk.start_line <= line <= chunk.end_line:
                return chunk
        return chunks[0]


def build_context(chunks: list[ChunkRecord]) -> GraphExtractionContext:
    chunks_by_path: dict[str, list[ChunkRecord]] = {}
    known_symbols: dict[tuple[str, str], ChunkRecord] = {}
    module_by_path: dict[str, str] = {}
    path_by_module: dict[str, str] = {}
    for chunk in chunks:
        relative_path = normalize_path(chunk.relative_path)
        chunks_by_path.setdefault(relative_path, []).append(chunk)
        if chunk.symbol_path:
            known_symbols[(relative_path, chunk.symbol_path)] = chunk
    for rows in chunks_by_path.values():
        rows.sort(key=lambda chunk: (chunk.start_line, chunk.point_id))
    known_files = set(chunks_by_path)
    for relative_path in sorted(known_files):
        module = module_name_for_path(relative_path)
        if module:
            module_by_path[relative_path] = module
            existing = path_by_module.get(module)
            if existing is None or _module_path_priority(relative_path) < _module_path_priority(existing):
                path_by_module[module] = relative_path
    return GraphExtractionContext(
        chunks=chunks,
        chunks_by_path=chunks_by_path,
        known_files=known_files,
        known_symbols=known_symbols,
        module_by_path=module_by_path,
        path_by_module=path_by_module,
    )


def source_text_for_chunks(chunks: list[ChunkRecord]) -> str:
    """Reconstruct a sparse source view while preserving original line offsets."""

    lines: list[str] = []
    current_line = 1
    for chunk in sorted(chunks, key=lambda item: (item.start_line, item.end_line, item.point_id)):
        if chunk.start_line > current_line:
            lines.extend("" for _ in range(chunk.start_line - current_line))
            current_line = chunk.start_line
        chunk_lines = chunk.text.splitlines() or [chunk.text]
        lines.extend(chunk_lines)
        current_line = max(current_line, chunk.start_line + len(chunk_lines))
        if chunk.end_line >= current_line:
            lines.extend("" for _ in range(chunk.end_line - current_line + 1))
            current_line = chunk.end_line + 1
    return "\n".join(lines)


def normalize_path(relative_path: str) -> str:
    return relative_path.replace("\\", "/").strip("/")


def payload_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def node_id(node_type: str, *parts: object) -> str:
    key = "|".join(str(part) for part in parts if part is not None)
    return f"{node_type}:{stable_json_hash([node_type, key])[:24]}"


def edge_id(source_node_id: str, target_node_id: str, edge_type: str) -> str:
    return f"edge:{stable_json_hash([source_node_id, target_node_id, edge_type])[:32]}"


def file_node_id(relative_path: str) -> str:
    return node_id("file", normalize_path(relative_path))


def chunk_node_id(chunk_id: str) -> str:
    return node_id("chunk", chunk_id)


def module_node_id(module_name: str) -> str:
    return node_id("module", module_name)


def symbol_node_id(relative_path: str, symbol_path: str) -> str:
    return node_id("symbol", normalize_path(relative_path), symbol_path)


def doc_section_node_id(relative_path: str, heading_path: str) -> str:
    return node_id("doc_section", normalize_path(relative_path), heading_path)


def route_node_id(route_kind: str, method: str | None, normalized_route: str) -> str:
    return node_id("route", route_key(route_kind=route_kind, method=method, normalized_route=normalized_route))


def test_case_node_id(relative_path: str, test_name: str) -> str:
    return node_id("test_case", normalize_path(relative_path), test_name)


def module_name_for_path(relative_path: str) -> str | None:
    path = normalize_path(relative_path)
    if not path.endswith(".py"):
        return None
    stem = path[:-3]
    if stem.endswith("/__init__"):
        stem = stem[: -len("/__init__")]
    parts = [part for part in stem.split("/") if part and part not in {"src"}]
    if not parts:
        return None
    return ".".join(parts)


def _module_path_priority(relative_path: str) -> tuple[int, str]:
    path = normalize_path(relative_path)
    return (0 if path.endswith("/__init__.py") else 1, path)


def route_key(*, route_kind: str, method: str | None, normalized_route: str) -> str:
    return f"{route_kind}|{(method or 'ANY').upper()}|{normalized_route}"


def normalize_route(surface: str) -> str:
    value = surface.strip().strip("\"'")
    if not value:
        return value
    value = re.sub(r"<(?:[^:<>]+:)?([^<>]+)>", r"{\1}", value)
    value = re.sub(r":([A-Za-z_][A-Za-z0-9_]*)", r"{\1}", value)
    value = re.sub(r"/+", "/", value)
    if not value.startswith("/") and not re.match(r"^[a-zA-Z]+:", value):
        value = "/" + value
    return value.rstrip("/") or "/"


def term_rows(
    *,
    node_id_value: str,
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
                "node_id": node_id_value,
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


def path_term_rows(*, node_id_value: str, relative_path: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    normalized = normalize_path(relative_path)
    for term in sparse_unique_terms(normalized):
        rows.append(
            {
                "node_id": node_id_value,
                "term": term,
                "normalized_term": term.lower(),
                "term_type": "path",
                "case_sensitive": False,
                "source": "path",
            }
        )
    return rows


def route_term_rows(
    *,
    node_id_value: str,
    surface: str,
    normalized_route: str,
    method: str | None,
    source: str,
) -> list[dict[str, object]]:
    terms = [surface, normalized_route]
    if method:
        terms.append(f"{method.upper()} {normalized_route}")
    terms.append(f"ANY {normalized_route}")
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for term in terms:
        canonical = term.lower()
        if canonical in seen:
            continue
        seen.add(canonical)
        rows.append(
            {
                "node_id": node_id_value,
                "term": term,
                "normalized_term": canonical,
                "term_type": "route",
                "case_sensitive": False,
                "source": source,
            }
        )
    return rows


def confidence_value(label: str) -> float:
    return {"high": 1.0, "medium": 0.7, "low": 0.4}.get(label, 0.5)
