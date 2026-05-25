"""Markdown/docs reference graph extraction."""

from __future__ import annotations

import posixpath
import re

from services.repo_semantic.graph.extractors.types import (
    GraphExtractionContext,
    GraphExtractionRows,
    confidence_value,
    doc_section_node_id,
    edge_id,
    file_node_id,
    node_id,
    normalize_path,
    normalize_route,
    payload_json,
    symbol_node_id,
)
from services.repo_semantic.lexical import analyze_sparse_text

EXTRACTOR_NAME = "markdown_refs"
EXTRACTOR_VERSION = "markdown_refs.v1"
MAX_REFS_PER_CHUNK = 120
CODE_SPAN_RE = re.compile(r"`([^`\n]{1,240})`")
LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
ROUTE_RE = re.compile(r"(/[A-Za-z0-9_./{}:<>\-]+)")


def extract(context: GraphExtractionContext) -> GraphExtractionRows:
    rows = GraphExtractionRows()
    for chunk in context.chunks:
        if chunk.scope != "docs" and chunk.language.lower() not in {"markdown", "md", "rst"}:
            continue
        source_node = _doc_source_node(chunk)
        emitted = 0
        candidates = [
            *CODE_SPAN_RE.findall(chunk.text),
            *LINK_RE.findall(chunk.text),
            *ROUTE_RE.findall(chunk.text),
        ]
        for candidate in candidates:
            if emitted >= MAX_REFS_PER_CHUNK:
                rows.warning_codes.append("markdown_ref_cap_reached")
                break
            emitted += 1
            _emit_reference(rows=rows, context=context, source_node=source_node, chunk_path=chunk.relative_path, value=candidate)
        for token in analyze_sparse_text(chunk.text):
            if emitted >= MAX_REFS_PER_CHUNK:
                rows.warning_codes.append("markdown_ref_cap_reached")
                break
            if token.token_type != "env_var":
                continue
            emitted += 1
            env_node = node_id("env_var", token.surface)
            rows.edges.append(
                _edge(source_node, env_node, "doc_references_env", chunk.relative_path, token.surface, "medium")
            )
    return rows


def _doc_source_node(chunk) -> str:
    if chunk.heading_path:
        return doc_section_node_id(chunk.relative_path, chunk.heading_path)
    return node_id("chunk", chunk.point_id)


def _emit_reference(
    *,
    rows: GraphExtractionRows,
    context: GraphExtractionContext,
    source_node: str,
    chunk_path: str,
    value: str,
) -> None:
    normalized_value = value.strip().strip("\"'")
    if not normalized_value:
        return
    path = _resolve_markdown_path(normalized_value, chunk_path, context.known_files)
    if path is not None:
        rows.edges.append(_edge(source_node, file_node_id(path), "doc_references_path", chunk_path, value, "high"))
        return
    route = normalize_route(normalized_value)
    if route in context.route_nodes_by_route:
        for route_node in context.route_nodes_by_route[route]:
            rows.edges.append(
                _edge(
                    source_node,
                    route_node,
                    "doc_references_route",
                    chunk_path,
                    value,
                    "high",
                    metadata={"normalized": route},
                )
            )
        return
    for (symbol_path, symbol_name), _chunk in context.known_symbols.items():
        if normalized_value == symbol_name:
            rows.edges.append(
                _edge(
                    source_node,
                    symbol_node_id(symbol_path, symbol_name),
                    "doc_references_symbol",
                    chunk_path,
                    value,
                    "high",
                )
            )
            return


def _resolve_markdown_path(value: str, chunk_path: str, known_files: set[str]) -> str | None:
    raw = value.split("#", 1)[0].split("?", 1)[0].strip()
    if not raw or raw.startswith("#") or "://" in raw or raw.startswith(("mailto:", "tel:")):
        return None
    normalized = normalize_path(posixpath.normpath(raw.lstrip("/")))
    candidates = [normalized]
    base_dir = posixpath.dirname(normalize_path(chunk_path))
    if base_dir and not raw.startswith("/"):
        candidates.append(normalize_path(posixpath.normpath(posixpath.join(base_dir, raw))))
    for candidate in dict.fromkeys(candidates):
        if candidate in known_files:
            return candidate
    return None


def _edge(
    source_node_id: str,
    target_node_id: str,
    edge_type: str,
    relative_path: str,
    surface: str,
    confidence: str,
    metadata: dict[str, str] | None = None,
) -> dict[str, object]:
    payload = {
        "confidence": confidence,
        "source": "markdown",
        "extractor_version": EXTRACTOR_VERSION,
        "surface": surface,
        **(metadata or {}),
    }
    return {
        "edge_id": edge_id(source_node_id, target_node_id, edge_type),
        "source_node_id": source_node_id,
        "target_node_id": target_node_id,
        "edge_type": edge_type,
        "confidence": confidence_value(confidence),
        "extractor": EXTRACTOR_NAME,
        "relative_path": relative_path,
        "payload_json": payload_json(payload),
    }
