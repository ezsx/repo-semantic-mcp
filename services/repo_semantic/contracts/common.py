"""Shared public contract primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SearchScope = Literal["all", "code", "docs"]
ChunkScope = Literal["code", "docs"]
SearchMode = Literal["semantic", "hybrid"]
RepoContextRoute = Literal["auto", "semantic", "hybrid", "exact_handoff"]
GraphMode = Literal["off", "auto", "expand"]
GraphNodeType = Literal[
    "file",
    "chunk",
    "symbol",
    "doc_section",
    "config_item",
    "env_var",
    "policy_scope",
    "module",
    "route",
    "test_case",
]
GraphEdgeType = Literal[
    "file_contains_chunk",
    "chunk_defines_symbol",
    "file_contains_doc_section",
    "chunk_mentions_path_or_symbol",
    "config_defines_env",
    "code_reads_env",
    "doc_references_symbol",
    "doc_references_config",
    "policy_applies_to_path",
    "file_defines_module",
    "module_imports_module",
    "file_imports_file",
    "route_defined_in_chunk",
    "route_handled_by_symbol",
    "test_targets_file",
    "test_targets_symbol",
    "doc_references_route",
    "doc_references_path",
    "doc_references_env",
]
EvidenceReason = Literal[
    "shared_graph_node",
    "direct_graph_edge",
    "term_bound_expansion",
    "file_containment",
]
GraphState = Literal[
    "missing",
    "ready",
    "partial",
    "stale",
    "incompatible",
    "building",
    "error",
]
SnippetMode = Literal["chunk_start", "query_centered"]
MatchType = Literal["semantic", "hybrid", "lexical", "exact"]
OriginBranch = Literal["dense", "sparse", "legacy_lexical", "graph"]
RepoStatus = Literal["registered", "indexed", "stale", "indexing", "error", "disabled"]
BackendRole = Literal["embedding", "reranker", "colbert", "llm_helper"]
StatusSeverity = Literal["info", "warning", "blocking"]
CompatibilityState = Literal["compatible", "incompatible", "unknown"]
FreshnessSeverity = Literal["none", "info", "warning", "blocking"]


@dataclass(slots=True)
class ChunkRecord:
    """Нормализованный чанк, который индексируется в vector store."""

    point_id: str
    scope: ChunkScope
    relative_path: str
    language: str
    chunk_type: str
    text: str
    start_line: int
    end_line: int
    content_hash: str
    source_mtime: float
    symbol_path: str | None = None
    heading_path: str | None = None
    domain_tags: list[str] = field(default_factory=list)
    is_generated: bool = False
    extra: dict[str, str] = field(default_factory=dict)
