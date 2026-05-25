"""Internal graph retrieval types."""

from __future__ import annotations

from dataclasses import dataclass, field

from services.repo_semantic.contracts.common import ChunkRecord, GraphEdgeType, GraphNodeType
from services.repo_semantic.contracts.repo_context import EvidencePath
from services.repo_semantic.contracts.repo_context import GraphBranchDiagnostics, RecommendedAction
from services.repo_semantic.contracts.search import SearchWarning


@dataclass(frozen=True, slots=True)
class GraphNode:
    node_id: str
    node_type: GraphNodeType
    key: str
    relative_path: str | None
    chunk_id: str | None
    start_line: int | None
    end_line: int | None


@dataclass(frozen=True, slots=True)
class GraphEdge:
    edge_id: str
    source_node_id: str
    target_node_id: str
    edge_type: GraphEdgeType
    confidence: float
    extractor: str
    relative_path: str | None
    payload_json: str
    neighbor_node_id: str
    neighbor_node_type: GraphNodeType
    neighbor_key: str
    neighbor_relative_path: str | None
    neighbor_chunk_id: str | None
    neighbor_start_line: int | None
    neighbor_end_line: int | None


@dataclass(frozen=True, slots=True)
class GraphChunkLink:
    node_id: str
    chunk_id: str
    relation: str
    weight: float
    scope: str
    relative_path: str
    start_line: int
    end_line: int


@dataclass(frozen=True, slots=True)
class GraphSeed:
    result_index: int
    query: str
    query_role: str
    query_weight: float
    pre_cap_rank: int
    chunk_id: str
    relative_path: str
    scope: str
    symbol_path: str | None = None
    heading_path: str | None = None
    matched_terms: tuple[str, ...] = ()


@dataclass(slots=True)
class GraphExpansionCandidate:
    chunk_id: str
    chunk: ChunkRecord
    score: float
    branch_rank: int = 0
    branch_score: float = 0.0
    seed_pre_cap_rank: int = 1_000_000
    origin_queries: list[str] = field(default_factory=list)
    origin_query_roles: dict[str, str] = field(default_factory=dict)
    matched_terms: list[str] = field(default_factory=list)
    evidence_paths: list[EvidencePath] = field(default_factory=list)


@dataclass(slots=True)
class GraphExpansionResult:
    mode_effective: str
    candidates: list[GraphExpansionCandidate] = field(default_factory=list)
    diagnostics: GraphBranchDiagnostics | None = None
    warnings: list[SearchWarning] = field(default_factory=list)
    actions: list[RecommendedAction] = field(default_factory=list)
