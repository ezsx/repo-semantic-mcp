"""Bounded graph expansion for repo_context_search."""

from __future__ import annotations

from collections.abc import Callable
import json
from math import sqrt

from services.repo_semantic.contracts.common import ChunkRecord, GraphMode
from services.repo_semantic.contracts.repo_context import (
    EvidencePath,
    EvidencePathStep,
    GraphBranchDiagnostics,
    RecommendedAction,
)
from services.repo_semantic.contracts.search import ExactAnchorAnalysis, SearchFilters, SearchResult, SearchWarning
from services.repo_semantic.contracts.status import PathFreshnessSummary
from services.repo_semantic.context.query_plan import ContextQueryItem
from services.repo_semantic.graph.freshness import (
    candidate_touches_invalidated_path,
    expanded_path_states,
    graph_path_guard,
)
from services.repo_semantic.graph.types import (
    GraphEdge,
    GraphExpansionCandidate,
    GraphExpansionResult,
    GraphNode,
    GraphSeed,
)
from services.repo_semantic.lexical import analyze_sparse_text
from services.repo_semantic.retrieval.constants import MAX_EXPLANATIONS, RRF_K
from services.repo_semantic.retrieval.types import _SearchExecution

MAX_GRAPH_SEEDS = 20
MAX_GRAPH_SEED_POOL = 50
MAX_GRAPH_CANDIDATES = 200
MAX_EVIDENCE_PATHS_PER_RESULT = 3
MAX_BOUND_NODES = 160
MAX_NEIGHBORS_PER_NODE = 30
MAX_GRAPH_SEEDS_PER_FILE_FIRST_PASS = 3
MAX_GRAPH_EDGE_PROJECTIONS = MAX_GRAPH_CANDIDATES
MAX_GRAPH_CHUNK_PROJECTIONS = MAX_GRAPH_CANDIDATES

ALLOWED_EDGE_TYPES = [
    "code_reads_env",
    "config_defines_env",
    "chunk_defines_symbol",
    "file_contains_doc_section",
    "chunk_mentions_path_or_symbol",
    "file_contains_chunk",
    "file_defines_module",
    "module_imports_module",
    "file_imports_file",
    "route_defined_in_chunk",
    "route_handled_by_symbol",
    "test_targets_file",
    "test_targets_symbol",
    "doc_references_route",
    "doc_references_symbol",
    "doc_references_path",
    "doc_references_env",
    "doc_references_config",
    "policy_applies_to_path",
]

EDGE_WEIGHTS = {
    "code_reads_env": 1.0,
    "config_defines_env": 1.0,
    "chunk_defines_symbol": 0.9,
    "file_contains_doc_section": 0.8,
    "chunk_mentions_path_or_symbol": 0.75,
    "file_contains_chunk": 0.55,
    "route_handled_by_symbol": 1.1,
    "route_defined_in_chunk": 1.0,
    "test_targets_symbol": 0.95,
    "test_targets_file": 0.85,
    "doc_references_route": 0.9,
    "doc_references_symbol": 0.85,
    "doc_references_path": 0.8,
    "doc_references_env": 0.85,
    "doc_references_config": 0.8,
    "file_imports_file": 0.75,
    "module_imports_module": 0.65,
    "file_defines_module": 0.65,
    "policy_applies_to_path": 0.75,
}

SUPPORT_EDGE_TYPES = [
    "route_handled_by_symbol",
    "route_defined_in_chunk",
    "test_targets_file",
    "test_targets_symbol",
    "doc_references_route",
    "doc_references_symbol",
    "doc_references_path",
    "file_imports_file",
    "module_imports_module",
]
IMPORT_EDGE_TYPES = {"file_imports_file", "module_imports_module", "file_defines_module"}
TEST_EDGE_TYPES = {"test_targets_file", "test_targets_symbol"}
DOC_EDGE_TYPES = {
    "file_contains_doc_section",
    "doc_references_route",
    "doc_references_symbol",
    "doc_references_path",
    "doc_references_env",
}
ROUTE_EDGE_TYPES = {"route_defined_in_chunk", "route_handled_by_symbol", "doc_references_route"}
CONFIG_EDGE_TYPES = {"code_reads_env", "config_defines_env", "doc_references_env", "doc_references_config"}
GENERAL_EDGE_TYPES = {
    "chunk_defines_symbol",
    "chunk_mentions_path_or_symbol",
    "file_contains_chunk",
    "policy_applies_to_path",
}

TERM_TYPE_ALLOWLIST = {
    "env_var",
    "route",
    "path",
    "cli_flag",
    "sql_identifier",
    "quoted_literal",
    "exact",
}

RELATIONSHIP_TERMS = {
    "related",
    "where used",
    "usage",
    "uses",
    "used by",
    "tests for",
    "docs for",
    "config for",
    "deploy for",
    "policy for",
    "handler for",
    "owner",
    "flow",
    "route",
    "contract",
}
INFRA_ANCHOR_TYPES = {"env_var", "route", "path", "cli_flag", "sql_identifier", "config_key"}

ReadChunk = Callable[[str, str], ChunkRecord | None]
MatchesFilters = Callable[[ChunkRecord, SearchFilters], bool]


def expand_graph_context(
    *,
    graph_read_provider,
    graph_mode: GraphMode,
    graph_auto_enabled: bool,
    route_used: str,
    query_items: list[ContextQueryItem],
    executions: list[tuple[str, str, float, _SearchExecution]],
    exact_anchor_analysis: ExactAnchorAnalysis,
    filters: SearchFilters,
    path_freshness: PathFreshnessSummary | None = None,
    read_chunk: ReadChunk,
    matches_filters: MatchesFilters,
) -> GraphExpansionResult:
    """Expand dense/sparse seeds through the read-only graph provider."""

    if graph_mode == "off" or route_used == "exact_handoff":
        return _skipped_result(
            graph_mode=graph_mode,
            skipped_reason="graph_disabled" if graph_mode == "off" else "graph_unavailable",
        )
    if graph_read_provider is None:
        warnings = []
        if graph_mode == "expand":
            warnings.append(
                SearchWarning(
                    code="graph_branch_unavailable",
                    severity="info",
                    detail="Graph expansion is unavailable because no read-only graph provider is configured.",
                )
            )
        return _skipped_result(
            graph_mode=graph_mode,
            skipped_reason="graph_unavailable",
            warnings=warnings,
        )
    if graph_mode == "auto" and not graph_auto_enabled:
        return _skipped_result(
            graph_mode=graph_mode,
            skipped_reason="graph_auto_compat_disabled",
        )

    seed_pool = _build_seed_pool(query_items=query_items, executions=executions)
    should_auto_expand = _should_auto_expand(
        query_items=query_items,
        seed_pool=seed_pool,
        exact_anchor_analysis=exact_anchor_analysis,
    )
    if graph_mode == "auto" and not should_auto_expand:
        return _skipped_result(
            graph_mode=graph_mode,
            skipped_reason="graph_auto_suppressed",
            seed_count=len(seed_pool),
        )

    status = graph_read_provider.graph_status(verify_source_revision=False)
    status_warning_codes = [str(code) for code in getattr(status, "warning_codes", []) if str(code)]
    status_warnings = [_status_warning_for_code(code) for code in status_warning_codes]
    if not status.expansion_allowed:
        warning, action = _status_warning_action(graph_mode=graph_mode, state=status.state)
        if status.state == "stale" and _stale_status_can_update_graph(status_warning_codes):
            action = RecommendedAction(
                code="update_graph",
                severity="warning",
                title="Update graph artifact",
                detail="Graph expansion is disabled because the graph artifact needs a bounded incremental update.",
                tool_hint="update_graph",
            )
        if status.state == "stale" and "graph_invalidated_paths_unknown" in status_warning_codes:
            action = None
        return GraphExpansionResult(
            mode_effective="off",
            diagnostics=GraphBranchDiagnostics(
                requested_mode=graph_mode,
                effective_mode="off",
                available=status.available,
                used=False,
                status_state=status.state,
                expansion_allowed=status.expansion_allowed,
                skipped_reason=f"graph_{status.state}",
                seed_count=len(seed_pool),
                warning_codes=[warning.code, *status_warning_codes],
            ),
            warnings=[warning, *status_warnings],
            actions=[action] if graph_mode == "expand" and action is not None else [],
        )

    path_guard = graph_path_guard(
        status=status,
        path_freshness=path_freshness,
        seeds=seed_pool[:MAX_GRAPH_SEEDS],
    )
    if path_guard.blocked_warning is not None:
        warning_codes = [*status_warning_codes, path_guard.blocked_warning.code]
        return GraphExpansionResult(
            mode_effective="off",
            diagnostics=GraphBranchDiagnostics(
                requested_mode=graph_mode,
                effective_mode="off",
                available=status.available,
                used=False,
                status_state=status.state,
                expansion_allowed=False,
                skipped_reason=path_guard.skipped_reason,
                seed_count=len(seed_pool),
                degraded=path_guard.degraded,
                seed_path_states=path_guard.seed_path_states,
                warning_codes=warning_codes,
            ),
            warnings=[*status_warnings, path_guard.blocked_warning],
        )
    bound_nodes_by_seed = _bind_seed_nodes(
        graph_read_provider=graph_read_provider,
        seeds=seed_pool[:MAX_GRAPH_SEEDS],
        exact_anchor_analysis=exact_anchor_analysis,
    )
    bound_seed_count = sum(1 for nodes in bound_nodes_by_seed.values() if nodes)
    if bound_seed_count == 0:
        return GraphExpansionResult(
            mode_effective="off",
            diagnostics=GraphBranchDiagnostics(
                requested_mode=graph_mode,
                effective_mode="off",
                available=True,
                used=False,
                status_state=status.state,
                expansion_allowed=True,
                skipped_reason="graph_no_bound_seeds",
                seed_count=len(seed_pool),
                bound_seed_count=0,
                degraded=path_guard.degraded,
                seed_path_states=path_guard.seed_path_states,
                warning_codes=[*status_warning_codes, "graph_no_bound_seeds"],
            ),
            warnings=[
                *status_warnings,
                SearchWarning(
                    code="graph_no_bound_seeds",
                    severity="info",
                    detail="No retrieved seed chunks could be bound to graph nodes.",
                )
            ],
        )

    expansion_stats = {"before_filter": 0, "after_filter": 0}
    query_intent = _query_intent(query_items, exact_anchor_analysis)
    candidates = _expand_candidates(
        graph_read_provider=graph_read_provider,
        seeds=seed_pool[:MAX_GRAPH_SEEDS],
        bound_nodes_by_seed=bound_nodes_by_seed,
        exact_anchor_analysis=exact_anchor_analysis,
        graph_mode=graph_mode,
        query_intent=query_intent,
        read_chunk=read_chunk,
        matches_filters=matches_filters,
        filters=filters,
        expansion_stats=expansion_stats,
    )
    expanded_path_states_by_path = expanded_path_states(candidates, path_guard.path_states)
    stale_candidate_count = sum(
        1
        for candidate in candidates
        if candidate_touches_invalidated_path(candidate, path_guard.path_states)
    )
    if stale_candidate_count:
        candidates = [
            candidate
            for candidate in candidates
            if not candidate_touches_invalidated_path(candidate, path_guard.path_states)
        ]
        expansion_stats["stale_path_filtered"] = stale_candidate_count
    candidates.sort(
        key=lambda candidate: (
            -candidate.score,
            candidate.seed_pre_cap_rank,
            candidate.chunk.relative_path,
            candidate.chunk.start_line,
            candidate.chunk_id,
        )
    )
    capped_candidates = candidates[:MAX_GRAPH_CANDIDATES]
    for rank, candidate in enumerate(capped_candidates, start=1):
        candidate.branch_rank = rank
        candidate.branch_score = candidate.score

    warning_codes: list[str] = list(status_warning_codes)
    warnings: list[SearchWarning] = list(status_warnings)
    if stale_candidate_count:
        warning_codes.append("graph_stale_expanded_path_overlap")
        warnings.append(
            SearchWarning(
                code="graph_stale_expanded_path_overlap",
                severity="warning",
                detail="Graph expansion discarded candidates that touched stale, deleted, or errored paths.",
            )
        )
    if len(candidates) > MAX_GRAPH_CANDIDATES or expansion_stats.get("budget_exhausted", 0):
        warning_codes.append("graph_candidate_cap_reached")
        warnings.append(
            SearchWarning(
                code="graph_candidate_cap_reached",
                severity="warning",
                detail="Graph candidate cap was reached; expansion stayed bounded.",
            )
        )
    if not capped_candidates:
        warning_codes.append("graph_no_candidates")
        warnings.append(
            SearchWarning(
                code="graph_no_candidates",
                severity="info",
                detail="Graph expansion found no additional filtered chunk candidates.",
            )
        )
    if expansion_stats["before_filter"] > expansion_stats["after_filter"]:
        warning_codes.append("graph_filter_post_applied")
        warnings.append(
            SearchWarning(
                code="graph_filter_post_applied",
                severity="info",
                detail="Graph expansion projected candidates before applying repo_context_search filters.",
            )
        )

    edge_types_used = sorted(
        {
            step.edge_type
            for candidate in capped_candidates
            for evidence in candidate.evidence_paths
            for step in evidence.path
            if step.edge_type is not None
        }
    )
    effective_mode = graph_mode if capped_candidates else "off"
    skipped_reason = None if capped_candidates else "graph_no_candidates"
    if not capped_candidates and stale_candidate_count:
        skipped_reason = "graph_stale_expanded_path_overlap"
    return GraphExpansionResult(
        mode_effective=effective_mode,
        candidates=capped_candidates,
        diagnostics=GraphBranchDiagnostics(
            requested_mode=graph_mode,
            effective_mode=effective_mode,
            available=True,
            used=bool(capped_candidates),
            status_state=status.state,
            expansion_allowed=True,
            skipped_reason=skipped_reason,
            seed_count=len(seed_pool),
            bound_seed_count=bound_seed_count,
            expanded_node_count=sum(len(nodes) for nodes in bound_nodes_by_seed.values()),
            graph_candidate_count=len(capped_candidates),
            final_graph_result_count=len(capped_candidates),
            graph_candidates_before_filter=expansion_stats["before_filter"],
            graph_candidates_after_filter=expansion_stats["after_filter"],
            graph_filter_post_families=_active_filter_families(filters)
            if expansion_stats["before_filter"] > expansion_stats["after_filter"]
            else [],
            degraded=path_guard.degraded,
            seed_path_states=path_guard.seed_path_states,
            expanded_path_states=expanded_path_states_by_path,
            edge_types_used=edge_types_used,  # type: ignore[arg-type]
            warning_codes=warning_codes,
        ),
        warnings=warnings,
    )


def _build_seed_pool(
    *,
    query_items: list[ContextQueryItem],
    executions: list[tuple[str, str, float, _SearchExecution]],
) -> list[GraphSeed]:
    del query_items
    entries: dict[str, dict[str, object]] = {}
    for query_text, role, weight, execution in executions:
        for rank, result in enumerate(execution.results, start=1):
            entry = entries.setdefault(
                result.chunk_id,
                {
                    "score": 0.0,
                    "best_rank": rank,
                    "best_query": query_text,
                    "best_role": role,
                    "best_weight": weight,
                    "relative_path": result.relative_path,
                    "scope": result.scope,
                    "symbol_path": result.symbol_path,
                    "heading_path": result.heading_path,
                    "matched_terms": [],
                },
            )
            entry["score"] = float(entry["score"]) + weight / (RRF_K + rank)
            if rank < int(entry["best_rank"]):
                entry["best_rank"] = rank
                entry["best_query"] = query_text
                entry["best_role"] = role
                entry["best_weight"] = weight
            matched_terms = entry["matched_terms"]
            assert isinstance(matched_terms, list)
            for term in result.matched_terms:
                if term not in matched_terms and len(matched_terms) < MAX_EXPLANATIONS:
                    matched_terms.append(term)

    ranked = [
        item
        for item in sorted(
            entries.items(),
            key=lambda item: (
                -float(item[1]["score"]),
                int(item[1]["best_rank"]),
                str(item[1]["relative_path"]),
                str(item[0]),
            ),
        )
        if not _low_value_seed_path(str(item[1]["relative_path"]))
    ]
    first_pass: list[tuple[str, dict[str, object]]] = []
    overflow: list[tuple[str, dict[str, object]]] = []
    per_file_counts: dict[str, int] = {}
    for chunk_id, entry in ranked:
        relative_path = str(entry["relative_path"])
        count = per_file_counts.get(relative_path, 0)
        if count < MAX_GRAPH_SEEDS_PER_FILE_FIRST_PASS:
            first_pass.append((chunk_id, entry))
            per_file_counts[relative_path] = count + 1
        else:
            overflow.append((chunk_id, entry))
    diversified = [*first_pass, *overflow]
    seeds: list[GraphSeed] = []
    for pool_rank, (chunk_id, entry) in enumerate(diversified[:MAX_GRAPH_SEED_POOL], start=1):
        seeds.append(
            GraphSeed(
                result_index=len(seeds),
                query=str(entry["best_query"]),
                query_role=str(entry["best_role"]),
                query_weight=float(entry["best_weight"]),
                pre_cap_rank=pool_rank,
                chunk_id=chunk_id,
                relative_path=str(entry["relative_path"]),
                scope=str(entry["scope"]),
                symbol_path=str(entry["symbol_path"]) if entry["symbol_path"] is not None else None,
                heading_path=str(entry["heading_path"]) if entry["heading_path"] is not None else None,
                matched_terms=tuple(str(term) for term in entry["matched_terms"]),  # type: ignore[index]
            )
        )
    return seeds


def _low_value_seed_path(relative_path: str) -> bool:
    normalized = relative_path.replace("\\", "/").lower()
    low_value_parts = (
        "/node_modules/",
        "/vendor/",
        "/dist/",
        "/build/",
        "/generated/",
        "/__generated__/",
    )
    bordered = f"/{normalized}"
    return normalized.endswith(".min.js") or any(part in bordered for part in low_value_parts)


def _should_auto_expand(
    *,
    query_items: list[ContextQueryItem],
    seed_pool: list[GraphSeed],
    exact_anchor_analysis: ExactAnchorAnalysis,
) -> bool:
    query_text = " ".join(item[0] for item in query_items).lower()
    relationship = any(term in query_text for term in RELATIONSHIP_TERMS)
    infra_anchor = any(anchor.anchor_type in INFRA_ANCHOR_TYPES for anchor in exact_anchor_analysis.anchors)
    cross_scope = len({seed.scope for seed in seed_pool}) > 1
    return bool((relationship or infra_anchor) and (seed_pool or cross_scope))


def _query_intent(
    query_items: list[ContextQueryItem],
    exact_anchor_analysis: ExactAnchorAnalysis,
) -> set[str]:
    """Infer bounded graph edge families from query text and exact anchors."""

    query_text = " ".join(item[0] for item in query_items).lower()
    intent: set[str] = set()
    if any(term in query_text for term in ("test", "tests", "pytest", "spec test", "unit test", "тест")):
        intent.add("tests")
    if any(term in query_text for term in ("doc", "docs", "documentation", "readme", "specification", "guide", "док")):
        intent.add("docs")
    if any(term in query_text for term in ("import", "imports", "dependency", "depends", "module", "caller", "callee")):
        intent.add("imports")
    if any(term in query_text for term in ("route", "endpoint", "handler", "http", "api/", "/api", "ручк")):
        intent.add("route")
    if any(term in query_text for term in ("handler", "owner", "implemented", "implementation", "runtime owner")):
        intent.add("handler")
    if any(term in query_text for term in ("config", "env", "setting", "settings", "deploy", "compose", "environment")):
        intent.add("config")
    if any(term in query_text for term in ("policy", "agent context", "agent_context")):
        intent.add("policy")
    if any(term in query_text for term in ("where used", "used by", "usage", "references", "referenced", "related")):
        intent.add("used_by")

    for anchor in exact_anchor_analysis.anchors:
        if anchor.anchor_type == "route":
            intent.add("route")
        elif anchor.anchor_type in {"env_var", "config_key", "cli_flag"}:
            intent.add("config")
        elif anchor.anchor_type in {"path", "file_name"}:
            intent.add("path")
        elif anchor.anchor_type == "symbol":
            intent.add("symbol")
        elif anchor.anchor_type == "sql_identifier":
            intent.add("config")
    return intent


def _edge_allowed(edge_type: str, *, graph_mode: GraphMode, query_intent: set[str]) -> bool:
    """Keep auto expansion deterministic and avoid noisy rich-edge families."""

    if edge_type in TEST_EDGE_TYPES:
        return bool(query_intent & {"tests", "used_by"})
    if edge_type in DOC_EDGE_TYPES:
        return bool(query_intent & {"docs", "route", "handler", "config", "policy", "used_by"})
    if edge_type in IMPORT_EDGE_TYPES:
        return bool(query_intent & {"imports", "used_by"})
    if edge_type in ROUTE_EDGE_TYPES:
        return bool(query_intent & {"route", "handler", "tests", "docs", "used_by"})
    if edge_type in CONFIG_EDGE_TYPES:
        return bool(query_intent & {"config", "used_by", "docs"})
    if edge_type == "policy_applies_to_path":
        return "policy" in query_intent
    if edge_type in GENERAL_EDGE_TYPES:
        return True
    return graph_mode == "expand"


def _edge_context_allowed(
    *,
    bound_node: GraphNode,
    edge: GraphEdge,
    query_intent: set[str],
) -> bool:
    docs_allowed = bool(query_intent & {"docs", "config", "policy", "used_by"})
    bound_docs = bound_node.node_type == "doc_section" or _is_docs_path(bound_node.relative_path)
    neighbor_docs = edge.neighbor_node_type == "doc_section" or _is_docs_path(edge.neighbor_relative_path)
    if edge.edge_type == "doc_references_route":
        if bound_docs and not neighbor_docs and bool(query_intent & {"route", "handler", "docs", "used_by"}):
            return True
        if neighbor_docs and not docs_allowed:
            return False
    docs_related = (
        bound_docs
        or neighbor_docs
    )
    if docs_related and not docs_allowed:
        return False
    return True


def _is_docs_path(relative_path: str | None) -> bool:
    if not relative_path:
        return False
    path = relative_path.replace("\\", "/").strip("/").lower()
    return path.startswith("docs/") or "/docs/" in f"/{path}/"


def _bind_seed_nodes(
    *,
    graph_read_provider,
    seeds: list[GraphSeed],
    exact_anchor_analysis: ExactAnchorAnalysis,
) -> dict[int, list[GraphNode]]:
    result: dict[int, list[GraphNode]] = {}
    chunk_nodes = graph_read_provider.find_nodes_for_chunks([seed.chunk_id for seed in seeds], limit=MAX_BOUND_NODES)
    nodes_by_chunk: dict[str, list[GraphNode]] = {}
    for node in chunk_nodes:
        if node.chunk_id:
            nodes_by_chunk.setdefault(node.chunk_id, []).append(node)

    file_nodes = graph_read_provider.find_file_nodes([seed.relative_path for seed in seeds], limit=MAX_BOUND_NODES)
    nodes_by_path: dict[str, list[GraphNode]] = {}
    for node in file_nodes:
        if node.relative_path:
            nodes_by_path.setdefault(node.relative_path, []).append(node)

    symbol_nodes = graph_read_provider.find_symbol_nodes(
        [
            (seed.relative_path, seed.symbol_path)
            for seed in seeds
            if seed.symbol_path
        ],
        limit=MAX_BOUND_NODES,
    )
    nodes_by_symbol: dict[tuple[str, str], list[GraphNode]] = {}
    for node in symbol_nodes:
        if node.relative_path:
            nodes_by_symbol.setdefault((node.relative_path, node.key), []).append(node)

    section_nodes = graph_read_provider.find_doc_section_nodes(
        [
            (seed.relative_path, seed.heading_path)
            for seed in seeds
            if seed.heading_path
        ],
        limit=MAX_BOUND_NODES,
    )
    nodes_by_section: dict[tuple[str, str], list[GraphNode]] = {}
    for node in section_nodes:
        if node.relative_path:
            nodes_by_section.setdefault((node.relative_path, node.key), []).append(node)

    for seed in seeds:
        term_nodes = graph_read_provider.find_term_nodes(
            _graph_terms_from_seed(seed, exact_anchor_analysis),
            limit=MAX_BOUND_NODES,
        )
        result[seed.result_index] = _dedupe_seed_nodes([
            *nodes_by_chunk.get(seed.chunk_id, []),
            *nodes_by_path.get(seed.relative_path, []),
            *nodes_by_symbol.get((seed.relative_path, seed.symbol_path or ""), []),
            *nodes_by_section.get((seed.relative_path, seed.heading_path or ""), []),
            *term_nodes,
        ])[:MAX_BOUND_NODES]
    return result


def _graph_terms_from_anchors(exact_anchor_analysis: ExactAnchorAnalysis) -> list[tuple[str, str | None]]:
    terms: list[tuple[str, str | None]] = []
    seen: set[tuple[str, str | None]] = set()
    for anchor in exact_anchor_analysis.anchors:
        for token in analyze_sparse_text(anchor.surface):
            if token.token_type not in TERM_TYPE_ALLOWLIST:
                continue
            if token.token_type == "exact" and len(token.canonical) < 4:
                continue
            key = (token.canonical, token.token_type)
            if key in seen:
                continue
            seen.add(key)
            terms.append(key)
    return terms


def _graph_terms_from_seed(
    seed: GraphSeed,
    exact_anchor_analysis: ExactAnchorAnalysis,
) -> list[tuple[str, str | None]]:
    terms: list[tuple[str, str | None]] = []
    seen: set[tuple[str, str | None]] = set()
    matched = {term.lower() for term in seed.matched_terms}
    seed_terms = list(seed.matched_terms)
    for anchor in exact_anchor_analysis.anchors:
        if anchor.surface.lower() in matched or anchor.canonical.lower() in matched:
            seed_terms.append(anchor.surface)
    for term in seed_terms:
        for token in analyze_sparse_text(term):
            if token.token_type not in TERM_TYPE_ALLOWLIST:
                continue
            key = (token.canonical, token.token_type)
            if key in seen:
                continue
            seen.add(key)
            terms.append(key)
            if len(terms) >= MAX_BOUND_NODES:
                return terms
    return terms


def _dedupe_seed_nodes(nodes: list[GraphNode]) -> list[GraphNode]:
    seen: set[str] = set()
    result: list[GraphNode] = []
    for node in nodes:
        if node.node_id in seen:
            continue
        seen.add(node.node_id)
        result.append(node)
    return result


def _expand_candidates(
    *,
    graph_read_provider,
    seeds: list[GraphSeed],
    bound_nodes_by_seed: dict[int, list[GraphNode]],
    exact_anchor_analysis: ExactAnchorAnalysis,
    graph_mode: GraphMode,
    query_intent: set[str],
    read_chunk: ReadChunk,
    matches_filters: MatchesFilters,
    filters: SearchFilters,
    expansion_stats: dict[str, int],
) -> list[GraphExpansionCandidate]:
    candidates: dict[str, GraphExpansionCandidate] = {}
    exact_bonus = 0.25 if exact_anchor_analysis.anchors else 0.0
    remaining_edges = MAX_GRAPH_EDGE_PROJECTIONS
    remaining_projections = MAX_GRAPH_CHUNK_PROJECTIONS
    primary_edge_types = _allowed_edge_types(ALLOWED_EDGE_TYPES, graph_mode=graph_mode, query_intent=query_intent)
    support_edge_types = _allowed_edge_types(SUPPORT_EDGE_TYPES, graph_mode=graph_mode, query_intent=query_intent)
    if not primary_edge_types:
        return []
    for seed in seeds:
        if remaining_edges <= 0 or remaining_projections <= 0:
            expansion_stats["budget_exhausted"] = 1
            break
        bound_nodes = bound_nodes_by_seed.get(seed.result_index, [])
        if not bound_nodes:
            continue
        for bound_node in bound_nodes:
            if remaining_edges <= 0 or remaining_projections <= 0:
                expansion_stats["budget_exhausted"] = 1
                break
            edges = graph_read_provider.neighbors(
                [bound_node.node_id],
                edge_types=primary_edge_types,
                max_neighbors_per_node=min(MAX_NEIGHBORS_PER_NODE, remaining_edges),
                max_total=min(MAX_GRAPH_CANDIDATES, remaining_edges),
            )
            remaining_edges -= len(edges)
            for edge in edges:
                if not _edge_allowed(edge.edge_type, graph_mode=graph_mode, query_intent=query_intent):
                    continue
                if not _edge_context_allowed(bound_node=bound_node, edge=edge, query_intent=query_intent):
                    continue
                remaining_projections = _project_graph_node(
                    candidates=candidates,
                    graph_read_provider=graph_read_provider,
                    seed=seed,
                    bound_node=bound_node,
                    edge=edge,
                    support_edge=None,
                    exact_bonus=exact_bonus,
                    remaining_projections=remaining_projections,
                    read_chunk=read_chunk,
                    matches_filters=matches_filters,
                    filters=filters,
                    expansion_stats=expansion_stats,
                )
                if remaining_edges <= 0 or remaining_projections <= 0:
                    expansion_stats["budget_exhausted"] = 1
                    break
                if not support_edge_types:
                    continue
                support_edges = graph_read_provider.neighbors(
                    [edge.neighbor_node_id],
                    edge_types=support_edge_types,
                    max_neighbors_per_node=min(MAX_NEIGHBORS_PER_NODE, remaining_edges),
                    max_total=min(MAX_GRAPH_CANDIDATES, remaining_edges),
                )
                remaining_edges -= len(support_edges)
                for support_edge in support_edges:
                    if support_edge.edge_id == edge.edge_id or support_edge.neighbor_node_id == bound_node.node_id:
                        continue
                    if not _edge_allowed(support_edge.edge_type, graph_mode=graph_mode, query_intent=query_intent):
                        continue
                    if not _edge_context_allowed(bound_node=bound_node, edge=support_edge, query_intent=query_intent):
                        continue
                    remaining_projections = _project_graph_node(
                        candidates=candidates,
                        graph_read_provider=graph_read_provider,
                        seed=seed,
                        bound_node=bound_node,
                        edge=edge,
                        support_edge=support_edge,
                        exact_bonus=exact_bonus,
                        remaining_projections=remaining_projections,
                        read_chunk=read_chunk,
                        matches_filters=matches_filters,
                        filters=filters,
                        expansion_stats=expansion_stats,
                    )
                    if remaining_edges <= 0 or remaining_projections <= 0:
                        expansion_stats["budget_exhausted"] = 1
                        break
    return list(candidates.values())


def _allowed_edge_types(
    edge_types: list[str],
    *,
    graph_mode: GraphMode,
    query_intent: set[str],
) -> list[str]:
    return [
        edge_type
        for edge_type in edge_types
        if _edge_allowed(edge_type, graph_mode=graph_mode, query_intent=query_intent)
    ]


def _project_graph_node(
    *,
    candidates: dict[str, GraphExpansionCandidate],
    graph_read_provider,
    seed: GraphSeed,
    bound_node: GraphNode,
    edge: GraphEdge,
    support_edge: GraphEdge | None,
    exact_bonus: float,
    remaining_projections: int,
    read_chunk: ReadChunk,
    matches_filters: MatchesFilters,
    filters: SearchFilters,
    expansion_stats: dict[str, int],
) -> int:
    if remaining_projections <= 0:
        expansion_stats["budget_exhausted"] = 1
        return remaining_projections

    target_node_id = support_edge.neighbor_node_id if support_edge is not None else edge.neighbor_node_id
    links = graph_read_provider.chunks_for_nodes(
        [target_node_id],
        limit=min(MAX_GRAPH_CANDIDATES, remaining_projections),
    )
    for link in links:
        if remaining_projections <= 0:
            expansion_stats["budget_exhausted"] = 1
            break
        remaining_projections -= 1
        if link.chunk_id == seed.chunk_id:
            continue
        expansion_stats["before_filter"] = expansion_stats.get("before_filter", 0) + 1
        chunk = read_chunk(link.scope, link.chunk_id)
        if chunk is None or not matches_filters(chunk, filters):
            continue
        expansion_stats["after_filter"] = expansion_stats.get("after_filter", 0) + 1
        score = _graph_path_score(
            seed=seed,
            edge=edge,
            support_edge=support_edge,
            link_weight=link.weight,
            exact_bonus=exact_bonus,
        )
        evidence = _evidence_path(
            seed=seed,
            bound_node=bound_node,
            edge=edge,
            support_edge=support_edge,
            chunk=chunk,
            score=score,
        )
        _upsert_candidate(
            candidates=candidates,
            chunk=chunk,
            seed=seed,
            score=score,
            evidence=evidence,
        )
    return remaining_projections


def _graph_path_score(
    *,
    seed: GraphSeed,
    edge: GraphEdge,
    support_edge: GraphEdge | None,
    link_weight: float,
    exact_bonus: float,
) -> float:
    seed_rank_prior = 1.0 / (1 + seed.pre_cap_rank)
    edge_weight = EDGE_WEIGHTS.get(edge.edge_type, 0.5)
    if support_edge is None:
        path_edge_weight = sqrt(edge_weight * max(0.1, link_weight))
        return seed_rank_prior + path_edge_weight + exact_bonus
    support_weight = EDGE_WEIGHTS.get(support_edge.edge_type, 0.5)
    path_edge_weight = sqrt(edge_weight * support_weight * max(0.1, link_weight))
    return seed_rank_prior + (path_edge_weight * 0.85) + exact_bonus - 0.15


def _upsert_candidate(
    *,
    candidates: dict[str, GraphExpansionCandidate],
    chunk: ChunkRecord,
    seed: GraphSeed,
    score: float,
    evidence: EvidencePath,
) -> None:
    candidate = candidates.get(chunk.point_id)
    if candidate is None:
        candidates[chunk.point_id] = GraphExpansionCandidate(
            chunk_id=chunk.point_id,
            chunk=chunk,
            score=score,
            seed_pre_cap_rank=seed.pre_cap_rank,
            origin_queries=[seed.query],
            origin_query_roles={seed.query: seed.query_role},
            matched_terms=list(seed.matched_terms)[:MAX_EXPLANATIONS],
            evidence_paths=[evidence],
        )
        return
    candidate.score = max(candidate.score, score)
    candidate.seed_pre_cap_rank = min(candidate.seed_pre_cap_rank, seed.pre_cap_rank)
    if seed.query not in candidate.origin_queries:
        candidate.origin_queries.append(seed.query)
    candidate.origin_query_roles.setdefault(seed.query, seed.query_role)
    candidate.evidence_paths.append(evidence)
    candidate.evidence_paths.sort(
        key=lambda path: (
            -(path.confidence or 0.0),
            path.seed_chunk_id or "",
            len(path.path),
        )
    )
    candidate.evidence_paths = candidate.evidence_paths[:MAX_EVIDENCE_PATHS_PER_RESULT]


def _evidence_path(
    *,
    seed: GraphSeed,
    bound_node: GraphNode,
    edge: GraphEdge,
    chunk: ChunkRecord,
    score: float,
    support_edge: GraphEdge | None = None,
) -> EvidencePath:
    path = [
        EvidencePathStep(
            node_type="chunk",
            key=seed.chunk_id,
            relative_path=seed.relative_path,
            chunk_id=seed.chunk_id,
        )
    ]
    bound_is_seed_chunk = bound_node.node_type == "chunk" and bound_node.chunk_id == seed.chunk_id
    if not bound_is_seed_chunk:
        path.append(
            EvidencePathStep(
                node_id=bound_node.node_id,
                node_type=bound_node.node_type,
                key=bound_node.key,
                relative_path=bound_node.relative_path,
                chunk_id=bound_node.chunk_id,
                start_line=bound_node.start_line,
                end_line=bound_node.end_line,
            )
        )
    path.append(
        EvidencePathStep(
            node_id=edge.neighbor_node_id,
            node_type=edge.neighbor_node_type,
            key=edge.neighbor_key,
            relative_path=edge.neighbor_relative_path,
            chunk_id=edge.neighbor_chunk_id,
            start_line=edge.neighbor_start_line,
            end_line=edge.neighbor_end_line,
            edge_type=edge.edge_type,
            metadata=_edge_metadata(edge),
        )
    )
    terminal_chunk_id = edge.neighbor_chunk_id
    if support_edge is not None:
        path.append(
            EvidencePathStep(
                node_id=support_edge.neighbor_node_id,
                node_type=support_edge.neighbor_node_type,
                key=support_edge.neighbor_key,
                relative_path=support_edge.neighbor_relative_path,
                chunk_id=support_edge.neighbor_chunk_id,
                start_line=support_edge.neighbor_start_line,
                end_line=support_edge.neighbor_end_line,
                edge_type=support_edge.edge_type,
                metadata=_edge_metadata(support_edge),
            )
        )
        terminal_chunk_id = support_edge.neighbor_chunk_id
    if terminal_chunk_id != chunk.point_id:
        path.append(
            EvidencePathStep(
                node_type="chunk",
                key=chunk.point_id,
                relative_path=chunk.relative_path,
                chunk_id=chunk.point_id,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
            )
        )
    reason = "shared_graph_node"
    if support_edge is not None:
        reason = "shared_graph_node"
    elif not bound_is_seed_chunk:
        reason = "term_bound_expansion"
    elif edge.neighbor_chunk_id == chunk.point_id:
        reason = "direct_graph_edge"
    elif edge.edge_type == "file_contains_chunk":
        reason = "file_containment"
    return EvidencePath(
        reason=reason,
        seed_chunk_id=seed.chunk_id,
        seed_relative_path=seed.relative_path,
        confidence=score,
        path=path,
    )


def _edge_metadata(edge: GraphEdge) -> dict[str, str]:
    if not edge.payload_json or edge.payload_json == "{}":
        return {}
    try:
        payload = json.loads(edge.payload_json)
    except json.JSONDecodeError:
        return {}
    result: dict[str, str] = {}
    for key in ("surface", "normalized", "normalized_route", "method", "framework"):
        value = payload.get(key)
        if value is not None:
            result[key] = str(value)
    return result


def _status_warning_action(
    *,
    graph_mode: GraphMode,
    state: str,
) -> tuple[SearchWarning, RecommendedAction | None]:
    code = f"graph_{state}"
    if state == "missing":
        return (
            SearchWarning(code=code, severity="warning", detail="Graph artifact is missing."),
            RecommendedAction(
                code="build_graph",
                severity="info",
                title="Build graph artifact",
                detail="Graph expansion is unavailable until build_graph is run explicitly.",
                tool_hint="build_graph",
            ),
        )
    if state == "building":
        return (
            SearchWarning(code=code, severity="info", detail="Graph artifact is currently building."),
            RecommendedAction(
                code="retry_with_graph_expand",
                severity="info",
                title="Retry after graph build",
                detail="Graph artifact is currently building; retry with graph_mode='expand' after graph_status reports ready.",
                tool_hint="repo_context_search(graph_mode='expand')",
            ),
        )
    return (
        SearchWarning(
            code=code,
            severity="warning",
            detail=f"Graph artifact state is {state}; graph expansion was skipped.",
        ),
        RecommendedAction(
            code="rebuild_graph",
            severity="warning",
            title="Rebuild graph artifact",
            detail="Graph expansion is disabled because the graph artifact is stale, incompatible, or errored.",
            tool_hint="rebuild_graph",
        ),
    )


def _status_warning_for_code(code: str) -> SearchWarning:
    """Return a route-level warning for graph status warning codes."""

    if code == "graph_source_index_contract_mismatch":
        return SearchWarning(
            code=code,
            severity="warning",
            detail=(
                "Graph artifact is behind the latest incremental index update; "
                "expansion is allowed in degraded mode and exact verification is recommended."
            ),
        )
    if code in {"graph_built_commit_mismatch", "graph_worktree_changes"}:
        return SearchWarning(
            code=code,
            severity="warning",
            detail="Graph artifact freshness differs from the current working tree.",
        )
    if code == "graph_invalidated_paths_unknown":
        return SearchWarning(
            code=code,
            severity="warning",
            detail="Graph artifact has stale source revision but invalidated paths are not known.",
        )
    if code == "graph_source_revision_hash_mismatch":
        return SearchWarning(
            code=code,
            severity="warning",
            detail="Graph artifact source hash differs from the current index.",
        )
    return SearchWarning(
        code=code,
        severity="info",
        detail="Graph status reported optional extractor coverage warnings.",
    )


def _stale_status_can_update_graph(warning_codes: list[str]) -> bool:
    if "graph_invalidated_paths_unknown" in warning_codes:
        return False
    if "graph_source_index_hard_contract_mismatch" in warning_codes:
        return False
    return any(
        code in warning_codes
        for code in (
            "graph_source_index_contract_mismatch",
            "graph_source_index_revision_mismatch",
            "graph_source_index_counts_mismatch",
            "graph_source_revision_hash_mismatch",
            "graph_file_errors",
        )
    )


def _active_filter_families(filters: SearchFilters) -> list[str]:
    families: list[str] = []
    if filters.path_prefix:
        families.append("path_prefix")
    for name in (
        "include_paths",
        "exclude_paths",
        "file_extensions",
        "languages",
        "chunk_types",
        "domain_tags",
    ):
        if getattr(filters, name):
            families.append(name)
    return families


def _skipped_result(
    *,
    graph_mode: GraphMode,
    skipped_reason: str,
    seed_count: int = 0,
    warnings: list[SearchWarning] | None = None,
) -> GraphExpansionResult:
    return GraphExpansionResult(
        mode_effective="off",
        diagnostics=GraphBranchDiagnostics(
            requested_mode=graph_mode,
            effective_mode="off",
            available=False,
            used=False,
            skipped_reason=skipped_reason,
            seed_count=seed_count,
            warning_codes=[warning.code for warning in warnings or []],
        ),
        warnings=warnings or [],
    )
