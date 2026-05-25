"""Query-aware graph freshness guards."""

from __future__ import annotations

from dataclasses import dataclass

from services.repo_semantic.contracts.search import SearchWarning
from services.repo_semantic.contracts.status import PathFreshnessSummary
from services.repo_semantic.graph.types import GraphExpansionCandidate, GraphSeed

INVALIDATED_PATH_STATES = {"stale", "deleted", "error", "unknown"}


@dataclass(frozen=True, slots=True)
class GraphPathGuard:
    """Decision state for degraded graph expansion path safety."""

    degraded: bool
    path_states: dict[str, str]
    seed_path_states: dict[str, str]
    blocked_warning: SearchWarning | None = None
    skipped_reason: str | None = None


def graph_path_guard(
    *,
    status,
    path_freshness: PathFreshnessSummary | None,
    seeds: list[GraphSeed],
) -> GraphPathGuard:
    """Return a query-aware degraded graph guard for retrieved seed paths."""

    degraded = bool(getattr(status, "state", None) == "stale" and getattr(status, "expansion_allowed", False))
    path_states, invalidated_count, invalidated_paths_known = invalidated_path_states(
        status=status,
        path_freshness=path_freshness,
    )
    seed_path_states = path_states_for_paths(
        [seed.relative_path for seed in seeds],
        path_states=path_states,
        include_ready=degraded,
    )
    if not degraded:
        return GraphPathGuard(
            degraded=False,
            path_states=path_states,
            seed_path_states=seed_path_states,
        )
    if invalidated_count > 0 and not invalidated_paths_known:
        return GraphPathGuard(
            degraded=True,
            path_states=path_states,
            seed_path_states=seed_path_states,
            blocked_warning=SearchWarning(
                code="graph_invalidated_paths_unknown",
                severity="warning",
                detail=(
                    "Graph expansion was skipped because degraded graph freshness has invalidated paths "
                    "that cannot be fully mapped to query paths."
                ),
            ),
            skipped_reason="graph_invalidated_paths_unknown",
        )
    if path_freshness is not None and not path_freshness.coverage_complete:
        return GraphPathGuard(
            degraded=True,
            path_states=path_states,
            seed_path_states=seed_path_states,
            blocked_warning=SearchWarning(
                code="graph_invalidated_paths_unknown",
                severity="warning",
                detail="Graph expansion was skipped because path freshness coverage is incomplete.",
            ),
            skipped_reason="graph_invalidated_paths_unknown",
        )
    stale_seed_paths = sorted(
        path
        for path, state in seed_path_states.items()
        if state in INVALIDATED_PATH_STATES
    )
    if stale_seed_paths:
        return GraphPathGuard(
            degraded=True,
            path_states=path_states,
            seed_path_states=seed_path_states,
            blocked_warning=SearchWarning(
                code="graph_stale_seed_path_overlap",
                severity="warning",
                detail="Graph expansion was skipped because stale graph paths overlap retrieved seed paths.",
            ),
            skipped_reason="graph_stale_seed_path_overlap",
        )
    return GraphPathGuard(
        degraded=True,
        path_states=path_states,
        seed_path_states=seed_path_states,
    )


def invalidated_path_states(
    *,
    status,
    path_freshness: PathFreshnessSummary | None,
) -> tuple[dict[str, str], int, bool]:
    """Return invalidated path states and whether the path set is complete."""

    path_states: dict[str, str] = {}
    status_count, status_paths = _status_invalidated_paths(status)
    if path_freshness is not None:
        for path in path_freshness.stale_paths_preview:
            normalized = normalize_graph_path(path)
            if normalized:
                path_states[normalized] = "stale"
        for path in path_freshness.deleted_paths_preview:
            normalized = normalize_graph_path(path)
            if normalized:
                path_states[normalized] = "deleted"
        for path in path_freshness.error_paths_preview:
            normalized = normalize_graph_path(path)
            if normalized:
                path_states[normalized] = "error"
        invalidated_count = (
            path_freshness.missing_from_index_count
            + path_freshness.stale_indexed_paths_count
            + path_freshness.deleted_indexed_paths_count
            + path_freshness.path_error_count
        )
        path_paths_count = len(path_states)
        path_paths_known = bool(path_freshness.coverage_complete) and invalidated_count <= path_paths_count
        for path in status_paths:
            path_states.setdefault(path, "stale")
        status_paths_known = status_count == 0 or status_count <= len(status_paths)
        if not path_paths_known or not status_paths_known:
            return path_states, max(invalidated_count + status_count, len(path_states) + 1), False
        return path_states, len(path_states), True

    for path in status_paths:
        path_states[path] = "stale"
    invalidated_paths_known = status_count == 0 or status_count <= len(status_paths)
    return path_states, status_count, invalidated_paths_known


def _status_invalidated_paths(status) -> tuple[int, list[str]]:
    invalidated_count = int(getattr(status, "stale_path_count", 0) or 0)
    paths: list[str] = []
    for path in getattr(status, "invalidated_paths_preview", []) or []:
        normalized = normalize_graph_path(path)
        if normalized:
            paths.append(normalized)
    return invalidated_count, list(dict.fromkeys(paths))


def path_states_for_paths(
    paths: list[str],
    *,
    path_states: dict[str, str],
    include_ready: bool,
) -> dict[str, str]:
    """Classify paths against known invalidated graph freshness states."""

    result: dict[str, str] = {}
    for path in paths:
        normalized = normalize_graph_path(path)
        if not normalized or normalized in result:
            continue
        state = path_states.get(normalized)
        if state is not None:
            result[normalized] = state
        elif include_ready:
            result[normalized] = "ready"
    return result


def expanded_path_states(
    candidates: list[GraphExpansionCandidate],
    path_states: dict[str, str],
) -> dict[str, str]:
    """Return path states touched by graph candidates and evidence paths."""

    paths: list[str] = []
    for candidate in candidates:
        paths.append(candidate.chunk.relative_path)
        for evidence in candidate.evidence_paths:
            if evidence.seed_relative_path:
                paths.append(evidence.seed_relative_path)
            for step in evidence.path:
                if step.relative_path:
                    paths.append(step.relative_path)
    return path_states_for_paths(paths, path_states=path_states, include_ready=True)


def candidate_touches_invalidated_path(
    candidate: GraphExpansionCandidate,
    path_states: dict[str, str],
) -> bool:
    """Return whether a graph candidate or its evidence touches invalidated paths."""

    if not path_states:
        return False
    paths = [candidate.chunk.relative_path]
    for evidence in candidate.evidence_paths:
        if evidence.seed_relative_path:
            paths.append(evidence.seed_relative_path)
        for step in evidence.path:
            if step.relative_path:
                paths.append(step.relative_path)
    return any(path_states.get(normalize_graph_path(path)) in INVALIDATED_PATH_STATES for path in paths)


def normalize_graph_path(path: str | None) -> str:
    """Normalize repo-relative graph paths for freshness comparisons."""

    if not path:
        return ""
    return str(path).replace("\\", "/").lstrip("./")
