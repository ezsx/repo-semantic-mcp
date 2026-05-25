"""Query planning helpers for repo_context_search."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from services.repo_semantic.contracts.common import RepoContextRoute
from services.repo_semantic.contracts.repo_context import ExactAnchorUsage
from services.repo_semantic.contracts.search import ExactAnchorAnalysis, ExactAnchorCandidate
from services.repo_semantic.retrieval.constants import MAX_CONTEXT_SUBQUERIES
from services.repo_semantic.lexical import analyze_exact_anchors

ContextQueryItem = tuple[str, str, float]
ContextRouteUsed = Literal["semantic", "hybrid", "exact_handoff"]


def normalize_context_queries(
    query: str,
    subqueries: list[str] | None,
    *,
    validate_query: Callable[[str], None],
) -> tuple[list[ContextQueryItem], int]:
    """Normalize context-search queries while preserving original first."""

    validate_query(query)
    items: list[ContextQueryItem] = [(query, "original", 1.0)]
    seen = {query.strip()}
    dropped = 0
    for raw in subqueries or []:
        item = str(raw).strip()
        if not item:
            dropped += 1
            continue
        validate_query(item)
        if item in seen:
            dropped += 1
            continue
        if len(items) - 1 >= MAX_CONTEXT_SUBQUERIES:
            dropped += 1
            continue
        seen.add(item)
        items.append((item, "subquery", 0.85))
    return items, dropped


def context_route_used(route: RepoContextRoute) -> ContextRouteUsed:
    if route == "auto":
        return "hybrid"
    if route in {"semantic", "hybrid", "exact_handoff"}:
        return route
    raise ValueError("route must be one of: auto, semantic, hybrid, exact_handoff")


def aggregate_exact_anchors(
    query_items: list[ContextQueryItem],
) -> tuple[ExactAnchorAnalysis, list[ExactAnchorUsage]]:
    anchors: list[ExactAnchorCandidate] = []
    usages: list[ExactAnchorUsage] = []
    seen: set[tuple[str, str, str]] = set()
    exact_heavy = False
    for query_text, role, _weight in query_items:
        analysis = analyze_exact_anchors(query_text)
        exact_heavy = exact_heavy or analysis.exact_anchor_heavy
        for anchor in analysis.anchors:
            usages.append(
                ExactAnchorUsage(
                    query=query_text,
                    query_role=role,  # type: ignore[arg-type]
                    anchor=anchor,
                )
            )
            key = (anchor.surface, anchor.canonical, anchor.anchor_type)
            if key in seen:
                continue
            seen.add(key)
            anchors.append(anchor)
    argv_hints = [
        [
            "rg",
            "--fixed-strings",
            "--line-number",
            "--max-count",
            "20",
            "-e",
            anchor.surface,
            ".",
        ]
        for anchor in anchors[:5]
    ]
    return (
        ExactAnchorAnalysis(
            anchors=anchors,
            exact_anchor_heavy=exact_heavy,
            exact_fallback_recommended=bool(anchors),
            argv_hints=argv_hints,
        ),
        usages,
    )
