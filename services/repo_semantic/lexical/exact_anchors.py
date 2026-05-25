"""Exact-anchor detection and local verification hints."""

from __future__ import annotations

from services.repo_semantic.lexical.analyzer import (
    CLI_FLAG_RE,
    ENV_VAR_RE,
    FILE_NAME_RE,
    KNOWN_FILE_EXTENSIONS,
    SQL_IDENTIFIER_RE,
    TOKEN_RE,
    sparse_unique_terms,
)
from services.repo_semantic.models import ExactAnchorAnalysis, ExactAnchorCandidate

ANCHOR_PRIORITY = {
    "env_var": 0,
    "route": 1,
    "path": 2,
    "file_name": 3,
    "cli_flag": 4,
    "quoted_literal": 5,
    "error_literal": 6,
    "sql_identifier": 7,
    "symbol": 8,
    "config_key": 9,
}


def _anchor_type_and_confidence(surface: str) -> tuple[str, str] | None:
    if not surface:
        return None
    if surface.startswith('"') and surface.endswith('"'):
        inner = surface[1:-1].replace('\\"', '"').strip()
        if inner:
            return "quoted_literal", "high"
        return None
    if CLI_FLAG_RE.match(surface):
        return "cli_flag", "high"
    if ENV_VAR_RE.match(surface):
        return "env_var", "high"
    if surface.startswith("/") and "/" in surface[1:]:
        return "route", "high"
    if "/" in surface:
        return "path", "high"
    if FILE_NAME_RE.match(surface):
        extension = surface.rsplit(".", 1)[-1].lower()
        if extension in KNOWN_FILE_EXTENSIONS:
            return "file_name", "medium"
    if "." in surface and not FILE_NAME_RE.match(surface):
        return "symbol", "medium"
    if "." in surface and any(char.isupper() for char in surface):
        return "symbol", "medium"
    if SQL_IDENTIFIER_RE.match(surface):
        return "sql_identifier", "high"
    if "_" in surface and any(part for part in surface.split("_")):
        return "config_key", "medium"
    return None


def _anchor_canonical(surface: str, anchor_type: str) -> str:
    if anchor_type == "quoted_literal":
        return surface[1:-1].replace('\\"', '"').strip().lower()
    if anchor_type == "route":
        return surface.lower()
    return surface.strip().lower()


def _outside_token_count(query: str, anchors: list[ExactAnchorCandidate]) -> int:
    if not anchors:
        return len(sparse_unique_terms(query))
    spans = sorted((anchor.start, anchor.end) for anchor in anchors)
    outside_parts: list[str] = []
    cursor = 0
    for start, end in spans:
        if cursor < start:
            outside_parts.append(query[cursor:start])
        cursor = max(cursor, end)
    if cursor < len(query):
        outside_parts.append(query[cursor:])
    outside_text = " ".join(outside_parts)
    return len(sparse_unique_terms(outside_text))


def analyze_exact_anchors(query: str) -> ExactAnchorAnalysis:
    """Detect exact-looking query anchors and safe local-rg guidance."""

    anchors: list[ExactAnchorCandidate] = []
    seen: set[tuple[str, str, str]] = set()
    for match in TOKEN_RE.finditer(query):
        surface = match.group(0).strip()
        verdict = _anchor_type_and_confidence(surface)
        if verdict is None:
            continue
        anchor_type, confidence = verdict
        canonical = _anchor_canonical(surface, anchor_type)
        if anchor_type == "quoted_literal":
            surface = surface[1:-1].replace('\\"', '"').strip()
        key = (surface, canonical, anchor_type)
        if key in seen:
            continue
        seen.add(key)
        anchors.append(
            ExactAnchorCandidate(
                surface=surface,
                canonical=canonical,
                start=match.start(),
                end=match.end(),
                anchor_type=anchor_type,  # type: ignore[arg-type]
                confidence=confidence,  # type: ignore[arg-type]
            )
        )

    anchors.sort(
        key=lambda anchor: (
            anchor.start,
            ANCHOR_PRIORITY.get(anchor.anchor_type, 99),
            -len(anchor.surface),
        )
    )
    anchors = anchors[:20]
    exact_anchor_heavy = any(anchor.confidence == "high" for anchor in anchors) and _outside_token_count(
        query,
        anchors,
    ) <= 4
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
    return ExactAnchorAnalysis(
        anchors=anchors,
        exact_anchor_heavy=exact_anchor_heavy,
        exact_fallback_recommended=bool(anchors),
        argv_hints=argv_hints,
    )
