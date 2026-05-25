"""Snippet and lexical explanation helpers."""

from __future__ import annotations

from services.repo_semantic.contracts.common import ChunkRecord
from services.repo_semantic.contracts.search import LineMatch
from services.repo_semantic.lexical import sparse_terms

from services.repo_semantic.retrieval.constants import (
    MAX_EXPLANATIONS,
    MAX_LINE_MATCHES,
    MAX_SNIPPET_CHARS,
)


def make_snippet(text: str, limit: int = 280) -> str:
    """Сжать чанк до короткого snippets для выдачи агенту."""

    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def matched_terms(query_tokens: list[str], text: str, *, max_terms: int = MAX_EXPLANATIONS) -> list[str]:
    """Return unique query tokens found in the chunk text."""

    text_tokens = set(sparse_terms(text))
    seen_canonical: set[str] = set()
    terms: list[str] = []
    for token in dict.fromkeys(query_tokens):
        if not token:
            continue
        canonical = token.lower()
        if canonical in seen_canonical or canonical not in text_tokens:
            continue
        seen_canonical.add(canonical)
        terms.append(token if token != canonical and token in text else canonical)
        if len(terms) >= max_terms:
            break
    return terms


def line_matches(
    chunk: ChunkRecord,
    matched_terms: list[str],
    *,
    max_matches: int = MAX_LINE_MATCHES,
    max_terms: int = MAX_EXPLANATIONS,
    snippet_limit: int = MAX_SNIPPET_CHARS,
) -> list[LineMatch]:
    """Collect bounded line-level matches for lexical explanations."""

    if not matched_terms:
        return []
    matches: list[LineMatch] = []
    for offset, line_text in enumerate(chunk.text.splitlines()):
        line_tokens = set(sparse_terms(line_text))
        line_terms = [
            term
            for term in matched_terms
            if term.lower() in line_tokens or (term != term.lower() and term in line_text)
        ]
        if not line_terms:
            continue
        matches.append(
            LineMatch(
                line=chunk.start_line + offset,
                text=make_snippet(line_text.strip(), limit=snippet_limit),
                matched_terms=line_terms[:max_terms],
            )
        )
        if len(matches) >= max_matches:
            break
    return matches


def query_centered_snippet(
    chunk: ChunkRecord,
    matched_terms: list[str],
    *,
    limit: int = MAX_SNIPPET_CHARS,
) -> tuple[str, int | None, int | None]:
    """Build a bounded snippet around the first query-term line."""

    lines = chunk.text.splitlines() or [chunk.text]
    center_index = 0
    if matched_terms:
        for index, line in enumerate(lines):
            line_tokens = set(sparse_terms(line))
            if any(
                term.lower() in line_tokens or (term != term.lower() and term in line)
                for term in matched_terms
            ):
                center_index = index
                break
    selected: list[tuple[int, str]] = [(center_index, lines[center_index].strip())]

    for index in range(center_index - 1, max(-1, center_index - 2), -1):
        candidate_rows = [(index, lines[index].strip()), *selected]
        if len("\n".join(row for _, row in candidate_rows)) <= limit:
            selected = candidate_rows
    for index in range(center_index + 1, min(len(lines), center_index + 3)):
        candidate_rows = [*selected, (index, lines[index].strip())]
        if len("\n".join(row for _, row in candidate_rows)) > limit:
            break
        selected = candidate_rows

    snippet = "\n".join(row for _, row in selected)
    if len(snippet) > limit:
        snippet = snippet[: limit - 3].rstrip() + "..."
    return snippet, chunk.start_line + selected[0][0], chunk.start_line + selected[-1][0]
