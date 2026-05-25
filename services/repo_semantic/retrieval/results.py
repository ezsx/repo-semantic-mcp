"""Search/read result assembly helpers."""

from __future__ import annotations

from pathlib import PurePosixPath
from collections.abc import Callable

from services.repo_semantic.lexical import analyze_sparse_text
from services.repo_semantic.models import (
    ChunkRecord,
    ReadChunkResult,
    SearchResult,
    SnippetMode,
)
from services.repo_semantic.retrieval.constants import (
    MAX_EXPLANATIONS,
    MAX_LINE_MATCHES,
    MAX_SNIPPET_CHARS,
)
from services.repo_semantic.retrieval.snippets import (
    line_matches as build_line_matches,
    make_snippet,
    matched_terms as find_matched_terms,
    query_centered_snippet as build_query_centered_snippet,
)


def query_terms_for_matching(query: str, *, tokenize: Callable[[str], list[str]]) -> list[str]:
    """Return display-oriented query terms while preserving canonical matching."""

    terms: list[str] = []
    for token in analyze_sparse_text(query):
        if token.surface and token.surface != token.canonical:
            terms.append(token.surface)
        terms.append(token.canonical)
    return list(dict.fromkeys(terms)) or tokenize(query)


def build_search_result(
    *,
    chunk: ChunkRecord,
    repo_root: str,
    score: float,
    dense_score: float | None = None,
    lexical_score: float | None = None,
    query_tokens: list[str] | None = None,
    snippet_mode: SnippetMode = "chunk_start",
    include_explanations: bool = True,
    match_type: str | None = None,
) -> SearchResult:
    """Build the public SearchResult for one ranked chunk."""

    tokens = query_tokens or []
    matched_terms = find_matched_terms(tokens, chunk.text)
    line_matches = build_line_matches(
        chunk,
        matched_terms,
        max_matches=MAX_LINE_MATCHES,
        max_terms=MAX_EXPLANATIONS,
        snippet_limit=MAX_SNIPPET_CHARS,
    )
    if snippet_mode == "query_centered":
        snippet, snippet_start_line, snippet_end_line = build_query_centered_snippet(
            chunk,
            matched_terms,
            limit=MAX_SNIPPET_CHARS,
        )
    else:
        snippet = make_snippet(chunk.text, limit=MAX_SNIPPET_CHARS)
        snippet_start_line = chunk.start_line
        snippet_end_line = chunk.start_line if snippet else None

    why_matched: list[str] = []
    if include_explanations:
        if dense_score is not None and dense_score > 0:
            why_matched.append("dense_vector_similarity")
        if lexical_score is not None and lexical_score > 0:
            why_matched.append("lexical_term_overlap")
        if snippet_mode == "query_centered" and not matched_terms:
            why_matched.append("query_centered_fallback_to_chunk_start")
        why_matched = why_matched[:MAX_EXPLANATIONS]

    return SearchResult(
        chunk_id=chunk.point_id,
        repo_root=repo_root,
        scope=chunk.scope,
        relative_path=chunk.relative_path,
        language=chunk.language,
        chunk_type=chunk.chunk_type,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        line_range=f"{chunk.start_line}-{chunk.end_line}",
        file_extension=PurePosixPath(chunk.relative_path).suffix.lower() or None,
        snippet=snippet,
        snippet_start_line=snippet_start_line,
        snippet_end_line=snippet_end_line,
        symbol_path=chunk.symbol_path,
        heading_path=chunk.heading_path,
        domain_tags=chunk.domain_tags,
        score=score,
        final_score=score,
        match_type=match_type,  # type: ignore[arg-type]
        dense_score=dense_score,
        lexical_score=lexical_score,
        matched_terms=matched_terms,
        why_matched=why_matched,
        line_matches=line_matches,
    )


def build_read_chunk_result(chunk: ChunkRecord) -> ReadChunkResult:
    """Build the public read-chunk result for one chunk."""

    return ReadChunkResult(
        chunk_id=chunk.point_id,
        scope=chunk.scope,
        relative_path=chunk.relative_path,
        language=chunk.language,
        chunk_type=chunk.chunk_type,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        symbol_path=chunk.symbol_path,
        heading_path=chunk.heading_path,
        text=chunk.text,
    )
