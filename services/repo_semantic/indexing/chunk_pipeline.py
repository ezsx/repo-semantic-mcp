"""Chunk normalization helpers for the indexing pipeline."""

from __future__ import annotations

from services.repo_semantic.contracts.common import ChunkRecord


def split_oversized_chunk(chunk: ChunkRecord, *, max_chars: int) -> list[ChunkRecord]:
    """Split a chunk into line-preserving parts that fit the embedding limit."""

    if len(chunk.text) <= max_chars:
        return [chunk]

    lines = chunk.text.splitlines(keepends=True)
    if not lines:
        return [chunk]

    result: list[ChunkRecord] = []
    buffer = ""
    part_index = 1
    part_start_line = chunk.start_line
    line_cursor = chunk.start_line

    def flush_buffer() -> None:
        nonlocal buffer, part_index, part_start_line, line_cursor
        if not buffer:
            return
        part_text = buffer
        newline_count = part_text.count("\n")
        result.append(
            ChunkRecord(
                point_id=f"{chunk.point_id}:part{part_index}",
                scope=chunk.scope,
                relative_path=chunk.relative_path,
                language=chunk.language,
                chunk_type=f"{chunk.chunk_type}_part",
                text=part_text,
                start_line=part_start_line,
                end_line=part_start_line + newline_count,
                content_hash=chunk.content_hash,
                source_mtime=chunk.source_mtime,
                symbol_path=chunk.symbol_path,
                heading_path=chunk.heading_path,
                domain_tags=list(chunk.domain_tags),
                is_generated=chunk.is_generated,
                extra={**chunk.extra, "split_part": str(part_index)},
            )
        )
        part_index += 1
        line_cursor = part_start_line + newline_count
        if part_text.endswith("\n"):
            line_cursor += 1
        buffer = ""
        part_start_line = line_cursor

    for line in lines:
        if len(line) > max_chars:
            flush_buffer()
            for offset in range(0, len(line), max_chars):
                segment = line[offset : offset + max_chars]
                buffer = segment
                part_start_line = line_cursor
                flush_buffer()
            continue

        if buffer and len(buffer) + len(line) > max_chars:
            flush_buffer()

        if not buffer:
            part_start_line = line_cursor
        buffer += line

    flush_buffer()
    return result or [chunk]


def normalize_chunks(chunks: list[ChunkRecord], *, max_chars: int) -> list[ChunkRecord]:
    """Return chunks after applying the oversized-chunk split policy."""

    normalized: list[ChunkRecord] = []
    for chunk in chunks:
        normalized.extend(split_oversized_chunk(chunk, max_chars=max_chars))
    return normalized
