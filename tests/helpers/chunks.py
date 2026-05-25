from __future__ import annotations

from services.repo_semantic.models import ChunkRecord


def make_chunk(
    chunk_id: str,
    relative_path: str,
    text: str,
    *,
    scope: str = "code",
    language: str = "python",
    chunk_type: str = "python_function",
    start_line: int = 1,
    symbol_path: str | None = None,
    heading_path: str | None = None,
    domain_tags: list[str] | None = None,
) -> ChunkRecord:
    line_count = max(1, text.count("\n") + 1)
    return ChunkRecord(
        point_id=chunk_id,
        scope=scope,  # type: ignore[arg-type]
        relative_path=relative_path,
        language=language,
        chunk_type=chunk_type,
        text=text,
        start_line=start_line,
        end_line=start_line + line_count - 1,
        content_hash=chunk_id,
        source_mtime=0.0,
        symbol_path=symbol_path,
        heading_path=heading_path,
        domain_tags=domain_tags or [],
    )
