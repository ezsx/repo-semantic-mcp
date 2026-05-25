"""Project Qdrant points back into repository chunk records."""

from __future__ import annotations

from services.repo_semantic.contracts.common import ChunkRecord


def point_to_chunk(point) -> ChunkRecord:
    """Convert a Qdrant point payload into a ChunkRecord."""

    payload = point.payload or {}
    return ChunkRecord(
        point_id=str(payload.get("chunk_id") or point.id),
        scope=payload["scope"],
        relative_path=payload["relative_path"],
        language=payload["language"],
        chunk_type=payload["chunk_type"],
        text=payload["text"],
        start_line=int(payload["start_line"]),
        end_line=int(payload["end_line"]),
        content_hash=payload["content_hash"],
        source_mtime=float(payload["source_mtime"]),
        symbol_path=payload.get("symbol_path"),
        heading_path=payload.get("heading_path"),
        domain_tags=list(payload.get("domain_tags") or []),
        is_generated=bool(payload.get("is_generated", False)),
        extra={},
    )
