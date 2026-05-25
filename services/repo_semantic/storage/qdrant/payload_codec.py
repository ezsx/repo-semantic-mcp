"""Encode repository chunks into Qdrant payloads and point vectors."""

from __future__ import annotations

from qdrant_client import models

from services.repo_semantic.contracts.common import ChunkRecord
from services.repo_semantic.lexical import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    EncodedSparseVector,
)


def path_payload_fields(relative_path: str) -> tuple[str | None, list[str], list[str]]:
    """Return file-extension, path-segment, and path-prefix payload fields."""

    path_parts = relative_path.replace("\\", "/").strip("/").split("/")
    path_prefixes = [
        "/".join(path_parts[: part_index])
        for part_index in range(1, len(path_parts) + 1)
        if path_parts[:part_index]
    ]
    file_name = relative_path.rsplit("/", 1)[-1]
    file_extension = "." + file_name.rsplit(".", 1)[-1].lower() if "." in file_name else None
    return file_extension, path_parts, path_prefixes


def chunk_payload(
    *,
    chunk: ChunkRecord,
    embedding_backend: str,
    embedding_model: str,
    schema_version: int,
    query_template_hash: str | None,
    document_prefix_hash: str | None,
    sparse_contract_hash: str | None,
    sparse_vocabulary_hash: str | None,
    sparse_corpus_stats_hash: str | None,
    lexical_analyzer_version: str | None,
) -> dict[str, object]:
    """Build the persisted Qdrant payload for one indexed chunk."""

    file_extension, path_segments, path_prefixes = path_payload_fields(chunk.relative_path)
    return {
        "chunk_id": chunk.point_id,
        "scope": chunk.scope,
        "relative_path": chunk.relative_path,
        "file_extension": file_extension,
        "path_segments": path_segments,
        "path_prefixes": path_prefixes,
        "language": chunk.language,
        "chunk_type": chunk.chunk_type,
        "text": chunk.text,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "content_hash": chunk.content_hash,
        "source_mtime": chunk.source_mtime,
        "symbol_path": chunk.symbol_path,
        "heading_path": chunk.heading_path,
        "domain_tags": chunk.domain_tags,
        "is_generated": chunk.is_generated,
        "embedding_backend": embedding_backend,
        "embedding_model": embedding_model,
        "index_schema_version": schema_version,
        "query_template_hash": query_template_hash,
        "document_prefix_hash": document_prefix_hash,
        "sparse_contract_hash": sparse_contract_hash,
        "sparse_vocabulary_hash": sparse_vocabulary_hash,
        "sparse_corpus_stats_hash": sparse_corpus_stats_hash,
        "lexical_analyzer_version": lexical_analyzer_version,
        **chunk.extra,
    }


def point_vector(
    *,
    dense_vector: list[float],
    dense_schema_kind: str,
    sparse_vector: EncodedSparseVector | None,
) -> list[float] | dict[str, list[float] | models.SparseVector]:
    """Build a Qdrant point vector for named or legacy dense schemas."""

    if sparse_vector is None:
        return (
            {DENSE_VECTOR_NAME: dense_vector}
            if dense_schema_kind == "named"
            else dense_vector
        )
    return {
        DENSE_VECTOR_NAME: dense_vector,
        SPARSE_VECTOR_NAME: models.SparseVector(
            indices=sparse_vector.indices,
            values=sparse_vector.values,
        ),
    }


def estimated_point_bytes(
    *,
    chunk: ChunkRecord,
    dense_vector: list[float],
    sparse_vector: EncodedSparseVector | None,
) -> int:
    """Return a conservative payload/vector byte estimate for batching."""

    return (
        len(chunk.text.encode("utf-8"))
        + len(chunk.relative_path.encode("utf-8"))
        + len(chunk.language.encode("utf-8"))
        + len(chunk.chunk_type.encode("utf-8"))
        + sum(len(tag.encode("utf-8")) for tag in chunk.domain_tags)
        + len(dense_vector) * 16
        + (len(sparse_vector.indices) * 16 if sparse_vector is not None else 0)
        + 2048
    )


def qdrant_point(
    *,
    point_id: str,
    chunk: ChunkRecord,
    dense_vector: list[float],
    dense_schema_kind: str,
    sparse_vector: EncodedSparseVector | None,
    embedding_backend: str,
    embedding_model: str,
    schema_version: int,
    query_template_hash: str | None,
    document_prefix_hash: str | None,
    sparse_contract_hash: str | None,
    sparse_vocabulary_hash: str | None,
    sparse_corpus_stats_hash: str | None,
    lexical_analyzer_version: str | None,
) -> models.PointStruct:
    """Build a Qdrant point from one chunk and its retrieval vectors."""

    return models.PointStruct(
        id=point_id,
        vector=point_vector(
            dense_vector=dense_vector,
            dense_schema_kind=dense_schema_kind,
            sparse_vector=sparse_vector,
        ),
        payload=chunk_payload(
            chunk=chunk,
            embedding_backend=embedding_backend,
            embedding_model=embedding_model,
            schema_version=schema_version,
            query_template_hash=query_template_hash,
            document_prefix_hash=document_prefix_hash,
            sparse_contract_hash=sparse_contract_hash,
            sparse_vocabulary_hash=sparse_vocabulary_hash,
            sparse_corpus_stats_hash=sparse_corpus_stats_hash,
            lexical_analyzer_version=lexical_analyzer_version,
        ),
    )
