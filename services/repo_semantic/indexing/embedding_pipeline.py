"""Embedding batching helpers for repository indexing."""

from __future__ import annotations

from services.repo_semantic.contracts.common import ChunkRecord
from services.repo_semantic.embeddings.base import EmbeddingProvider
from services.repo_semantic.logging import jlog


def embed_chunks(
    *,
    chunks: list[ChunkRecord],
    scope: str,
    embedding_provider: EmbeddingProvider,
    max_batch_docs: int,
    max_batch_chars: int,
) -> list[list[float]]:
    """Build embeddings for chunks in bounded document/character batches."""

    if not chunks:
        return []

    vectors: list[list[float]] = []
    batch: list[ChunkRecord] = []
    batch_chars = 0
    batch_index = 0
    total_chunks = len(chunks)

    def flush_batch() -> None:
        nonlocal batch, batch_chars, vectors, batch_index
        if not batch:
            return
        batch_index += 1
        vectors.extend(
            embedding_provider.embed_documents([chunk.text for chunk in batch])
        )
        if batch_index == 1 or batch_index % 25 == 0 or len(vectors) == total_chunks:
            jlog(
                "info",
                "semantic_embedding_progress",
                scope=scope,
                embedded=len(vectors),
                total=total_chunks,
                batch_index=batch_index,
            )
        batch = []
        batch_chars = 0

    for chunk in chunks:
        chunk_chars = len(chunk.text)
        would_exceed_docs = len(batch) >= max_batch_docs
        would_exceed_chars = bool(batch) and batch_chars + chunk_chars > max_batch_chars
        if would_exceed_docs or would_exceed_chars:
            flush_batch()
        batch.append(chunk)
        batch_chars += chunk_chars

    flush_batch()
    return vectors
