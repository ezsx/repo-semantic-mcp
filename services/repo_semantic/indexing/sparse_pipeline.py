"""Sparse manifest compatibility and indexing helpers."""

from __future__ import annotations

from services.repo_semantic.contracts.common import ChunkRecord
from services.repo_semantic.lexical import (
    LEXICAL_ANALYZER_VERSION,
    SPARSE_ENCODER_KIND,
    EncodedSparseVector,
    SparseManifest,
    encode_sparse_document,
    sparse_contract_hash,
)


def compatible_sparse_manifest(
    manifest: SparseManifest | None,
    *,
    schema_version: int,
) -> SparseManifest | None:
    """Return manifest only when incremental sparse updates are contract-safe."""

    if manifest is None:
        return None
    if manifest.schema_version != schema_version:
        return None
    if manifest.lexical_analyzer_version != LEXICAL_ANALYZER_VERSION:
        return None
    if manifest.sparse_encoder_kind != SPARSE_ENCODER_KIND:
        return None
    expected_contract_hash = sparse_contract_hash(schema_version=schema_version)
    if manifest.immutable_sparse_contract_hash != expected_contract_hash:
        return None
    return manifest


def should_upsert_sparse_vectors(
    manifest: SparseManifest | None,
    *,
    current_collection_exists: bool,
    sparse_vector_available: bool,
) -> bool:
    """Return whether incremental writes can include sparse vectors."""

    if manifest is None:
        return False
    return not current_collection_exists or sparse_vector_available


def encode_sparse_documents(
    chunks: list[ChunkRecord],
    manifest: SparseManifest,
) -> list[EncodedSparseVector]:
    """Encode chunks with the manifest vocabulary for Qdrant sparse vectors."""

    return [encode_sparse_document(chunk.text, manifest) for chunk in chunks]
