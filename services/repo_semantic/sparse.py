"""Compatibility facade for deterministic sparse lexical retrieval.

New code should import from ``services.repo_semantic.lexical``. This module
keeps the original public surface stable for existing callers and tests.
"""

from __future__ import annotations

from services.repo_semantic.lexical import (
    BM25_B,
    BM25_K1,
    DENSE_VECTOR_NAME,
    LEXICAL_ANALYZER_VERSION,
    SPARSE_ENCODER_KIND,
    SPARSE_MANIFEST_CONTRACT_VERSION,
    SPARSE_VECTOR_NAME,
    EncodedSparseVector,
    SparseManifest,
    SparseManifestStore,
    SparseToken,
    analyze_exact_anchors,
    analyze_sparse_text,
    build_sparse_manifest,
    encode_sparse_document,
    encode_sparse_query,
    extend_manifest_for_incremental_chunks,
    mark_manifest_stale,
    sparse_contract_hash,
    sparse_terms,
    sparse_unique_terms,
)

__all__ = [
    "BM25_B",
    "BM25_K1",
    "DENSE_VECTOR_NAME",
    "EncodedSparseVector",
    "LEXICAL_ANALYZER_VERSION",
    "SPARSE_ENCODER_KIND",
    "SPARSE_MANIFEST_CONTRACT_VERSION",
    "SPARSE_VECTOR_NAME",
    "SparseManifest",
    "SparseManifestStore",
    "SparseToken",
    "analyze_exact_anchors",
    "analyze_sparse_text",
    "build_sparse_manifest",
    "encode_sparse_document",
    "encode_sparse_query",
    "extend_manifest_for_incremental_chunks",
    "mark_manifest_stale",
    "sparse_contract_hash",
    "sparse_terms",
    "sparse_unique_terms",
]
