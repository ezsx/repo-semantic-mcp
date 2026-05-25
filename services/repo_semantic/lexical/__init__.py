"""Code-aware lexical retrieval primitives."""

from __future__ import annotations

from services.repo_semantic.lexical.analyzer import (
    analyze_sparse_text,
    sparse_terms,
    sparse_unique_terms,
)
from services.repo_semantic.lexical.constants import (
    BM25_B,
    BM25_K1,
    DENSE_VECTOR_NAME,
    LEXICAL_ANALYZER_VERSION,
    SPARSE_ENCODER_KIND,
    SPARSE_MANIFEST_CONTRACT_VERSION,
    SPARSE_VECTOR_NAME,
)
from services.repo_semantic.lexical.encoder import (
    encode_sparse_document,
    encode_sparse_query,
)
from services.repo_semantic.lexical.exact_anchors import analyze_exact_anchors
from services.repo_semantic.lexical.manifest import (
    SparseManifest,
    SparseManifestStore,
    build_sparse_manifest,
    extend_manifest_for_incremental_chunks,
    mark_manifest_stale,
    sparse_contract_hash,
)
from services.repo_semantic.lexical.types import EncodedSparseVector, SparseToken

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
