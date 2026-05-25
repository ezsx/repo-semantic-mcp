"""Sparse BM25 vector encoding."""

from __future__ import annotations

from collections import Counter

from services.repo_semantic.lexical.analyzer import sparse_terms
from services.repo_semantic.lexical.constants import BM25_B, BM25_K1
from services.repo_semantic.lexical.manifest import SparseManifest, fallback_idf
from services.repo_semantic.lexical.types import EncodedSparseVector


def encode_sparse_document(text: str, manifest: SparseManifest) -> EncodedSparseVector:
    """Encode chunk text as a BM25-weighted sparse document vector."""

    terms = sparse_terms(text)
    if not terms:
        return EncodedSparseVector(indices=[], values=[])
    term_counts = Counter(term for term in terms if term in manifest.token_to_id)
    if not term_counts:
        return EncodedSparseVector(indices=[], values=[])

    doc_len = len(terms)
    avg_doc_len = manifest.avg_doc_len if manifest.avg_doc_len > 0 else max(doc_len, 1)
    denominator_base = BM25_K1 * (1.0 - BM25_B + BM25_B * (doc_len / avg_doc_len))
    pairs: list[tuple[int, float]] = []
    for term, tf in term_counts.items():
        token_id = manifest.token_to_id[term]
        idf_value = manifest.idf.get(term, fallback_idf(manifest))
        weight = idf_value * ((tf * (BM25_K1 + 1.0)) / (tf + denominator_base))
        if weight > 0:
            pairs.append((token_id, float(weight)))
    pairs.sort(key=lambda item: item[0])
    return EncodedSparseVector(
        indices=[index for index, _ in pairs],
        values=[value for _, value in pairs],
    )


def encode_sparse_query(query: str, manifest: SparseManifest) -> EncodedSparseVector:
    """Encode query text using manifest vocabulary without rewriting raw dense text."""

    counts = Counter(term for term in sparse_terms(query) if term in manifest.token_to_id)
    pairs = sorted(
        (manifest.token_to_id[term], float(tf))
        for term, tf in counts.items()
        if tf > 0
    )
    return EncodedSparseVector(
        indices=[index for index, _ in pairs],
        values=[value for _, value in pairs],
    )
