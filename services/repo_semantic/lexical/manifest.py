"""Sparse manifest model, hashing, and filesystem persistence."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from services.repo_semantic.config import SemanticMcpSettings
from services.repo_semantic.lexical.analyzer import sparse_terms, sparse_unique_terms
from services.repo_semantic.lexical.constants import (
    BM25_B,
    BM25_K1,
    LEXICAL_ANALYZER_VERSION,
    SPARSE_ENCODER_KIND,
    SPARSE_MANIFEST_CONTRACT_VERSION,
)
from services.repo_semantic.models import ChunkRecord


class SparseManifest(BaseModel):
    """Persisted sparse vocabulary and BM25 corpus statistics for one scope."""

    contract_version: str = SPARSE_MANIFEST_CONTRACT_VERSION
    scope: Literal["code", "docs"]
    schema_version: int
    lexical_analyzer_version: str = LEXICAL_ANALYZER_VERSION
    sparse_encoder_kind: str = SPARSE_ENCODER_KIND
    bm25_k1: float = BM25_K1
    bm25_b: float = BM25_B
    token_to_id: dict[str, int] = Field(default_factory=dict)
    doc_freqs: dict[str, int] = Field(default_factory=dict)
    idf: dict[str, float] = Field(default_factory=dict)
    doc_count: int = 0
    avg_doc_len: float = 0.0
    vocabulary_hash: str
    corpus_stats_hash: str
    immutable_sparse_contract_hash: str
    manifest_content_hash: str
    sparse_stats_stale: bool = False
    intentionally_empty: bool = False
    built_at: str
    updated_at: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_json(payload: object) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def sparse_contract_hash(*, schema_version: int) -> str:
    """Hash the immutable sparse encoder contract, excluding corpus data."""

    return _sha256_json(
        {
            "contract_version": SPARSE_MANIFEST_CONTRACT_VERSION,
            "schema_version": schema_version,
            "lexical_analyzer_version": LEXICAL_ANALYZER_VERSION,
            "sparse_encoder_kind": SPARSE_ENCODER_KIND,
            "bm25_k1": BM25_K1,
            "bm25_b": BM25_B,
        }
    )


def _idf(doc_count: int, doc_freq: int) -> float:
    if doc_count <= 0 or doc_freq <= 0:
        return 0.0
    return math.log(((doc_count - doc_freq + 0.5) / (doc_freq + 0.5)) + 1.0)


def _vocabulary_hash(token_to_id: dict[str, int]) -> str:
    return _sha256_json({"token_to_id": token_to_id})


def _corpus_stats_hash(
    *,
    doc_count: int,
    avg_doc_len: float,
    doc_freqs: dict[str, int],
    idf: dict[str, float],
) -> str:
    return _sha256_json(
        {
            "doc_count": doc_count,
            "avg_doc_len": round(avg_doc_len, 12),
            "doc_freqs": doc_freqs,
            "idf": {term: round(value, 12) for term, value in idf.items()},
        }
    )


def _manifest_content_hash(manifest_payload: dict[str, object]) -> str:
    excluded = {"built_at", "updated_at", "manifest_content_hash"}
    return _sha256_json(
        {
            key: value
            for key, value in manifest_payload.items()
            if key not in excluded
        }
    )


def build_sparse_manifest(
    *,
    scope: Literal["code", "docs"],
    chunks: list[ChunkRecord],
    schema_version: int,
) -> SparseManifest:
    """Build a deterministic manifest from the full scoped chunk corpus."""

    ordered_chunks = sorted(
        chunks,
        key=lambda chunk: (chunk.relative_path, chunk.start_line, chunk.end_line, chunk.point_id),
    )
    docs_terms = [sparse_terms(chunk.text) for chunk in ordered_chunks]
    doc_count = len(docs_terms)
    doc_lengths = [len(terms) for terms in docs_terms]
    avg_doc_len = sum(doc_lengths) / doc_count if doc_count else 0.0

    doc_freq_counter: Counter[str] = Counter()
    for terms in docs_terms:
        doc_freq_counter.update(set(terms))
    doc_freqs = dict(sorted(doc_freq_counter.items()))
    token_to_id = {term: index for index, term in enumerate(sorted(doc_freqs), start=1)}
    idf = {term: _idf(doc_count, doc_freq) for term, doc_freq in doc_freqs.items()}

    payload = {
        "contract_version": SPARSE_MANIFEST_CONTRACT_VERSION,
        "scope": scope,
        "schema_version": schema_version,
        "lexical_analyzer_version": LEXICAL_ANALYZER_VERSION,
        "sparse_encoder_kind": SPARSE_ENCODER_KIND,
        "bm25_k1": BM25_K1,
        "bm25_b": BM25_B,
        "token_to_id": token_to_id,
        "doc_freqs": doc_freqs,
        "idf": idf,
        "doc_count": doc_count,
        "avg_doc_len": avg_doc_len,
        "vocabulary_hash": _vocabulary_hash(token_to_id),
        "corpus_stats_hash": _corpus_stats_hash(
            doc_count=doc_count,
            avg_doc_len=avg_doc_len,
            doc_freqs=doc_freqs,
            idf=idf,
        ),
        "immutable_sparse_contract_hash": sparse_contract_hash(schema_version=schema_version),
        "manifest_content_hash": "",
        "sparse_stats_stale": False,
        "intentionally_empty": doc_count == 0,
        "built_at": _utc_now(),
        "updated_at": _utc_now(),
    }
    payload["manifest_content_hash"] = _manifest_content_hash(payload)
    return SparseManifest.model_validate(payload)


def extend_manifest_for_incremental_chunks(
    manifest: SparseManifest,
    chunks: list[ChunkRecord],
) -> SparseManifest:
    """Append new vocabulary ids while preserving existing term ids."""

    if not chunks:
        return manifest

    token_to_id = dict(manifest.token_to_id)
    max_id = max(token_to_id.values(), default=0)
    new_terms = sorted(
        {
            term
            for chunk in chunks
            for term in sparse_unique_terms(chunk.text)
            if term not in token_to_id
        }
    )
    for term in new_terms:
        max_id += 1
        token_to_id[term] = max_id

    payload = manifest.model_dump(mode="python")
    payload["token_to_id"] = token_to_id
    payload["vocabulary_hash"] = _vocabulary_hash(token_to_id)
    payload["sparse_stats_stale"] = True
    payload["intentionally_empty"] = False
    payload["updated_at"] = _utc_now()
    payload["manifest_content_hash"] = _manifest_content_hash(payload)
    return SparseManifest.model_validate(payload)


def mark_manifest_stale(manifest: SparseManifest) -> SparseManifest:
    """Return a copy marked with non-blocking stale sparse statistics."""

    payload = manifest.model_dump(mode="python")
    payload["sparse_stats_stale"] = True
    payload["updated_at"] = _utc_now()
    payload["manifest_content_hash"] = _manifest_content_hash(payload)
    return SparseManifest.model_validate(payload)


def fallback_idf(manifest: SparseManifest) -> float:
    """Return a stable fallback IDF for terms not present in manifest stats."""

    return _idf(max(manifest.doc_count, 1), 1)


class SparseManifestStore:
    """Filesystem persistence for per-scope sparse manifests."""

    def __init__(self, settings: SemanticMcpSettings) -> None:
        self._settings = settings

    @property
    def root(self) -> Path:
        return (
            self._settings.registry_db_path.parent
            / "sparse_manifests"
            / self._settings.repo_key_slug
            / self._settings.profile_slug
        )

    def path_for_scope(self, scope: Literal["code", "docs"] | str) -> Path:
        if scope not in {"code", "docs"}:
            raise ValueError(f"Unsupported scope: {scope}")
        return self.root / f"{scope}.json"

    def save(self, manifest: SparseManifest) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = manifest.model_dump(mode="json")
        self.path_for_scope(manifest.scope).write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    def load(self, scope: Literal["code", "docs"] | str) -> SparseManifest | None:
        path = self.path_for_scope(scope)
        if not path.exists():
            return None
        try:
            return SparseManifest.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def delete(self, scope: Literal["code", "docs"] | str) -> None:
        path = self.path_for_scope(scope)
        if path.exists():
            path.unlink()
