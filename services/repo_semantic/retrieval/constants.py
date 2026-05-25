"""Retrieval constants shared across search helpers."""

from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[A-Za-zА-Яа-я0-9_./:-]+", re.UNICODE)
MAX_QUERY_CHARS = 1000
MAX_TOP_K = 100
MAX_FILTER_VALUES = 50
MAX_GLOB_CHARS = 240
MAX_SNIPPET_CHARS = 500
MAX_LINE_MATCHES = 5
MAX_EXPLANATIONS = 8
MAX_CONTEXT_SUBQUERIES = 8
MAX_CONTEXT_READ_ACTIONS = 5
MAX_CONTEXT_FILE_GROUPS = 20
MAX_CONTEXT_GROUP_CHUNK_IDS = 10
MAX_CONTEXT_GROUP_LINE_RANGES = 5
MAX_CONTEXT_UNCOVERED_TERMS = 10
DENSE_FILTERED_CANDIDATE_CAP = 320
SPARSE_CANDIDATE_CAP = 500
RRF_K = 60
RRF_DENSE_WEIGHT = 1.0
RRF_SPARSE_WEIGHT = 1.2
FILTER_PAYLOAD_FIELDS = {
    "path_prefixes",
    "file_extension",
    "language",
    "chunk_type",
    "domain_tags",
}
EXPECTED_PAYLOAD_INDEX_FIELDS = {
    "scope",
    "relative_path",
    "file_extension",
    "language",
    "chunk_type",
    "domain_tags",
    "is_generated",
    "content_hash",
    "path_prefixes",
}
CONTEXT_REBUILD_CODES = {
    "sparse_manifest_missing",
    "sparse_analyzer_version_mismatch",
    "sparse_encoder_kind_mismatch",
    "sparse_contract_mismatch",
    "sparse_vector_missing",
    "legacy_dense_only_collection",
}
UNCOVERED_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "where",
    "which",
    "with",
    "use",
    "used",
    "uses",
    "a",
    "в",
    "где",
    "для",
    "и",
    "или",
    "как",
    "на",
    "не",
    "по",
    "с",
    "что",
    "это",
}
