"""Search filter normalization, matching, and Qdrant pushdown planning."""

from __future__ import annotations

from fnmatch import fnmatchcase
from pathlib import PurePosixPath
import re

from qdrant_client import models as qdrant_models

from services.repo_semantic.models import (
    ChunkRecord,
    PayloadCapabilities,
    PayloadIndexStatus,
    SearchFilters,
)
from services.repo_semantic.retrieval.constants import (
    EXPECTED_PAYLOAD_INDEX_FIELDS,
    MAX_FILTER_VALUES,
    MAX_GLOB_CHARS,
)
from services.repo_semantic.retrieval.types import _FilterPlan


def normalize_filter_values(
    values: list[str] | None,
    *,
    family: str,
    lower: bool = False,
) -> list[str]:
    """Normalize generic non-path filter values with a bounded cardinality."""

    if values is None:
        return []
    if len(values) > MAX_FILTER_VALUES:
        raise ValueError(f"{family} has too many values; max {MAX_FILTER_VALUES}")
    normalized: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item:
            continue
        normalized.append(item.lower() if lower else item)
    return list(dict.fromkeys(normalized))


def normalize_file_extensions(values: list[str] | None) -> list[str]:
    """Normalize extension filters to lower-case values with a leading dot."""

    normalized = normalize_filter_values(values, family="file_extensions", lower=True)
    result: list[str] = []
    for value in normalized:
        if "/" in value or "\\" in value:
            raise ValueError("file_extensions must not contain path separators")
        result.append(value if value.startswith(".") else f".{value}")
    return list(dict.fromkeys(result))


def normalize_filter_path(value: str, *, family: str, allow_glob: bool) -> str:
    """Normalize repo-relative POSIX paths and reject unsafe path shapes."""

    raw = str(value).strip()
    if not raw:
        raise ValueError(f"{family} contains an empty path")
    if len(raw) > MAX_GLOB_CHARS:
        raise ValueError(f"{family} path is too long; max {MAX_GLOB_CHARS} characters")
    if raw.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", raw):
        raise ValueError(f"{family} must be repo-relative")
    normalized = raw.replace("\\", "/").rstrip("/")
    if not normalized:
        raise ValueError(f"{family} contains an empty path")
    segments = normalized.split("/")
    if any(segment == "" or segment == ".." for segment in segments):
        raise ValueError(f"{family} contains unsafe path segments")
    if not allow_glob and any("*" in segment or "?" in segment or "[" in segment for segment in segments):
        raise ValueError(f"{family} must be a plain path prefix")
    return normalized


def normalize_path_filters(
    values: list[str] | None,
    *,
    family: str,
) -> list[str]:
    """Normalize include/exclude glob-like repo-relative path filters."""

    if values is None:
        return []
    if len(values) > MAX_FILTER_VALUES:
        raise ValueError(f"{family} has too many values; max {MAX_FILTER_VALUES}")
    normalized = [
        normalize_filter_path(value, family=family, allow_glob=True)
        for value in values
    ]
    return list(dict.fromkeys(normalized))


def build_filters(
    *,
    path_prefix: str | None = None,
    include_paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
    file_extensions: list[str] | None = None,
    languages: list[str] | None = None,
    chunk_types: list[str] | None = None,
    domain_tags: list[str] | None = None,
) -> SearchFilters:
    """Build the normalized internal filter model shared by search modes."""

    normalized_path_prefix = (
        normalize_filter_path(path_prefix, family="path_prefix", allow_glob=False)
        if path_prefix is not None and str(path_prefix).strip()
        else None
    )
    return SearchFilters(
        path_prefix=normalized_path_prefix,
        include_paths=normalize_path_filters(include_paths, family="include_paths"),
        exclude_paths=normalize_path_filters(exclude_paths, family="exclude_paths"),
        file_extensions=normalize_file_extensions(file_extensions),
        languages=normalize_filter_values(languages, family="languages"),
        chunk_types=normalize_filter_values(chunk_types, family="chunk_types"),
        domain_tags=normalize_filter_values(domain_tags, family="domain_tags"),
    )


def filters_active(filters: SearchFilters) -> bool:
    """Return whether client-side filters can reduce candidate recall."""

    return bool(
        filters.path_prefix
        or filters.include_paths
        or filters.exclude_paths
        or filters.file_extensions
        or filters.languages
        or filters.chunk_types
        or filters.domain_tags
    )


def normalize_chunk_path(relative_path: str) -> str:
    """Normalize indexed paths for matching without changing stored payloads."""

    return relative_path.replace("\\", "/").strip("/")


def prefix_matches(relative_path: str, path_prefix: str) -> bool:
    """Boundary-aware prefix predicate for the legacy path_prefix alias."""

    normalized_path = normalize_chunk_path(relative_path)
    return normalized_path == path_prefix or normalized_path.startswith(f"{path_prefix}/")


def path_glob_matches(relative_path: str, pattern: str) -> bool:
    """Match repo-relative paths with fnmatchcase plus directory exactness."""

    normalized_path = normalize_chunk_path(relative_path)
    return fnmatchcase(normalized_path, pattern)


def matches_filters(chunk: ChunkRecord, filters: SearchFilters) -> bool:
    """Return whether a chunk passes client-side filters."""

    relative_path = normalize_chunk_path(chunk.relative_path)
    if filters.path_prefix and not prefix_matches(relative_path, filters.path_prefix):
        return False
    if filters.include_paths and not any(
        path_glob_matches(relative_path, pattern)
        for pattern in filters.include_paths
    ):
        return False
    if filters.exclude_paths and any(
        path_glob_matches(relative_path, pattern)
        for pattern in filters.exclude_paths
    ):
        return False
    if filters.file_extensions and PurePosixPath(relative_path).suffix.lower() not in set(filters.file_extensions):
        return False
    if filters.languages and chunk.language not in set(filters.languages):
        return False
    if filters.chunk_types and chunk.chunk_type not in set(filters.chunk_types):
        return False
    if filters.domain_tags and not set(filters.domain_tags).intersection(chunk.domain_tags):
        return False
    return True


def payload_index_statuses(
    *,
    scope: str,
    collection_name: str,
    index_fields: set[str] | None,
) -> list[PayloadIndexStatus]:
    """Return best-effort payload index presence for status diagnostics."""

    statuses: list[PayloadIndexStatus] = []
    for field_name in sorted(EXPECTED_PAYLOAD_INDEX_FIELDS):
        present = field_name in index_fields if index_fields is not None else None
        warning_code = None
        if index_fields is None:
            warning_code = "payload_index_presence_not_checked"
        elif not present:
            warning_code = "payload_index_missing"
        statuses.append(
            PayloadIndexStatus(
                scope=scope,  # type: ignore[arg-type]
                collection_name=collection_name,
                field_name=field_name,
                present=present,
                warning_code=warning_code,
            )
        )
    return statuses


def filter_plan_for_scope(
    *,
    scope: str,
    collection_name: str,
    payload_fields: set[str],
    filters: SearchFilters,
) -> _FilterPlan:
    """Build Qdrant payload filter plan while preserving legacy fallback semantics."""

    pushed: list[str] = []
    post: list[str] = []
    required_fields: list[str] = []
    missing_fields: list[str] = []
    conditions: list[qdrant_models.FieldCondition] = []

    def add_condition(
        family: str,
        field_name: str,
        condition,
    ) -> None:
        required_fields.append(field_name)
        if field_name not in payload_fields:
            post.append(family)
            missing_fields.append(field_name)
            return
        pushed.append(family)
        conditions.append(condition)

    if filters.path_prefix:
        add_condition(
            "path_prefix",
            "path_prefixes",
            qdrant_models.FieldCondition(
                key="path_prefixes",
                match=qdrant_models.MatchValue(value=filters.path_prefix),
            ),
        )
    if filters.file_extensions:
        add_condition(
            "file_extensions",
            "file_extension",
            qdrant_models.FieldCondition(
                key="file_extension",
                match=qdrant_models.MatchAny(any=filters.file_extensions),
            ),
        )
    if filters.languages:
        add_condition(
            "languages",
            "language",
            qdrant_models.FieldCondition(
                key="language",
                match=qdrant_models.MatchAny(any=filters.languages),
            ),
        )
    if filters.chunk_types:
        add_condition(
            "chunk_types",
            "chunk_type",
            qdrant_models.FieldCondition(
                key="chunk_type",
                match=qdrant_models.MatchAny(any=filters.chunk_types),
            ),
        )
    if filters.domain_tags:
        add_condition(
            "domain_tags",
            "domain_tags",
            qdrant_models.FieldCondition(
                key="domain_tags",
                match=qdrant_models.MatchAny(any=filters.domain_tags),
            ),
        )
    if filters.include_paths:
        post.append("include_paths")
    if filters.exclude_paths:
        post.append("exclude_paths")

    best_effort = bool(post)
    warning_codes: list[str] = []
    if pushed:
        warning_codes.append("qdrant_filter_pushdown_used")
    if missing_fields:
        warning_codes.append("qdrant_filter_pushdown_partial")
    if best_effort:
        warning_codes.append("qdrant_filter_post_filter_best_effort")
    qdrant_filter = qdrant_models.Filter(must=conditions) if conditions else None
    return _FilterPlan(
        qdrant_filter=qdrant_filter,
        pushed_families=list(dict.fromkeys(pushed)),
        post_filter_families=list(dict.fromkeys(post)),
        required_payload_fields=list(dict.fromkeys(required_fields)),
        missing_payload_fields=sorted(set(missing_fields)),
        best_effort=best_effort,
        warning_codes=list(dict.fromkeys(warning_codes)),
        payload_capabilities=PayloadCapabilities(
            scope=scope,  # type: ignore[arg-type]
            collection_name=collection_name,
            checked=bool(payload_fields),
            available_fields=sorted(payload_fields),
            missing_fields=sorted(set(missing_fields)),
        ),
    )


def merge_filter_plans(plans: list[_FilterPlan]) -> tuple[list[str], list[str], bool, list[PayloadCapabilities]]:
    """Merge per-scope filter plans into response diagnostics."""

    pushed = sorted({family for plan in plans for family in plan.pushed_families})
    post = sorted({family for plan in plans for family in plan.post_filter_families})
    best_effort = any(plan.best_effort for plan in plans)
    capabilities = [plan.payload_capabilities for plan in plans]
    return pushed, post, best_effort, capabilities
