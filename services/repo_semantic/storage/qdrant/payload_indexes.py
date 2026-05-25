"""Qdrant payload index specifications."""

from __future__ import annotations

from qdrant_client import models


def payload_index_specs() -> dict[str, models.PayloadSchemaType]:
    """Return payload indexes used by filtering and status diagnostics."""

    return {
        "relative_path": models.PayloadSchemaType.KEYWORD,
        "scope": models.PayloadSchemaType.KEYWORD,
        "file_extension": models.PayloadSchemaType.KEYWORD,
        "language": models.PayloadSchemaType.KEYWORD,
        "chunk_type": models.PayloadSchemaType.KEYWORD,
        "domain_tags": models.PayloadSchemaType.KEYWORD,
        "is_generated": models.PayloadSchemaType.BOOL,
        "content_hash": models.PayloadSchemaType.KEYWORD,
        "path_prefixes": models.PayloadSchemaType.KEYWORD,
    }
