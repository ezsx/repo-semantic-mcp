"""Qdrant collection naming helpers."""

from __future__ import annotations

from services.repo_semantic.config import SemanticMcpSettings


def collection_name_for_schema(
    settings: SemanticMcpSettings,
    *,
    scope: str,
    schema_version: int,
) -> str:
    """Return physical collection name for a logical scope and schema."""

    if scope == "code":
        suffix = "code"
    elif scope == "docs":
        suffix = "docs"
    else:
        raise ValueError(f"Unsupported scope: {scope}")
    return (
        f"{settings.SEMANTIC_MCP_COLLECTION_PREFIX}_{settings.repo_key_slug}_"
        f"{settings.profile_slug}_{settings.embedding_model_slug}_"
        f"{suffix}_v{schema_version}"
    )


def current_collection_name(settings: SemanticMcpSettings, *, scope: str) -> str:
    """Return current physical collection name for a logical scope."""

    return collection_name_for_schema(
        settings,
        scope=scope,
        schema_version=settings.SEMANTIC_MCP_INDEX_SCHEMA_VERSION,
    )
