"""Embedding providers for semantic MCP."""

from collections.abc import Callable

from services.repo_semantic.config import SemanticMcpSettings
from services.repo_semantic.embeddings.base import EmbeddingProvider


def _build_fastembed_provider(settings: SemanticMcpSettings) -> EmbeddingProvider:
    """Построить локальный FastEmbed provider."""

    from services.repo_semantic.embeddings.fastembed import FastEmbedProvider

    return FastEmbedProvider(
        model_name=settings.SEMANTIC_MCP_EMBEDDING_MODEL,
        query_template=settings.SEMANTIC_MCP_QUERY_TEMPLATE,
        document_prefix=settings.SEMANTIC_MCP_DOCUMENT_PREFIX,
        profile_name=settings.SEMANTIC_MCP_PROFILE_NAME,
    )


def _build_http_embedding_provider(settings: SemanticMcpSettings) -> EmbeddingProvider:
    """Построить HTTP provider для совместимых embedding backends."""

    from services.repo_semantic.embeddings.tei import TeiProvider

    if not settings.SEMANTIC_MCP_TEI_URL:
        raise RuntimeError(
            f"SEMANTIC_MCP_TEI_URL is required for {settings.SEMANTIC_MCP_EMBEDDING_BACKEND} backend"
        )
    return TeiProvider(
        base_url=settings.SEMANTIC_MCP_TEI_URL,
        model_name=settings.SEMANTIC_MCP_EMBEDDING_MODEL,
        backend_name=settings.SEMANTIC_MCP_EMBEDDING_BACKEND.strip().lower(),
        health_path=settings.SEMANTIC_MCP_EMBEDDING_HEALTH_PATH or "/health",
        query_template=settings.SEMANTIC_MCP_QUERY_TEMPLATE,
        document_prefix=settings.SEMANTIC_MCP_DOCUMENT_PREFIX,
        query_prompt_name=settings.SEMANTIC_MCP_TEI_QUERY_PROMPT_NAME,
        document_prompt_name=settings.SEMANTIC_MCP_TEI_DOCUMENT_PROMPT_NAME,
        profile_name=settings.SEMANTIC_MCP_PROFILE_NAME,
    )


_EMBEDDING_BACKEND_BUILDERS: dict[str, Callable[[SemanticMcpSettings], EmbeddingProvider]] = {
    "fastembed_local": _build_fastembed_provider,
    "tei_http": _build_http_embedding_provider,
    "tei": _build_http_embedding_provider,
    "openai_embeddings_http": _build_http_embedding_provider,
    "custom_http": _build_http_embedding_provider,
}


def build_embedding_provider(settings: SemanticMcpSettings) -> EmbeddingProvider:
    """Создать embedding provider по env contract."""

    backend = settings.SEMANTIC_MCP_EMBEDDING_BACKEND.strip().lower()
    builder = _EMBEDDING_BACKEND_BUILDERS.get(backend)
    if builder is None:
        raise RuntimeError(f"Unsupported embedding backend: {settings.SEMANTIC_MCP_EMBEDDING_BACKEND}")
    return builder(settings)
