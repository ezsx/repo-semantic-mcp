"""Embedding providers for semantic MCP."""

from services.repo_semantic.config import SemanticMcpSettings
from services.repo_semantic.embeddings.base import EmbeddingProvider
from services.repo_semantic.embeddings.fastembed import FastEmbedProvider
from services.repo_semantic.embeddings.tei import TeiProvider


def build_embedding_provider(settings: SemanticMcpSettings) -> EmbeddingProvider:
    """Создать embedding provider по env contract."""

    backend = settings.SEMANTIC_MCP_EMBEDDING_BACKEND.strip().lower()
    if backend == "fastembed_local":
        return FastEmbedProvider(
            model_name=settings.SEMANTIC_MCP_EMBEDDING_MODEL,
            query_template=settings.SEMANTIC_MCP_QUERY_TEMPLATE,
            document_prefix=settings.SEMANTIC_MCP_DOCUMENT_PREFIX,
            profile_name=settings.SEMANTIC_MCP_PROFILE_NAME,
        )
    if backend == "tei_http":
        if not settings.SEMANTIC_MCP_TEI_URL:
            raise RuntimeError("SEMANTIC_MCP_TEI_URL is required for tei_http backend")
        return TeiProvider(
            base_url=settings.SEMANTIC_MCP_TEI_URL,
            model_name=settings.SEMANTIC_MCP_EMBEDDING_MODEL,
            query_template=settings.SEMANTIC_MCP_QUERY_TEMPLATE,
            document_prefix=settings.SEMANTIC_MCP_DOCUMENT_PREFIX,
            query_prompt_name=settings.SEMANTIC_MCP_TEI_QUERY_PROMPT_NAME,
            document_prompt_name=settings.SEMANTIC_MCP_TEI_DOCUMENT_PROMPT_NAME,
            profile_name=settings.SEMANTIC_MCP_PROFILE_NAME,
        )
    raise RuntimeError(f"Unsupported embedding backend: {settings.SEMANTIC_MCP_EMBEDDING_BACKEND}")
