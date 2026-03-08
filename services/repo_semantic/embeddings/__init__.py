"""Embedding providers for semantic MCP."""

from services.repo_semantic.config import SemanticMcpSettings
from services.repo_semantic.embeddings.base import EmbeddingProvider
from services.repo_semantic.embeddings.fastembed import FastEmbedProvider
from services.repo_semantic.embeddings.tei import TeiProvider


def build_embedding_provider(settings: SemanticMcpSettings) -> EmbeddingProvider:
    """Создать embedding provider по env contract."""

    backend = settings.SEMANTIC_MCP_EMBEDDING_BACKEND.strip().lower()
    if backend == "fastembed_local":
        return FastEmbedProvider(model_name=settings.SEMANTIC_MCP_EMBEDDING_MODEL)
    if backend == "tei_http":
        if not settings.SEMANTIC_MCP_TEI_URL:
            raise RuntimeError("SEMANTIC_MCP_TEI_URL is required for tei_http backend")
        return TeiProvider(
            base_url=settings.SEMANTIC_MCP_TEI_URL,
            model_name=settings.SEMANTIC_MCP_EMBEDDING_MODEL,
        )
    raise RuntimeError(f"Unsupported embedding backend: {settings.SEMANTIC_MCP_EMBEDDING_BACKEND}")

