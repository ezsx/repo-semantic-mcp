"""Local embedding provider powered by FastEmbed."""

from __future__ import annotations

from fastembed import TextEmbedding

from services.repo_semantic.embeddings.base import EmbeddingProvider


class FastEmbedProvider(EmbeddingProvider):
    """Локальный embedding provider для быстрого первого rollout."""

    def __init__(self, model_name: str) -> None:
        """Инициализировать FastEmbed модель."""

        self._model_name = model_name
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Векторизовать список документов."""

        if not texts:
            return []
        return [list(vector) for vector in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        """Векторизовать поисковый запрос."""

        return self.embed_documents([text])[0]

    def backend_name(self) -> str:
        """Вернуть backend name."""

        return "fastembed_local"

    def model_name(self) -> str:
        """Вернуть имя embedding модели."""

        return self._model_name

    def healthcheck(self) -> None:
        """Проверить, что модель уже может строить embeddings."""

        self.embed_query("semantic healthcheck")

