"""Local embedding provider powered by FastEmbed."""

from __future__ import annotations

from fastembed import TextEmbedding

from services.repo_semantic.embeddings.base import EmbeddingProvider


class FastEmbedProvider(EmbeddingProvider):
    """Локальный embedding provider для быстрого первого rollout."""

    def __init__(
        self,
        model_name: str,
        query_template: str = "{query}",
        document_prefix: str = "",
        profile_name: str = "",
    ) -> None:
        """Инициализировать FastEmbed модель."""

        self._model_name = model_name
        self._query_template = query_template
        self._document_prefix = document_prefix
        self._profile_name = profile_name.strip()
        self._model = TextEmbedding(model_name=model_name)

    def _format_documents(self, texts: list[str]) -> list[str]:
        """Подготовить документы перед embedding."""

        if not self._document_prefix:
            return texts
        return [f"{self._document_prefix}{text}" for text in texts]

    def _format_query(self, text: str) -> str:
        """Подготовить запрос перед embedding."""

        return self._query_template.replace("{query}", text)

    def index_profile(self) -> str:
        """Вернуть профиль индекса, влияющий на совместимость dense vectors."""

        if self._profile_name:
            return self._profile_name
        if self._query_template != "{query}" or self._document_prefix:
            return "formatted"
        return "default"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Векторизовать список документов."""

        if not texts:
            return []
        return [list(vector) for vector in self._model.embed(self._format_documents(texts))]

    def embed_query(self, text: str) -> list[float]:
        """Векторизовать поисковый запрос."""

        return [list(vector) for vector in self._model.embed([self._format_query(text)])][0]

    def backend_name(self) -> str:
        """Вернуть backend name."""

        return "fastembed_local"

    def model_name(self) -> str:
        """Вернуть имя embedding модели."""

        return self._model_name

    def healthcheck(self) -> None:
        """Проверить, что модель уже может строить embeddings."""

        self.embed_query("semantic healthcheck")
