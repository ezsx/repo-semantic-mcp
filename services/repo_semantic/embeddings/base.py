"""Base protocol for embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Абстракция над локальным или удалённым embedding backend."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Преобразовать список документов в dense vectors."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Преобразовать пользовательский запрос в dense vector."""

    @abstractmethod
    def backend_name(self) -> str:
        """Вернуть имя backend provider."""

    @abstractmethod
    def model_name(self) -> str:
        """Вернуть имя embedding модели."""

    @abstractmethod
    def healthcheck(self) -> None:
        """Проверить доступность backend и бросить исключение при ошибке."""

