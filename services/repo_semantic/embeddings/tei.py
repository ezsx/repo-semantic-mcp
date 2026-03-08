"""Remote embedding provider for Hugging Face TEI."""

from __future__ import annotations

import httpx

from services.repo_semantic.embeddings.base import EmbeddingProvider


class TeiProvider(EmbeddingProvider):
    """Embedding provider, работающий через HTTP API TEI."""

    def __init__(self, base_url: str, model_name: str, timeout_sec: int = 60) -> None:
        """Сохранить параметры подключения к TEI."""

        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._timeout_sec = timeout_sec

    def _post_openai_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Отправить запрос в OpenAI-compatible `/v1/embeddings` endpoint."""

        response = httpx.post(
            f"{self._base_url}/v1/embeddings",
            json={"input": texts, "model": self._model_name},
            timeout=self._timeout_sec,
        )
        response.raise_for_status()
        payload = response.json()
        return [item["embedding"] for item in payload["data"]]

    def _post_native_embed(self, texts: list[str]) -> list[list[float]]:
        """Отправить запрос в native `/embed` endpoint TEI."""

        response = httpx.post(
            f"{self._base_url}/embed",
            json={"inputs": texts},
            timeout=self._timeout_sec,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and "embeddings" in payload:
            return payload["embeddings"]
        return payload

    def _embed_with_split(self, texts: list[str]) -> list[list[float]]:
        """Повторить embedding, дробя батч при `413 Payload Too Large`."""

        if not texts:
            return []

        try:
            return self._post_openai_embeddings(texts)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 413:
                raise
        except httpx.HTTPError:
            pass

        try:
            return self._post_native_embed(texts)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 413:
                raise
        except httpx.HTTPError:
            pass

        if len(texts) == 1:
            return [self._embed_single_text_with_split(texts[0])]

        middle = max(1, len(texts) // 2)
        return self._embed_with_split(texts[:middle]) + self._embed_with_split(texts[middle:])

    def _embed_single_text_with_split(self, text: str) -> list[float]:
        """Построить embedding для одного oversized текста через усреднение частей."""

        if len(text) <= 1:
            raise RuntimeError("TEI rejected even a minimal single-document request")

        split_at = max(1, len(text) // 2)
        left = text[:split_at]
        right = text[split_at:]
        left_vector = self._embed_with_split([left])[0]
        right_vector = self._embed_with_split([right])[0]
        return [
            (left_value + right_value) / 2.0
            for left_value, right_value in zip(left_vector, right_vector, strict=True)
        ]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Преобразовать документы в embeddings через TEI."""

        return self._embed_with_split(texts)

    def embed_query(self, text: str) -> list[float]:
        """Преобразовать поисковый запрос в embedding."""

        return self.embed_documents([text])[0]

    def backend_name(self) -> str:
        """Вернуть backend name."""

        return "tei_http"

    def model_name(self) -> str:
        """Вернуть имя embedding модели."""

        return self._model_name

    def healthcheck(self) -> None:
        """Проверить доступность TEI сервиса."""

        response = httpx.get(f"{self._base_url}/health", timeout=10)
        response.raise_for_status()
