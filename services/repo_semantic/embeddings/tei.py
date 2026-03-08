"""Remote embedding provider for Hugging Face TEI."""

from __future__ import annotations

import httpx

from services.repo_semantic.embeddings.base import EmbeddingProvider


class TeiProvider(EmbeddingProvider):
    """Embedding provider, работающий через HTTP API TEI."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        timeout_sec: int = 60,
        query_template: str = "{query}",
        document_prefix: str = "",
        query_prompt_name: str | None = None,
        document_prompt_name: str | None = None,
        profile_name: str = "",
    ) -> None:
        """Сохранить параметры подключения к TEI."""

        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._timeout_sec = timeout_sec
        self._query_template = query_template
        self._document_prefix = document_prefix
        self._query_prompt_name = query_prompt_name.strip() if query_prompt_name else None
        self._document_prompt_name = document_prompt_name.strip() if document_prompt_name else None
        self._profile_name = profile_name.strip()

    def _format_documents(self, texts: list[str]) -> list[str]:
        """Подготовить документы перед embedding."""

        if not self._document_prefix:
            return texts
        return [f"{self._document_prefix}{text}" for text in texts]

    def _format_query(self, text: str) -> str:
        """Подготовить запрос перед embedding."""

        return self._query_template.replace("{query}", text)

    def _post_openai_embeddings(
        self,
        texts: list[str],
        prompt_name: str | None = None,
    ) -> list[list[float]]:
        """Отправить запрос в OpenAI-compatible `/v1/embeddings` endpoint."""

        payload = {"input": texts, "model": self._model_name}
        if prompt_name:
            payload["prompt_name"] = prompt_name
        response = httpx.post(
            f"{self._base_url}/v1/embeddings",
            json=payload,
            timeout=self._timeout_sec,
        )
        response.raise_for_status()
        payload = response.json()
        return [item["embedding"] for item in payload["data"]]

    def _post_native_embed(
        self,
        texts: list[str],
        prompt_name: str | None = None,
    ) -> list[list[float]]:
        """Отправить запрос в native `/embed` endpoint TEI."""

        payload = {"inputs": texts}
        if prompt_name:
            payload["prompt_name"] = prompt_name
        response = httpx.post(
            f"{self._base_url}/embed",
            json=payload,
            timeout=self._timeout_sec,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and "embeddings" in payload:
            return payload["embeddings"]
        return payload

    def _embed_with_split(
        self,
        texts: list[str],
        prompt_name: str | None = None,
    ) -> list[list[float]]:
        """Повторить embedding, дробя батч при `413 Payload Too Large`."""

        if not texts:
            return []

        try:
            return self._post_openai_embeddings(texts, prompt_name=prompt_name)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 413:
                raise
        except httpx.HTTPError:
            pass

        try:
            return self._post_native_embed(texts, prompt_name=prompt_name)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 413:
                raise
        except httpx.HTTPError:
            pass

        if len(texts) == 1:
            return [self._embed_single_text_with_split(texts[0], prompt_name=prompt_name)]

        middle = max(1, len(texts) // 2)
        return self._embed_with_split(
            texts[:middle],
            prompt_name=prompt_name,
        ) + self._embed_with_split(
            texts[middle:],
            prompt_name=prompt_name,
        )

    def _embed_single_text_with_split(
        self,
        text: str,
        prompt_name: str | None = None,
    ) -> list[float]:
        """Построить embedding для одного oversized текста через усреднение частей."""

        if len(text) <= 1:
            raise RuntimeError("TEI rejected even a minimal single-document request")

        split_at = max(1, len(text) // 2)
        left = text[:split_at]
        right = text[split_at:]
        left_vector = self._embed_with_split([left], prompt_name=prompt_name)[0]
        right_vector = self._embed_with_split([right], prompt_name=prompt_name)[0]
        return [
            (left_value + right_value) / 2.0
            for left_value, right_value in zip(left_vector, right_vector, strict=True)
        ]

    def index_profile(self) -> str:
        """Вернуть профиль индекса, влияющий на совместимость dense vectors."""

        if self._profile_name:
            return self._profile_name
        if (
            self._query_template != "{query}"
            or self._document_prefix
            or self._query_prompt_name
            or self._document_prompt_name
        ):
            return "formatted"
        return "default"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Преобразовать документы в embeddings через TEI."""

        return self._embed_with_split(
            self._format_documents(texts),
            prompt_name=self._document_prompt_name,
        )

    def embed_query(self, text: str) -> list[float]:
        """Преобразовать поисковый запрос в embedding."""

        return self._embed_with_split(
            [self._format_query(text)],
            prompt_name=self._query_prompt_name,
        )[0]

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
