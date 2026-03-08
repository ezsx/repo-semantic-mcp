"""Qdrant-backed vector store for repo semantic search."""

from __future__ import annotations

from typing import Iterable
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from services.repo_semantic.config import SemanticMcpSettings
from services.repo_semantic.logging import jlog
from services.repo_semantic.models import ChunkRecord


class QdrantStore:
    """Обертка над Qdrant с двумя логическими коллекциями: code и docs."""

    def __init__(self, settings: SemanticMcpSettings) -> None:
        """Создать Qdrant client из env settings."""

        self._settings = settings
        self._client = QdrantClient(
            url=settings.SEMANTIC_MCP_QDRANT_URL,
            api_key=settings.SEMANTIC_MCP_QDRANT_API_KEY,
            timeout=60,
        )

    @property
    def client(self) -> QdrantClient:
        """Вернуть underlying Qdrant client."""

        return self._client

    def collection_name(self, scope: str) -> str:
        """Вернуть physical collection name по logical scope."""

        if scope == "code":
            return self._settings.collection_code
        if scope == "docs":
            return self._settings.collection_docs
        raise ValueError(f"Unsupported scope: {scope}")

    def collection_exists(self, scope: str) -> bool:
        """Проверить существование коллекции."""

        return self._client.collection_exists(self.collection_name(scope))

    def recreate_collection(self, scope: str, vector_size: int) -> None:
        """Пересоздать коллекцию с нужным размером dense vector."""

        collection_name = self.collection_name(scope)
        if self._client.collection_exists(collection_name):
            self._client.delete_collection(collection_name)
        self._client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    def ensure_collection(self, scope: str, vector_size: int) -> None:
        """Создать коллекцию, если она еще не существует."""

        if not self.collection_exists(scope):
            self.recreate_collection(scope, vector_size)

    def _point_id(self, chunk_id: str) -> str:
        """Преобразовать внешний chunk id в Qdrant-safe UUID."""

        return str(uuid5(NAMESPACE_URL, chunk_id))

    def upsert_chunks(
        self,
        scope: str,
        chunks: list[ChunkRecord],
        vectors: list[list[float]],
        embedding_backend: str,
        embedding_model: str,
        schema_version: int,
    ) -> None:
        """Upsert chunk payloads и их dense vectors в Qdrant."""

        if not chunks:
            return

        max_points = max(1, self._settings.SEMANTIC_MCP_QDRANT_UPSERT_BATCH_POINTS)
        max_bytes = max(1, self._settings.SEMANTIC_MCP_QDRANT_UPSERT_MAX_BYTES)

        batch: list[models.PointStruct] = []
        batch_estimated_bytes = 0
        uploaded_points = 0
        batch_index = 0
        total_points = len(chunks)

        def flush_batch() -> None:
            """Отправить накопленный батч векторных точек в Qdrant."""

            nonlocal batch, batch_estimated_bytes, uploaded_points, batch_index
            if not batch:
                return
            batch_index += 1
            self._client.upsert(collection_name=self.collection_name(scope), points=batch)
            uploaded_points += len(batch)
            jlog(
                "info",
                "semantic_qdrant_upsert_progress",
                scope=scope,
                uploaded=uploaded_points,
                total=total_points,
                batch_index=batch_index,
            )
            batch = []
            batch_estimated_bytes = 0

        for chunk, vector in zip(chunks, vectors, strict=True):
            payload = {
                "chunk_id": chunk.point_id,
                "scope": chunk.scope,
                "relative_path": chunk.relative_path,
                "language": chunk.language,
                "chunk_type": chunk.chunk_type,
                "text": chunk.text,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "content_hash": chunk.content_hash,
                "source_mtime": chunk.source_mtime,
                "symbol_path": chunk.symbol_path,
                "heading_path": chunk.heading_path,
                "domain_tags": chunk.domain_tags,
                "is_generated": chunk.is_generated,
                "embedding_backend": embedding_backend,
                "embedding_model": embedding_model,
                "index_schema_version": schema_version,
                **chunk.extra,
            }
            point = models.PointStruct(
                id=self._point_id(chunk.point_id),
                vector=vector,
                payload=payload,
            )
            estimated_point_bytes = (
                len(chunk.text.encode("utf-8"))
                + len(chunk.relative_path.encode("utf-8"))
                + len(chunk.language.encode("utf-8"))
                + len(chunk.chunk_type.encode("utf-8"))
                + sum(len(tag.encode("utf-8")) for tag in chunk.domain_tags)
                + len(vector) * 16
                + 2048
            )
            if batch and (
                len(batch) >= max_points
                or batch_estimated_bytes + estimated_point_bytes > max_bytes
            ):
                flush_batch()
            batch.append(point)
            batch_estimated_bytes += estimated_point_bytes

        flush_batch()

    def search(self, scope: str, query_vector: list[float], limit: int):
        """Выполнить dense search по scope collection."""

        if not self.collection_exists(scope):
            return []
        response = self._client.query_points(
            collection_name=self.collection_name(scope),
            query=query_vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return response.points

    def get_chunk(self, scope: str, chunk_id: str):
        """Получить point по идентификатору чанка."""

        if not self.collection_exists(scope):
            return None
        result = self._client.retrieve(
            collection_name=self.collection_name(scope),
            ids=[self._point_id(chunk_id)],
            with_payload=True,
            with_vectors=False,
        )
        return result[0] if result else None

    def scroll_chunks(self, scope: str) -> list:
        """Прочитать все чанки коллекции с payload для lexical поиска."""

        if not self.collection_exists(scope):
            return []

        collection_name = self.collection_name(scope)
        offset = None
        result = []
        while True:
            page, offset = self._client.scroll(
                collection_name=collection_name,
                with_payload=True,
                with_vectors=False,
                limit=256,
                offset=offset,
            )
            result.extend(page)
            if offset is None:
                break
        return result

    def delete_file_chunks(self, scope: str, relative_path: str) -> None:
        """Удалить все point'ы, соответствующие конкретному файлу."""

        if not self.collection_exists(scope):
            return
        points = [
            point.id
            for point in self.scroll_chunks(scope)
            if (point.payload or {}).get("relative_path") == relative_path
        ]
        if points:
            self._client.delete(
                collection_name=self.collection_name(scope),
                points_selector=models.PointIdsList(points=points),
            )

    def count(self, scope: str) -> int:
        """Вернуть количество point'ов в коллекции."""

        if not self.collection_exists(scope):
            return 0
        return int(
            self._client.count(
                collection_name=self.collection_name(scope),
                exact=True,
            ).count
        )

    def healthcheck(self) -> None:
        """Проверить доступность Qdrant."""

        self._client.get_collections()

    def iter_existing_scopes(self) -> Iterable[str]:
        """Вернуть список реально существующих logical collections."""

        for scope in ("code", "docs"):
            if self.collection_exists(scope):
                yield scope
