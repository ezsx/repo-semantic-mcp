"""Qdrant-backed vector store for repo semantic search."""

from __future__ import annotations

from typing import Iterable, Iterator
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from services.repo_semantic.config import SemanticMcpSettings
from services.repo_semantic.logging import jlog
from services.repo_semantic.models import ChunkRecord
from services.repo_semantic.lexical import (
    DENSE_VECTOR_NAME,
    EncodedSparseVector,
    SPARSE_VECTOR_NAME,
)
from services.repo_semantic.storage.qdrant.payload_codec import (
    estimated_point_bytes,
    qdrant_point,
)
from services.repo_semantic.storage.qdrant.collection_names import (
    collection_name_for_schema,
    current_collection_name,
)
from services.repo_semantic.storage.qdrant.payload_indexes import payload_index_specs


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

        return current_collection_name(self._settings, scope=scope)

    def _collection_name_for_schema(self, scope: str, schema_version: int) -> str:
        """Return physical collection name for a logical scope and schema."""

        return collection_name_for_schema(
            self._settings,
            scope=scope,
            schema_version=schema_version,
        )

    def read_collection_name(self, scope: str) -> str:
        """Return current collection or a previous dense-compatible collection."""

        current_name = self.collection_name(scope)
        if self._client.collection_exists(current_name):
            return current_name
        for schema_version in range(
            self._settings.SEMANTIC_MCP_INDEX_SCHEMA_VERSION - 1,
            0,
            -1,
        ):
            legacy_name = self._collection_name_for_schema(scope, schema_version)
            if self._client.collection_exists(legacy_name):
                return legacy_name
        return current_name

    def read_collection_exists(self, scope: str) -> bool:
        """Return whether current or readable legacy collection exists."""

        return self._client.collection_exists(self.read_collection_name(scope))

    def current_collection_exists(self, scope: str) -> bool:
        """Return whether the current schema collection exists."""

        return self._client.collection_exists(self.collection_name(scope))

    def collection_exists(self, scope: str) -> bool:
        """Проверить существование текущей коллекции."""

        return self.current_collection_exists(scope)

    def _target_collection_name_for_read(self, scope: str) -> str | None:
        collection_name = self.read_collection_name(scope)
        if not self._client.collection_exists(collection_name):
            return None
        return collection_name

    def _target_collection_name_for_write(self, scope: str) -> str | None:
        collection_name = self.collection_name(scope)
        if not self._client.collection_exists(collection_name):
            return None
        return collection_name

    def delete_collection(self, scope: str) -> None:
        """Delete current and older schema collections for an explicit rebuild path."""

        for schema_version in range(1, self._settings.SEMANTIC_MCP_INDEX_SCHEMA_VERSION + 1):
            collection_name = self._collection_name_for_schema(scope, schema_version)
            if self._client.collection_exists(collection_name):
                self._client.delete_collection(collection_name)

    def recreate_collection(
        self,
        scope: str,
        vector_size: int,
        *,
        sparse_enabled: bool = False,
    ) -> None:
        """Recreate collection with dense vectors and optional sparse vectors."""

        collection_name = self.collection_name(scope)
        if self._client.collection_exists(collection_name):
            self._client.delete_collection(collection_name)
        vectors_config: models.VectorParams | dict[str, models.VectorParams]
        if sparse_enabled:
            vectors_config = {
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                )
            }
        else:
            vectors_config = models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            )
        self._client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=(
                {
                    SPARSE_VECTOR_NAME: models.SparseVectorParams(
                        index=models.SparseIndexParams(),
                        modifier=None,
                    )
                }
                if sparse_enabled
                else None
            ),
        )
        self._create_payload_indexes(collection_name)

    def ensure_collection(
        self,
        scope: str,
        vector_size: int,
        *,
        sparse_enabled: bool = False,
    ) -> None:
        """Создать коллекцию, если она еще не существует."""

        if not self.collection_exists(scope):
            self.recreate_collection(scope, vector_size, sparse_enabled=sparse_enabled)

    def _create_payload_indexes(self, collection_name: str) -> None:
        """Create best-effort payload indexes used by filtering and diagnostics."""

        for field_name, field_schema in payload_index_specs().items():
            try:
                self._client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                )
            except Exception as exc:  # noqa: BLE001
                jlog(
                    "warning",
                    "semantic_qdrant_payload_index_create_failed",
                    collection=collection_name,
                    field_name=field_name,
                    error=str(exc),
                )

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
        query_template_hash: str | None = None,
        document_prefix_hash: str | None = None,
        sparse_vectors: list[EncodedSparseVector] | None = None,
        sparse_contract_hash: str | None = None,
        sparse_vocabulary_hash: str | None = None,
        sparse_corpus_stats_hash: str | None = None,
        lexical_analyzer_version: str | None = None,
    ) -> None:
        """Upsert chunk payloads and dense/sparse vectors into Qdrant."""

        if not chunks:
            return
        if sparse_vectors is not None and len(sparse_vectors) != len(chunks):
            raise ValueError("sparse_vectors length must match chunks length")

        max_points = max(1, self._settings.SEMANTIC_MCP_QDRANT_UPSERT_BATCH_POINTS)
        max_bytes = max(1, self._settings.SEMANTIC_MCP_QDRANT_UPSERT_MAX_BYTES)

        batch: list[models.PointStruct] = []
        batch_estimated_bytes = 0
        uploaded_points = 0
        batch_index = 0
        total_points = len(chunks)
        dense_schema_kind = self.dense_schema_kind(scope)

        def flush_batch() -> None:
            """Отправить накопленный батч векторных точек в Qdrant."""

            nonlocal batch, batch_estimated_bytes, uploaded_points, batch_index
            if not batch:
                return
            batch_index += 1
            collection_name = self._target_collection_name_for_write(scope)
            if collection_name is None:
                raise RuntimeError(f"Cannot upsert chunks; current collection is missing for scope {scope}")
            self._client.upsert(collection_name=collection_name, points=batch)
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

        for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
            sparse_vector = sparse_vectors[index] if sparse_vectors is not None else None
            point = qdrant_point(
                point_id=self._point_id(chunk.point_id),
                chunk=chunk,
                dense_vector=vector,
                dense_schema_kind=dense_schema_kind,
                sparse_vector=sparse_vector,
                embedding_backend=embedding_backend,
                embedding_model=embedding_model,
                schema_version=schema_version,
                query_template_hash=query_template_hash,
                document_prefix_hash=document_prefix_hash,
                sparse_contract_hash=sparse_contract_hash,
                sparse_vocabulary_hash=sparse_vocabulary_hash,
                sparse_corpus_stats_hash=sparse_corpus_stats_hash,
                lexical_analyzer_version=lexical_analyzer_version,
            )
            point_bytes = estimated_point_bytes(
                chunk=chunk,
                dense_vector=vector,
                sparse_vector=sparse_vector,
            )
            if batch and (
                len(batch) >= max_points
                or batch_estimated_bytes + point_bytes > max_bytes
            ):
                flush_batch()
            batch.append(point)
            batch_estimated_bytes += point_bytes

        flush_batch()

    def search(
        self,
        scope: str,
        query_vector: list[float],
        limit: int,
        query_filter: models.Filter | None = None,
    ):
        """Выполнить dense search по scope collection."""

        collection_name = self._target_collection_name_for_read(scope)
        if collection_name is None:
            return []
        using = DENSE_VECTOR_NAME if self.dense_schema_kind(scope) == "named" else None
        response = self._client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=using,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return response.points

    def search_sparse(
        self,
        scope: str,
        indices: list[int],
        values: list[float],
        limit: int,
        query_filter: models.Filter | None = None,
    ):
        """Run Qdrant sparse-vector search for a scoped collection."""

        collection_name = self._target_collection_name_for_read(scope)
        if not indices or not values or collection_name is None:
            return []
        if not self.sparse_vector_available(scope):
            return []
        response = self._client.query_points(
            collection_name=collection_name,
            query=models.SparseVector(indices=indices, values=values),
            using=SPARSE_VECTOR_NAME,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return response.points

    def payload_fields(self, scope: str) -> set[str]:
        """Return payload field names from a bounded sample of one point."""

        if not self.current_collection_exists(scope):
            return set()
        collection_name = self._target_collection_name_for_read(scope)
        if collection_name is None:
            return set()
        page, _ = self._client.scroll(
            collection_name=collection_name,
            with_payload=True,
            with_vectors=False,
            limit=1,
        )
        if not page:
            return set()
        return set((page[0].payload or {}).keys())

    def payload_index_fields(self, scope: str) -> set[str] | None:
        """Return declared payload-index field names when Qdrant exposes them."""

        if not self.current_collection_exists(scope):
            return set()
        collection_name = self._target_collection_name_for_read(scope)
        if collection_name is None:
            return set()
        try:
            info = self._client.get_collection(collection_name)
        except Exception:  # noqa: BLE001
            return None
        payload_schema = getattr(info, "payload_schema", None)
        if payload_schema is None:
            return None
        if isinstance(payload_schema, dict):
            return {str(field_name) for field_name in payload_schema.keys()}
        return None

    def get_chunk(self, scope: str, chunk_id: str):
        """Получить point по идентификатору чанка."""

        collection_name = self._target_collection_name_for_read(scope)
        if collection_name is None:
            return None
        result = self._client.retrieve(
            collection_name=collection_name,
            ids=[self._point_id(chunk_id)],
            with_payload=True,
            with_vectors=False,
        )
        return result[0] if result else None

    def scroll_chunks(self, scope: str) -> Iterator[object]:
        """Прочитать все чанки коллекции с payload для lexical поиска."""

        collection_name = self._target_collection_name_for_read(scope)
        if collection_name is None:
            return

        offset = None
        while True:
            page, offset = self._client.scroll(
                collection_name=collection_name,
                with_payload=True,
                with_vectors=False,
                limit=256,
                offset=offset,
            )
            yield from page
            if offset is None:
                break

    def collection_embedding_contract(self, scope: str) -> dict[str, str] | None:
        """Вернуть embedding contract первого чанка коллекции для drift-check."""

        collection_name = self._target_collection_name_for_read(scope)
        if collection_name is None:
            return None

        page, _ = self._client.scroll(
            collection_name=collection_name,
            with_payload=True,
            with_vectors=False,
            limit=1,
        )
        if not page:
            return None

        payload = page[0].payload or {}
        backend = payload.get("embedding_backend")
        model = payload.get("embedding_model")
        schema_version = payload.get("index_schema_version")
        query_template_hash = payload.get("query_template_hash")
        document_prefix_hash = payload.get("document_prefix_hash")
        if not backend and not model and schema_version is None:
            return None
        result = {
            "embedding_backend": str(backend or ""),
            "embedding_model": str(model or ""),
        }
        if schema_version is not None:
            result["index_schema_version"] = str(schema_version)
        if query_template_hash:
            result["query_template_hash"] = str(query_template_hash)
        if document_prefix_hash:
            result["document_prefix_hash"] = str(document_prefix_hash)
        dense_schema_kind = self.dense_schema_kind(scope)
        if dense_schema_kind:
            result["dense_schema_kind"] = dense_schema_kind
        result["sparse_vector_available"] = str(self.sparse_vector_available(scope)).lower()
        return result

    def dense_schema_kind(self, scope: str) -> str:
        """Return named/unnamed dense vector schema shape for compatibility."""

        collection_name = self._target_collection_name_for_read(scope)
        if collection_name is None:
            return "missing"
        try:
            info = self._client.get_collection(collection_name)
            vectors = getattr(getattr(info.config, "params", None), "vectors", None)
        except Exception:  # noqa: BLE001
            return "unknown"
        if isinstance(vectors, dict):
            return "named" if DENSE_VECTOR_NAME in vectors else "unknown"
        if vectors is not None:
            return "unnamed_legacy"
        return "unknown"

    def sparse_vector_available(self, scope: str) -> bool:
        """Return whether the collection declares the expected sparse vector."""

        collection_name = self._target_collection_name_for_read(scope)
        if collection_name is None:
            return False
        try:
            info = self._client.get_collection(collection_name)
            sparse_vectors = getattr(getattr(info.config, "params", None), "sparse_vectors", None)
        except Exception:  # noqa: BLE001
            return False
        return isinstance(sparse_vectors, dict) and SPARSE_VECTOR_NAME in sparse_vectors

    def delete_file_chunks(self, scope: str, relative_path: str) -> bool:
        """Delete all points belonging to a specific file.

        Uses Qdrant-native filter delete — atomic and O(indexed) instead of
        scrolling the entire collection and filtering in Python.
        """

        if not self.current_collection_exists(scope):
            return not self.read_collection_exists(scope)
        self._client.delete(
            collection_name=self.collection_name(scope),
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="relative_path",
                            match=models.MatchValue(value=relative_path),
                        ),
                    ],
                ),
            ),
        )
        return True

    def count(self, scope: str) -> int:
        """Вернуть количество point'ов в коллекции."""

        collection_name = self._target_collection_name_for_read(scope)
        if collection_name is None:
            return 0
        return int(
            self._client.count(
                collection_name=collection_name,
                exact=True,
            ).count
        )

    def healthcheck(self) -> None:
        """Проверить доступность Qdrant."""

        self._client.get_collections()

    def iter_existing_scopes(self) -> Iterable[str]:
        """Вернуть список реально существующих logical collections."""

        for scope in ("code", "docs"):
            if self.read_collection_exists(scope):
                yield scope

    def purge_stale_collections(self) -> list[str]:
        """Удалить коллекции, принадлежащие другим репозиториям.

        Определяет «чужие» коллекции по префиксу: все коллекции, начинающиеся с
        SEMANTIC_MCP_COLLECTION_PREFIX + '_', кроме текущих collection_code и
        collection_docs, считаются устаревшими и удаляются.

        Это позволяет безболезненно переключаться между репозиториями: при следующем
        старте данные прошлого проекта не занимают место в Qdrant.
        """

        prefix = self._settings.SEMANTIC_MCP_COLLECTION_PREFIX
        current = {
            self._collection_name_for_schema(scope, schema_version)
            for scope in ("code", "docs")
            for schema_version in range(1, self._settings.SEMANTIC_MCP_INDEX_SCHEMA_VERSION + 1)
        }

        all_names = [c.name for c in self._client.get_collections().collections]
        stale = [name for name in all_names if name.startswith(f"{prefix}_") and name not in current]

        for name in stale:
            self._client.delete_collection(name)
            jlog("info", "semantic_stale_collection_deleted", collection=name)

        if stale:
            jlog("info", "semantic_stale_collections_purged", count=len(stale))

        return stale
