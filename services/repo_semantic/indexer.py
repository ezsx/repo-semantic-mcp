"""Repository indexing pipeline for semantic MCP."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from services.repo_semantic.chunkers.factory import (
    build_chunks_for_file,
    classify_scope,
    is_text_like,
    should_index_path,
)
from services.repo_semantic.config import SemanticMcpSettings
from services.repo_semantic.embeddings.base import EmbeddingProvider
from services.repo_semantic.logging import jlog
from services.repo_semantic.models import ChunkRecord
from services.repo_semantic.qdrant_store import QdrantStore


class RepositoryIndexer:
    """Построить и поддерживать semantic индекс по рабочей копии репозитория."""

    def __init__(
        self,
        settings: SemanticMcpSettings,
        embedding_provider: EmbeddingProvider,
        store: QdrantStore,
    ) -> None:
        """Сохранить зависимости индексации."""

        self._settings = settings
        self._embedding_provider = embedding_provider
        self._store = store
        self.last_full_build_ts: str | None = None

    def iter_indexable_paths(self) -> list[Path]:
        """Вернуть все индексируемые текстовые файлы репозитория."""

        result: list[Path] = []
        for path in self._settings.repo_root.rglob("*"):
            if not path.is_file():
                continue
            if not is_text_like(path):
                continue
            relative_path = path.relative_to(self._settings.repo_root).as_posix()
            if should_index_path(
                relative_path,
                self._settings.SEMANTIC_MCP_INCLUDE_GLOBS,
                self._settings.SEMANTIC_MCP_EXCLUDE_GLOBS,
            ):
                result.append(path)
        return result

    def _split_oversized_chunk(self, chunk: ChunkRecord) -> list[ChunkRecord]:
        """Разрезать слишком крупный чанк на более мелкие линейные части."""

        max_chars = self._settings.SEMANTIC_MCP_MAX_CHUNK_CHARS
        if len(chunk.text) <= max_chars:
            return [chunk]

        lines = chunk.text.splitlines(keepends=True)
        if not lines:
            return [chunk]

        result: list[ChunkRecord] = []
        buffer = ""
        part_index = 1
        part_start_line = chunk.start_line
        line_cursor = chunk.start_line

        def flush_buffer() -> None:
            """Сохранить накопленную часть oversized чанка."""

            nonlocal buffer, part_index, part_start_line, line_cursor
            if not buffer:
                return
            part_text = buffer
            newline_count = part_text.count("\n")
            result.append(
                ChunkRecord(
                    point_id=f"{chunk.point_id}:part{part_index}",
                    scope=chunk.scope,
                    relative_path=chunk.relative_path,
                    language=chunk.language,
                    chunk_type=f"{chunk.chunk_type}_part",
                    text=part_text,
                    start_line=part_start_line,
                    end_line=part_start_line + newline_count,
                    content_hash=chunk.content_hash,
                    source_mtime=chunk.source_mtime,
                    symbol_path=chunk.symbol_path,
                    heading_path=chunk.heading_path,
                    domain_tags=list(chunk.domain_tags),
                    is_generated=chunk.is_generated,
                    extra={**chunk.extra, "split_part": str(part_index)},
                )
            )
            part_index += 1
            line_cursor = part_start_line + newline_count
            if part_text.endswith("\n"):
                line_cursor += 1
            buffer = ""
            part_start_line = line_cursor

        for line in lines:
            if len(line) > max_chars:
                flush_buffer()
                for offset in range(0, len(line), max_chars):
                    segment = line[offset : offset + max_chars]
                    buffer = segment
                    part_start_line = line_cursor
                    flush_buffer()
                continue

            if buffer and len(buffer) + len(line) > max_chars:
                flush_buffer()

            if not buffer:
                part_start_line = line_cursor
            buffer += line

        flush_buffer()
        return result or [chunk]

    def _normalize_chunks(self, chunks: list[ChunkRecord]) -> list[ChunkRecord]:
        """Привести чанки к размеру, безопасному для embedding backend."""

        normalized: list[ChunkRecord] = []
        for chunk in chunks:
            normalized.extend(self._split_oversized_chunk(chunk))
        return normalized

    def _embed_chunks(self, chunks: list[ChunkRecord], scope: str) -> list[list[float]]:
        """Построить embeddings для списка чанков управляемыми батчами."""

        if not chunks:
            return []

        vectors: list[list[float]] = []
        batch: list[ChunkRecord] = []
        batch_chars = 0
        batch_index = 0
        total_chunks = len(chunks)

        def flush_batch() -> None:
            """Отправить накопленный батч в embedding backend."""

            nonlocal batch, batch_chars, vectors, batch_index
            if not batch:
                return
            batch_index += 1
            vectors.extend(
                self._embedding_provider.embed_documents([chunk.text for chunk in batch])
            )
            if batch_index == 1 or batch_index % 25 == 0 or len(vectors) == total_chunks:
                jlog(
                    "info",
                    "semantic_embedding_progress",
                    scope=scope,
                    embedded=len(vectors),
                    total=total_chunks,
                    batch_index=batch_index,
                )
            batch = []
            batch_chars = 0

        for chunk in chunks:
            chunk_chars = len(chunk.text)
            would_exceed_docs = len(batch) >= self._settings.SEMANTIC_MCP_EMBED_BATCH_DOCS
            would_exceed_chars = (
                bool(batch)
                and batch_chars + chunk_chars > self._settings.SEMANTIC_MCP_EMBED_BATCH_CHARS
            )
            if would_exceed_docs or would_exceed_chars:
                flush_batch()
            batch.append(chunk)
            batch_chars += chunk_chars

        flush_batch()
        return vectors

    def rebuild_index(self) -> dict[str, int]:
        """Полностью перестроить code и docs коллекции."""

        chunks_by_scope: dict[str, list[ChunkRecord]] = {"code": [], "docs": []}
        for file_path in self.iter_indexable_paths():
            raw_chunks = build_chunks_for_file(
                file_path=file_path,
                repo_root=self._settings.repo_root,
                include_globs=self._settings.SEMANTIC_MCP_INCLUDE_GLOBS,
                exclude_globs=self._settings.SEMANTIC_MCP_EXCLUDE_GLOBS,
            )
            for chunk in self._normalize_chunks(raw_chunks):
                chunks_by_scope[chunk.scope].append(chunk)

        result: dict[str, int] = {}
        for scope, chunks in chunks_by_scope.items():
            if not chunks:
                result[scope] = 0
                continue
            vectors = self._embed_chunks(chunks, scope=scope)
            self._store.recreate_collection(scope, len(vectors[0]))
            self._store.upsert_chunks(
                scope=scope,
                chunks=chunks,
                vectors=vectors,
                embedding_backend=self._embedding_provider.backend_name(),
                embedding_model=self._embedding_provider.model_name(),
                schema_version=self._settings.SEMANTIC_MCP_INDEX_SCHEMA_VERSION,
            )
            result[scope] = len(chunks)

        self.last_full_build_ts = datetime.now(timezone.utc).isoformat()
        jlog("info", "semantic_index_rebuilt", **result)
        return result

    def reconcile_index(self) -> dict[str, int]:
        """Сверить текущий индекс с рабочей копией и дозаписать только отличия."""

        current_paths: dict[str, float] = {}
        for path in self.iter_indexable_paths():
            relative_path = path.relative_to(self._settings.repo_root).as_posix()
            current_paths[relative_path] = path.stat().st_mtime

        indexed_paths: dict[str, float] = {}
        for scope in ("code", "docs"):
            for point in self._store.scroll_chunks(scope):
                payload = point.payload or {}
                relative_path = payload.get("relative_path")
                if not relative_path or relative_path in indexed_paths:
                    continue
                try:
                    indexed_paths[relative_path] = float(payload.get("source_mtime") or 0.0)
                except (TypeError, ValueError):
                    indexed_paths[relative_path] = 0.0

        touched: list[str] = []
        for relative_path, current_mtime in current_paths.items():
            indexed_mtime = indexed_paths.get(relative_path)
            if indexed_mtime is None or abs(indexed_mtime - current_mtime) > 1e-6:
                touched.append(relative_path)

        for relative_path in indexed_paths:
            if relative_path not in current_paths:
                touched.append(relative_path)

        touched = sorted(set(touched))
        if not touched:
            jlog("info", "semantic_index_reconcile_noop")
            return {"paths": 0, "code": 0, "docs": 0}

        result = self.reindex_paths(touched)
        jlog(
            "info",
            "semantic_index_reconciled",
            paths=len(touched),
            code_chunks=result.get("code", 0),
            docs_chunks=result.get("docs", 0),
        )
        return {"paths": len(touched), **result}

    def reindex_paths(self, relative_paths: list[str]) -> dict[str, int]:
        """Переиндексировать конкретные файлы по относительным путям."""

        affected_by_scope: dict[str, list[ChunkRecord]] = {"code": [], "docs": []}

        for relative_path in sorted(set(path.replace("\\", "/") for path in relative_paths)):
            # Удаляем во всех scope, чтобы корректно переживать перенос между code/docs
            # и появление файлов, пока стек был остановлен.
            for scope in ("code", "docs"):
                self._store.delete_file_chunks(scope, relative_path)

            file_path = self._settings.repo_root / relative_path
            if not file_path.exists() or not file_path.is_file():
                continue
            raw_chunks = build_chunks_for_file(
                file_path=file_path,
                repo_root=self._settings.repo_root,
                include_globs=self._settings.SEMANTIC_MCP_INCLUDE_GLOBS,
                exclude_globs=self._settings.SEMANTIC_MCP_EXCLUDE_GLOBS,
            )
            for chunk in self._normalize_chunks(raw_chunks):
                affected_by_scope[chunk.scope].append(chunk)

        result: dict[str, int] = {}
        for scope, chunks in affected_by_scope.items():
            if chunks:
                vectors = self._embed_chunks(chunks, scope=scope)
                self._store.ensure_collection(scope, len(vectors[0]))
                self._store.upsert_chunks(
                    scope=scope,
                    chunks=chunks,
                    vectors=vectors,
                    embedding_backend=self._embedding_provider.backend_name(),
                    embedding_model=self._embedding_provider.model_name(),
                    schema_version=self._settings.SEMANTIC_MCP_INDEX_SCHEMA_VERSION,
                )
            result[scope] = len(chunks)

        if relative_paths:
            jlog(
                "info",
                "semantic_index_paths_reindexed",
                paths=len(relative_paths),
                code_chunks=result.get("code", 0),
                docs_chunks=result.get("docs", 0),
            )
        return result
