"""Repository indexing pipeline for semantic MCP."""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from collections.abc import Iterator
from pathlib import Path

from services.repo_semantic.chunkers.factory import (
    build_chunks_for_text,
    classify_scope,
    is_text_like,
    should_index_path,
)
from services.repo_semantic.config import SemanticMcpSettings
from services.repo_semantic.embeddings.base import EmbeddingProvider
from services.repo_semantic.indexing.chunk_pipeline import (
    normalize_chunks,
    split_oversized_chunk,
)
from services.repo_semantic.indexing.embedding_pipeline import embed_chunks
from services.repo_semantic.indexing.sparse_pipeline import (
    compatible_sparse_manifest,
    encode_sparse_documents,
    should_upsert_sparse_vectors,
)
from services.repo_semantic.logging import jlog
from services.repo_semantic.models import ChunkRecord
from services.repo_semantic.path_manifest import (
    PathManifestStore,
    PathManifestRecord,
    manifest_record_for_chunks,
    normalize_manifest_relative_path,
)
from services.repo_semantic.qdrant_store import QdrantStore
from services.repo_semantic.storage.qdrant.point_projection import point_to_chunk
from services.repo_semantic.lexical import (
    LEXICAL_ANALYZER_VERSION,
    SparseManifest,
    SparseManifestStore,
    build_sparse_manifest,
    extend_manifest_for_incremental_chunks,
    mark_manifest_stale,
)


BACKFILLED_INDEXED_AT = "1970-01-01T00:00:00+00:00"


class IndexablePathScanBoundsExceeded(RuntimeError):
    """Raised when repository path discovery exceeds configured safe bounds."""

    def __init__(self, summary: dict[str, object]) -> None:
        super().__init__("indexable_path_scan_bounds_exceeded")
        self.summary = summary


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
        self._sparse_manifests = SparseManifestStore(settings)
        self.path_manifest = PathManifestStore(settings)
        self.last_full_build_ts: str | None = None
        self.last_incremental_update_ts: str | None = None

    def _compatible_sparse_manifest(self, manifest: SparseManifest | None) -> SparseManifest | None:
        """Return manifest only when incremental sparse updates are contract-safe."""

        return compatible_sparse_manifest(
            manifest,
            schema_version=self._settings.SEMANTIC_MCP_INDEX_SCHEMA_VERSION,
        )

    def iter_indexable_paths(
        self,
        *,
        max_paths_scanned: int | None = None,
        max_duration_ms: int | None = None,
    ) -> Iterator[Path]:
        """Вернуть все индексируемые текстовые файлы репозитория."""

        started_at = time.monotonic()
        paths_scanned = 0
        for path in self._settings.repo_root.rglob("*"):
            paths_scanned += 1
            duration_ms = int((time.monotonic() - started_at) * 1000)
            bounds_exceeded: list[str] = []
            if max_paths_scanned is not None and paths_scanned > max_paths_scanned:
                bounds_exceeded.append("max_scan_paths")
            if max_duration_ms is not None and duration_ms > max_duration_ms:
                bounds_exceeded.append("max_duration_ms")
            if bounds_exceeded:
                raise IndexablePathScanBoundsExceeded(
                    {
                        "paths_scanned": paths_scanned,
                        "max_paths_scanned": max_paths_scanned,
                        "max_duration_ms": max_duration_ms,
                        "bounds_exceeded": bounds_exceeded,
                        "duration_ms": duration_ms,
                        "warning_codes": ["startup_reconcile_bounds_exceeded"],
                    }
                )
            if not path.is_file():
                continue
            if not is_text_like(path):
                continue
            relative_path = path.relative_to(self._settings.repo_root).as_posix()
            if should_index_path(
                relative_path,
                self._settings.effective_include_globs,
                self._settings.SEMANTIC_MCP_EXCLUDE_GLOBS,
            ):
                yield path

    def _split_oversized_chunk(self, chunk: ChunkRecord) -> list[ChunkRecord]:
        """Разрезать слишком крупный чанк на более мелкие линейные части."""

        return split_oversized_chunk(
            chunk,
            max_chars=self._settings.SEMANTIC_MCP_MAX_CHUNK_CHARS,
        )

    def _normalize_chunks(self, chunks: list[ChunkRecord]) -> list[ChunkRecord]:
        """Привести чанки к размеру, безопасному для embedding backend."""

        return normalize_chunks(
            chunks,
            max_chars=self._settings.SEMANTIC_MCP_MAX_CHUNK_CHARS,
        )

    def _embed_chunks(self, chunks: list[ChunkRecord], scope: str) -> list[list[float]]:
        """Построить embeddings для списка чанков управляемыми батчами."""

        return embed_chunks(
            chunks=chunks,
            scope=scope,
            embedding_provider=self._embedding_provider,
            max_batch_docs=self._settings.SEMANTIC_MCP_EMBED_BATCH_DOCS,
            max_batch_chars=self._settings.SEMANTIC_MCP_EMBED_BATCH_CHARS,
        )

    def _settings_contract_hashes(self) -> tuple[str, str]:
        """Return hashes for embedding text-shaping contract fields."""

        query_template_hash = hashlib.sha256(
            self._settings.SEMANTIC_MCP_QUERY_TEMPLATE.encode("utf-8")
        ).hexdigest()
        document_prefix_hash = hashlib.sha256(
            self._settings.SEMANTIC_MCP_DOCUMENT_PREFIX.encode("utf-8")
        ).hexdigest()
        return query_template_hash, document_prefix_hash

    def _read_file_snapshot(self, file_path: Path) -> tuple[str, str, float, int]:
        """Read one file once and return text plus the exact bytes hash used."""

        data = file_path.read_bytes()
        stat = file_path.stat()
        return (
            data.decode("utf-8", errors="ignore"),
            hashlib.sha256(data).hexdigest(),
            float(stat.st_mtime),
            int(len(data)),
        )

    def _build_chunks_for_snapshot(
        self,
        *,
        file_path: Path,
        text: str,
        source_mtime: float,
    ) -> list[ChunkRecord]:
        """Build chunks from an already-read file snapshot."""

        return build_chunks_for_text(
            file_path=file_path,
            repo_root=self._settings.repo_root,
            text=text,
            source_mtime=source_mtime,
            doc_prefixes=self._settings.effective_doc_prefixes,
        )

    def rebuild_index(self) -> dict[str, int]:
        """Полностью перестроить code и docs коллекции."""

        chunks_by_scope: dict[str, list[ChunkRecord]] = {"code": [], "docs": []}
        manifest_records = []
        indexed_at = datetime.now(timezone.utc).isoformat()
        for file_path in self.iter_indexable_paths():
            text, file_hash, source_mtime, file_size = self._read_file_snapshot(file_path)
            raw_chunks = self._build_chunks_for_snapshot(
                file_path=file_path,
                text=text,
                source_mtime=source_mtime,
            )
            chunks = self._normalize_chunks(raw_chunks)
            relative_path = file_path.relative_to(self._settings.repo_root).as_posix()
            manifest_records.append(
                manifest_record_for_chunks(
                    settings=self._settings,
                    relative_path=relative_path,
                    chunks=chunks,
                    indexed_at=indexed_at,
                    content_hash=file_hash,
                    source_mtime=source_mtime,
                    file_size=file_size,
                    scope=classify_scope(relative_path, self._settings.effective_doc_prefixes),
                )
            )
            for chunk in chunks:
                chunks_by_scope[chunk.scope].append(chunk)

        result: dict[str, int] = {}
        query_template_hash, document_prefix_hash = self._settings_contract_hashes()
        for scope, chunks in chunks_by_scope.items():
            sparse_manifest = build_sparse_manifest(
                scope=scope,  # type: ignore[arg-type]
                chunks=chunks,
                schema_version=self._settings.SEMANTIC_MCP_INDEX_SCHEMA_VERSION,
            )
            self._sparse_manifests.save(sparse_manifest)
            if not chunks:
                self._store.delete_collection(scope)
                result[scope] = 0
                continue
            vectors = self._embed_chunks(chunks, scope=scope)
            sparse_vectors = encode_sparse_documents(chunks, sparse_manifest)
            self._store.recreate_collection(scope, len(vectors[0]), sparse_enabled=True)
            self._store.upsert_chunks(
                scope=scope,
                chunks=chunks,
                vectors=vectors,
                embedding_backend=self._embedding_provider.backend_name(),
                embedding_model=self._embedding_provider.model_name(),
                schema_version=self._settings.SEMANTIC_MCP_INDEX_SCHEMA_VERSION,
                query_template_hash=query_template_hash,
                document_prefix_hash=document_prefix_hash,
                sparse_vectors=sparse_vectors,
                sparse_contract_hash=sparse_manifest.immutable_sparse_contract_hash,
                sparse_vocabulary_hash=sparse_manifest.vocabulary_hash,
                sparse_corpus_stats_hash=sparse_manifest.corpus_stats_hash,
                lexical_analyzer_version=LEXICAL_ANALYZER_VERSION,
            )
            result[scope] = len(chunks)

        self.path_manifest.replace_all(manifest_records)

        self.last_full_build_ts = indexed_at
        self.last_incremental_update_ts = self.last_full_build_ts
        jlog("info", "semantic_index_rebuilt", **result)
        return result

    def reconcile_index(self) -> dict[str, object]:
        """Сверить текущий индекс с рабочей копией и дозаписать только отличия."""

        started_at = time.monotonic()
        max_reconcile_paths = self._settings.SEMANTIC_MCP_RECONCILE_MAX_PATHS
        max_deleted_paths = self._settings.SEMANTIC_MCP_RECONCILE_MAX_DELETED_PATHS
        max_points_deleted = self._settings.SEMANTIC_MCP_RECONCILE_MAX_POINTS_DELETED
        max_duration_ms = self._settings.SEMANTIC_MCP_RECONCILE_MAX_DURATION_MS
        max_scan_paths = self._settings.SEMANTIC_MCP_RECONCILE_MAX_SCAN_PATHS

        def _summary(
            *,
            touched: list[str],
            deleted_paths: list[str],
            indexed_point_counts: dict[str, int],
            bounds_exceeded: list[str],
        ) -> dict[str, object]:
            points_deleted = sum(indexed_point_counts.get(relative_path, 0) for relative_path in touched)
            return {
                "paths": len(touched),
                "code": 0,
                "docs": 0,
                "paths_scanned": len(current_paths),
                "paths_reindexed": 0,
                "paths_deleted": len(deleted_paths),
                "touched_paths": list(touched),
                "deleted_paths": list(deleted_paths),
                "points_deleted": points_deleted,
                "max_paths_scanned": max_scan_paths,
                "max_paths_reindexed": max_reconcile_paths,
                "max_deleted_paths": max_deleted_paths,
                "max_points_deleted": max_points_deleted,
                "max_duration_ms": max_duration_ms,
                "bounds_exceeded": bounds_exceeded,
                "mutation_started": False,
                "duration_ms": int((time.monotonic() - started_at) * 1000),
                "warning_codes": ["startup_reconcile_bounds_exceeded"] if bounds_exceeded else [],
            }

        current_paths: dict[str, float] = {}
        try:
            path_iter = self.iter_indexable_paths(
                max_paths_scanned=max_scan_paths,
                max_duration_ms=max_duration_ms,
            )
            for path in path_iter:
                relative_path = path.relative_to(self._settings.repo_root).as_posix()
                try:
                    current_paths[relative_path] = path.stat().st_mtime
                except FileNotFoundError:
                    continue
        except IndexablePathScanBoundsExceeded as exc:
            return {
                **_summary(
                    touched=[],
                    deleted_paths=[],
                    indexed_point_counts={},
                    bounds_exceeded=list(exc.summary.get("bounds_exceeded") or []),
                ),
                "paths_scanned": exc.summary.get("paths_scanned", len(current_paths)),
                "duration_ms": exc.summary.get("duration_ms", 0),
            }

        indexed_paths: dict[str, float] = {}
        indexed_point_counts: dict[str, int] = {}
        indexed_chunks_by_path: dict[str, list[ChunkRecord]] = {}
        zero_chunk_records_by_path = self.path_manifest.ready_zero_chunk_records()
        early_touched_paths: set[str] = set()
        early_deleted_paths: set[str] = set()
        planned_points_deleted = 0
        for scope in ("code", "docs"):
            for point in self._store.scroll_chunks(scope):
                payload = point.payload or {}
                if (
                    max_duration_ms is not None
                    and int((time.monotonic() - started_at) * 1000) > max_duration_ms
                ):
                    return _summary(
                        touched=sorted(early_touched_paths),
                        deleted_paths=sorted(early_deleted_paths),
                        indexed_point_counts=indexed_point_counts,
                        bounds_exceeded=["max_duration_ms"],
                    )
                relative_path = normalize_manifest_relative_path(str(payload.get("relative_path") or ""))
                if not relative_path:
                    continue
                try:
                    chunk = point_to_chunk(point)
                except Exception:  # noqa: BLE001
                    chunk = None
                if chunk is not None:
                    indexed_chunks_by_path.setdefault(relative_path, []).append(chunk)
                indexed_point_counts[relative_path] = indexed_point_counts.get(relative_path, 0) + 1
                try:
                    indexed_mtime_for_point = float(payload.get("source_mtime") or 0.0)
                except (TypeError, ValueError):
                    indexed_mtime_for_point = 0.0
                current_mtime_for_point = current_paths.get(relative_path)
                point_would_be_deleted = (
                    current_mtime_for_point is None
                    or abs(indexed_mtime_for_point - current_mtime_for_point) > 1e-6
                )
                if point_would_be_deleted:
                    early_touched_paths.add(relative_path)
                    planned_points_deleted += 1
                    if current_mtime_for_point is None:
                        early_deleted_paths.add(relative_path)
                    if max_reconcile_paths is not None and len(early_touched_paths) > max_reconcile_paths:
                        return _summary(
                            touched=sorted(early_touched_paths),
                            deleted_paths=sorted(early_deleted_paths),
                            indexed_point_counts=indexed_point_counts,
                            bounds_exceeded=["max_reconcile_paths"],
                        )
                    if max_deleted_paths is not None and len(early_deleted_paths) > max_deleted_paths:
                        return _summary(
                            touched=sorted(early_touched_paths),
                            deleted_paths=sorted(early_deleted_paths),
                            indexed_point_counts=indexed_point_counts,
                            bounds_exceeded=["max_deleted_paths"],
                        )
                    if max_points_deleted is not None and planned_points_deleted > max_points_deleted:
                        return _summary(
                            touched=sorted(early_touched_paths),
                            deleted_paths=sorted(early_deleted_paths),
                            indexed_point_counts=indexed_point_counts,
                            bounds_exceeded=["max_points_deleted"],
                        )
                if relative_path in indexed_paths:
                    continue
                indexed_paths[relative_path] = indexed_mtime_for_point

        for relative_path, record in zero_chunk_records_by_path.items():
            if relative_path in current_paths and relative_path not in indexed_paths:
                indexed_paths[relative_path] = record.source_mtime
                indexed_point_counts.setdefault(relative_path, 0)

        touched: list[str] = []
        for relative_path, current_mtime in current_paths.items():
            indexed_mtime = indexed_paths.get(relative_path)
            if indexed_mtime is None or abs(indexed_mtime - current_mtime) > 1e-6:
                touched.append(relative_path)

        deleted_paths: list[str] = []
        for relative_path in indexed_paths:
            if relative_path not in current_paths:
                deleted_paths.append(relative_path)
                touched.append(relative_path)

        touched = sorted(set(touched))
        points_deleted = sum(indexed_point_counts.get(relative_path, 0) for relative_path in touched)
        duration_ms = int((time.monotonic() - started_at) * 1000)
        bounds_exceeded: list[str] = []
        if max_scan_paths is not None and len(current_paths) > max_scan_paths:
            bounds_exceeded.append("max_scan_paths")
        if max_reconcile_paths is not None and len(touched) > max_reconcile_paths:
            bounds_exceeded.append("max_reconcile_paths")
        if max_deleted_paths is not None and len(deleted_paths) > max_deleted_paths:
            bounds_exceeded.append("max_deleted_paths")
        if max_points_deleted is not None and points_deleted > max_points_deleted:
            bounds_exceeded.append("max_points_deleted")
        if max_duration_ms is not None and duration_ms > max_duration_ms:
            bounds_exceeded.append("max_duration_ms")
        base_summary = _summary(
            touched=touched,
            deleted_paths=deleted_paths,
            indexed_point_counts=indexed_point_counts,
            bounds_exceeded=bounds_exceeded,
        )
        if bounds_exceeded:
            jlog(
                "warning",
                "semantic_index_reconcile_bounds_exceeded",
                paths=len(touched),
                deleted_paths=len(deleted_paths),
                points_deleted=base_summary["points_deleted"],
                paths_scanned=len(current_paths),
                bounds_exceeded=bounds_exceeded,
            )
            return base_summary
        if not touched:
            manifest_backfill = self._backfill_complete_path_manifest_from_index_snapshot(
                current_paths=current_paths,
                indexed_chunks_by_path=indexed_chunks_by_path,
                zero_chunk_records_by_path=zero_chunk_records_by_path,
            )
            jlog("info", "semantic_index_reconcile_noop")
            return {**base_summary, **manifest_backfill}

        result = self.reindex_paths(touched)
        duration_ms = int((time.monotonic() - started_at) * 1000)
        jlog(
            "info",
            "semantic_index_reconciled",
            paths=len(touched),
            code_chunks=result.get("code", 0),
            docs_chunks=result.get("docs", 0),
        )
        return {
            **base_summary,
            "code": result.get("code", 0),
            "docs": result.get("docs", 0),
            "paths_reindexed": len(touched),
            "mutation_started": True,
            "duration_ms": duration_ms,
        }

    def _backfill_complete_path_manifest_from_index_snapshot(
        self,
        *,
        current_paths: dict[str, float],
        indexed_chunks_by_path: dict[str, list[ChunkRecord]],
        zero_chunk_records_by_path: dict[str, PathManifestRecord] | None = None,
    ) -> dict[str, object]:
        """Repair a missing/incomplete path manifest from already-indexed chunks.

        This is safe only for a no-op reconcile: the caller has already scanned
        the working tree and Qdrant, found no stale/missing/deleted paths, and
        therefore does not need to mutate the vector index. The repair writes
        diagnostic path metadata only; it does not rebuild, embed, upsert, or
        delete vector points.
        """

        if not current_paths:
            self.path_manifest.replace_all([])
            return {
                "path_manifest_backfilled": True,
                "path_manifest_coverage_complete": True,
                "path_manifest_backfilled_paths": 0,
            }

        records = []
        zero_chunk_records_by_path = zero_chunk_records_by_path or {}
        missing_manifest_paths: list[str] = []
        for relative_path in sorted(current_paths):
            chunks = indexed_chunks_by_path.get(relative_path)
            file_path = self._settings.repo_root / relative_path
            try:
                file_size = int(file_path.stat().st_size)
            except OSError:
                missing_manifest_paths.append(relative_path)
                continue
            if not chunks:
                zero_chunk_record = zero_chunk_records_by_path.get(relative_path)
                if zero_chunk_record is None:
                    missing_manifest_paths.append(relative_path)
                    continue
                records.append(
                    manifest_record_for_chunks(
                        settings=self._settings,
                        relative_path=relative_path,
                        chunks=[],
                        indexed_at=zero_chunk_record.indexed_at,
                        content_hash=zero_chunk_record.content_hash,
                        source_mtime=zero_chunk_record.source_mtime,
                        file_size=file_size,
                        scope=zero_chunk_record.scope,
                    )
                )
                continue
            first_chunk = chunks[0]
            records.append(
                manifest_record_for_chunks(
                    settings=self._settings,
                    relative_path=relative_path,
                    chunks=chunks,
                    indexed_at=self.last_incremental_update_ts
                    or self.last_full_build_ts
                    or BACKFILLED_INDEXED_AT,
                    content_hash=first_chunk.content_hash,
                    source_mtime=first_chunk.source_mtime,
                    file_size=file_size,
                    scope=classify_scope(relative_path, self._settings.effective_doc_prefixes),
                )
            )

        if missing_manifest_paths:
            return {
                "path_manifest_backfilled": False,
                "path_manifest_coverage_complete": False,
                "path_manifest_missing_paths": missing_manifest_paths[:12],
            }

        self.path_manifest.replace_all(records)
        return {
            "path_manifest_backfilled": True,
            "path_manifest_coverage_complete": True,
            "path_manifest_backfilled_paths": len(records),
        }

    def reindex_paths(self, relative_paths: list[str]) -> dict[str, int]:
        """Переиндексировать конкретные файлы по относительным путям."""

        affected_by_scope: dict[str, list[ChunkRecord]] = {"code": [], "docs": []}
        chunks_by_path: dict[str, list[ChunkRecord]] = {}
        delete_confirmed_by_path: dict[str, bool] = {}
        manifest_snapshot_by_path: dict[str, tuple[str, float, int, str]] = {}
        zero_chunk_records: list = []
        deleted_paths: list[str] = []
        indexed_at = datetime.now(timezone.utc).isoformat()
        sparse_manifests = {
            scope: self._compatible_sparse_manifest(self._sparse_manifests.load(scope))
            for scope in ("code", "docs")
        }

        normalized_relative_paths = {
            normalized
            for raw_path in relative_paths
            if (normalized := normalize_manifest_relative_path(raw_path)) is not None
        }
        for relative_path in sorted(normalized_relative_paths):
            # Удаляем во всех scope, чтобы корректно переживать перенос между code/docs
            # и появление файлов, пока стек был остановлен.
            deletes_confirmed = True
            for scope in ("code", "docs"):
                delete_result = self._store.delete_file_chunks(scope, relative_path)
                if delete_result is False:
                    deletes_confirmed = False
            delete_confirmed_by_path[relative_path] = deletes_confirmed

            file_path = self._settings.repo_root / relative_path
            if not file_path.exists() or not file_path.is_file():
                if deletes_confirmed:
                    deleted_paths.append(relative_path)
                continue
            if (
                not should_index_path(
                    relative_path,
                    self._settings.effective_include_globs,
                    self._settings.SEMANTIC_MCP_EXCLUDE_GLOBS,
                )
                or not is_text_like(file_path)
            ):
                if deletes_confirmed:
                    deleted_paths.append(relative_path)
                continue
            text, file_hash, source_mtime, file_size = self._read_file_snapshot(file_path)
            raw_chunks = self._build_chunks_for_snapshot(
                file_path=file_path,
                text=text,
                source_mtime=source_mtime,
            )
            chunks = self._normalize_chunks(raw_chunks)
            manifest_snapshot_by_path[relative_path] = (
                file_hash,
                source_mtime,
                file_size,
                classify_scope(relative_path, self._settings.effective_doc_prefixes),
            )
            if not chunks and deletes_confirmed:
                zero_chunk_records.append(
                    manifest_record_for_chunks(
                        settings=self._settings,
                        relative_path=relative_path,
                        chunks=[],
                        indexed_at=indexed_at,
                        content_hash=file_hash,
                        source_mtime=source_mtime,
                        file_size=file_size,
                        scope=classify_scope(relative_path, self._settings.effective_doc_prefixes),
                    )
                )
            for chunk in chunks:
                affected_by_scope[chunk.scope].append(chunk)
                chunks_by_path.setdefault(relative_path, []).append(chunk)

        result: dict[str, int] = {}
        successful_chunks_by_path: dict[str, list[ChunkRecord]] = {}
        query_template_hash, document_prefix_hash = self._settings_contract_hashes()
        for scope, chunks in affected_by_scope.items():
            sparse_manifest = sparse_manifests.get(scope)
            if sparse_manifest is not None and relative_paths:
                sparse_manifest = mark_manifest_stale(sparse_manifest)
            if sparse_manifest is not None and chunks:
                sparse_manifest = extend_manifest_for_incremental_chunks(sparse_manifest, chunks)
            if sparse_manifest is not None:
                self._sparse_manifests.save(sparse_manifest)
            if chunks:
                current_collection_exists = (
                    self._store.current_collection_exists(scope)
                    if hasattr(self._store, "current_collection_exists")
                    else self._store.collection_exists(scope)
                )
                if not current_collection_exists:
                    jlog(
                        "warning",
                        "semantic_incremental_reindex_skipped_current_collection_missing",
                        scope=scope,
                        chunks=len(chunks),
                    )
                    result[scope] = 0
                    continue
                vectors = self._embed_chunks(chunks, scope=scope)
                store_sparse_available = (
                    hasattr(self._store, "sparse_vector_available")
                    and self._store.sparse_vector_available(scope)
                )
                sparse_upsert_enabled = should_upsert_sparse_vectors(
                    sparse_manifest,
                    current_collection_exists=current_collection_exists,
                    sparse_vector_available=store_sparse_available,
                )
                sparse_vectors = (
                    encode_sparse_documents(chunks, sparse_manifest)
                    if sparse_upsert_enabled and sparse_manifest is not None
                    else None
                )
                self._store.ensure_collection(
                    scope,
                    len(vectors[0]),
                    sparse_enabled=sparse_upsert_enabled,
                )
                self._store.upsert_chunks(
                    scope=scope,
                    chunks=chunks,
                    vectors=vectors,
                    embedding_backend=self._embedding_provider.backend_name(),
                    embedding_model=self._embedding_provider.model_name(),
                    schema_version=self._settings.SEMANTIC_MCP_INDEX_SCHEMA_VERSION,
                    query_template_hash=query_template_hash,
                    document_prefix_hash=document_prefix_hash,
                    sparse_vectors=sparse_vectors,
                    sparse_contract_hash=sparse_manifest.immutable_sparse_contract_hash
                    if sparse_upsert_enabled and sparse_manifest is not None
                    else None,
                    sparse_vocabulary_hash=sparse_manifest.vocabulary_hash
                    if sparse_upsert_enabled and sparse_manifest is not None
                    else None,
                    sparse_corpus_stats_hash=sparse_manifest.corpus_stats_hash
                    if sparse_upsert_enabled and sparse_manifest is not None
                    else None,
                    lexical_analyzer_version=LEXICAL_ANALYZER_VERSION
                    if sparse_upsert_enabled and sparse_manifest is not None
                    else None,
                )
                for chunk in chunks:
                    path_chunks = chunks_by_path.get(chunk.relative_path, [])
                    if path_chunks and delete_confirmed_by_path.get(chunk.relative_path):
                        successful_chunks_by_path[chunk.relative_path] = path_chunks
            result[scope] = len(chunks)

        if relative_paths:
            if successful_chunks_by_path:
                manifest_records = [
                    manifest_record_for_chunks(
                        settings=self._settings,
                        relative_path=relative_path,
                        chunks=chunks,
                        indexed_at=indexed_at,
                        content_hash=manifest_snapshot_by_path[relative_path][0],
                        source_mtime=manifest_snapshot_by_path[relative_path][1],
                        file_size=manifest_snapshot_by_path[relative_path][2],
                        scope=manifest_snapshot_by_path[relative_path][3],
                    )
                    for relative_path, chunks in successful_chunks_by_path.items()
                ]
                self.path_manifest.upsert_ready_records(manifest_records)
            if zero_chunk_records:
                self.path_manifest.upsert_ready_records(zero_chunk_records)
            if deleted_paths:
                self.path_manifest.mark_deleted(deleted_paths)
            self.last_incremental_update_ts = indexed_at
            jlog(
                "info",
                "semantic_index_paths_reindexed",
                paths=len(relative_paths),
                code_chunks=result.get("code", 0),
                docs_chunks=result.get("docs", 0),
            )
        return result
