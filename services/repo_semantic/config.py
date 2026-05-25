"""Конфигурация repo semantic MCP."""

from __future__ import annotations

import hashlib
from pathlib import Path
from pathlib import PureWindowsPath
import re

from pydantic import Field, PrivateAttr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from services.repo_semantic.backend_catalog import get_backend_catalog_entry


def _default_repo_root() -> str:
    """Вернуть корень репозитория из расположения текущего файла."""

    return str(Path(__file__).resolve().parents[2])


def _slugify_embedding_model(model_name: str) -> str:
    """Преобразовать имя embedding модели в короткий slug для имен коллекций."""

    raw = model_name.strip().lower()
    tail = raw.split("/")[-1]
    slug = re.sub(r"[^a-z0-9]+", "_", tail).strip("_")
    return slug or "model"


def _slugify_backend_id(raw_value: str) -> str:
    """Нормализовать backend id или его части до filesystem-safe slug."""

    return _slugify_embedding_model(raw_value)


def build_repo_key(repo_root: Path) -> str:
    """Собрать стабильный ключ репозитория из имени папки и пути."""

    normalized = normalize_repo_identity(repo_root)
    repo_name = re.sub(r"[^a-z0-9]+", "_", Path(normalized).name.strip().lower()).strip("_") or "repo"
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:8]
    return f"{repo_name}_{digest}"


def normalize_repo_identity(repo_root: str | Path) -> str:
    """Нормализовать repo identity к стабильной строке вне зависимости от host OS."""

    raw = str(repo_root).strip()
    if not raw:
        raise ValueError("repo_root must not be empty")

    if re.match(r"^[A-Za-z]:[\\/]", raw):
        win = PureWindowsPath(raw)
        drive = win.drive.upper()
        tail = "/".join(part.lower() for part in win.parts[1:] if part not in ("\\", "/"))
        return f"{drive}/{tail}" if tail else f"{drive}/"

    if raw.startswith("\\\\"):
        unc = raw.replace("\\", "/").lstrip("/").lower()
        return f"//{unc}"

    return str(Path(raw).resolve())


def _default_registry_db_path() -> str:
    """Вернуть default path для lifecycle registry SQLite."""

    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "artifacts" / "repo_semantic_registry.sqlite3")


class SemanticMcpSettings(BaseSettings):
    """Настройки semantic MCP, Qdrant и indexing pipeline."""

    # Кэш для результатов auto-detection (вычисляется один раз при первом обращении)
    _globs_cache: list[str] | None = PrivateAttr(default=None)
    _doc_prefixes_cache: tuple[str, ...] | None = PrivateAttr(default=None)

    SEMANTIC_MCP_REPO_ROOT: str = Field(default_factory=_default_repo_root)
    SEMANTIC_MCP_LOGICAL_REPO_ROOT: str | None = None
    SEMANTIC_MCP_QDRANT_URL: str = "http://qdrant:6333"
    SEMANTIC_MCP_QDRANT_API_KEY: str | None = None
    SEMANTIC_MCP_REGISTRY_DB_PATH: str = Field(default_factory=_default_registry_db_path)
    SEMANTIC_MCP_COLLECTION_PREFIX: str = "repo_semantic"
    SEMANTIC_MCP_REPO_KEY: str | None = None
    SEMANTIC_MCP_INDEX_SCHEMA_VERSION: int = 3

    SEMANTIC_MCP_EMBEDDING_BACKEND: str = "tei_http"
    SEMANTIC_MCP_EMBEDDING_BACKEND_ID: str | None = None
    SEMANTIC_MCP_EMBEDDING_MODEL: str = "intfloat/multilingual-e5-small"
    SEMANTIC_MCP_TEI_URL: str | None = None
    SEMANTIC_MCP_PROFILE_NAME: str = "cpu_e5"
    SEMANTIC_MCP_QUERY_TEMPLATE: str = "query: {query}"
    SEMANTIC_MCP_DOCUMENT_PREFIX: str = "passage: "
    SEMANTIC_MCP_TEI_QUERY_PROMPT_NAME: str | None = None
    SEMANTIC_MCP_TEI_DOCUMENT_PROMPT_NAME: str | None = None
    SEMANTIC_MCP_EMBEDDING_LOCATION_TYPE: str = "unknown"
    SEMANTIC_MCP_EMBEDDING_DEVICE_TYPE: str = "unknown"
    SEMANTIC_MCP_EMBEDDING_MANAGED_BY_SERVICE: bool = False
    SEMANTIC_MCP_EMBEDDING_AUTOSTART_POLICY: str = "manual"
    SEMANTIC_MCP_EMBEDDING_HEALTH_PATH: str | None = "/health"
    SEMANTIC_MCP_RERANKER_BACKEND: str = "custom_http"
    SEMANTIC_MCP_RERANKER_BACKEND_ID: str | None = None
    SEMANTIC_MCP_RERANKER_MODEL: str | None = None
    SEMANTIC_MCP_RERANKER_URL: str | None = None
    SEMANTIC_MCP_RERANKER_LOCATION_TYPE: str = "unknown"
    SEMANTIC_MCP_RERANKER_DEVICE_TYPE: str = "unknown"
    SEMANTIC_MCP_RERANKER_MANAGED_BY_SERVICE: bool = False
    SEMANTIC_MCP_RERANKER_AUTOSTART_POLICY: str = "manual"
    SEMANTIC_MCP_RERANKER_HEALTH_PATH: str | None = "/health"
    SEMANTIC_MCP_COLBERT_BACKEND: str = "custom_http"
    SEMANTIC_MCP_COLBERT_BACKEND_ID: str | None = None
    SEMANTIC_MCP_COLBERT_MODEL: str | None = None
    SEMANTIC_MCP_COLBERT_URL: str | None = None
    SEMANTIC_MCP_COLBERT_LOCATION_TYPE: str = "unknown"
    SEMANTIC_MCP_COLBERT_DEVICE_TYPE: str = "unknown"
    SEMANTIC_MCP_COLBERT_MANAGED_BY_SERVICE: bool = False
    SEMANTIC_MCP_COLBERT_AUTOSTART_POLICY: str = "manual"
    SEMANTIC_MCP_COLBERT_HEALTH_PATH: str | None = "/health"

    SEMANTIC_MCP_TRANSPORT: str = "stdio"
    SEMANTIC_MCP_HTTP_HOST: str = "0.0.0.0"
    SEMANTIC_MCP_HTTP_PORT: int = 8011
    SEMANTIC_MCP_DEPENDENCY_WAIT_ATTEMPTS: int = 90
    SEMANTIC_MCP_DEPENDENCY_WAIT_DELAY_SEC: int = 2

    SEMANTIC_MCP_WATCH_ENABLED: bool = False
    SEMANTIC_MCP_WATCH_DEBOUNCE_SEC: int = 3
    SEMANTIC_MCP_RECONCILE_MAX_PATHS: int = 250
    SEMANTIC_MCP_RECONCILE_MAX_DELETED_PATHS: int = 100
    SEMANTIC_MCP_RECONCILE_MAX_POINTS_DELETED: int = 5000
    SEMANTIC_MCP_RECONCILE_MAX_DURATION_MS: int = 180000
    SEMANTIC_MCP_RECONCILE_MAX_SCAN_PATHS: int = 50000
    SEMANTIC_MCP_STATUS_MAX_SCAN_PATHS: int = 250000
    SEMANTIC_MCP_STATUS_MAX_DURATION_MS: int = 5000
    SEMANTIC_MCP_STATUS_MAX_HASHED_PATHS: int = 200
    SEMANTIC_MCP_GRAPH_INVALIDATED_PATH_LIMIT: int = 200
    SEMANTIC_MCP_GRAPH_UPDATE_MAX_PATHS: int = 100
    SEMANTIC_MCP_GRAPH_UPDATE_MAX_CHUNKS_SCANNED: int = 50000
    SEMANTIC_MCP_GRAPH_UPDATE_MAX_CHANGED_CHUNKS: int = 3000
    SEMANTIC_MCP_GRAPH_UPDATE_MAX_ROWS_DELETED: int = 200000
    SEMANTIC_MCP_GRAPH_UPDATE_MAX_ROWS_UPSERTED: int = 200000
    SEMANTIC_MCP_GRAPH_UPDATE_MAX_DURATION_MS: int = 180000
    SEMANTIC_MCP_GRAPH_UPDATE_LARGE_MAX_ROWS_DELETED: int = 1000000
    SEMANTIC_MCP_GRAPH_UPDATE_LARGE_MAX_ROWS_UPSERTED: int = 1000000
    SEMANTIC_MCP_GRAPH_UPDATE_LARGE_MAX_DURATION_MS: int = 600000
    SEMANTIC_MCP_GRAPH_UPDATE_DISCOVERY_MAX_PATHS: int = 1000
    SEMANTIC_MCP_GRAPH_UPDATE_DISCOVERY_MAX_CHUNKS: int = 50000
    SEMANTIC_MCP_GRAPH_UPDATE_MAX_REVERSE_DEPENDENT_PATHS: int = 100
    SEMANTIC_MCP_GRAPH_UPDATE_AUTO_BATCH_PATHS: int = 50
    SEMANTIC_MCP_GRAPH_UPDATE_AUTO_BATCHES: int = 5
    SEMANTIC_MCP_GRAPH_UPDATE_ON_WATCHER: bool = True
    SEMANTIC_MCP_GRAPH_UPDATE_ON_STARTUP_RECONCILE: bool = True
    SEMANTIC_MCP_EMBED_BATCH_DOCS: int = 24
    SEMANTIC_MCP_EMBED_BATCH_CHARS: int = 12000
    SEMANTIC_MCP_MAX_CHUNK_CHARS: int = 800
    SEMANTIC_MCP_QDRANT_UPSERT_BATCH_POINTS: int = 16
    SEMANTIC_MCP_QDRANT_UPSERT_MAX_BYTES: int = 24 * 1024 * 1024
    SEMANTIC_MCP_REPO_CONTEXT_GRAPH_AUTO_ENABLED: bool = False

    # "auto" → auto-detect из layout репозитория (see auto_detect.py).
    # Для явного переопределения передайте CSV: apps/**,src/**,docs/**
    SEMANTIC_MCP_INCLUDE_GLOBS: list[str] = Field(default_factory=lambda: ["auto"])

    # Префиксы путей, которые classify_scope относит к docs-корпусу.
    # Пусто → auto-detect (обнаружить реальные doc-директории в репо).
    SEMANTIC_MCP_DOC_PATH_PREFIXES: list[str] = Field(default_factory=list)
    SEMANTIC_MCP_EXCLUDE_GLOBS: list[str] = Field(
        default_factory=lambda: [
            ".git/**",
            ".venv/**",
            "venv/**",
            "**/__pycache__/**",
            "node_modules/**",
            "artifacts/**",
            "tmp/**",
            "temp/**",
            "**/*.png",
            "**/*.jpg",
            "**/*.jpeg",
            "**/*.gif",
            "**/*.webp",
            "**/*.ico",
            "**/*.svg",
            "**/*.zip",
            "**/*.gz",
            "**/*.tar",
            "**/*.7z",
            "**/*.pdf",
            "**/*.db",
            "**/*.sqlite",
            "**/.env*",
            "**/*.pem",
            "**/*.key",
            "**/*.crt",
            "**/*.pfx",
            "**/*.kdbx",
        ]
    )
    SEMANTIC_MCP_LOG_LEVEL: str = "info"

    @field_validator(
        "SEMANTIC_MCP_INCLUDE_GLOBS",
        "SEMANTIC_MCP_EXCLUDE_GLOBS",
        "SEMANTIC_MCP_DOC_PATH_PREFIXES",
        mode="before",
    )
    @classmethod
    def _split_csv_lists(cls, value):
        """Разрешить список globs/prefixes через CSV-строку в env."""

        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator(
        "SEMANTIC_MCP_QUERY_TEMPLATE",
        "SEMANTIC_MCP_DOCUMENT_PREFIX",
        "SEMANTIC_MCP_PROFILE_NAME",
        mode="before",
    )
    @classmethod
    def _normalize_string_fields(cls, value):
        """Сохранить пустые строки и декодировать escaped newlines в env."""

        if isinstance(value, str):
            return value.replace("\\n", "\n")
        return value

    @property
    def repo_root(self) -> Path:
        """Вернуть корень репозитория как Path."""

        return Path(self.SEMANTIC_MCP_REPO_ROOT).resolve()

    @property
    def registry_db_path(self) -> Path:
        """Вернуть путь к SQLite registry."""

        return Path(self.SEMANTIC_MCP_REGISTRY_DB_PATH).resolve()

    @property
    def logical_repo_root(self) -> Path:
        """Вернуть логический repo root как Path, если это локальный posix path."""

        if self.SEMANTIC_MCP_LOGICAL_REPO_ROOT:
            return Path(self.SEMANTIC_MCP_LOGICAL_REPO_ROOT).resolve()
        return self.repo_root

    @property
    def logical_repo_identity(self) -> str:
        """Вернуть логическую repo identity для registry/status/collection names."""

        if self.SEMANTIC_MCP_LOGICAL_REPO_ROOT:
            return normalize_repo_identity(self.SEMANTIC_MCP_LOGICAL_REPO_ROOT)
        return normalize_repo_identity(self.repo_root)

    @property
    def collection_code(self) -> str:
        """Имя коллекции для code corpus."""

        return (
            f"{self.SEMANTIC_MCP_COLLECTION_PREFIX}_{self.repo_key_slug}_{self.profile_slug}_"
            f"{self.embedding_model_slug}_code_v"
            f"{self.SEMANTIC_MCP_INDEX_SCHEMA_VERSION}"
        )

    @property
    def collection_docs(self) -> str:
        """Имя коллекции для docs corpus."""

        return (
            f"{self.SEMANTIC_MCP_COLLECTION_PREFIX}_{self.repo_key_slug}_{self.profile_slug}_"
            f"{self.embedding_model_slug}_docs_v"
            f"{self.SEMANTIC_MCP_INDEX_SCHEMA_VERSION}"
        )

    @property
    def embedding_model_slug(self) -> str:
        """Вернуть нормализованный slug embedding модели."""

        return _slugify_embedding_model(self.SEMANTIC_MCP_EMBEDDING_MODEL)

    @property
    def embedding_backend_slug(self) -> str:
        """Вернуть нормализованный slug embedding backend type."""

        return _slugify_backend_id(self.SEMANTIC_MCP_EMBEDDING_BACKEND)

    @property
    def profile_slug(self) -> str:
        """Вернуть нормализованный slug профиля индекса."""

        return _slugify_embedding_model(self.SEMANTIC_MCP_PROFILE_NAME)

    @property
    def repo_key_slug(self) -> str:
        """Вернуть стабильный ключ текущего target repo."""

        if self.SEMANTIC_MCP_REPO_KEY:
            return _slugify_embedding_model(self.SEMANTIC_MCP_REPO_KEY)
        return build_repo_key(self.logical_repo_identity)

    @property
    def embedding_backend_id(self) -> str:
        """Вернуть stable backend id для текущего embedding runtime."""

        if self.SEMANTIC_MCP_EMBEDDING_BACKEND_ID:
            return self.SEMANTIC_MCP_EMBEDDING_BACKEND_ID.strip()
        return (
            f"embedding/{self.embedding_backend_slug}_"
            f"{self.profile_slug}_{self.embedding_model_slug}"
        )

    @property
    def embedding_backend_transport(self) -> str:
        """Вернуть transport текущего embedding backend."""

        backend = self.SEMANTIC_MCP_EMBEDDING_BACKEND.strip().lower()
        if backend in {"tei_http", "tei", "openai_embeddings_http", "custom_http"}:
            return "http"
        if backend == "fastembed_local":
            return "local"
        return "unknown"

    @staticmethod
    def _transport_for_backend_type(backend_type: str) -> str:
        """Разрешить transport по backend_type для optional backend roles."""

        backend = backend_type.strip().lower()
        if backend in {"tei_http", "tei", "openai_embeddings_http", "custom_http"}:
            return "http"
        if backend == "fastembed_local":
            return "local"
        return "unknown"

    @property
    def embedding_backend_location_type(self) -> str:
        """Вернуть location type embedding backend с мягким fallback по endpoint."""

        configured = self.SEMANTIC_MCP_EMBEDDING_LOCATION_TYPE.strip().lower()
        if configured and configured != "unknown":
            return configured

        endpoint = (self.embedding_backend_endpoint or "").lower()
        backend = self.SEMANTIC_MCP_EMBEDDING_BACKEND.strip().lower()
        if backend == "fastembed_local":
            return "host"
        if endpoint.startswith("http://tei") or endpoint.startswith("https://tei"):
            return "docker"
        if "host.docker.internal" in endpoint or "127.0.0.1" in endpoint or "localhost" in endpoint:
            return "host"
        if endpoint:
            return "remote"
        return "unknown"

    @property
    def embedding_backend_device_type(self) -> str:
        """Вернуть device type embedding backend с мягким fallback по backend type."""

        configured = self.SEMANTIC_MCP_EMBEDDING_DEVICE_TYPE.strip().lower()
        if configured and configured != "unknown":
            return configured

        backend = self.SEMANTIC_MCP_EMBEDDING_BACKEND.strip().lower()
        if backend == "fastembed_local":
            return "cpu"
        return "unknown"

    @property
    def embedding_backend_managed_by_service(self) -> bool:
        """Определить, управляется ли backend самим repo-semantic-mcp runtime/compose."""

        if self.SEMANTIC_MCP_EMBEDDING_MANAGED_BY_SERVICE:
            return True
        endpoint = (self.embedding_backend_endpoint or "").lower()
        return endpoint.startswith("http://tei") or endpoint.startswith("https://tei")

    @property
    def embedding_backend_endpoint(self) -> str | None:
        """Вернуть endpoint текущего embedding backend, если он сетевой."""

        backend = self.SEMANTIC_MCP_EMBEDDING_BACKEND.strip().lower()
        if backend in {"tei_http", "tei"}:
            return self.SEMANTIC_MCP_TEI_URL
        return None

    def current_embedding_backend_payload(self) -> dict[str, object]:
        """Собрать backend registry payload для текущего embedding runtime."""

        payload = {
            "backend_id": self.embedding_backend_id,
            "role": "embedding",
            "backend_type": self.SEMANTIC_MCP_EMBEDDING_BACKEND.strip().lower(),
            "transport": self.embedding_backend_transport,
            "endpoint": self.embedding_backend_endpoint,
            "managed_by_service": self.embedding_backend_managed_by_service,
            "autostart_policy": self.SEMANTIC_MCP_EMBEDDING_AUTOSTART_POLICY,
            "health_path": self.SEMANTIC_MCP_EMBEDDING_HEALTH_PATH,
            "location_type": self.embedding_backend_location_type,
            "device_type": self.embedding_backend_device_type,
            "model_name": self.SEMANTIC_MCP_EMBEDDING_MODEL,
            "config_blob": {
                "profile_name": self.SEMANTIC_MCP_PROFILE_NAME,
                "query_template": self.SEMANTIC_MCP_QUERY_TEMPLATE,
                "document_prefix": self.SEMANTIC_MCP_DOCUMENT_PREFIX,
                "tei_query_prompt_name": self.SEMANTIC_MCP_TEI_QUERY_PROMPT_NAME,
                "tei_document_prompt_name": self.SEMANTIC_MCP_TEI_DOCUMENT_PROMPT_NAME,
            },
        }
        catalog_entry = get_backend_catalog_entry(self.embedding_backend_id)
        if catalog_entry is not None:
            catalog_payload = catalog_entry.to_registry_payload()
            payload["backend_type"] = catalog_payload["backend_type"]
            payload["transport"] = catalog_payload["transport"]
            payload["managed_by_service"] = catalog_payload["managed_by_service"]
            payload["autostart_policy"] = catalog_payload["autostart_policy"]
            payload["health_path"] = catalog_payload["health_path"]
            payload["location_type"] = catalog_payload["location_type"]
            payload["device_type"] = catalog_payload["device_type"]
            payload["config_blob"] = {
                **catalog_payload["config_blob"],
                **payload["config_blob"],
            }
            if payload["endpoint"] is None:
                payload["endpoint"] = catalog_payload["endpoint"]
            if payload["model_name"] is None:
                payload["model_name"] = catalog_payload["model_name"]
        return payload

    def _optional_backend_payload(
        self,
        *,
        role: str,
        backend_type: str,
        backend_id: str | None,
        endpoint: str | None,
        model_name: str | None,
        location_type: str,
        device_type: str,
        managed_by_service: bool,
        autostart_policy: str,
        health_path: str | None,
    ) -> dict[str, object] | None:
        """Собрать registry payload для optional inference backend roles."""

        effective_backend_id = backend_id.strip() if backend_id else ""
        effective_backend_type = backend_type.strip().lower()
        if not effective_backend_id and not endpoint and not model_name:
            return None
        if not effective_backend_id:
            model_slug = _slugify_backend_id(model_name or effective_backend_type or role)
            effective_backend_id = f"{role}/{_slugify_backend_id(effective_backend_type)}_{model_slug}"

        payload = {
            "backend_id": effective_backend_id,
            "role": role,
            "backend_type": effective_backend_type,
            "transport": self._transport_for_backend_type(effective_backend_type),
            "endpoint": endpoint,
            "managed_by_service": managed_by_service,
            "autostart_policy": autostart_policy,
            "health_path": health_path,
            "location_type": location_type,
            "device_type": device_type,
            "model_name": model_name,
            "config_blob": {},
        }
        catalog_entry = get_backend_catalog_entry(effective_backend_id)
        if catalog_entry is not None:
            catalog_payload = catalog_entry.to_registry_payload()
            payload["backend_type"] = catalog_payload["backend_type"]
            payload["transport"] = catalog_payload["transport"]
            payload["managed_by_service"] = catalog_payload["managed_by_service"]
            payload["autostart_policy"] = catalog_payload["autostart_policy"]
            payload["health_path"] = catalog_payload["health_path"]
            payload["location_type"] = catalog_payload["location_type"]
            payload["device_type"] = catalog_payload["device_type"]
            payload["config_blob"] = dict(catalog_payload["config_blob"])
            if payload["endpoint"] is None:
                payload["endpoint"] = catalog_payload["endpoint"]
            if payload["model_name"] is None:
                payload["model_name"] = catalog_payload["model_name"]
        return payload

    def current_optional_backend_payloads(self) -> list[dict[str, object]]:
        """Собрать registry payloads для optional inference roles."""

        payloads: list[dict[str, object]] = []
        reranker = self._optional_backend_payload(
            role="reranker",
            backend_type=self.SEMANTIC_MCP_RERANKER_BACKEND,
            backend_id=self.SEMANTIC_MCP_RERANKER_BACKEND_ID,
            endpoint=self.SEMANTIC_MCP_RERANKER_URL,
            model_name=self.SEMANTIC_MCP_RERANKER_MODEL,
            location_type=self.SEMANTIC_MCP_RERANKER_LOCATION_TYPE,
            device_type=self.SEMANTIC_MCP_RERANKER_DEVICE_TYPE,
            managed_by_service=self.SEMANTIC_MCP_RERANKER_MANAGED_BY_SERVICE,
            autostart_policy=self.SEMANTIC_MCP_RERANKER_AUTOSTART_POLICY,
            health_path=self.SEMANTIC_MCP_RERANKER_HEALTH_PATH,
        )
        if reranker is not None:
            payloads.append(reranker)
        colbert = self._optional_backend_payload(
            role="colbert",
            backend_type=self.SEMANTIC_MCP_COLBERT_BACKEND,
            backend_id=self.SEMANTIC_MCP_COLBERT_BACKEND_ID,
            endpoint=self.SEMANTIC_MCP_COLBERT_URL,
            model_name=self.SEMANTIC_MCP_COLBERT_MODEL,
            location_type=self.SEMANTIC_MCP_COLBERT_LOCATION_TYPE,
            device_type=self.SEMANTIC_MCP_COLBERT_DEVICE_TYPE,
            managed_by_service=self.SEMANTIC_MCP_COLBERT_MANAGED_BY_SERVICE,
            autostart_policy=self.SEMANTIC_MCP_COLBERT_AUTOSTART_POLICY,
            health_path=self.SEMANTIC_MCP_COLBERT_HEALTH_PATH,
        )
        if colbert is not None:
            payloads.append(colbert)
        return payloads

    def update_globs(self, new_globs: list[str]) -> None:
        """Обновить include globs в runtime и сбросить кэш авто-детектирования.

        Принимает либо конкретный список глобов, либо ["auto"] для сброса
        к авто-детектированию. Используется MCP-инструментом update_include_globs.
        """

        object.__setattr__(self, "SEMANTIC_MCP_INCLUDE_GLOBS", new_globs)
        object.__setattr__(self, "_globs_cache", None)
        object.__setattr__(self, "_doc_prefixes_cache", None)

    def update_doc_prefixes(self, new_prefixes: list[str]) -> None:
        """Обновить doc prefixes в runtime и сбросить связанные кэши."""

        object.__setattr__(self, "SEMANTIC_MCP_DOC_PATH_PREFIXES", new_prefixes)
        object.__setattr__(self, "_doc_prefixes_cache", None)

    def apply_repo_registry_config(
        self,
        *,
        include_globs: list[str],
        doc_prefixes: list[str],
    ) -> None:
        """Применить repo-specific config из persistent registry к runtime settings."""

        self.update_globs(list(include_globs))
        self.update_doc_prefixes(list(doc_prefixes))

    def clone_with_overrides(self, **overrides: object) -> "SemanticMcpSettings":
        """Построить независимую копию settings с явными override-полями."""

        data = self.model_dump(mode="python")
        data.update(overrides)
        return SemanticMcpSettings(**data)

    @property
    def effective_include_globs(self) -> list[str]:
        """Вернуть итоговые include globs: авто-обнаружение или явный список.

        Если SEMANTIC_MCP_INCLUDE_GLOBS == ["auto"] или пусто — сканирует
        верхний уровень repo_root и строит globs из реального layout.
        Результат кэшируется, чтобы не сканировать диск при каждом вызове.
        """

        if self._globs_cache is None:
            globs = self.SEMANTIC_MCP_INCLUDE_GLOBS
            if not globs or globs == ["auto"]:
                from services.repo_semantic.auto_detect import detect_repo_globs

                result = detect_repo_globs(self.repo_root)
            else:
                result = list(globs)
            object.__setattr__(self, "_globs_cache", result)
        return self._globs_cache  # type: ignore[return-value]

    @property
    def effective_doc_prefixes(self) -> tuple[str, ...]:
        """Вернуть префиксы путей для classify_scope (docs-корпус).

        Если SEMANTIC_MCP_DOC_PATH_PREFIXES пусто — определяет doc-директории
        из реального layout репозитория.
        Результат кэшируется.
        """

        if self._doc_prefixes_cache is None:
            if self.SEMANTIC_MCP_DOC_PATH_PREFIXES:
                result: tuple[str, ...] = tuple(self.SEMANTIC_MCP_DOC_PATH_PREFIXES)
            else:
                from services.repo_semantic.auto_detect import detect_doc_prefixes

                result = detect_doc_prefixes(self.repo_root)
            object.__setattr__(self, "_doc_prefixes_cache", result)
        return self._doc_prefixes_cache  # type: ignore[return-value]

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=True,
        enable_decoding=False,
        env_ignore_empty=False,
    )
