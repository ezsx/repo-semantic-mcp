"""Каталог известных inference backend'ов и profile defaults для helper/runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from services.repo_semantic.models import BackendRole


@dataclass(frozen=True, slots=True)
class BackendCatalogEntry:
    """Описание известного backend'а для orchestration и runtime registry."""

    backend_id: str
    role: BackendRole
    backend_type: str
    transport: str
    endpoint: str | None = None
    managed_by_service: bool = False
    autostart_policy: str = "manual"
    health_path: str | None = "/health"
    location_type: str = "unknown"
    device_type: str = "unknown"
    model_name: str | None = None
    env_candidates: tuple[str, ...] = ()
    config_blob: dict[str, Any] = field(default_factory=dict)

    def to_registry_payload(self) -> dict[str, Any]:
        """Преобразовать catalog entry к payload формату backend registry."""

        return {
            "backend_id": self.backend_id,
            "role": self.role,
            "backend_type": self.backend_type,
            "transport": self.transport,
            "endpoint": self.endpoint,
            "managed_by_service": self.managed_by_service,
            "autostart_policy": self.autostart_policy,
            "health_path": self.health_path,
            "location_type": self.location_type,
            "device_type": self.device_type,
            "model_name": self.model_name,
            "config_blob": dict(self.config_blob),
        }


BACKEND_CATALOG: dict[str, BackendCatalogEntry] = {
    "embedding/cpu_e5_bundled_tei": BackendCatalogEntry(
        backend_id="embedding/cpu_e5_bundled_tei",
        role="embedding",
        backend_type="tei_http",
        transport="http",
        endpoint="http://tei:80",
        managed_by_service=True,
        autostart_policy="explicit",
        health_path="/health",
        location_type="docker",
        device_type="cpu",
        model_name="intfloat/multilingual-e5-small",
        env_candidates=(".env.cpu", ".env.cpu.example", ".env", ".env.example"),
        config_blob={
            "launch_mode": "bundled_compose",
            "compose_variant": "cpu",
            "use_gpu_compose": False,
        },
    ),
    "embedding/gpu_pplx_wsl_managed": BackendCatalogEntry(
        backend_id="embedding/gpu_pplx_wsl_managed",
        role="embedding",
        backend_type="tei_http",
        transport="http",
        endpoint="http://host.docker.internal:8084",
        managed_by_service=False,
        autostart_policy="explicit",
        health_path="/health",
        location_type="wsl",
        device_type="gpu",
        model_name="pplx-embed-v1",
        env_candidates=(".env.gpu", ".env.gpu.example"),
        config_blob={
            "launch_mode": "managed_host",
            "compose_variant": "mcp_only",
            "use_gpu_compose": False,
            "managed_start_kind": "wsl_embedder",
        },
    ),
    "embedding/gpu_qwen3_external_manual": BackendCatalogEntry(
        backend_id="embedding/gpu_qwen3_external_manual",
        role="embedding",
        backend_type="tei_http",
        transport="http",
        endpoint="http://host.docker.internal:8084",
        managed_by_service=False,
        autostart_policy="manual",
        health_path="/health",
        location_type="host",
        device_type="gpu",
        model_name="Qwen/Qwen3-Embedding-0.6B",
        env_candidates=(".env.gpu.qwen3", ".env.gpu.qwen3.example"),
        config_blob={
            "launch_mode": "external_manual",
            "compose_variant": "mcp_only",
            "use_gpu_compose": False,
        },
    ),
    "embedding/gpu_qwen3_bundled_tei": BackendCatalogEntry(
        backend_id="embedding/gpu_qwen3_bundled_tei",
        role="embedding",
        backend_type="tei_http",
        transport="http",
        endpoint="http://tei:80",
        managed_by_service=True,
        autostart_policy="explicit",
        health_path="/health",
        location_type="docker",
        device_type="gpu",
        model_name="Qwen/Qwen3-Embedding-0.6B",
        env_candidates=(".env.gpu", ".env.gpu.example", ".env.gpu.qwen3", ".env.gpu.qwen3.example"),
        config_blob={
            "launch_mode": "bundled_compose",
            "compose_variant": "gpu",
            "use_gpu_compose": True,
        },
    ),
    "embedding/gpu_bge_m3_bundled_tei": BackendCatalogEntry(
        backend_id="embedding/gpu_bge_m3_bundled_tei",
        role="embedding",
        backend_type="tei_http",
        transport="http",
        endpoint="http://tei:80",
        managed_by_service=True,
        autostart_policy="explicit",
        health_path="/health",
        location_type="docker",
        device_type="gpu",
        model_name="BAAI/bge-m3",
        env_candidates=(".env.gpu.bge-m3", ".env.gpu.bge-m3.example"),
        config_blob={
            "launch_mode": "bundled_compose",
            "compose_variant": "gpu",
            "use_gpu_compose": True,
        },
    ),
    "embedding/gpu_pplx_v100_infinity": BackendCatalogEntry(
        backend_id="embedding/gpu_pplx_v100_infinity",
        role="embedding",
        backend_type="tei_http",
        transport="http",
        endpoint="http://tei:80",
        managed_by_service=True,
        autostart_policy="explicit",
        health_path="/health",
        location_type="docker",
        device_type="gpu",
        model_name="pplx-embed-v1",
        env_candidates=(".env.gpu.pplx-v100-infinity", ".env.gpu.pplx-v100-infinity.example"),
        config_blob={
            "launch_mode": "bundled_compose",
            "compose_variant": "gpu_pplx_v100_infinity",
            "use_gpu_compose": False,
            "compose_extra_files": ("docker-compose.repo-semantic-search.gpu-pplx-v100-infinity.yml",),
        },
    ),
    "reranker/http_external_manual": BackendCatalogEntry(
        backend_id="reranker/http_external_manual",
        role="reranker",
        backend_type="custom_http",
        transport="http",
        endpoint=None,
        managed_by_service=False,
        autostart_policy="manual",
        health_path="/health",
        location_type="host",
        device_type="unknown",
        model_name=None,
        config_blob={
            "launch_mode": "external_manual",
        },
    ),
    "colbert/http_external_manual": BackendCatalogEntry(
        backend_id="colbert/http_external_manual",
        role="colbert",
        backend_type="custom_http",
        transport="http",
        endpoint=None,
        managed_by_service=False,
        autostart_policy="manual",
        health_path="/health",
        location_type="host",
        device_type="unknown",
        model_name=None,
        config_blob={
            "launch_mode": "external_manual",
        },
    ),
}

PROFILE_DEFAULT_BACKENDS: dict[str, str] = {
    "cpu": "embedding/cpu_e5_bundled_tei",
    "gpu": "embedding/gpu_qwen3_bundled_tei",
    "gpu-qwen3": "embedding/gpu_qwen3_bundled_tei",
    "gpu-bge-m3": "embedding/gpu_bge_m3_bundled_tei",
    "gpu-pplx-v100-infinity": "embedding/gpu_pplx_v100_infinity",
}

BACKEND_DEFAULT_PROFILES: dict[str, str] = {
    backend_id: profile for profile, backend_id in PROFILE_DEFAULT_BACKENDS.items()
}


def get_backend_catalog_entry(backend_id: str) -> BackendCatalogEntry | None:
    """Вернуть catalog entry по backend id."""

    return BACKEND_CATALOG.get(backend_id)


def default_backend_id_for_profile(profile: str) -> str | None:
    """Вернуть backend id по profile alias."""

    return PROFILE_DEFAULT_BACKENDS.get(profile)


def env_candidates_for_profile(profile: str) -> tuple[str, ...]:
    """Вернуть список env-файлов для profile через catalog."""

    backend_id = default_backend_id_for_profile(profile)
    entry = get_backend_catalog_entry(backend_id) if backend_id else None
    return entry.env_candidates if entry is not None else ()


def preferred_profile_for_backend(backend_id: str) -> str | None:
    """Вернуть профиль, который по умолчанию соответствует backend id."""

    return BACKEND_DEFAULT_PROFILES.get(backend_id)
