"""Backend registry public contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from services.repo_semantic.contracts.common import BackendRole


class BackendRegistryEntry(BaseModel):
    """Persistent backend registry entry для inference role."""

    backend_id: str
    role: BackendRole
    backend_type: str
    transport: str
    endpoint: str | None = None
    managed_by_service: bool = False
    autostart_policy: str = "manual"
    health_path: str | None = None
    location_type: str = "unknown"
    device_type: str = "unknown"
    model_name: str | None = None
    config_blob: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class BackendStatusResult(BaseModel):
    """Статус backend entry и его связи с текущим runtime."""

    entry: BackendRegistryEntry
    selected_for_role: bool
    runtime_backend_id: str | None = None
    runtime_backend_active: bool = False
    runtime_switch_required: bool = False
    healthy: bool | None = None
    health_error: str | None = None
