"""Backend lifecycle/status helpers used by MCP wrappers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx

from services.repo_semantic.backend_catalog import preferred_profile_for_backend
from services.repo_semantic.models import BackendRole


def runtime_embedding_backend_id(runtime: Any) -> str:
    """Return the backend id used by the current embedding runtime."""

    return runtime.indexer._settings.embedding_backend_id


def selected_backend_for_role(runtime: Any, role: BackendRole):
    """Return the registry-selected backend for a role."""

    return runtime.registry.get_role_backend(role)


def embedding_backend_switch_payload(runtime: Any) -> dict[str, object] | None:
    """Return a pending embedding backend switch payload, if one is required."""

    selected = selected_backend_for_role(runtime, "embedding")
    runtime_backend_id = runtime_embedding_backend_id(runtime)
    if selected is None or selected.backend_id == runtime_backend_id:
        return None
    return {
        "backend_switch_required": True,
        "requested_backend_id": selected.backend_id,
        "runtime_backend_id": runtime_backend_id,
        "reason": "selected embedding backend differs from the current runtime backend",
        "status": runtime.search_service.index_status().model_dump(),
    }


def backend_status_payload(runtime: Any, backend_entry) -> dict[str, object]:
    """Build a status payload for one backend registry entry."""

    selected = selected_backend_for_role(runtime, backend_entry.role)
    runtime_backend_id = runtime_embedding_backend_id(runtime) if backend_entry.role == "embedding" else None
    runtime_backend_active = bool(runtime_backend_id and backend_entry.backend_id == runtime_backend_id)
    runtime_switch_required = bool(selected and runtime_backend_id and selected.backend_id != runtime_backend_id)
    healthy: bool | None = None
    health_error: str | None = None

    try:
        if runtime_backend_active and backend_entry.role == "embedding":
            runtime.embedding_provider.healthcheck()
            healthy = True
        elif backend_entry.transport == "http" and backend_entry.endpoint and backend_entry.health_path:
            response = httpx.get(
                f"{backend_entry.endpoint.rstrip('/')}{backend_entry.health_path}",
                timeout=10,
            )
            response.raise_for_status()
            healthy = True
    except Exception as exc:  # noqa: BLE001
        healthy = False
        health_error = str(exc)

    return {
        "entry": backend_entry.model_dump(),
        "selected_for_role": bool(selected and selected.backend_id == backend_entry.backend_id),
        "runtime_backend_id": runtime_backend_id,
        "runtime_backend_active": runtime_backend_active,
        "runtime_switch_required": runtime_switch_required,
        "launch_mode": backend_entry.config_blob.get("launch_mode"),
        "preferred_profile": preferred_profile_for_backend(backend_entry.backend_id),
        "healthy": healthy,
        "health_error": health_error,
    }


def resolve_backend_entry(runtime: Any, role: BackendRole | None, backend_id: str | None):
    """Resolve a backend entry by explicit id or current role binding."""

    if backend_id is None and role is None:
        raise RuntimeError("Either role or backend_id must be provided")
    entry = runtime.registry.get_backend(backend_id) if backend_id is not None else runtime.registry.get_role_backend(role)
    if entry is None:
        identifier = backend_id if backend_id is not None else role
        raise RuntimeError(f"Backend is not registered: {identifier}")
    return entry


def default_action_plan_repo_root(runtime: Any, *, runtime_repo_root: Callable[[Any], str]) -> str:
    """Return repo_root used by default for an operator action plan."""

    active_repo = runtime.registry.get_active_repo()
    if active_repo is not None:
        return active_repo.repo_root
    return runtime_repo_root(runtime)


def _powershell_quote(value: str) -> str:
    """Safely quote a value for PowerShell command hints."""

    return "'" + value.replace("'", "''") + "'"


def _host_ensure_command_hint(backend_entry, repo_root: str) -> str | None:
    """Build a host-side command hint for the selected backend/profile."""

    profile = preferred_profile_for_backend(backend_entry.backend_id)
    if profile is None:
        return None
    return (
        "pwsh -File scripts/agents/ensure_repo_semantic_search.ps1 "
        f"-Profile {profile} -TargetRepoPath {_powershell_quote(repo_root)}"
    )


def backend_action_plan_payload(
    runtime: Any,
    backend_entry,
    *,
    target_repo_root: str,
) -> dict[str, object]:
    """Build an operator-facing action plan for explicit backend lifecycle."""

    status = backend_status_payload(runtime, backend_entry)
    launch_mode = str(backend_entry.config_blob.get("launch_mode") or "unknown")
    preferred_profile = preferred_profile_for_backend(backend_entry.backend_id)
    host_ensure_command = _host_ensure_command_hint(backend_entry, target_repo_root)
    manual_backend_start_required = launch_mode == "external_manual" and not status["healthy"]
    runtime_switch_required = bool(status["runtime_switch_required"])

    steps: list[dict[str, object]] = []
    state = "ready"
    summary = "Backend is ready for the current runtime."

    if manual_backend_start_required:
        state = "manual_backend_start_required"
        if runtime_switch_required:
            summary = (
                "The selected manual-only backend is not healthy yet; start it first, then perform the explicit host-side runtime switch."
            )
        else:
            summary = "The selected backend is manual-only and is not healthy yet."
    elif runtime_switch_required:
        state = "runtime_switch_required"
        summary = "Selected backend differs from the current runtime backend; explicit host-side runtime switch is required."
    elif status["healthy"] is False:
        state = "backend_unhealthy"
        summary = "Backend is registered but currently unhealthy."
    elif status["healthy"] is None:
        state = "health_unknown"
        summary = "Backend health could not be determined from the current runtime."

    if launch_mode == "external_manual":
        steps.append(
            {
                "kind": "manual_backend_start",
                "description": (
                    "Start the external backend manually and make sure the configured endpoint responds."
                ),
                "endpoint": backend_entry.endpoint,
                "health_path": backend_entry.health_path,
            }
        )
    if host_ensure_command is not None and state != "ready":
        steps.append(
            {
                "kind": "host_runtime_ensure",
                "description": (
                    "Run the host-side ensure helper so repo-semantic-search uses the selected backend/profile for this repo."
                ),
                "command": host_ensure_command,
            }
        )
    steps.append(
        {
            "kind": "verify",
            "description": "Verify backend and repo status after the host-side action completes.",
            "mcp_tools": ["get_backend_status", "get_repo_status", "index_status"],
        }
    )

    return {
        "backend_id": backend_entry.backend_id,
        "role": backend_entry.role,
        "repo_root": target_repo_root,
        "launch_mode": launch_mode,
        "preferred_profile": preferred_profile,
        "operator_action_state": state,
        "summary": summary,
        "host_action_required": state != "ready",
        "manual_backend_start_required": manual_backend_start_required,
        "host_ensure_command": host_ensure_command,
        "status": status,
        "steps": steps,
    }
