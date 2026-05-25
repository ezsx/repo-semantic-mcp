"""Repository lifecycle/registry operation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from services.repo_semantic.auto_detect import detect_doc_prefixes, detect_repo_globs
from services.repo_semantic.config import build_repo_key


RUNTIME_SWITCH_COMMAND_HINT = (
    "Use scripts/agents/start_repo_semantic_for_project.* or "
    "ensure_repo_semantic_search.* -TargetRepoPath <repo_root> on the host."
)


def current_runtime_repo_state(runtime: Any) -> str:
    """Determine repo state from the current runtime index contents."""

    total_points = sum(runtime.indexer._store.count(scope) for scope in ("code", "docs"))
    return "indexed" if total_points > 0 else "registered"


def runtime_repo_root(runtime: Any) -> str:
    """Return the logical repo root for the current runtime."""

    return runtime.indexer._settings.logical_repo_identity


def repo_entry_or_raise(runtime: Any, repo_root: str):
    """Return a registry entry for a repo or raise a clear error."""

    entry = runtime.registry.get_repo(repo_root)
    if entry is None:
        raise RuntimeError(f"Repo is not registered: {repo_root}")
    return entry


def upsert_registered_repo(
    runtime: Any,
    *,
    repo_root: str,
    include_globs: list[str],
    doc_prefixes: list[str],
    active: bool,
):
    """Create or update a repo registry entry without losing metadata."""

    existing = runtime.registry.get_repo(repo_root)
    effective_status = existing.status if existing is not None else "registered"
    if existing is not None:
        config_changed = (
            list(existing.include_globs) != list(include_globs)
            or list(existing.doc_prefixes) != list(doc_prefixes)
        )
        if config_changed and existing.status == "indexed":
            effective_status = "stale"
    return runtime.registry.upsert_repo(
        repo_root,
        repo_key=existing.repo_key if existing is not None else build_repo_key(repo_root),
        display_name=existing.display_name if existing is not None else repo_root.rstrip("/").split("/")[-1],
        status=effective_status,
        active=active,
        index_profile=existing.index_profile if existing is not None else runtime.search_service.index_status().index_profile,
        include_globs=include_globs,
        doc_prefixes=doc_prefixes,
        exclude_globs=existing.exclude_globs
        if existing is not None
        else list(runtime.indexer._settings.SEMANTIC_MCP_EXCLUDE_GLOBS),
        last_full_build_ts=existing.last_full_build_ts if existing is not None else None,
        last_incremental_update_ts=existing.last_incremental_update_ts if existing is not None else None,
        last_error=existing.last_error if existing is not None else None,
        watch_enabled=existing.watch_enabled if existing is not None else False,
        watch_running=False,
        code_points_count=existing.code_points_count if existing is not None else 0,
        docs_points_count=existing.docs_points_count if existing is not None else 0,
    )


def runtime_switch_required_payload(runtime: Any, repo_root: str) -> dict[str, object]:
    """Build the standard cross-repo lifecycle payload."""

    entry = repo_entry_or_raise(runtime, repo_root)
    return {
        "repo_root": entry.repo_root,
        "runtime_repo_root": runtime_repo_root(runtime),
        "requires_runtime_switch": True,
        "runtime_switch_command_hint": RUNTIME_SWITCH_COMMAND_HINT,
        "status": entry.model_dump(),
    }


def register_repo_payload(
    runtime: Any,
    *,
    target_repo_root: str,
    include_globs: list[str] | None,
    doc_prefixes: list[str] | None,
    activate: bool,
) -> dict[str, object]:
    """Register a repo in the registry without running indexing."""

    existing = runtime.registry.get_repo(target_repo_root)
    runtime_visible = target_repo_root == runtime_repo_root(runtime)
    target_path = Path(target_repo_root)
    path_visible = runtime_visible or (target_path.exists() and target_path.is_dir())
    if not path_visible and existing is None and include_globs is None and doc_prefixes is None:
        resolved_globs = ["auto"]
        resolved_doc_prefixes = []
    else:
        if not path_visible and existing is None:
            resolved_globs = list(include_globs) if include_globs is not None else ["auto"]
            resolved_doc_prefixes = list(doc_prefixes) if doc_prefixes is not None else []
        else:
            resolved_globs = (
                list(include_globs)
                if include_globs is not None
                else (
                    list(existing.include_globs)
                    if existing is not None
                    else detect_repo_globs(target_path)
                )
            )
            resolved_doc_prefixes = (
                list(doc_prefixes)
                if doc_prefixes is not None
                else (
                    list(existing.doc_prefixes)
                    if existing is not None
                    else list(detect_doc_prefixes(target_path))
                )
            )
    resolved_active = (
        True if activate and target_repo_root == runtime_repo_root(runtime) else (existing.active if existing is not None else False)
    )
    entry = upsert_registered_repo(
        runtime,
        repo_root=target_repo_root,
        include_globs=resolved_globs,
        doc_prefixes=resolved_doc_prefixes,
        active=resolved_active,
    )
    if target_repo_root == runtime_repo_root(runtime):
        runtime.indexer._settings.apply_repo_registry_config(
            include_globs=entry.include_globs,
            doc_prefixes=entry.doc_prefixes,
        )
        runtime.search_service.sync_registry_state(
            status=entry.status,
            active=entry.active,
            watch_running=bool(runtime.watcher and runtime.watcher.is_running),
            last_error=entry.last_error,
        )
        return runtime.registry.get_repo(target_repo_root).model_dump()
    payload = runtime.registry.get_repo(target_repo_root).model_dump()
    if activate:
        payload["requires_runtime_switch"] = True
        payload["runtime_repo_root"] = runtime_repo_root(runtime)
        payload["runtime_switch_command_hint"] = RUNTIME_SWITCH_COMMAND_HINT
    return payload


def activate_repo_payload(runtime: Any, *, requested_repo_root: str) -> dict[str, object]:
    """Activate the current runtime repo or return an explicit switch payload."""

    active_runtime_root = runtime_repo_root(runtime)
    entry = repo_entry_or_raise(runtime, requested_repo_root)
    if requested_repo_root != active_runtime_root:
        return {
            "active_repo_root": runtime.search_service.index_status().active_repo_root,
            "runtime_repo_root": active_runtime_root,
            "requires_runtime_switch": True,
            "active_switch_applied": False,
            "runtime_switch_command_hint": RUNTIME_SWITCH_COMMAND_HINT,
            "status": entry.model_dump(),
        }
    runtime.indexer._settings.apply_repo_registry_config(
        include_globs=entry.include_globs,
        doc_prefixes=entry.doc_prefixes,
    )
    runtime_state = current_runtime_repo_state(runtime)
    effective_status = (
        "indexed"
        if entry.status in {"registered", "error"} and runtime_state == "indexed"
        else entry.status
    )
    runtime.search_service.sync_registry_state(
        status=effective_status,
        active=True,
        last_error=entry.last_error,
    )
    return {
        "active_repo_root": active_runtime_root,
        "runtime_repo_root": active_runtime_root,
        "requires_runtime_switch": False,
        "active_switch_applied": True,
        "status": runtime.search_service.index_status().model_dump(),
    }
