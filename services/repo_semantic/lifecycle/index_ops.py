"""Index lifecycle operation handlers used by MCP wrappers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from services.repo_semantic.lifecycle.lease import lifecycle_mutation


def rebuild_current_index(
    runtime: Any,
    *,
    sync_watch_running_on_start: bool = True,
    use_lease: bool = True,
) -> dict[str, object]:
    """Rebuild the current runtime index and return the public MCP payload."""

    if use_lease:
        with lifecycle_mutation(runtime, "rebuild_index"):
            payload = rebuild_current_index(
                runtime,
                sync_watch_running_on_start=sync_watch_running_on_start,
                use_lease=False,
            )
        payload["status"] = runtime.search_service.index_status().model_dump()
        return payload

    if runtime.watcher and runtime.watcher.is_running:
        runtime.watcher.stop()
    if sync_watch_running_on_start:
        runtime.search_service.sync_registry_state(status="indexing", watch_running=False)
    else:
        runtime.search_service.sync_registry_state(status="indexing")
    try:
        result = runtime.indexer.rebuild_index()
        runtime.search_service.invalidate_cache()
        runtime.search_service.sync_registry_state(
            status="indexed",
            record_index_revision=True,
        )
    except Exception as exc:
        runtime.search_service.sync_registry_state(
            status="error",
            watch_running=False,
            last_error=str(exc),
        )
        raise
    return {
        "rebuild": result,
        "status": runtime.search_service.index_status().model_dump(),
    }


def reindex_current_paths(
    runtime: Any,
    *,
    paths: list[str],
    current_runtime_repo_state: Callable[[Any], str],
    use_lease: bool = True,
) -> dict[str, int]:
    """Reindex selected paths in the current runtime repo."""

    if use_lease:
        with lifecycle_mutation(runtime, "reindex_paths"):
            return reindex_current_paths(
                runtime,
                paths=paths,
                current_runtime_repo_state=current_runtime_repo_state,
                use_lease=False,
            )

    runtime.search_service.sync_registry_state(status="indexing")
    try:
        result = runtime.indexer.reindex_paths(paths)
        runtime.search_service.invalidate_cache()
        runtime.search_service.sync_registry_state(
            status=current_runtime_repo_state(runtime),
            record_index_revision=True,
        )
    except Exception as exc:
        runtime.search_service.sync_registry_state(
            status="error",
            last_error=str(exc),
        )
        raise
    return result


def update_include_globs_and_rebuild(
    runtime: Any,
    *,
    globs: list[str],
) -> dict[str, object]:
    """Apply include globs and rebuild the current runtime index."""

    with lifecycle_mutation(runtime, "update_include_globs"):
        settings = runtime.indexer._settings
        old_globs = settings.effective_include_globs

        settings.update_globs(globs)
        new_globs = settings.effective_include_globs

        rebuild_payload = rebuild_current_index(
            runtime,
            sync_watch_running_on_start=False,
            use_lease=False,
        )
    rebuild_payload["status"] = runtime.search_service.index_status().model_dump()
    return {
        "old_globs": old_globs,
        "new_globs": new_globs,
        **rebuild_payload,
    }
