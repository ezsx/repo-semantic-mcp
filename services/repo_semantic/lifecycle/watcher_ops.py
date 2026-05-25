"""Watcher lifecycle operation helpers."""

from __future__ import annotations

from typing import Any

from services.repo_semantic.lifecycle.lease import lifecycle_mutation
from services.repo_semantic.registry import LifecycleLeaseActiveError
from services.repo_semantic.watcher import StartupReconcileBoundsExceeded


def _startup_reconcile_bounds_action(startup_reconcile: dict[str, object]) -> dict[str, object]:
    """Build a permission-required action for bounded startup reconcile failures."""

    return {
        "code": "startup_reconcile_bounds_exceeded",
        "severity": "warning",
        "title": "Startup reconcile exceeds safe bounds",
        "detail": (
            "Watcher startup stopped before mutation because the incremental "
            "repair plan exceeded configured safe-recovery bounds."
        ),
        "reason_codes": list(startup_reconcile.get("warning_codes") or ["startup_reconcile_bounds_exceeded"]),
        "requires_user_permission": True,
        "safe_auto_run": False,
        "destructive": False,
        "expensive": False,
    }


def _idempotent_start_retry_payload(runtime: Any) -> dict[str, object]:
    """Return current watcher status for a repeated start_watcher request."""

    return {
        "watch_running": bool(runtime.watcher and runtime.watcher.is_running),
        "startup_reconcile": {
            "paths": 0,
            "code": 0,
            "docs": 0,
            "skipped": True,
            "reason": "idempotent_retry_in_progress",
        },
        "idempotent_retry": True,
        "operation_in_progress": bool(runtime.watcher and not runtime.watcher.is_running),
        "status": None,
        "status_skipped_reason": "idempotent_retry_in_progress",
    }


def start_watcher_for_runtime(
    runtime: Any,
    *,
    idempotency_key: str | None = None,
) -> dict[str, object]:
    """Start the configured watcher for the current runtime repo."""

    if runtime.watcher is None:
        raise RuntimeError("Watcher is disabled by configuration for this runtime")
    runtime.search_service.ensure_search_available()
    try:
        if runtime.watcher.is_running:
            with lifecycle_mutation(runtime, "start_watcher", idempotency_key=idempotency_key):
                runtime.search_service.sync_registry_state(
                    status="indexed",
                    watch_running=True,
                )
                status = runtime.search_service.index_status().model_dump()
            return {
                "watch_running": True,
                "startup_reconcile": {"paths": 0, "code": 0, "docs": 0, "skipped": True},
                "status": status,
            }
        with lifecycle_mutation(runtime, "start_watcher", idempotency_key=idempotency_key):
            indexer = getattr(runtime, "indexer", getattr(runtime.watcher, "_indexer", None))
            settings = getattr(indexer, "_settings", None)
            try:
                initial_snapshot = runtime.watcher.capture_snapshot(
                    max_paths=getattr(settings, "SEMANTIC_MCP_RECONCILE_MAX_SCAN_PATHS", None),
                    max_duration_ms=getattr(settings, "SEMANTIC_MCP_RECONCILE_MAX_DURATION_MS", None),
                )
            except StartupReconcileBoundsExceeded as exc:
                return {
                    "watch_running": False,
                    "startup_reconcile": exc.summary,
                    "permission_required": True,
                    "recommended_actions": [_startup_reconcile_bounds_action(exc.summary)],
                    "status": None,
                    "status_skipped_reason": "startup_reconcile_bounds_exceeded",
                }
            startup_reconcile = runtime.watcher.run_startup_reconcile()
            if startup_reconcile.get("bounds_exceeded"):
                return {
                    "watch_running": False,
                    "startup_reconcile": startup_reconcile,
                    "permission_required": True,
                    "recommended_actions": [_startup_reconcile_bounds_action(startup_reconcile)],
                    "status": None,
                    "status_skipped_reason": "startup_reconcile_bounds_exceeded",
                }
            runtime.watcher.start(initial_snapshot=initial_snapshot)
            runtime.search_service.sync_registry_state(
                status="indexed",
                watch_running=True,
                record_index_revision=True,
            )
    except LifecycleLeaseActiveError as exc:
        if idempotency_key and exc.existing_operation == "start_watcher" and exc.same_idempotency_key:
            return _idempotent_start_retry_payload(runtime)
        raise
    status = runtime.search_service.index_status().model_dump()
    return {
        "watch_running": runtime.watcher.is_running,
        "startup_reconcile": startup_reconcile,
        "status": status,
    }


def stop_watcher_for_runtime(runtime: Any) -> dict[str, object]:
    """Stop the watcher for the current runtime repo."""

    if runtime.watcher is None:
        return {
            "watch_running": False,
            "status": runtime.search_service.index_status().model_dump(),
        }
    with lifecycle_mutation(runtime, "stop_watcher"):
        runtime.watcher.stop()
        runtime.search_service.sync_registry_state(
            status=runtime.search_service.index_status().repo_state,
            watch_running=False,
        )
    return {
        "watch_running": runtime.watcher.is_running,
        "status": runtime.search_service.index_status().model_dump(),
    }
