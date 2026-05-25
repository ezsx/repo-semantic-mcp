"""Lifecycle mutation lease helpers."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import threading
from typing import Any, Iterator

from services.repo_semantic.logging import jlog
from services.repo_semantic.registry import LifecycleLease


def _runtime_lease_identity(runtime: Any) -> tuple[object | None, str | None, str, str]:
    registry = getattr(runtime, "registry", None)
    indexer = getattr(runtime, "indexer", None)
    settings = getattr(indexer, "_settings", None)
    embedding_provider = getattr(runtime, "embedding_provider", None)
    if registry is None or settings is None:
        return None, None, "", ""
    repo_root = getattr(settings, "logical_repo_identity", None) or getattr(
        settings,
        "SEMANTIC_MCP_REPO_ROOT",
        None,
    )
    index_profile = (
        embedding_provider.index_profile()
        if embedding_provider is not None and hasattr(embedding_provider, "index_profile")
        else getattr(settings, "SEMANTIC_MCP_PROFILE_NAME", "default")
    )
    backend_id = getattr(settings, "embedding_backend_id", "unknown")
    return registry, repo_root, index_profile, backend_id


def _start_lease_heartbeat(
    registry: Any,
    lease: LifecycleLease,
    *,
    ttl_sec: int,
    interval_sec: float | None = None,
) -> tuple[threading.Event, threading.Thread]:
    stop_event = threading.Event()
    heartbeat_interval_sec = interval_sec if interval_sec is not None else max(1.0, min(30.0, ttl_sec / 3))

    def _heartbeat_loop() -> None:
        while not stop_event.wait(heartbeat_interval_sec):
            try:
                heartbeat_updated = registry.heartbeat_lifecycle_lease(lease, ttl_sec=ttl_sec)
            except Exception as exc:  # noqa: BLE001
                jlog(
                    "warning",
                    "lifecycle_lease_heartbeat_failed",
                    operation=lease.operation,
                    error=str(exc),
                )
                continue
            if not heartbeat_updated:
                break

    thread = threading.Thread(
        target=_heartbeat_loop,
        name=f"repo-semantic-lease-heartbeat-{lease.operation}",
        daemon=True,
    )
    thread.start()
    return stop_event, thread


@contextmanager
def lifecycle_mutation_for_identity(
    *,
    registry: Any,
    repo_root: str,
    index_profile: str,
    backend_id: str,
    operation: str,
    ttl_sec: int = 900,
    idempotency_key: str | None = None,
) -> Iterator[LifecycleLease | None]:
    """Serialize lifecycle mutations for a known repo/profile/backend identity."""

    lease = registry.acquire_lifecycle_lease(
        repo_root,
        index_profile=index_profile,
        backend_id=backend_id,
        operation=operation,
        ttl_sec=ttl_sec,
        idempotency_key=idempotency_key,
    )
    stop_heartbeat, heartbeat_thread = _start_lease_heartbeat(
        registry,
        lease,
        ttl_sec=ttl_sec,
    )
    try:
        yield lease
    except Exception as exc:
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=1)
        registry.release_lifecycle_lease(
            lease,
            status="error",
            error=str(exc),
        )
        raise
    else:
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=1)
        registry.release_lifecycle_lease(
            lease,
            status="success",
        )


@contextmanager
def lifecycle_mutation(
    runtime: Any,
    operation: str,
    *,
    enabled: bool = True,
    ttl_sec: int = 900,
    idempotency_key: str | None = None,
) -> Iterator[LifecycleLease | None]:
    """Serialize repo/profile/backend lifecycle mutations when registry exists."""

    if not enabled:
        with nullcontext(None) as lease:
            yield lease
        return
    registry, repo_root, index_profile, backend_id = _runtime_lease_identity(runtime)
    if registry is None or repo_root is None:
        with nullcontext(None) as lease:
            yield lease
        return
    with lifecycle_mutation_for_identity(
        registry=registry,
        repo_root=repo_root,
        index_profile=index_profile,
        backend_id=backend_id,
        operation=operation,
        ttl_sec=ttl_sec,
        idempotency_key=idempotency_key,
    ) as lease:
        yield lease


@contextmanager
def lifecycle_global_mutation(
    runtime: Any,
    operation: str,
    *,
    enabled: bool = True,
    ttl_sec: int = 900,
    idempotency_key: str | None = None,
    scope: str = "registry",
) -> Iterator[LifecycleLease | None]:
    """Serialize global registry/backend mutations across active repos."""

    if not enabled:
        with nullcontext(None) as lease:
            yield lease
        return
    registry = getattr(runtime, "registry", None)
    if registry is None:
        with nullcontext(None) as lease:
            yield lease
        return
    repo_root = str(getattr(registry, "db_path", "repo_semantic_registry"))
    with lifecycle_mutation_for_identity(
        registry=registry,
        repo_root=repo_root,
        index_profile=f"global:{scope}",
        backend_id="global",
        operation=operation,
        ttl_sec=ttl_sec,
        idempotency_key=idempotency_key,
    ) as lease:
        yield lease
