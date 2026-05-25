"""Agent-facing readiness and safe recovery helper for repo-semantic-search.

The helper is intentionally conservative:

- it never builds or rebuilds the vector index;
- it never builds or rebuilds graph artifacts;
- it never switches repo, profile, runtime, backend, or collections;
- in safe-recover mode it may start the watcher for the already active repo when
  status says this is required and not blocked.

It exists for Codex/agent sessions where the direct MCP tool may be absent from
the active toolset. Readiness is checked through the host HTTP fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

try:
    from scripts.agents import repo_semantic_call_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    import repo_semantic_call_tool  # type: ignore[no-redef]


DEFAULT_URL = repo_semantic_call_tool.DEFAULT_URL
DEFAULT_CONTAINER_NAME = "repo-semantic-mcp"
READY_STATES_OK = {"ready", "usable_degraded", "search_only"}
PERMISSION_ACTION_CODES = {
    "activate_repo",
    "active_repo_mismatch",
    "backend_switch_required",
    "build_graph",
    "build_index",
    "build_index_required",
    "embedding_contract_mismatch",
    "explicit_rebuild_required",
    "rebuild_graph",
    "rebuild_index",
    "runtime_repo_mismatch",
    "runtime_switch_required",
    "switch_backend",
    "switch_runtime",
}
_HTTP_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _base_url_from_mcp_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme and parsed.netloc:
        return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
    normalized = url.rstrip("/")
    if normalized.endswith("/mcp"):
        return normalized[: -len("/mcp")]
    return normalized


def _normalize_repo_path(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.replace("\\", "/").rstrip("/")
    lowered = normalized.lower()
    if lowered.startswith("/mnt/") and len(normalized) >= 7 and normalized[6] == "/":
        drive = normalized[5]
        if drive.isalpha():
            normalized = f"{drive.upper()}:{normalized[6:]}"
    if len(normalized) >= 2 and normalized[1] == ":":
        normalized = normalized[0].upper() + normalized[1:]
    return normalized.lower()


def _safe_recovery_idempotency_key(
    *,
    action: str,
    repo: str | None,
    profile: str | None,
    url: str,
    extra: dict[str, Any] | None = None,
) -> str:
    """Return a stable non-secret idempotency key for safe recovery actions."""

    fingerprint_input = json.dumps(
        {
            "action": action,
            "repo": _normalize_repo_path(repo) or "",
            "profile": profile or "",
            "base_url": _base_url_from_mcp_url(url),
            "extra": extra or {},
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    fingerprint = hashlib.sha256(fingerprint_input.encode("utf-8")).hexdigest()[:20]
    return f"repo-semantic-safe-recover:{action}:v1:{fingerprint}"


def _http_get_json(base_url: str, path: str, timeout_sec: int) -> tuple[bool, Any, str | None]:
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    try:
        with _HTTP_OPENER.open(url, timeout=timeout_sec) as response:
            body = response.read().decode("utf-8", errors="replace")
        return True, json.loads(body) if body.strip() else {}, None
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(body) if body.strip() else {}
        except json.JSONDecodeError:
            payload = {"body": body}
        return False, payload, f"http_{exc.code}"
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return False, None, str(exc)


def _call_tool_payload(
    tool_name: str,
    *,
    arguments: dict[str, Any] | None,
    url: str,
    timeout_sec: int,
) -> tuple[bool, Any, str | None]:
    try:
        envelope = repo_semantic_call_tool.call_tool_result(
            tool_name,
            arguments=arguments or {},
            url=url,
            timeout_sec=timeout_sec,
        )
    except Exception as exc:  # noqa: BLE001
        return False, None, str(exc)
    if not envelope.get("ok"):
        error = envelope.get("error") or envelope
        return False, envelope, json.dumps(error, ensure_ascii=False)
    return True, envelope.get("result"), None


def _start_existing_backend_container(container_name: str, timeout_sec: int) -> tuple[bool, str | None]:
    """Start an existing backend container without compose/build/switch behavior."""

    try:
        completed = subprocess.run(
            ["docker", "start", container_name],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip() or f"docker start exited {completed.returncode}"
        return False, detail
    return True, None


def _wait_for_http_readiness(
    base_url: str,
    *,
    timeout_sec: int,
    poll_interval_sec: float = 2.0,
) -> dict[str, str]:
    """Poll healthz/readyz after a container start until ready or timeout."""

    deadline = time.monotonic() + max(1, timeout_sec)
    statuses: dict[str, str] = {}
    while True:
        health_ok, _, health_error = _http_get_json(base_url, "healthz", timeout_sec=5)
        ready_ok, ready_payload, ready_error = _http_get_json(base_url, "readyz", timeout_sec=5)
        statuses["healthz"] = "ok" if health_ok else f"error:{health_error}"
        if ready_ok:
            statuses["readyz"] = "ok"
        elif isinstance(ready_payload, dict):
            phase = ready_payload.get("phase") or ready_payload.get("bootstrap_phase")
            suffix = f":{phase}" if phase else ""
            statuses["readyz"] = f"error:{ready_error}{suffix}"
        else:
            statuses["readyz"] = f"error:{ready_error}"
        if health_ok and ready_ok:
            return statuses
        if time.monotonic() >= deadline:
            return statuses
        time.sleep(poll_interval_sec)


def _warning_codes(status: dict[str, Any]) -> list[str]:
    warnings = status.get("warnings")
    if not isinstance(warnings, list):
        return []
    codes: list[str] = []
    for warning in warnings:
        if isinstance(warning, dict) and isinstance(warning.get("code"), str):
            codes.append(warning["code"])
    return codes


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _status_action_codes(status: dict[str, Any]) -> list[str]:
    availability = status.get("availability") if isinstance(status.get("availability"), dict) else {}
    actions = availability.get("next_actions")
    if not isinstance(actions, list):
        return []
    codes: list[str] = []
    for action in actions:
        if isinstance(action, dict) and isinstance(action.get("code"), str):
            codes.append(action["code"])
    return codes


def _next_action(
    *,
    code: str,
    severity: str,
    title: str,
    detail: str | None = None,
    safe_auto_run: bool = False,
    requires_user_permission: bool = True,
    destructive: bool = False,
    expensive: bool = False,
    reason_codes: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "title": title,
        "detail": detail,
        "reason_codes": reason_codes or [],
        "safe_auto_run": safe_auto_run,
        "requires_user_permission": requires_user_permission,
        "destructive": destructive,
        "expensive": expensive,
    }


def _repo_match_details(status: dict[str, Any], requested_repo: str | None) -> tuple[bool, bool, bool]:
    if not requested_repo:
        return True, True, True
    requested = _normalize_repo_path(requested_repo)
    runtime_repo = _normalize_repo_path(status.get("repo_root"))
    active_repo = _normalize_repo_path(status.get("active_repo_root"))
    runtime_matches = requested == runtime_repo
    active_matches = requested == active_repo
    return runtime_matches and active_matches, runtime_matches, active_matches


def _repo_matches(status: dict[str, Any], requested_repo: str | None) -> bool:
    return _repo_match_details(status, requested_repo)[0]


def _profile_matches(status: dict[str, Any], requested_profile: str | None) -> bool:
    if not requested_profile:
        return True
    return status.get("index_profile") == requested_profile


def _status_capabilities(status: dict[str, Any], http_fallback_available: bool) -> dict[str, bool]:
    retrieval = status.get("retrieval") if isinstance(status.get("retrieval"), dict) else {}
    graph = status.get("graph") if isinstance(status.get("graph"), dict) else {}
    watcher = status.get("watcher") if isinstance(status.get("watcher"), dict) else {}
    return {
        "search": bool(status.get("search_available")),
        "dense": bool(retrieval.get("dense_available")),
        "sparse": bool(retrieval.get("sparse_available")),
        "graph": bool(graph.get("available") and graph.get("expansion_allowed")),
        "watcher": bool(watcher.get("running")),
        "http_fallback": http_fallback_available,
    }


def _readiness_from_status(
    status: dict[str, Any],
    *,
    repo_matches: bool,
    runtime_repo_matches: bool,
    active_repo_matches: bool,
    profile_matches: bool,
    require_graph: bool,
) -> tuple[str, list[dict[str, Any]]]:
    actions: list[dict[str, Any]] = []
    search_available = bool(status.get("search_available"))
    graph = status.get("graph") if isinstance(status.get("graph"), dict) else {}
    graph_allowed = bool(graph.get("available") and graph.get("expansion_allowed"))

    if not repo_matches:
        code = "runtime_repo_mismatch" if not runtime_repo_matches else "active_repo_mismatch"
        title = (
            "Runtime repo does not match requested repo"
            if code == "runtime_repo_mismatch"
            else "Active repo does not match requested repo"
        )
        actions.append(
            _next_action(
                code=code,
                severity="blocking",
                title=title,
                detail="Safe recovery must not activate or switch repos.",
                requires_user_permission=True,
                reason_codes=[code],
            )
        )
        return "needs_user_permission", actions

    if not profile_matches:
        actions.append(
            _next_action(
                code="profile_mismatch",
                severity="blocking",
                title="Index profile does not match requested profile",
                detail="Safe recovery must not switch profiles or backends.",
                requires_user_permission=True,
                reason_codes=["profile_mismatch"],
            )
        )
        return "needs_user_permission", actions

    if not search_available:
        reason_codes = _dedupe(
            _status_action_codes(status)
            + [
                code
                for code, active in (
                    ("backend_switch_required", bool(status.get("backend_switch_required"))),
                    ("runtime_switch_required", bool(status.get("runtime_switch_required"))),
                )
                if active
            ]
        ) or ["search_unavailable"]
        requires_permission = any(code in PERMISSION_ACTION_CODES for code in reason_codes)
        actions.append(
            _next_action(
                code="search_unavailable",
                severity="blocking" if requires_permission else "error",
                title="Semantic search is unavailable",
                detail=status.get("reason_if_unavailable"),
                requires_user_permission=requires_permission,
                reason_codes=reason_codes,
            )
        )
        return ("needs_user_permission" if requires_permission else "blocked"), actions

    if require_graph and not graph_allowed:
        reason_codes = list(graph.get("warning_codes") or [])
        if graph.get("state") in {None, "missing"}:
            reason_codes.append("graph_missing")
        reason_codes = _dedupe(reason_codes)
        if graph.get("update_required"):
            safe_auto_run = bool(graph.get("update_safe_auto_run"))
            blocked_reason = graph.get("update_blocked_reason")
            actions.append(
                _next_action(
                    code="update_graph",
                    severity="warning",
                    title="Graph is stale but incrementally repairable",
                    detail=(
                        "Run safe recovery so update_graph can refresh the graph without a rebuild."
                        if safe_auto_run
                        else str(blocked_reason or "Graph update requires explicit permission.")
                    ),
                    requires_user_permission=not safe_auto_run,
                    safe_auto_run=safe_auto_run,
                    destructive=False,
                    expensive=False,
                    reason_codes=reason_codes or ["graph_update_required"],
                )
            )
            return ("usable_degraded" if safe_auto_run else "needs_user_permission"), actions
        if "graph_invalidated_paths_unknown" in reason_codes:
            actions.append(
                _next_action(
                    code="graph_invalidated_paths_unknown",
                    severity="warning",
                    title="Graph invalidated paths are unknown",
                    detail=(
                        "Run safe recovery to start the watcher and reconcile path metadata; "
                        "do not rebuild graph as the first response."
                    ),
                    requires_user_permission=False,
                    expensive=False,
                    reason_codes=reason_codes,
                )
            )
            return "usable_degraded", actions
        actions.append(
            _next_action(
                code="build_graph",
                severity="warning",
                title="Graph artifact is required but unavailable",
                detail="Graph lifecycle is permission-required in v1.",
                requires_user_permission=True,
                destructive=False,
                expensive=True,
                reason_codes=reason_codes,
            )
        )
        return "needs_user_permission", actions

    warnings = _warning_codes(status)
    if status.get("index_stale") or warnings:
        return "usable_degraded", actions
    if not graph_allowed:
        return "search_only", actions
    return "ready", actions


def ensure_ready(
    *,
    repo: str | None,
    profile: str | None = None,
    url: str = DEFAULT_URL,
    timeout_sec: int = 30,
    mode: str = "diagnose",
    start_backend: bool = False,
    backend_container_name: str = DEFAULT_CONTAINER_NAME,
    start_watcher: bool = True,
    start_watcher_idempotency_key: str | None = None,
    probe_query: str | None = None,
    require_graph: bool = False,
) -> dict[str, Any]:
    """Check repo-semantic readiness and perform bounded safe recovery."""

    base_url = _base_url_from_mcp_url(url)
    result: dict[str, Any] = {
        "ok": False,
        "readiness": "unknown",
        "repo_root": None,
        "profile": None,
        "runtime_backend_id": None,
        "selected_backend_id": None,
        "repo_matches": False,
        "runtime_repo_matches": False,
        "active_repo_matches": False,
        "profile_matches": False,
        "search_available": False,
        "capabilities": {
            "search": False,
            "dense": False,
            "sparse": False,
            "graph": False,
            "watcher": False,
            "http_fallback": False,
        },
        "direct_mcp_available": None,
        "http_fallback_available": False,
        "dependency_statuses": {},
        "backend_started": False,
        "watcher_started": False,
        "startup_reconcile_ran": False,
        "graph_update_attempted": False,
        "graph_update_succeeded": False,
        "graph_update_skipped_reason": None,
        "graph_update_bounds_exceeded": [],
        "graph_available": False,
        "graph_expansion_allowed": False,
        "exact_verification_required": False,
        "freshness_state": "unknown",
        "blocking_codes": [],
        "warnings": [],
        "auto_recovery_attempted": [],
        "auto_recovery_idempotency_keys": {},
        "watcher_start_confirmed_after_transport_error": False,
        "watcher_start_in_progress": False,
        "permission_required": False,
        "permission_required_actions": [],
        "next_actions": [],
        "exit_code": 1,
    }

    for path in ("healthz", "readyz"):
        ok, payload, error = _http_get_json(base_url, path, timeout_sec)
        result["dependency_statuses"][path] = "ok" if ok else f"error:{error}"

    if (
        str(result["dependency_statuses"].get("healthz", "")).startswith("ok")
        and not str(result["dependency_statuses"].get("readyz", "")).startswith("ok")
    ):
        result["dependency_statuses"].update(
            _wait_for_http_readiness(
                base_url,
                timeout_sec=max(timeout_sec, 60),
            )
        )

    if (
        mode == "safe-recover"
        and start_backend
        and not str(result["dependency_statuses"].get("healthz", "")).startswith("ok")
    ):
        result["auto_recovery_attempted"].append("start_backend_container")
        started, start_error = _start_existing_backend_container(
            backend_container_name,
            timeout_sec=max(timeout_sec, 60),
        )
        result["backend_started"] = started
        if started:
            result["dependency_statuses"].update(
                _wait_for_http_readiness(base_url, timeout_sec=max(timeout_sec, 60))
            )
        else:
            result["warnings"].append(start_error or "docker start failed")

    ok, status, error = _call_tool_payload(
        "index_status",
        arguments={},
        url=url,
        timeout_sec=timeout_sec,
    )
    result["http_fallback_available"] = ok
    result["capabilities"]["http_fallback"] = ok
    if not ok or not isinstance(status, dict):
        result["readiness"] = "blocked"
        result["blocking_codes"] = ["http_fallback_unavailable"]
        result["warnings"] = [error or "index_status unavailable through HTTP fallback"]
        result["next_actions"] = [
            _next_action(
                code="check_backend",
                severity="blocking",
                title="repo-semantic-search HTTP fallback is unavailable",
                detail=error,
                requires_user_permission=True,
                reason_codes=["http_fallback_unavailable"],
            )
        ]
        result["permission_required"] = True
        result["permission_required_actions"] = list(result["next_actions"])
        result["exit_code"] = 1
        return result

    repo_matches, runtime_repo_matches, active_repo_matches = _repo_match_details(status, repo)
    profile_matches = _profile_matches(status, profile)
    watcher_recovery_failed = False
    watcher_recovery_error: str | None = None
    watcher_permission_actions: list[dict[str, Any]] = []
    graph_update_permission_actions: list[dict[str, Any]] = []
    probe_failed = False
    probe_error_detail: str | None = None

    def apply_status(current: dict[str, Any]) -> None:
        graph = current.get("graph") if isinstance(current.get("graph"), dict) else {}
        backend = current.get("backend") if isinstance(current.get("backend"), dict) else {}
        freshness = current.get("freshness") if isinstance(current.get("freshness"), dict) else {}
        result["repo_root"] = current.get("repo_root")
        result["profile"] = current.get("index_profile")
        result["runtime_backend_id"] = (
            backend.get("runtime_embedding_backend_id") or current.get("embedding_backend_id")
        )
        result["selected_backend_id"] = (
            backend.get("selected_embedding_backend_id") or current.get("selected_embedding_backend_id")
        )
        repo_ok, runtime_ok, active_ok = _repo_match_details(current, repo)
        result["repo_matches"] = repo_ok
        result["runtime_repo_matches"] = runtime_ok
        result["active_repo_matches"] = active_ok
        result["profile_matches"] = _profile_matches(current, profile)
        result["search_available"] = bool(current.get("search_available"))
        result["capabilities"] = _status_capabilities(current, bool(result["http_fallback_available"]))
        result["graph_available"] = bool(graph.get("available"))
        result["graph_expansion_allowed"] = bool(graph.get("expansion_allowed"))
        result["exact_verification_required"] = bool(
            current.get("index_stale")
            or freshness.get("policy", {}).get("exact_fallback_recommended")
        )
        freshness_state = freshness.get("primary_state") if isinstance(freshness, dict) else None
        result["freshness_state"] = freshness_state or ("stale" if current.get("index_stale") else "fresh")
        result["warnings"] = _warning_codes(current)

    apply_status(status)

    if (
        mode == "safe-recover"
        and start_watcher
        and repo_matches
        and profile_matches
        and status.get("search_available")
    ):
        watcher = status.get("watcher") if isinstance(status.get("watcher"), dict) else {}
        if watcher.get("start_required") and watcher.get("start_blocked_reason") is None:
            result["auto_recovery_attempted"].append("start_watcher")
            watcher_idempotency_key = start_watcher_idempotency_key or _safe_recovery_idempotency_key(
                action="start_watcher",
                repo=repo,
                profile=profile,
                url=url,
            )
            result["auto_recovery_idempotency_keys"]["start_watcher"] = watcher_idempotency_key
            watcher_arguments: dict[str, Any] = {"idempotency_key": watcher_idempotency_key}
            if repo:
                watcher_arguments["repo_root"] = repo
            watcher_ok, watcher_payload, watcher_error = _call_tool_payload(
                "start_watcher",
                arguments=watcher_arguments,
                url=url,
                timeout_sec=max(timeout_sec, 120),
            )
            if watcher_ok and isinstance(watcher_payload, dict):
                result["watcher_started"] = bool(watcher_payload.get("watch_running"))
                result["startup_reconcile_ran"] = "startup_reconcile" in watcher_payload
                embedded_status = watcher_payload.get("status")
                watcher_refresh_skipped = bool(
                    watcher_payload.get("idempotent_retry")
                    or watcher_payload.get("status_skipped_reason")
                )
                if watcher_payload.get("operation_in_progress"):
                    result["watcher_start_in_progress"] = True
                    result["warnings"].append("start_watcher idempotent retry is still in progress")
                if watcher_payload.get("permission_required"):
                    raw_actions = watcher_payload.get("recommended_actions")
                    watcher_permission_actions = [
                        action
                        for action in raw_actions
                        if isinstance(action, dict)
                    ] or [
                        _next_action(
                            code="start_watcher_permission_required",
                            severity="warning",
                            title="Watcher startup requires permission",
                            detail="Safe recovery stopped before watcher startup.",
                            requires_user_permission=True,
                            reason_codes=["start_watcher_permission_required"],
                        )
                    ]
                if isinstance(embedded_status, dict):
                    status = embedded_status
                    apply_status(status)
                elif not watcher_permission_actions and not watcher_refresh_skipped:
                    refresh_ok, refreshed_status, _ = _call_tool_payload(
                        "index_status",
                        arguments={},
                        url=url,
                        timeout_sec=timeout_sec,
                    )
                    if refresh_ok and isinstance(refreshed_status, dict):
                        status = refreshed_status
                        apply_status(status)
            else:
                health_ok, health_payload, health_error = _http_get_json(
                    base_url,
                    "healthz",
                    timeout_sec=min(timeout_sec, 10),
                )
                result["dependency_statuses"]["healthz_after_start_watcher_error"] = (
                    "ok" if health_ok else f"error:{health_error}"
                )
                if (
                    health_ok
                    and isinstance(health_payload, dict)
                    and health_payload.get("watch_running")
                ):
                    result["watcher_started"] = True
                    result["capabilities"]["watcher"] = True
                    result["watcher_start_confirmed_after_transport_error"] = True
                    result["warnings"].append(
                        "start_watcher transport failed, but healthz confirms watcher is running"
                    )
                else:
                    watcher_recovery_failed = True
                    watcher_recovery_error = watcher_error or health_error or "start_watcher failed"
                    result["warnings"].append(watcher_recovery_error)

    graph = status.get("graph") if isinstance(status.get("graph"), dict) else {}
    if (
        mode == "safe-recover"
        and repo_matches
        and profile_matches
        and status.get("search_available")
        and graph.get("update_required")
        and graph.get("update_safe_auto_run")
    ):
        result["auto_recovery_attempted"].append("update_graph")
        graph_idempotency_key = _safe_recovery_idempotency_key(
            action="update_graph",
            repo=repo,
            profile=profile,
            url=url,
            extra={
                "indexed_commit_hash": status.get("indexed_commit_hash"),
                "indexed_branch": status.get("indexed_branch"),
                "last_incremental_update_ts": status.get("last_incremental_update_ts"),
                "last_full_build_ts": status.get("last_full_build_ts"),
                "collections": status.get("collections"),
                "built_commit": graph.get("built_commit"),
                "built_at": graph.get("built_at"),
                "update_path_count": graph.get("update_path_count"),
                "update_paths_preview": graph.get("update_paths_preview"),
                "warning_codes": graph.get("warning_codes"),
            },
        )
        result["auto_recovery_idempotency_keys"]["update_graph"] = graph_idempotency_key
        graph_arguments: dict[str, Any] = {
            "paths": [],
            "idempotency_key": graph_idempotency_key,
        }
        if repo:
            graph_arguments["repo_root"] = repo
        graph_ok, graph_payload, graph_error = _call_tool_payload(
            "update_graph",
            arguments=graph_arguments,
            url=url,
            timeout_sec=max(timeout_sec, 180),
        )
        result["graph_update_attempted"] = True
        if graph_ok and isinstance(graph_payload, dict):
            result["graph_update_succeeded"] = bool(graph_payload.get("updated"))
            result["graph_update_skipped_reason"] = graph_payload.get("skip_reason")
            result["graph_update_bounds_exceeded"] = list(graph_payload.get("bounds_exceeded") or [])
            raw_actions = graph_payload.get("recommended_actions")
            graph_update_permission_actions = [
                action
                for action in raw_actions or []
                if isinstance(action, dict) and action.get("requires_user_permission")
            ]
            embedded_status = graph_payload.get("status")
            if isinstance(embedded_status, dict):
                graph_status = embedded_status
                status = {
                    **status,
                    "graph": {
                        "available": graph_status.get("available"),
                        "state": graph_status.get("state"),
                        "expansion_allowed": graph_status.get("expansion_allowed"),
                        "schema_version": graph_status.get("schema_version"),
                        "built_commit": graph_status.get("built_commit"),
                        "built_at": graph_status.get("built_at"),
                        "extractor_versions_hash": graph_status.get("extractor_versions_hash"),
                        "file_coverage_ratio": graph_status.get("file_coverage_ratio"),
                        "stale_path_count": graph_status.get("stale_path_count"),
                        "invalidated_paths_preview": graph_status.get("invalidated_paths_preview") or [],
                        "update_available": graph_status.get("update_available", False),
                        "update_required": graph_status.get("update_required", False),
                        "update_safe_auto_run": graph_status.get("update_safe_auto_run", False),
                        "update_blocked_reason": graph_status.get("update_blocked_reason"),
                        "update_path_count": graph_status.get("update_path_count"),
                        "update_paths_preview": graph_status.get("update_paths_preview") or [],
                        "rebuild_required": graph_status.get("rebuild_required", False),
                        "warning_codes": graph_status.get("warning_codes") or [],
                    },
                }
                apply_status(status)
            elif graph_payload.get("operation_in_progress"):
                result["warnings"].append("update_graph idempotent retry is still in progress")
        else:
            result["graph_update_skipped_reason"] = "update_graph_transport_failed"
            result["warnings"].append(graph_error or "update_graph failed")

    readiness, actions = _readiness_from_status(
        status,
        repo_matches=bool(result["repo_matches"]),
        runtime_repo_matches=bool(result["runtime_repo_matches"]),
        active_repo_matches=bool(result["active_repo_matches"]),
        profile_matches=bool(result["profile_matches"]),
        require_graph=require_graph,
    )
    if result["graph_update_attempted"]:
        suppress_update_graph_actions = bool(
            result["graph_update_skipped_reason"] or graph_update_permission_actions
        )
        actions = [
            action
            for action in actions
            if action.get("code") not in {"build_graph", "rebuild_graph"}
            and not (
                suppress_update_graph_actions
                and action.get("code") == "update_graph"
            )
        ]
        if result["graph_update_skipped_reason"] == "lifecycle_operation_in_progress":
            actions.append(
                _next_action(
                    code="lifecycle_operation_in_progress",
                    severity="warning",
                    title="Graph update is already in progress",
                    detail="Retry readiness after the active lifecycle operation completes.",
                    requires_user_permission=False,
                    reason_codes=["lifecycle_operation_in_progress"],
                )
            )
            if readiness == "needs_user_permission":
                readiness = "usable_degraded"
        elif result["graph_update_skipped_reason"] == "update_graph_transport_failed":
            actions.append(
                _next_action(
                    code="update_graph",
                    severity="blocking",
                    title="Graph update transport failed",
                    detail="Retry readiness after the MCP transport is healthy.",
                    requires_user_permission=False,
                    reason_codes=["update_graph_transport_failed"],
                )
            )
            readiness = "blocked"
        elif result["graph_update_skipped_reason"] and not graph_update_permission_actions:
            actions.append(
                _next_action(
                    code="update_graph",
                    severity="warning",
                    title="Graph update did not complete",
                    detail=str(result["graph_update_skipped_reason"]),
                    requires_user_permission=bool(result["graph_update_bounds_exceeded"]),
                    reason_codes=[str(result["graph_update_skipped_reason"])],
                )
            )
            if result["graph_update_bounds_exceeded"]:
                readiness = "needs_user_permission"
            elif readiness == "needs_user_permission":
                readiness = "blocked"
    if watcher_recovery_failed:
        actions.append(
            _next_action(
                code="start_watcher_failed",
                severity="blocking",
                title="Watcher safe recovery failed",
                detail=watcher_recovery_error,
                requires_user_permission=True,
                reason_codes=["start_watcher_failed"],
            )
        )
        readiness = "needs_user_permission"
    if watcher_permission_actions:
        actions.extend(watcher_permission_actions)
        readiness = "needs_user_permission"
    if graph_update_permission_actions:
        actions.extend(graph_update_permission_actions)
        readiness = "needs_user_permission"
    if result["watcher_start_in_progress"] and readiness == "ready":
        readiness = "usable_degraded"
    result["readiness"] = readiness
    result["next_actions"].extend(actions)

    if probe_query and result["search_available"]:
        graph_mode = "expand" if require_graph else "off"
        probe_ok, _, probe_error = _call_tool_payload(
            "repo_context_search",
            arguments={
                "query": probe_query,
                "top_k": 1,
                "graph_mode": graph_mode,
                "scope": "all",
            },
            url=url,
            timeout_sec=max(timeout_sec, 120),
        )
        result["dependency_statuses"]["repo_context_search_probe"] = (
            "ok" if probe_ok else f"error:{probe_error}"
        )
        if not probe_ok:
            probe_failed = True
            probe_error_detail = probe_error or "repo_context_search probe failed"

    if probe_failed:
        result["blocking_codes"].append("repo_context_search_probe_failed")
        result["next_actions"].append(
            _next_action(
                code="repo_context_search_probe_failed",
                severity="blocking",
                title="Repo context search probe failed",
                detail=probe_error_detail,
                requires_user_permission=False,
                reason_codes=["repo_context_search_probe_failed"],
            )
        )
        if readiness in READY_STATES_OK:
            readiness = "blocked"
            result["readiness"] = readiness

    result["permission_required_actions"] = [
        action for action in result["next_actions"] if action.get("requires_user_permission")
    ]
    result["permission_required"] = bool(result["permission_required_actions"])
    result["ok"] = readiness in READY_STATES_OK
    if result["ok"]:
        result["exit_code"] = 0
    elif readiness == "needs_user_permission":
        result["exit_code"] = 3
    else:
        result["exit_code"] = 1
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Expected logical repo root.")
    parser.add_argument("--profile", default=None, help="Expected index profile.")
    parser.add_argument("--url", default=DEFAULT_URL, help="HTTP MCP endpoint URL.")
    parser.add_argument(
        "--mode",
        choices=["diagnose", "safe-recover", "permissioned-lifecycle"],
        default="diagnose",
    )
    parser.add_argument(
        "--start-backend",
        action="store_true",
        help="Start the existing backend container if healthz is unavailable.",
    )
    parser.add_argument(
        "--backend-container-name",
        default=os.getenv("SEMANTIC_MCP_CONTAINER_NAME", DEFAULT_CONTAINER_NAME),
        help="Existing backend container name used by --start-backend.",
    )
    parser.add_argument("--start-watcher", dest="start_watcher", action="store_true", default=None)
    parser.add_argument("--no-start-watcher", dest="start_watcher", action="store_false")
    parser.add_argument("--start-watcher-idempotency-key", default=None)
    parser.add_argument("--diagnose-only", action="store_true", help="Alias for --mode diagnose.")
    parser.add_argument("--probe-query", default=None)
    parser.add_argument("--require-graph", action="store_true")
    parser.add_argument("--timeout-sec", type=int, default=30)
    parser.add_argument("--json", action="store_true", help="Kept for explicit agent scripts; output is always JSON.")
    ns = parser.parse_args()

    mode = "diagnose" if ns.diagnose_only else ns.mode
    start_watcher = (mode == "safe-recover" if ns.start_watcher is None else ns.start_watcher) and (
        mode == "safe-recover"
    )
    payload = ensure_ready(
        repo=ns.repo,
        profile=ns.profile,
        url=ns.url,
        timeout_sec=ns.timeout_sec,
        mode=mode,
        start_backend=ns.start_backend,
        backend_container_name=ns.backend_container_name,
        start_watcher=start_watcher,
        start_watcher_idempotency_key=ns.start_watcher_idempotency_key,
        probe_query=ns.probe_query,
        require_graph=ns.require_graph,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return int(payload.get("exit_code", 1))


if __name__ == "__main__":
    raise SystemExit(main())
