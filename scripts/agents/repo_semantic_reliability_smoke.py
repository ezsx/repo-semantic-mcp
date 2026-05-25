"""Safe reliability smoke harness for repo-semantic-search.

The smoke is intentionally agent-facing and conservative. It verifies that the
HTTP fallback path, readiness diagnostics, search probe, and graph status are
usable for a target repository without running rebuilds, graph builds, backend
switches, repo activation, or source mutations.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import re
from typing import Any

try:
    from scripts.agents import repo_semantic_call_tool
    from scripts.agents import repo_semantic_ensure_ready as ensure
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    import repo_semantic_call_tool  # type: ignore[no-redef]
    import repo_semantic_ensure_ready as ensure  # type: ignore[no-redef]


DEFAULT_URL = repo_semantic_call_tool.DEFAULT_URL
DEFAULT_PROBE_QUERY = "repo semantic reliability smoke index status graph watcher"
CONTRACT_VERSION = "repo_semantic_reliability_smoke.v1"
FORBIDDEN_LIFECYCLE_ACTIONS = {
    "activate_repo",
    "build_graph",
    "build_index",
    "drop_collection",
    "rebuild_graph",
    "rebuild_index",
    "switch_backend",
    "switch_runtime",
}
SAFE_RECOVERY_ACTIONS = {"start_backend_container", "start_watcher", "update_graph"}
SECRET_KEY_MARKERS = {
    "api-key",
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret",
    "token",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _redact_text(value: str) -> str:
    redacted = re.sub(
        r"(?i)(authorization\s*[:=]\s*)(bearer\s+)?[^\s,;}]+",
        r"\1<redacted>",
        value,
    )
    redacted = re.sub(
        r"(?i)((?:api[-_]?key|token|secret|password)\s*[:=]\s*)[^\s,;}]+",
        r"\1<redacted>",
        redacted,
    )
    return redacted


def _is_secret_key(key: object) -> bool:
    if not isinstance(key, str):
        return False
    normalized = key.lower().replace(" ", "_")
    return any(marker in normalized for marker in SECRET_KEY_MARKERS)


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, dict):
        return {
            key: "<redacted>" if _is_secret_key(key) else _redact(item)
            for key, item in value.items()
        }
    return value


def _scenario(
    name: str,
    *,
    ok: bool,
    code: str,
    detail: str | None = None,
    data: dict[str, Any] | None = None,
    skipped: bool = False,
) -> dict[str, Any]:
    return {
        "name": name,
        "ok": ok,
        "skipped": skipped,
        "code": code,
        "detail": _redact(detail),
        "data": _redact(data or {}),
    }


def _call_tool_result(
    tool_name: str,
    *,
    arguments: dict[str, Any] | None,
    url: str,
    timeout_sec: int,
) -> dict[str, Any]:
    return repo_semantic_call_tool.call_tool_result(
        tool_name,
        arguments=arguments or {},
        url=url,
        timeout_sec=timeout_sec,
    )


def _tool_scenario(
    name: str,
    tool_name: str,
    *,
    arguments: dict[str, Any] | None,
    url: str,
    timeout_sec: int,
) -> tuple[dict[str, Any], Any | None]:
    try:
        envelope = _call_tool_result(
            tool_name,
            arguments=arguments or {},
            url=url,
            timeout_sec=timeout_sec,
        )
    except Exception as exc:  # noqa: BLE001
        return (
            _scenario(
                name,
                ok=False,
                code="unexpected_tool_call_exception",
                detail=f"{type(exc).__name__}: {exc}",
                data={"tool_name": tool_name, "retryable": False},
            ),
            None,
        )
    if envelope.get("ok"):
        result = envelope.get("result")
        return (
            _scenario(
                name,
                ok=True,
                code="ok",
                data={"tool_name": tool_name},
            ),
            result,
        )
    return (
        _scenario(
            name,
            ok=False,
            code=str(envelope.get("error_code") or "tool_call_failed"),
            detail=json.dumps(envelope.get("error") or envelope, ensure_ascii=False),
            data={"tool_name": tool_name, "retryable": bool(envelope.get("retryable"))},
        ),
        None,
    )


def _repo_matches(status: dict[str, Any] | None, repo: str) -> bool:
    if not isinstance(status, dict):
        return False
    requested = ensure._normalize_repo_path(repo)
    return requested == ensure._normalize_repo_path(status.get("repo_root"))


def run_smoke(
    *,
    repo: str,
    profile: str | None = None,
    url: str = DEFAULT_URL,
    timeout_sec: int = 30,
    allow_start_watcher: bool = False,
    start_backend: bool = False,
    allow_update_graph: bool = False,
    require_graph: bool = False,
    probe_query: str = DEFAULT_PROBE_QUERY,
) -> dict[str, Any]:
    """Run a no-rebuild reliability smoke and return a stable JSON payload."""

    started_at = _utc_now()
    scenarios: list[dict[str, Any]] = []
    lifecycle_counters = {
        action: 0
        for action in sorted(FORBIDDEN_LIFECYCLE_ACTIONS | SAFE_RECOVERY_ACTIONS)
    }
    effective_allow_update_graph = allow_update_graph or require_graph
    mode = (
        "safe-recover"
        if allow_start_watcher or start_backend or effective_allow_update_graph
        else "diagnose"
    )
    allowed_recovery_actions: set[str] = set()
    if allow_start_watcher:
        allowed_recovery_actions.add("start_watcher")
    if start_backend:
        allowed_recovery_actions.add("start_backend_container")
    if effective_allow_update_graph:
        allowed_recovery_actions.add("update_graph")

    try:
        readiness = ensure.ensure_ready(
            repo=repo,
            profile=profile,
            url=url,
            timeout_sec=timeout_sec,
            mode=mode,
            start_backend=start_backend,
            start_watcher=allow_start_watcher,
            probe_query=None,
            require_graph=require_graph,
        )
    except Exception as exc:  # noqa: BLE001
        readiness = {
            "ok": False,
            "readiness": "blocked",
            "permission_required": False,
            "permission_required_actions": [],
            "auto_recovery_attempted": [],
            "repo_root": None,
            "search_available": False,
            "freshness_state": "unknown",
            "graph_expansion_allowed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    for action in readiness.get("auto_recovery_attempted", []):
        if action in lifecycle_counters:
            lifecycle_counters[action] += 1
    attempted_recovery_actions = [
        str(action)
        for action in readiness.get("auto_recovery_attempted", [])
    ]
    disallowed_recovery_actions = [
        action
        for action in attempted_recovery_actions
        if action not in allowed_recovery_actions
    ]
    scenarios.append(
        _scenario(
            "ensure_ready",
            ok=bool(readiness.get("ok")),
            code=str(readiness.get("readiness") or "unknown"),
            data={
                "repo_root": readiness.get("repo_root"),
                "search_available": readiness.get("search_available"),
                "freshness_state": readiness.get("freshness_state"),
                "graph_expansion_allowed": readiness.get("graph_expansion_allowed"),
                "permission_required": readiness.get("permission_required"),
                "watcher_start_in_progress": readiness.get("watcher_start_in_progress"),
                "watcher_start_confirmed_after_transport_error": readiness.get(
                    "watcher_start_confirmed_after_transport_error"
                ),
                "error": readiness.get("error"),
            },
        )
    )
    scenarios.append(
        _scenario(
            "safe_recovery_action_policy",
            ok=not disallowed_recovery_actions,
            code="ok" if not disallowed_recovery_actions else "disallowed_safe_recovery_action",
            data={
                "auto_recovery_attempted": attempted_recovery_actions,
                "allowed_recovery_actions": sorted(allowed_recovery_actions),
                "auto_recovery_idempotency_keys": readiness.get("auto_recovery_idempotency_keys", {}),
                "disallowed_actions": disallowed_recovery_actions,
            },
        )
    )
    permission_actions = readiness.get("permission_required_actions", [])
    permission_actions_present = bool(permission_actions) or not bool(readiness.get("permission_required"))
    scenarios.append(
        _scenario(
            "permission_required_actions_present",
            ok=permission_actions_present,
            code="ok" if permission_actions_present else "permission_required_without_actions",
            data={
                "permission_required": readiness.get("permission_required"),
                "permission_required_actions": permission_actions,
            },
        )
    )

    status_scenario, status_payload = _tool_scenario(
        "index_status_http_fallback",
        "index_status",
        arguments={},
        url=url,
        timeout_sec=timeout_sec,
    )
    scenarios.append(status_scenario)
    status = status_payload if isinstance(status_payload, dict) else None

    scenarios.append(
        _scenario(
            "repo_root_match",
            ok=_repo_matches(status, repo),
            code="ok" if _repo_matches(status, repo) else "repo_root_mismatch",
            data={
                "expected_repo": repo,
                "actual_repo": status.get("repo_root") if isinstance(status, dict) else None,
            },
        )
    )
    status_repo_matches = _repo_matches(status, repo)
    search_available = bool(status and status.get("search_available") and status_repo_matches)
    scenarios.append(
        _scenario(
            "search_available",
            ok=search_available,
            code="ok"
            if search_available
            else "repo_root_mismatch"
            if status and not status_repo_matches
            else "search_unavailable",
            data={
                "reason_if_unavailable": status.get("reason_if_unavailable") if status else None,
                "repo_root_match": status_repo_matches,
            },
        )
    )

    if search_available:
        probe_scenario, probe_payload = _tool_scenario(
            "repo_context_search_probe",
            "repo_context_search",
            arguments={
                "query": probe_query,
                "top_k": 1,
                "graph_mode": "expand" if require_graph else "off",
                "scope": "all",
            },
            url=url,
            timeout_sec=max(timeout_sec, 120),
        )
        result_count = len(probe_payload.get("results", [])) if isinstance(probe_payload, dict) else 0
        probe_scenario["data"]["result_count"] = result_count
        scenarios.append(probe_scenario)
    else:
        scenarios.append(
            _scenario(
                "repo_context_search_probe",
                ok=True,
                skipped=True,
                code="skipped_repo_mismatch"
                if status and not status_repo_matches
                else "skipped_search_unavailable",
            )
        )

    graph_scenario, graph_payload = _tool_scenario(
        "graph_status",
        "graph_status",
        arguments={},
        url=url,
        timeout_sec=timeout_sec,
    )
    if isinstance(graph_payload, dict):
        graph_scenario["data"].update(
            {
                "state": graph_payload.get("state"),
                "available": graph_payload.get("available"),
                "expansion_allowed": graph_payload.get("expansion_allowed"),
            }
        )
    scenarios.append(graph_scenario)

    forbidden_executed = [
        action
        for action, count in lifecycle_counters.items()
        if action in FORBIDDEN_LIFECYCLE_ACTIONS and count > 0
    ]
    scenarios.append(
        _scenario(
            "no_forbidden_lifecycle",
            ok=not forbidden_executed,
            code="ok" if not forbidden_executed else "forbidden_lifecycle_attempted",
            data={"forbidden_actions": forbidden_executed},
        )
    )

    failed = [scenario for scenario in scenarios if not scenario["ok"] and not scenario["skipped"]]
    permission_required = bool(readiness.get("permission_required"))
    exit_code = 0
    if failed:
        exit_code = 3 if permission_required else 1

    return {
        "contract_version": CONTRACT_VERSION,
        "ok": not failed,
        "repo": repo,
        "profile": profile,
        "url": _redact(url),
        "mode": mode,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "readiness": readiness.get("readiness"),
        "permission_required": permission_required,
        "permission_required_actions": _redact(readiness.get("permission_required_actions", [])),
        "auto_recovery_attempted": _redact(attempted_recovery_actions),
        "auto_recovery_idempotency_keys": _redact(
            readiness.get("auto_recovery_idempotency_keys", {})
        ),
        "lifecycle_operation_counters": lifecycle_counters,
        "forbidden_lifecycle_detected": bool(forbidden_executed),
        "scenarios": scenarios,
        "exit_code": exit_code,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Expected logical repo root.")
    parser.add_argument("--profile", default=None, help="Expected index profile.")
    parser.add_argument("--url", default=DEFAULT_URL, help="HTTP MCP endpoint URL.")
    parser.add_argument("--timeout-sec", type=int, default=30)
    parser.add_argument("--probe-query", default=DEFAULT_PROBE_QUERY)
    parser.add_argument("--require-graph", action="store_true")
    parser.add_argument(
        "--allow-update-graph",
        action="store_true",
        help=(
            "Allow safe-recover mode to call bounded update_graph. "
            "--require-graph also enables this safe action."
        ),
    )
    parser.add_argument(
        "--allow-start-watcher",
        action="store_true",
        help="Allow safe-recover mode to call start_watcher for the already active repo.",
    )
    parser.add_argument(
        "--start-backend",
        action="store_true",
        help="Allow safe-recover mode to start the existing backend container.",
    )
    parser.add_argument("--json", action="store_true", help="Kept for explicit agent scripts; output is always JSON.")
    ns = parser.parse_args()

    payload = run_smoke(
        repo=ns.repo,
        profile=ns.profile,
        url=ns.url,
        timeout_sec=ns.timeout_sec,
        allow_start_watcher=ns.allow_start_watcher,
        start_backend=ns.start_backend,
        allow_update_graph=ns.allow_update_graph,
        require_graph=ns.require_graph,
        probe_query=ns.probe_query,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return int(payload.get("exit_code", 1))


if __name__ == "__main__":
    raise SystemExit(main())
