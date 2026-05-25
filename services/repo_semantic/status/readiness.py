"""Search readiness helpers for index status and search preflight."""

from __future__ import annotations

from services.repo_semantic.contracts.registry import RepoRegistryEntry
from services.repo_semantic.contracts.status import StatusAction


def search_unavailable_codes(
    repo_entry: RepoRegistryEntry | None,
    *,
    selected_backend_id: str | None,
    runtime_backend_id: str,
    runtime_repo_is_active: bool,
    contract_issue: str | None = None,
) -> list[str]:
    """Return stable machine-readable reasons why search is unavailable."""

    if repo_entry is None:
        return ["repo_not_registered"]
    if selected_backend_id and selected_backend_id != runtime_backend_id:
        return ["backend_switch_required"]
    if not runtime_repo_is_active:
        return ["runtime_repo_not_active"]
    if repo_entry.status == "registered":
        return ["index_missing"]
    if repo_entry.status == "stale":
        return ["lifecycle_stale_blocking"]
    if repo_entry.status == "indexing":
        return ["indexing_in_progress"]
    if repo_entry.status == "error":
        return ["index_error"]
    if repo_entry.status == "disabled":
        return ["repo_disabled"]
    if contract_issue:
        return ["embedding_contract_mismatch"]
    if repo_entry.code_points_count + repo_entry.docs_points_count <= 0:
        return ["index_empty"]
    return []


def reason_for_unavailable_codes(
    codes: list[str],
    repo_entry: RepoRegistryEntry | None,
    *,
    contract_issue: str | None = None,
) -> str | None:
    """Return the legacy prose reason for existing consumers."""

    if not codes:
        return None
    first = codes[0]
    if first == "repo_not_registered":
        return "repo is not registered"
    if first == "backend_switch_required":
        return "selected embedding backend differs from the current runtime backend"
    if first == "runtime_repo_not_active":
        return "runtime repo is not active"
    if first == "index_missing":
        return "index has not been built yet"
    if first == "lifecycle_stale_blocking":
        return "index is stale; explicit rebuild is required"
    if first == "indexing_in_progress":
        return "index build is in progress"
    if first == "index_error":
        return repo_entry.last_error if repo_entry and repo_entry.last_error else "repo is in error state"
    if first == "repo_disabled":
        return "repo is disabled"
    if first == "embedding_contract_mismatch":
        return contract_issue or "indexed vectors are incompatible with the current runtime"
    if first == "index_empty":
        return "index has no points"
    return "search is unavailable"


def status_action_for_code(code: str) -> StatusAction:
    """Map stable availability code to a next-action hint."""

    if code == "backend_switch_required":
        return StatusAction(
            code="backend_switch_required",
            severity="blocking",
            title="Selected embedding backend differs from runtime backend",
            tool_hint="get_backend_action_plan(role='embedding')",
        )
    if code == "runtime_repo_not_active":
        return StatusAction(
            code="runtime_switch_required",
            severity="blocking",
            title="Runtime repo is not the active repo",
            host_command_hint="pwsh -File scripts/agents/start_repo_semantic_for_project.ps1 -RepoPath <repo_root>",
        )
    if code == "index_missing":
        return StatusAction(
            code="build_index_required",
            severity="blocking",
            title="Index has not been built",
            tool_hint="build_index(force_rebuild=false)",
        )
    if code == "lifecycle_stale_blocking":
        return StatusAction(
            code="explicit_rebuild_required",
            severity="blocking",
            title="Index lifecycle state is stale",
            tool_hint="build_index(force_rebuild=true)",
        )
    if code == "indexing_in_progress":
        return StatusAction(
            code="wait_for_indexing",
            severity="blocking",
            title="Index build is in progress",
            tool_hint="index_status",
        )
    if code == "index_error":
        return StatusAction(
            code="inspect_index_error",
            severity="blocking",
            title="Repo is in index error state",
            tool_hint="index_status",
        )
    if code == "embedding_contract_mismatch":
        return StatusAction(
            code="explicit_rebuild_required",
            severity="blocking",
            title="Stored vectors are incompatible with current embedding runtime",
            tool_hint="build_index(force_rebuild=true)",
        )
    return StatusAction(
        code=code,
        severity="blocking",
        title="Search is unavailable",
        tool_hint="index_status",
    )
