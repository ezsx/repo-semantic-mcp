"""Git freshness status assembly helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from services.repo_semantic.contracts.registry import RepoRegistryEntry
from services.repo_semantic.contracts.status import (
    FreshnessPolicy,
    GitWorktreeStatus,
    IndexFreshnessStatus,
    PathFreshnessSummary,
    StatusWarning,
)
from services.repo_semantic.git_status import (
    tracked_indexable_changes_after,
    untracked_indexable_changes_after,
)

FRESHNESS_STATE_PRECEDENCE = [
    "unknown",
    "blocking_stale",
    "contract_stale",
    "path_stale",
    "graph_degraded",
    "sparse_stats_stale",
    "soft_stale",
    "head_stale",
    "fresh",
]
_STATE_SEVERITY = {
    "unknown": "info",
    "blocking_stale": "blocking",
    "contract_stale": "warning",
    "path_stale": "warning",
    "graph_degraded": "warning",
    "sparse_stats_stale": "warning",
    "soft_stale": "warning",
    "head_stale": "warning",
    "fresh": "none",
}
_SEVERITY_RANK = {"none": 0, "info": 1, "warning": 2, "blocking": 3}


@dataclass(frozen=True, slots=True)
class FreshnessStatusBuild:
    """Assembled git/freshness status state."""

    git_status: GitWorktreeStatus
    freshness: IndexFreshnessStatus
    warnings: list[StatusWarning]
    stale_reasons: list[str]
    indexed_branch: str | None
    indexed_commit_hash: str | None


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _primary_state(states: list[str]) -> str:
    for state in FRESHNESS_STATE_PRECEDENCE:
        if state in states:
            return state
    return states[0] if states else "fresh"


def _max_severity(*values: str) -> str:
    return max(values or ("none",), key=lambda item: _SEVERITY_RANK.get(item, 0))


def add_freshness_states(
    freshness: IndexFreshnessStatus,
    *,
    states: list[str],
    reason_codes: list[str],
    exact_fallback_recommended: bool = False,
) -> IndexFreshnessStatus:
    """Return freshness with additional typed non-blocking states."""

    additional_states = [state for state in states if state and state != "fresh"]
    if not additional_states and not reason_codes:
        return freshness

    merged_states = _dedupe(
        [state for state in freshness.states if state != "fresh"] + additional_states
    )
    if not merged_states:
        merged_states = ["fresh"]
    primary_state = _primary_state(merged_states)
    policy = freshness.policy or FreshnessPolicy()
    max_additional_severity = _max_severity(
        *[_STATE_SEVERITY.get(state, "none") for state in additional_states]
    )
    merged_policy = policy.model_copy(
        update={
            "stale_severity": _max_severity(policy.stale_severity, max_additional_severity),
            "exact_fallback_recommended": (
                policy.exact_fallback_recommended or exact_fallback_recommended
            ),
        }
    )
    return freshness.model_copy(
        update={
            "primary_state": primary_state,
            "states": merged_states,
            "reason_codes": _dedupe([*freshness.reason_codes, *reason_codes]),
            "policy": merged_policy,
        }
    )


def build_freshness_status(
    *,
    git_snapshot,
    repo_entry: RepoRegistryEntry | None,
    repo_root: Path,
    path_freshness: PathFreshnessSummary | None = None,
) -> FreshnessStatusBuild:
    """Build git status, freshness policy, and freshness warnings."""

    git_status = GitWorktreeStatus(
        is_git_repo=git_snapshot.is_git_repo,
        branch=git_snapshot.branch,
        head_commit=git_snapshot.head_commit,
        worktree_dirty=git_snapshot.worktree_dirty,
        changed_files_count=git_snapshot.changed_files_count,
        untracked_files_count=git_snapshot.untracked_files_count,
        changed_indexable_files_count=git_snapshot.changed_indexable_files_count,
        untracked_indexable_files_count=git_snapshot.untracked_indexable_files_count,
        changed_indexable_files_sample=git_snapshot.changed_indexable_files_sample,
        untracked_indexable_files_sample=git_snapshot.untracked_indexable_files_sample,
        error=git_snapshot.error,
    )
    stale_reasons: list[str] = []
    indexed_commit_hash = repo_entry.indexed_commit_hash if repo_entry else None
    indexed_branch = repo_entry.indexed_branch if repo_entry else None
    indexed_at = (
        repo_entry.last_incremental_update_ts or repo_entry.last_full_build_ts
        if repo_entry
        else None
    )
    if indexed_commit_hash and git_status.head_commit and indexed_commit_hash != git_status.head_commit:
        stale_reasons.append("index_built_for_different_head")
    if tracked_indexable_changes_after(git_snapshot, indexed_at):
        stale_reasons.append("indexable_tracked_files_changed")
    if untracked_indexable_changes_after(git_snapshot, indexed_at):
        stale_reasons.append("indexable_untracked_files_present")

    index_revision_unknown = bool(git_status.is_git_repo and not indexed_commit_hash)
    freshness_unknown = index_revision_unknown
    path_reason_codes: list[str] = []
    path_state = None
    if path_freshness is not None:
        if not path_freshness.coverage_complete:
            freshness_unknown = True
            path_state = "unknown"
            path_reason_codes.append(
                path_freshness.coverage_error_code or "path_freshness_incomplete"
            )
        path_stale_count = (
            path_freshness.missing_from_index_count
            + path_freshness.stale_indexed_paths_count
            + path_freshness.deleted_indexed_paths_count
            + path_freshness.path_error_count
        )
        if path_stale_count > 0:
            path_state = "path_stale"
            if path_freshness.missing_from_index_count:
                path_reason_codes.append("path_manifest_missing_indexed_paths")
            if path_freshness.stale_indexed_paths_count:
                path_reason_codes.append("path_manifest_stale_indexed_paths")
            if path_freshness.deleted_indexed_paths_count:
                path_reason_codes.append("path_manifest_deleted_indexed_paths")
            if path_freshness.path_error_count:
                path_reason_codes.append("path_manifest_error_paths")
            if "path_freshness_stale" not in stale_reasons:
                stale_reasons.append("path_freshness_stale")
    states: list[str] = []
    if freshness_unknown:
        states.append("unknown")
    if "index_built_for_different_head" in stale_reasons:
        states.append("head_stale")
    if any(
        reason in stale_reasons
        for reason in ("indexable_tracked_files_changed", "indexable_untracked_files_present")
    ):
        states.append("path_stale")
    if path_state == "path_stale" and "path_stale" not in states:
        states.append("path_stale")
    if (
        path_freshness is not None
        and path_freshness.coverage_complete
        and "index_built_for_different_head" in stale_reasons
        and "path_stale" not in states
    ):
        states.append("soft_stale")
    if not states:
        states.append("fresh")
    primary_state = _primary_state(states)
    reason_codes = list(stale_reasons)
    if index_revision_unknown:
        reason_codes.append("indexed_commit_hash_unknown")
    reason_codes.extend(code for code in path_reason_codes if code)
    freshness_severity = "none"
    if stale_reasons:
        freshness_severity = "warning"
    elif freshness_unknown:
        freshness_severity = "info"

    host_side_hint = None
    if stale_reasons or freshness_unknown:
        host_side_hint = (
            "Use repair-first readiness before considering rebuilds. "
            "Host helper example: py -3.12 scripts/agents/repo_semantic_ensure_ready.py "
            f"--repo '{repo_root}' --mode safe-recover --start-watcher --json. "
            "For exact identifiers, verify against the working tree with local rg before editing."
        )

    freshness = IndexFreshnessStatus(
        indexed_branch=indexed_branch,
        indexed_commit_hash=indexed_commit_hash,
        current_branch=git_status.branch,
        current_head_commit=git_status.head_commit,
        primary_state=primary_state,
        states=states,
        reason_codes=reason_codes,
        head_mismatch=bool(
            indexed_commit_hash and git_status.head_commit and indexed_commit_hash != git_status.head_commit
        ),
        indexable_worktree_changes=git_status.changed_indexable_files_count > 0,
        untracked_indexable_files=git_status.untracked_indexable_files_count > 0,
        stale=bool(stale_reasons),
        freshness_unknown=freshness_unknown,
        stale_reasons=stale_reasons,
        host_side_hint=host_side_hint,
        policy=FreshnessPolicy(
            search_allowed_when_stale=True,
            stale_severity=freshness_severity,  # type: ignore[arg-type]
            requires_rebuild=False,
            exact_fallback_recommended=bool(stale_reasons),
        ),
    )

    warnings: list[StatusWarning] = []
    if freshness_unknown:
        warnings.append(
            StatusWarning(
                code=path_reason_codes[0] if path_reason_codes else "indexed_commit_hash_unknown",
                severity="info",
                detail="Path freshness coverage is incomplete or index revision metadata is unavailable.",
            )
        )
    if stale_reasons:
        warnings.append(
            StatusWarning(
                code="freshness_stale",
                severity="warning",
                detail="Git worktree or HEAD differs from the indexed revision; verify exact matches before editing.",
            )
        )

    return FreshnessStatusBuild(
        git_status=git_status,
        freshness=freshness,
        warnings=warnings,
        stale_reasons=stale_reasons,
        indexed_branch=indexed_branch,
        indexed_commit_hash=indexed_commit_hash,
    )
