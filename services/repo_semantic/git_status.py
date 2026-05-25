"""Small git diagnostics helpers for status payloads."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from services.repo_semantic.chunkers.factory import is_text_like, should_index_path


@dataclass(slots=True)
class GitWorktreeSnapshot:
    """Read-only snapshot of the target repository git state."""

    is_git_repo: bool
    branch: str | None = None
    head_commit: str | None = None
    worktree_dirty: bool = False
    changed_files_count: int = 0
    untracked_files_count: int = 0
    changed_indexable_files_count: int = 0
    untracked_indexable_files_count: int = 0
    changed_indexable_files_sample: list[str] = field(default_factory=list)
    untracked_indexable_files_sample: list[str] = field(default_factory=list)
    changed_indexable_latest_mtime: float | None = None
    untracked_indexable_latest_mtime: float | None = None
    changed_indexable_missing_count: int = 0
    untracked_indexable_missing_count: int = 0
    error: str | None = None


def _run_git(repo_root: Path, *args: str) -> str:
    """Run a git command without invoking a shell."""

    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout.rstrip("\r\n")


def _status_path(line: str) -> tuple[str, str] | None:
    """Return status code and normalized path from porcelain v1 output."""

    if len(line) < 4:
        return None
    code = line[:2]
    path = line[3:].strip()
    if " -> " in path:
        path = path.rsplit(" -> ", 1)[1]
    path = path.replace("\\", "/")
    return code, path


def _is_indexable_status_path(
    repo_root: Path,
    relative_path: str,
    include_globs: list[str],
    exclude_globs: list[str],
) -> bool:
    """Check whether a git status path belongs to the semantic corpus."""

    if not should_index_path(relative_path, include_globs, exclude_globs):
        return False
    path = repo_root / relative_path
    if path.exists():
        return path.is_file() and is_text_like(path)
    return True


def inspect_git_worktree(
    repo_root: Path,
    *,
    include_globs: list[str],
    exclude_globs: list[str],
    sample_limit: int = 12,
) -> GitWorktreeSnapshot:
    """Collect cheap branch/head/dirty diagnostics for a repository."""

    try:
        inside_worktree = _run_git(repo_root, "rev-parse", "--is-inside-work-tree")
    except Exception as exc:  # noqa: BLE001
        return GitWorktreeSnapshot(is_git_repo=False, error=str(exc))

    if inside_worktree.lower() != "true":
        return GitWorktreeSnapshot(is_git_repo=False)

    try:
        branch = _run_git(repo_root, "branch", "--show-current") or None
        head_commit = _run_git(repo_root, "rev-parse", "HEAD") or None
        status_output = _run_git(repo_root, "status", "--porcelain=v1", "--untracked-files=all")
    except Exception as exc:  # noqa: BLE001
        return GitWorktreeSnapshot(is_git_repo=True, error=str(exc))

    changed_indexable: list[str] = []
    untracked_indexable: list[str] = []
    changed_latest_mtime: float | None = None
    untracked_latest_mtime: float | None = None
    changed_missing_count = 0
    untracked_missing_count = 0
    changed_files_count = 0
    untracked_files_count = 0

    for line in status_output.splitlines():
        parsed = _status_path(line)
        if parsed is None:
            continue
        code, relative_path = parsed
        is_untracked = code == "??"
        if is_untracked:
            untracked_files_count += 1
        else:
            changed_files_count += 1

        if not _is_indexable_status_path(repo_root, relative_path, include_globs, exclude_globs):
            continue
        path = repo_root / relative_path
        if path.exists():
            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = None
            if mtime is None and is_untracked:
                untracked_missing_count += 1
            elif mtime is None:
                changed_missing_count += 1
            elif is_untracked:
                untracked_latest_mtime = max(untracked_latest_mtime or mtime, mtime)
            else:
                changed_latest_mtime = max(changed_latest_mtime or mtime, mtime)
        elif is_untracked:
            untracked_missing_count += 1
        else:
            changed_missing_count += 1
        if is_untracked:
            untracked_indexable.append(relative_path)
        else:
            changed_indexable.append(relative_path)

    return GitWorktreeSnapshot(
        is_git_repo=True,
        branch=branch,
        head_commit=head_commit,
        worktree_dirty=bool(status_output),
        changed_files_count=changed_files_count,
        untracked_files_count=untracked_files_count,
        changed_indexable_files_count=len(changed_indexable),
        untracked_indexable_files_count=len(untracked_indexable),
        changed_indexable_files_sample=changed_indexable[:sample_limit],
        untracked_indexable_files_sample=untracked_indexable[:sample_limit],
        changed_indexable_latest_mtime=changed_latest_mtime,
        untracked_indexable_latest_mtime=untracked_latest_mtime,
        changed_indexable_missing_count=changed_missing_count,
        untracked_indexable_missing_count=untracked_missing_count,
    )


def inspect_git_revision(repo_root: Path) -> GitWorktreeSnapshot:
    """Collect branch/head diagnostics without scanning worktree changes."""

    try:
        inside_worktree = _run_git(repo_root, "rev-parse", "--is-inside-work-tree")
    except Exception as exc:  # noqa: BLE001
        return GitWorktreeSnapshot(is_git_repo=False, error=str(exc))

    if inside_worktree.lower() != "true":
        return GitWorktreeSnapshot(is_git_repo=False)

    try:
        branch = _run_git(repo_root, "branch", "--show-current") or None
        head_commit = _run_git(repo_root, "rev-parse", "HEAD") or None
    except Exception as exc:  # noqa: BLE001
        return GitWorktreeSnapshot(is_git_repo=True, error=str(exc))

    return GitWorktreeSnapshot(
        is_git_repo=True,
        branch=branch,
        head_commit=head_commit,
    )


def _iso_timestamp(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _changes_after_index(
    *,
    count: int,
    latest_mtime: float | None,
    missing_count: int,
    indexed_at: str | None,
) -> bool:
    if count <= 0:
        return False
    indexed_ts = _iso_timestamp(indexed_at)
    if indexed_ts is None:
        return True
    if missing_count > 0 or latest_mtime is None:
        return True
    return latest_mtime > indexed_ts + 1e-6


def tracked_indexable_changes_after(snapshot: GitWorktreeSnapshot, indexed_at: str | None) -> bool:
    """Return whether tracked indexable changes are newer than an index update timestamp."""

    return _changes_after_index(
        count=snapshot.changed_indexable_files_count,
        latest_mtime=getattr(snapshot, "changed_indexable_latest_mtime", None),
        missing_count=getattr(snapshot, "changed_indexable_missing_count", 0),
        indexed_at=indexed_at,
    )


def untracked_indexable_changes_after(snapshot: GitWorktreeSnapshot, indexed_at: str | None) -> bool:
    """Return whether untracked indexable files are newer than an index update timestamp."""

    return _changes_after_index(
        count=snapshot.untracked_indexable_files_count,
        latest_mtime=getattr(snapshot, "untracked_indexable_latest_mtime", None),
        missing_count=getattr(snapshot, "untracked_indexable_missing_count", 0),
        indexed_at=indexed_at,
    )
