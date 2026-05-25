"""Repo registry public contracts."""

from __future__ import annotations

from pydantic import BaseModel, Field

from services.repo_semantic.contracts.common import RepoStatus


class RepoRegistryEntry(BaseModel):
    """Persistent registry entry для конкретного repo."""

    repo_root: str
    repo_key: str
    display_name: str
    status: RepoStatus
    active: bool
    index_profile: str
    include_globs: list[str] = Field(default_factory=list)
    doc_prefixes: list[str] = Field(default_factory=list)
    exclude_globs: list[str] = Field(default_factory=list)
    last_full_build_ts: str | None = None
    last_incremental_update_ts: str | None = None
    indexed_branch: str | None = None
    indexed_commit_hash: str | None = None
    last_error: str | None = None
    watch_enabled: bool
    watch_running: bool
    code_points_count: int
    docs_points_count: int
    created_at: str
    updated_at: str
