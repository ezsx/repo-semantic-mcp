from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import unittest

from services.repo_semantic.models import RepoRegistryEntry
from services.repo_semantic.status.freshness import add_freshness_states, build_freshness_status


def _entry() -> RepoRegistryEntry:
    return RepoRegistryEntry(
        repo_root="C:/repo",
        repo_key="repo",
        display_name="repo",
        status="indexed",
        active=True,
        index_profile="cpu_e5",
        include_globs=["**/*"],
        doc_prefixes=[],
        exclude_globs=[".git/**"],
        last_full_build_ts="now",
        last_incremental_update_ts="now",
        indexed_branch="main",
        indexed_commit_hash="old",
        watch_enabled=False,
        watch_running=False,
        code_points_count=1,
        docs_points_count=0,
        created_at="now",
        updated_at="now",
    )


class FreshnessStatusTests(unittest.TestCase):
    def test_build_freshness_status_reports_head_and_worktree_stale(self) -> None:
        snapshot = SimpleNamespace(
            is_git_repo=True,
            branch="main",
            head_commit="new",
            worktree_dirty=True,
            changed_files_count=1,
            untracked_files_count=1,
            changed_indexable_files_count=1,
            untracked_indexable_files_count=1,
            changed_indexable_files_sample=["src/app.py"],
            untracked_indexable_files_sample=["src/new.py"],
            error=None,
        )

        build = build_freshness_status(
            git_snapshot=snapshot,
            repo_entry=_entry(),
            repo_root=Path("C:/repo"),
        )

        self.assertTrue(build.freshness.stale)
        self.assertEqual(build.freshness.primary_state, "path_stale")
        self.assertEqual(build.freshness.states, ["head_stale", "path_stale"])
        self.assertEqual(
            build.freshness.reason_codes,
            [
                "index_built_for_different_head",
                "indexable_tracked_files_changed",
                "indexable_untracked_files_present",
            ],
        )
        self.assertEqual(
            build.stale_reasons,
            [
                "index_built_for_different_head",
                "indexable_tracked_files_changed",
                "indexable_untracked_files_present",
            ],
        )
        self.assertEqual([warning.code for warning in build.warnings], ["freshness_stale"])

    def test_dirty_files_older_than_index_update_do_not_make_index_stale(self) -> None:
        indexed_at = datetime.now(timezone.utc).isoformat()
        entry = _entry().model_copy(
            update={
                "last_full_build_ts": indexed_at,
                "last_incremental_update_ts": indexed_at,
                "indexed_commit_hash": "head",
            }
        )
        snapshot = SimpleNamespace(
            is_git_repo=True,
            branch="main",
            head_commit="head",
            worktree_dirty=True,
            changed_files_count=1,
            untracked_files_count=1,
            changed_indexable_files_count=1,
            untracked_indexable_files_count=1,
            changed_indexable_files_sample=["src/app.py"],
            untracked_indexable_files_sample=["src/new.py"],
            changed_indexable_latest_mtime=1.0,
            untracked_indexable_latest_mtime=1.0,
            changed_indexable_missing_count=0,
            untracked_indexable_missing_count=0,
            error=None,
        )

        build = build_freshness_status(
            git_snapshot=snapshot,
            repo_entry=entry,
            repo_root=Path("C:/repo"),
        )

        self.assertFalse(build.freshness.stale)
        self.assertEqual(build.freshness.primary_state, "fresh")
        self.assertEqual(build.freshness.states, ["fresh"])
        self.assertEqual(build.freshness.reason_codes, [])
        self.assertEqual(build.stale_reasons, [])

    def test_add_freshness_states_applies_taxonomy_precedence(self) -> None:
        snapshot = SimpleNamespace(
            is_git_repo=True,
            branch="main",
            head_commit="head",
            worktree_dirty=False,
            changed_files_count=0,
            untracked_files_count=0,
            changed_indexable_files_count=0,
            untracked_indexable_files_count=0,
            changed_indexable_files_sample=[],
            untracked_indexable_files_sample=[],
            error=None,
        )
        build = build_freshness_status(
            git_snapshot=snapshot,
            repo_entry=_entry().model_copy(update={"indexed_commit_hash": "head"}),
            repo_root=Path("C:/repo"),
        )

        updated = add_freshness_states(
            build.freshness,
            states=["sparse_stats_stale", "graph_degraded"],
            reason_codes=["sparse_stats_stale", "graph_stale"],
        )

        self.assertFalse(updated.stale)
        self.assertEqual(updated.primary_state, "graph_degraded")
        self.assertEqual(updated.states, ["sparse_stats_stale", "graph_degraded"])
        self.assertEqual(updated.reason_codes, ["sparse_stats_stale", "graph_stale"])
        self.assertEqual(updated.policy.stale_severity, "warning")
        self.assertFalse(updated.policy.exact_fallback_recommended)


if __name__ == "__main__":
    unittest.main()
