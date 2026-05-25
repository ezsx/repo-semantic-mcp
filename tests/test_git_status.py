from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from services.repo_semantic.git_status import inspect_git_worktree


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


class GitStatusTests(unittest.TestCase):
    def test_inspect_git_worktree_reports_indexable_changes(self) -> None:
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            _git(repo_root, "init")
            _git(repo_root, "config", "user.email", "tests@example.local")
            _git(repo_root, "config", "user.name", "Tests")
            tracked = repo_root / "service.py"
            tracked.write_text("print('v1')\n", encoding="utf-8")
            _git(repo_root, "add", "service.py")
            _git(repo_root, "commit", "-m", "initial")

            tracked.write_text("print('v2')\n", encoding="utf-8")
            (repo_root / "notes.md").write_text("# Notes\n", encoding="utf-8")
            (repo_root / "image.png").write_bytes(b"not-indexable")

            snapshot = inspect_git_worktree(
                repo_root,
                include_globs=["*", "**/*"],
                exclude_globs=[".git/**"],
            )

            self.assertTrue(snapshot.is_git_repo)
            self.assertIsNotNone(snapshot.head_commit)
            self.assertTrue(snapshot.worktree_dirty)
            self.assertEqual(snapshot.changed_indexable_files_count, 1)
            self.assertEqual(snapshot.untracked_indexable_files_count, 1)
            self.assertIn("service.py", snapshot.changed_indexable_files_sample)
            self.assertIn("notes.md", snapshot.untracked_indexable_files_sample)


if __name__ == "__main__":
    unittest.main()
