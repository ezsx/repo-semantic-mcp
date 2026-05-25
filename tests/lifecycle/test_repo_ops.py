from __future__ import annotations

from types import SimpleNamespace
import unittest

from services.repo_semantic.lifecycle.repo_ops import (
    activate_repo_payload,
    current_runtime_repo_state,
    register_repo_payload,
    runtime_switch_required_payload,
)
from services.repo_semantic.models import RepoRegistryEntry


def _entry(
    repo_root: str,
    *,
    status: str = "registered",
    active: bool = False,
    include_globs: list[str] | None = None,
    doc_prefixes: list[str] | None = None,
) -> RepoRegistryEntry:
    return RepoRegistryEntry(
        repo_root=repo_root,
        repo_key="repo-key",
        display_name="repo",
        status=status,
        active=active,
        index_profile="cpu_e5",
        include_globs=include_globs or ["src/**"],
        doc_prefixes=doc_prefixes or ["docs/"],
        exclude_globs=[".git/**"],
        watch_enabled=False,
        watch_running=False,
        code_points_count=0,
        docs_points_count=0,
        created_at="now",
        updated_at="now",
    )


class FakeRegistry:
    def __init__(self, entries: dict[str, RepoRegistryEntry] | None = None) -> None:
        self.entries = entries or {}
        self.upsert_kwargs: dict[str, object] | None = None

    def get_repo(self, repo_root: str) -> RepoRegistryEntry | None:
        return self.entries.get(repo_root)

    def upsert_repo(self, repo_root: str, **kwargs: object) -> RepoRegistryEntry:
        self.upsert_kwargs = {"repo_root": repo_root, **kwargs}
        entry = _entry(
            repo_root,
            status=str(kwargs["status"]),
            active=bool(kwargs["active"]),
            include_globs=list(kwargs["include_globs"]),
            doc_prefixes=list(kwargs["doc_prefixes"]),
        )
        self.entries[repo_root] = entry
        return entry


class RepoOpsTests(unittest.TestCase):
    def test_current_runtime_repo_state_uses_index_counts(self) -> None:
        runtime = SimpleNamespace(
            indexer=SimpleNamespace(_store=SimpleNamespace(count=lambda scope: 1 if scope == "code" else 0)),
        )

        self.assertEqual(current_runtime_repo_state(runtime), "indexed")

    def test_runtime_switch_payload_is_explicit(self) -> None:
        runtime = SimpleNamespace(
            registry=FakeRegistry({"C:/other": _entry("C:/other", status="indexed")}),
            indexer=SimpleNamespace(_settings=SimpleNamespace(logical_repo_identity="C:/runtime")),
        )

        payload = runtime_switch_required_payload(runtime, "C:/other")

        self.assertTrue(payload["requires_runtime_switch"])
        self.assertEqual(payload["runtime_repo_root"], "C:/runtime")
        self.assertEqual(payload["status"]["repo_root"], "C:/other")

    def test_register_external_repo_uses_auto_globs_and_switch_hint(self) -> None:
        registry = FakeRegistry()
        runtime = SimpleNamespace(
            registry=registry,
            watcher=None,
            indexer=SimpleNamespace(
                _settings=SimpleNamespace(
                    logical_repo_identity="C:/runtime",
                    SEMANTIC_MCP_EXCLUDE_GLOBS=[".git/**"],
                )
            ),
            search_service=SimpleNamespace(index_status=lambda: SimpleNamespace(index_profile="cpu_e5")),
        )

        payload = register_repo_payload(
            runtime,
            target_repo_root="Z:/missing",
            include_globs=None,
            doc_prefixes=None,
            activate=True,
        )

        self.assertEqual(registry.upsert_kwargs["include_globs"], ["auto"])
        self.assertEqual(registry.upsert_kwargs["doc_prefixes"], [])
        self.assertTrue(payload["requires_runtime_switch"])

    def test_activate_current_repo_applies_config_and_syncs_registry_state(self) -> None:
        entry = _entry("C:/runtime", status="registered")
        registry = FakeRegistry({"C:/runtime": entry})
        sync_calls: list[dict[str, object]] = []
        applied: dict[str, object] = {}
        runtime = SimpleNamespace(
            registry=registry,
            indexer=SimpleNamespace(
                _settings=SimpleNamespace(
                    logical_repo_identity="C:/runtime",
                    apply_repo_registry_config=lambda **kwargs: applied.update(kwargs),
                ),
                _store=SimpleNamespace(count=lambda _scope: 1),
            ),
            search_service=SimpleNamespace(
                sync_registry_state=lambda **kwargs: sync_calls.append(kwargs),
                index_status=lambda: SimpleNamespace(
                    active_repo_root="C:/runtime",
                    model_dump=lambda: {"active": True},
                ),
            ),
        )

        payload = activate_repo_payload(runtime, requested_repo_root="C:/runtime")

        self.assertTrue(payload["active_switch_applied"])
        self.assertEqual(applied, {"include_globs": ["src/**"], "doc_prefixes": ["docs/"]})
        self.assertEqual(sync_calls[0]["status"], "indexed")


if __name__ == "__main__":
    unittest.main()
