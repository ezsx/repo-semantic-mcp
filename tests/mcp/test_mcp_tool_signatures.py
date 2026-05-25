from __future__ import annotations

import asyncio
import importlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from services.repo_semantic import models
from tests.helpers.assertions import assert_signature
from tests.helpers.fakes import install_rank_bm25_stub
from tests.helpers.mcp import install_mcp_stub


SEARCH_PARAMS = [
    "query",
    "top_k",
    "scope",
    "path_prefix",
    "chunk_types",
    "domain_tags",
    "include_paths",
    "exclude_paths",
    "file_extensions",
    "languages",
    "max_results_per_file",
    "snippet_mode",
    "include_explanations",
]

SCOPED_SEARCH_PARAMS = [
    "query",
    "top_k",
    "path_prefix",
    "chunk_types",
    "domain_tags",
    "include_paths",
    "exclude_paths",
    "file_extensions",
    "languages",
    "max_results_per_file",
    "snippet_mode",
    "include_explanations",
]

SEARCH_DEFAULTS = {
    "top_k": 10,
    "scope": "all",
    "path_prefix": None,
    "chunk_types": None,
    "domain_tags": None,
    "include_paths": None,
    "exclude_paths": None,
    "file_extensions": None,
    "languages": None,
    "max_results_per_file": None,
    "snippet_mode": "chunk_start",
    "include_explanations": True,
}

SCOPED_SEARCH_DEFAULTS = {
    key: value
    for key, value in SEARCH_DEFAULTS.items()
    if key != "scope"
}


class McpToolSignatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        install_rank_bm25_stub()
        handle = install_mcp_stub()
        try:
            cls.mcp_server = importlib.import_module("services.repo_semantic.mcp_server")
        finally:
            handle.restore()

    def test_public_mcp_registration_inventory_is_stable(self) -> None:
        self.assertEqual(
            [item["name"] for item in self.mcp_server.mcp.tool_registrations],
            [
                "semantic_search",
                "semantic_search_code",
                "semantic_search_docs",
                "hybrid_search",
                "hybrid_search_code",
                "hybrid_search_docs",
                "search_v2",
                "repo_context_search",
                "find_similar_chunk",
                "read_chunk",
                "index_status",
                "graph_status",
                "update_graph",
                "build_graph",
                "rebuild_graph",
                "rebuild_index",
                "reindex_paths",
                "update_include_globs",
                "list_repos",
                "list_backends",
                "get_backend_status",
                "set_role_backend",
                "get_backend_action_plan",
                "ensure_backend",
                "list_legacy_repo_entries",
                "prune_legacy_repo_entries",
                "get_repo_status",
                "register_repo",
                "activate_repo",
                "build_index",
                "start_watcher",
                "stop_watcher",
            ],
        )
        self.assertEqual(
            [item["args"][0] for item in self.mcp_server.mcp.resource_registrations],
            ["index://status", "index://collections", "index://config"],
        )
        self.assertEqual(
            [(item["args"][0], tuple(item["kwargs"]["methods"])) for item in self.mcp_server.mcp.custom_route_registrations],
            [("/healthz", ("GET",)), ("/readyz", ("GET",)), ("/statusz", ("GET",))],
        )

    def test_readyz_uses_lightweight_bootstrap_payload(self) -> None:
        with patch.object(
            self.mcp_server,
            "_bootstrap_payload",
            return_value={"ready": True, "phase": "ready"},
        ) as bootstrap_payload:
            response = asyncio.run(self.mcp_server.route_readyz(None))

        self.assertEqual(response.status_code, 200)
        bootstrap_payload.assert_called_once_with(include_index_status=False)

    def test_legacy_search_tool_signatures_are_stable(self) -> None:
        assert_signature(self, self.mcp_server.semantic_search, SEARCH_PARAMS, SEARCH_DEFAULTS)
        assert_signature(self, self.mcp_server.hybrid_search, SEARCH_PARAMS, SEARCH_DEFAULTS)
        for name in [
            "semantic_search_code",
            "semantic_search_docs",
            "hybrid_search_code",
            "hybrid_search_docs",
        ]:
            assert_signature(self, getattr(self.mcp_server, name), SCOPED_SEARCH_PARAMS, SCOPED_SEARCH_DEFAULTS)

    def test_search_v2_signature_is_stable(self) -> None:
        assert_signature(
            self,
            self.mcp_server.search_v2,
            [
                "query",
                "mode",
                "top_k",
                "scope",
                "path_prefix",
                "chunk_types",
                "domain_tags",
                "include_paths",
                "exclude_paths",
                "file_extensions",
                "languages",
                "max_results_per_file",
                "snippet_mode",
                "include_explanations",
            ],
            {
                "mode": "hybrid",
                "top_k": 10,
                "scope": "all",
                "path_prefix": None,
                "chunk_types": None,
                "domain_tags": None,
                "include_paths": None,
                "exclude_paths": None,
                "file_extensions": None,
                "languages": None,
                "max_results_per_file": None,
                "snippet_mode": "chunk_start",
                "include_explanations": True,
            },
        )

    def test_repo_context_search_signature_is_stable(self) -> None:
        assert_signature(
            self,
            self.mcp_server.repo_context_search,
            [
                "query",
                "subqueries",
                "route",
                "graph_mode",
                "scope",
                "path_prefix",
                "include_paths",
                "exclude_paths",
                "file_extensions",
                "languages",
                "chunk_types",
                "domain_tags",
                "top_k",
                "max_results_per_file",
                "snippet_mode",
                "include_diagnostics",
                "include_explanations",
            ],
            {
                "subqueries": None,
                "route": "auto",
                "graph_mode": "auto",
                "scope": "all",
                "path_prefix": None,
                "include_paths": None,
                "exclude_paths": None,
                "file_extensions": None,
                "languages": None,
                "chunk_types": None,
                "domain_tags": None,
                "top_k": 20,
                "max_results_per_file": 3,
                "snippet_mode": "query_centered",
                "include_diagnostics": True,
                "include_explanations": True,
            },
        )

    def test_status_and_lifecycle_tool_signatures_are_stable(self) -> None:
        assert_signature(self, self.mcp_server.find_similar_chunk, ["scope", "chunk_id", "top_k"], {"top_k": 10})
        assert_signature(self, self.mcp_server.read_chunk, ["scope", "chunk_id"], {})
        assert_signature(self, self.mcp_server.index_status, [], {})
        assert_signature(self, self.mcp_server.graph_status, [], {})
        assert_signature(
            self,
            self.mcp_server.update_graph,
            ["paths", "repo_root", "idempotency_key", "allow_large_update"],
            {"paths": None, "repo_root": None, "idempotency_key": None, "allow_large_update": False},
        )
        assert_signature(
            self,
            self.mcp_server.build_graph,
            ["repo_root", "force_rebuild"],
            {"repo_root": None, "force_rebuild": False},
        )
        assert_signature(self, self.mcp_server.rebuild_graph, ["repo_root"], {"repo_root": None})
        assert_signature(self, self.mcp_server.rebuild_index, [], {})
        assert_signature(self, self.mcp_server.reindex_paths, ["paths"], {})
        assert_signature(self, self.mcp_server.update_include_globs, ["globs"], {})
        assert_signature(
            self,
            self.mcp_server.build_index,
            ["repo_root", "force_rebuild"],
            {"repo_root": None, "force_rebuild": False},
        )
        assert_signature(
            self,
            self.mcp_server.start_watcher,
            ["repo_root", "idempotency_key"],
            {"repo_root": None, "idempotency_key": None},
        )
        assert_signature(self, self.mcp_server.stop_watcher, [], {})

    def test_build_graph_skip_uses_graph_build_result_shape(self) -> None:
        class FakeSearchService:
            def index_status(self):
                return SimpleNamespace(search_available=False)

        class FakeGraphService:
            graph_path = "C:/tmp/graph.sqlite3"

            def graph_status(self, *, verify_source_revision: bool = True):
                return models.GraphStatusResult(
                    repo_root="C:/repo",
                    profile="test-profile",
                    graph_path=self.graph_path,
                    available=False,
                    state="missing",
                    expansion_allowed=False,
                    warning_codes=["graph_missing"],
                )

        self.mcp_server.configure_runtime(
            self.mcp_server.AppRuntime(
                search_service=FakeSearchService(),
                indexer=SimpleNamespace(
                    _settings=SimpleNamespace(logical_repo_identity="C:/repo")
                ),
                watcher=None,
                registry=SimpleNamespace(status_summary=lambda: {}),
                embedding_provider=SimpleNamespace(),
                graph_service=FakeGraphService(),
            )
        )
        self.mcp_server.mark_bootstrap_ready()

        payload = self.mcp_server.build_graph()

        self.assertEqual(payload["contract_version"], "graph_build.v1")
        self.assertFalse(payload["built"])
        self.assertFalse(payload["rebuilt"])
        self.assertEqual(payload["state"], "missing")
        self.assertIn("graph_build_index_unavailable", payload["warning_codes"])

    def test_repo_backend_and_registry_tool_signatures_are_stable(self) -> None:
        assert_signature(self, self.mcp_server.list_repos, [], {})
        assert_signature(self, self.mcp_server.list_backends, ["role"], {"role": None})
        assert_signature(self, self.mcp_server.get_backend_status, ["role", "backend_id"], {"role": None, "backend_id": None})
        assert_signature(self, self.mcp_server.set_role_backend, ["role", "backend_id"], {})
        assert_signature(
            self,
            self.mcp_server.get_backend_action_plan,
            ["role", "backend_id", "repo_root"],
            {"role": None, "backend_id": None, "repo_root": None},
        )
        assert_signature(
            self,
            self.mcp_server.ensure_backend,
            ["role", "backend_id", "repo_root"],
            {"role": None, "backend_id": None, "repo_root": None},
        )
        assert_signature(self, self.mcp_server.list_legacy_repo_entries, [], {})
        assert_signature(self, self.mcp_server.prune_legacy_repo_entries, ["force_remove_active"], {"force_remove_active": False})
        assert_signature(self, self.mcp_server.get_repo_status, ["repo_root"], {"repo_root": None})
        assert_signature(
            self,
            self.mcp_server.register_repo,
            ["repo_root", "include_globs", "doc_prefixes", "activate"],
            {"include_globs": None, "doc_prefixes": None, "activate": False},
        )
        assert_signature(self, self.mcp_server.activate_repo, ["repo_root"], {})

    def test_search_envelope_tools_return_plain_dict_payloads(self) -> None:
        class Dumpable:
            def __init__(self, payload: dict[str, object]) -> None:
                self._payload = payload

            def model_dump(self):
                return dict(self._payload)

        class FakeSearchService:
            def search_v2(self, **kwargs):
                return Dumpable({"contract_version": "search.v2", "query": kwargs["query"]})

            def repo_context_search(self, **kwargs):
                return Dumpable(
                    {
                        "contract_version": "repo_context_search.v1",
                        "query": kwargs["query"],
                    }
                )

        self.mcp_server.configure_runtime(
            self.mcp_server.AppRuntime(
                search_service=FakeSearchService(),
                indexer=SimpleNamespace(),
                watcher=None,
                registry=SimpleNamespace(status_summary=lambda: {}),
                embedding_provider=SimpleNamespace(),
            )
        )
        self.mcp_server.mark_bootstrap_ready()

        self.assertEqual(self.mcp_server.search_v2("needle"), {"contract_version": "search.v2", "query": "needle"})
        self.assertEqual(
            self.mcp_server.repo_context_search("needle"),
            {"contract_version": "repo_context_search.v1", "query": "needle"},
        )

    def test_legacy_search_payload_shape_convention_is_stable(self) -> None:
        rows = [{"chunk_id": "a"}]
        self.assertIs(self.mcp_server._search_payload(rows), rows)
        self.assertEqual(self.mcp_server._search_payload([]), "[]")


if __name__ == "__main__":
    unittest.main()
