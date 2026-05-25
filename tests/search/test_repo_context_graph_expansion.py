from __future__ import annotations

import unittest
from types import SimpleNamespace
from tempfile import TemporaryDirectory

from services.repo_semantic.graph import GraphService
from services.repo_semantic.models import ChunkRecord, PathFreshnessSummary

from tests.helpers.chunks import make_chunk as _chunk
from tests.helpers.fakes import FakePoint
from tests.helpers.service_factory import build_search_service


class _NoInvalidatedPathManifest:
    def invalidated_paths_after(self, timestamp: str, *, limit: int = 12):
        return 0, [], True


class _InvalidatedPathManifest:
    def __init__(self, paths: list[str]) -> None:
        self._paths = paths

    def invalidated_paths_after(self, timestamp: str, *, limit: int = 12):
        preview = self._paths[:limit]
        return len(preview), preview, len(self._paths) <= limit


def _attach_graph_service(service, *, path_manifest=None) -> GraphService:
    graph_service = GraphService(
        settings=service._settings,
        chunk_store=service._store,
        registry=service._registry,
        path_manifest=path_manifest,
    )
    service._graph_read_provider = graph_service.read_provider()
    return graph_service


def _with_path_freshness(service, path_freshness: PathFreshnessSummary) -> None:
    status = service.index_status()
    service.index_status = lambda: status.model_copy(update={"path_freshness": path_freshness})


def _mark_graph_degraded(service) -> None:
    entry = service._registry.get_repo(service._settings.logical_repo_identity)
    assert entry is not None
    service._registry.upsert_repo(
        service._settings.logical_repo_identity,
        status=entry.status,
        active=entry.active,
        index_profile=entry.index_profile,
        include_globs=entry.include_globs,
        doc_prefixes=entry.doc_prefixes,
        exclude_globs=entry.exclude_globs,
        last_full_build_ts=entry.last_full_build_ts,
        last_incremental_update_ts="2026-05-06T01:00:00+00:00",
        indexed_branch=entry.indexed_branch,
        indexed_commit_hash=entry.indexed_commit_hash,
        code_points_count=entry.code_points_count,
        docs_points_count=entry.docs_points_count,
    )


class RepoContextGraphExpansionTests(unittest.TestCase):
    def _service(self, tmp: str, chunks: list[tuple[ChunkRecord, float]]):
        return build_search_service(tmp, chunks)

    def test_auto_graph_mode_is_compat_disabled_by_default(self) -> None:
        with TemporaryDirectory() as tmp:
            runtime = _chunk(
                "runtime",
                "services/api/settings.py",
                "DATABASE_URL = os.environ['DATABASE_URL']",
            )
            deploy = _chunk(
                "deploy",
                "deploy/compose.yml",
                "DATABASE_URL: postgres://db",
                language="yaml",
                chunk_type="yaml_document",
            )
            service = self._service(tmp, [(runtime, 0.9), (deploy, 0.2)])
            graph_service = _attach_graph_service(service, path_manifest=_NoInvalidatedPathManifest())
            graph_service.build_graph()

            payload = service.repo_context_search(
                "where is DATABASE_URL used",
                graph_mode="auto",
                top_k=5,
            ).model_dump()

            self.assertEqual(payload["graph_mode_requested"], "auto")
            self.assertEqual(payload["graph_mode_effective"], "off")
            self.assertFalse(payload["diagnostics"]["graph_used"])
            self.assertEqual(
                payload["diagnostics"]["graph_diagnostics"]["skipped_reason"],
                "graph_auto_compat_disabled",
            )
            self.assertFalse(any(result["evidence_paths"] for result in payload["results"]))

    def test_explicit_expand_adds_graph_origin_and_evidence_paths(self) -> None:
        with TemporaryDirectory() as tmp:
            unrelated = _chunk(
                "unrelated",
                "services/api/unrelated.py",
                "unrelated helper",
            )
            runtime = _chunk(
                "runtime",
                "services/api/settings.py",
                "DATABASE_URL = os.environ['DATABASE_URL']",
            )
            deploy = _chunk(
                "deploy",
                "deploy/compose.yml",
                "DATABASE_URL: postgres://db",
                language="yaml",
                chunk_type="yaml_document",
            )
            service = self._service(tmp, [(unrelated, 0.99), (runtime, 0.95), (deploy, 0.05)])
            graph_service = _attach_graph_service(service, path_manifest=_NoInvalidatedPathManifest())
            graph_service.build_graph()

            payload = service.repo_context_search(
                "where is DATABASE_URL used",
                graph_mode="expand",
                top_k=5,
            ).model_dump()

            graph_results = [
                result
                for result in payload["results"]
                if "graph" in result["origin_branches"]
            ]
            self.assertEqual(payload["graph_mode_effective"], "expand")
            self.assertTrue(payload["diagnostics"]["graph_used"])
            self.assertGreaterEqual(payload["diagnostics"]["graph_diagnostics"]["bound_seed_count"], 1)
            self.assertTrue(graph_results)
            self.assertTrue(any(result["evidence_paths"] for result in graph_results))
            self.assertIn("graph", graph_results[0]["branch_ranks"])
            self.assertIn("graph", graph_results[0]["branch_scores"])
            seed_ids = {
                evidence["seed_chunk_id"]
                for result in graph_results
                for evidence in result["evidence_paths"]
            }
            self.assertNotIn("unrelated", seed_ids)

    def test_explicit_expand_links_route_to_docs_and_tests(self) -> None:
        with TemporaryDirectory() as tmp:
            handler = _chunk(
                "handler",
                "services/api/checkout_flow.py",
                (
                    "from fastapi import APIRouter\n"
                    "router = APIRouter()\n\n"
                    "@router.get('/api/v1/connect')\n"
                    "def connect_endpoint():\n"
                    "    return {'ok': True}\n"
                ),
                symbol_path="connect_endpoint",
            )
            tests = _chunk(
                "tests",
                "tests/test_checkout_flow.py",
                (
                    "from services.api.checkout_flow import connect_endpoint\n\n"
                    "def test_connect_endpoint():\n"
                    "    assert connect_endpoint()\n"
                ),
                symbol_path="test_connect_endpoint",
            )
            docs = _chunk(
                "docs",
                "docs/checkout-flow.md",
                "# Checkout Flow\nCall `/api/v1/connect` from the frontend.\n",
                scope="docs",
                language="markdown",
                chunk_type="markdown_section",
                heading_path="Checkout Flow",
            )
            service = self._service(tmp, [(handler, 0.95), (tests, 0.1), (docs, 0.1)])
            graph_service = _attach_graph_service(service, path_manifest=_NoInvalidatedPathManifest())
            graph_service.build_graph()

            payload = service.repo_context_search(
                "find tests and docs for /api/v1/connect handler",
                graph_mode="expand",
                top_k=10,
            ).model_dump()

            graph_results = [
                result
                for result in payload["results"]
                if "graph" in result["origin_branches"]
            ]
            graph_paths = {result["relative_path"] for result in graph_results}
            edge_types = {
                step["edge_type"]
                for result in graph_results
                for evidence in result["evidence_paths"]
                for step in evidence["path"]
                if step["edge_type"]
            }
            route_metadata = [
                step["metadata"]
                for result in graph_results
                for evidence in result["evidence_paths"]
                for step in evidence["path"]
                if step["edge_type"] in {"route_handled_by_symbol", "doc_references_route"}
            ]
            self.assertEqual(payload["graph_mode_effective"], "expand")
            self.assertIn("tests/test_checkout_flow.py", graph_paths)
            self.assertIn("docs/checkout-flow.md", graph_paths)
            self.assertIn("route_handled_by_symbol", edge_types)
            self.assertIn("test_targets_symbol", edge_types)
            self.assertIn("doc_references_route", edge_types)
            self.assertTrue(
                any(
                    metadata.get("surface") == "/api/v1/connect"
                    and metadata.get("normalized") == "/api/v1/connect"
                    for metadata in route_metadata
                )
            )

    def test_explicit_expand_route_handler_query_does_not_pull_docs_or_tests(self) -> None:
        with TemporaryDirectory() as tmp:
            handler = _chunk(
                "handler",
                "services/api/checkout_flow.py",
                (
                    "from fastapi import APIRouter\n"
                    "router = APIRouter()\n\n"
                    "@router.get('/api/v1/connect')\n"
                    "def connect_endpoint():\n"
                    "    return {'ok': True}\n"
                ),
                symbol_path="connect_endpoint",
            )
            tests = _chunk(
                "tests",
                "tests/test_checkout_flow.py",
                (
                    "from services.api.checkout_flow import connect_endpoint\n\n"
                    "def test_connect_endpoint():\n"
                    "    assert connect_endpoint()\n"
                ),
                symbol_path="test_connect_endpoint",
            )
            docs = _chunk(
                "docs",
                "docs/checkout-flow.md",
                "# Checkout Flow\nCall `/api/v1/connect` from the frontend.\n",
                scope="docs",
                language="markdown",
                chunk_type="markdown_section",
                heading_path="Checkout Flow",
            )
            service = self._service(tmp, [(handler, 0.95), (tests, 0.1), (docs, 0.1)])
            graph_service = _attach_graph_service(service, path_manifest=_NoInvalidatedPathManifest())
            graph_service.build_graph()

            payload = service.repo_context_search(
                "where is /api/v1/connect implemented",
                graph_mode="expand",
                top_k=10,
            ).model_dump()

            graph_paths = {
                result["relative_path"]
                for result in payload["results"]
                if "graph" in result["origin_branches"]
            }
            self.assertNotIn("tests/test_checkout_flow.py", graph_paths)
            self.assertNotIn("docs/checkout-flow.md", graph_paths)

    def test_explicit_expand_docs_seed_can_link_route_to_handler(self) -> None:
        with TemporaryDirectory() as tmp:
            handler = _chunk(
                "handler",
                "services/api/checkout_flow.py",
                (
                    "from fastapi import APIRouter\n"
                    "router = APIRouter()\n\n"
                    "@router.get('/api/v1/connect')\n"
                    "def connect_endpoint():\n"
                    "    return {'ok': True}\n"
                ),
                symbol_path="connect_endpoint",
            )
            docs = _chunk(
                "docs",
                "docs/checkout-flow.md",
                "# Checkout Flow\nThe checkout flow route is `/api/v1/connect`.\n",
                scope="docs",
                language="markdown",
                chunk_type="markdown_section",
                heading_path="Checkout Flow",
            )
            service = self._service(tmp, [(docs, 0.95), (handler, 0.1)])
            graph_service = _attach_graph_service(service, path_manifest=_NoInvalidatedPathManifest())
            graph_service.build_graph()

            payload = service.repo_context_search(
                "where is the checkout flow route implemented",
                graph_mode="expand",
                top_k=10,
            ).model_dump()

            handler_results = [
                result
                for result in payload["results"]
                if result["relative_path"] == "services/api/checkout_flow.py"
            ]
            edge_types = {
                step["edge_type"]
                for result in handler_results
                for evidence in result["evidence_paths"]
                for step in evidence["path"]
                if step["edge_type"]
            }
            route_metadata = [
                step["metadata"]
                for result in handler_results
                for evidence in result["evidence_paths"]
                for step in evidence["path"]
                if step["edge_type"] == "doc_references_route"
            ]
            self.assertTrue(any("graph" in result["origin_branches"] for result in handler_results))
            self.assertIn("doc_references_route", edge_types)
            self.assertTrue(any(metadata.get("normalized") == "/api/v1/connect" for metadata in route_metadata))

    def test_explicit_expand_links_import_target_to_importer(self) -> None:
        with TemporaryDirectory() as tmp:
            target = _chunk(
                "target",
                "services/user/domain/subscriptions.py",
                "def load_subscription(user_id):\n    return user_id\n",
                symbol_path="load_subscription",
            )
            importer = _chunk(
                "importer",
                "services/api/checkout_flow.py",
                (
                    "from services.user.domain.subscriptions import load_subscription\n\n"
                    "def connect_endpoint(user_id):\n"
                    "    return load_subscription(user_id)\n"
                ),
                symbol_path="connect_endpoint",
            )
            service = self._service(tmp, [(target, 0.95), (importer, 0.1)])
            graph_service = _attach_graph_service(service, path_manifest=_NoInvalidatedPathManifest())
            graph_service.build_graph()

            payload = service.repo_context_search(
                "what imports services.user.domain.subscriptions",
                graph_mode="expand",
                top_k=10,
            ).model_dump()

            importer_results = [
                result
                for result in payload["results"]
                if result["relative_path"] == "services/api/checkout_flow.py"
            ]
            edge_types = {
                step["edge_type"]
                for result in importer_results
                for evidence in result["evidence_paths"]
                for step in evidence["path"]
                if step["edge_type"]
            }
            self.assertTrue(any("graph" in result["origin_branches"] for result in importer_results))
            self.assertIn("file_defines_module", edge_types)
            self.assertIn("module_imports_module", edge_types)

    def test_explicit_expand_missing_graph_recommends_build_graph(self) -> None:
        with TemporaryDirectory() as tmp:
            service = self._service(
                tmp,
                [(_chunk("runtime", "services/api/settings.py", "DATABASE_URL"), 0.9)],
            )
            _attach_graph_service(service)

            payload = service.repo_context_search(
                "where is DATABASE_URL used",
                graph_mode="expand",
                top_k=3,
            ).model_dump()

            self.assertEqual(payload["graph_mode_effective"], "off")
            self.assertIn("graph_missing", {warning["code"] for warning in payload["warnings"]})
            self.assertIn("build_graph", {action["code"] for action in payload["recommended_next_actions"]})

    def test_auto_missing_graph_does_not_recommend_lifecycle(self) -> None:
        with TemporaryDirectory() as tmp:
            service = self._service(
                tmp,
                [(_chunk("runtime", "services/api/settings.py", "DATABASE_URL"), 0.9)],
            )
            service._settings.SEMANTIC_MCP_REPO_CONTEXT_GRAPH_AUTO_ENABLED = True
            _attach_graph_service(service)

            payload = service.repo_context_search(
                "where is DATABASE_URL used",
                graph_mode="auto",
                top_k=3,
            ).model_dump()

            self.assertEqual(payload["graph_mode_effective"], "off")
            self.assertIn("graph_missing", {warning["code"] for warning in payload["warnings"]})
            self.assertNotIn("build_graph", {action["code"] for action in payload["recommended_next_actions"]})
            self.assertNotIn("rebuild_graph", {action["code"] for action in payload["recommended_next_actions"]})

    def test_graph_provider_error_fails_closed_and_preserves_exact_actions(self) -> None:
        class RaisingProvider:
            def graph_status(self, *, verify_source_revision: bool = True):
                raise RuntimeError("boom")

        with TemporaryDirectory() as tmp:
            service = self._service(
                tmp,
                [(_chunk("runtime", "services/api/settings.py", "DATABASE_URL"), 0.9)],
            )
            service._graph_read_provider = RaisingProvider()

            payload = service.repo_context_search(
                "where is DATABASE_URL used",
                graph_mode="expand",
                top_k=3,
            ).model_dump()

            self.assertEqual(payload["graph_mode_effective"], "off")
            self.assertIn("graph_error", {warning["code"] for warning in payload["warnings"]})
            self.assertIn("rebuild_graph", {action["code"] for action in payload["recommended_next_actions"]})
            self.assertIn("run_local_rg", {action["code"] for action in payload["recommended_next_actions"]})
            self.assertTrue(payload["results"])

    def test_empty_graph_expansion_reports_effective_off(self) -> None:
        with TemporaryDirectory() as tmp:
            service = self._service(
                tmp,
                [(_chunk("runtime", "services/api/settings.py", "DATABASE_URL"), 0.9)],
            )
            graph_service = _attach_graph_service(service, path_manifest=_NoInvalidatedPathManifest())
            graph_service.build_graph()

            payload = service.repo_context_search(
                "where is DATABASE_URL used",
                graph_mode="expand",
                top_k=3,
            ).model_dump()

            self.assertEqual(payload["graph_mode_effective"], "off")
            self.assertFalse(payload["diagnostics"]["graph_used"])
            self.assertEqual(payload["diagnostics"]["graph_diagnostics"]["skipped_reason"], "graph_no_candidates")
            self.assertIn("graph_no_candidates", {warning["code"] for warning in payload["warnings"]})

    def test_graph_filter_diagnostics_report_post_filtering(self) -> None:
        with TemporaryDirectory() as tmp:
            runtime = _chunk(
                "runtime",
                "services/api/settings.py",
                "DATABASE_URL = os.environ['DATABASE_URL']",
            )
            deploy = _chunk(
                "deploy",
                "deploy/compose.yml",
                "DATABASE_URL: postgres://db",
                language="yaml",
                chunk_type="yaml_document",
            )
            service = self._service(tmp, [(runtime, 0.95), (deploy, 0.9)])
            graph_service = _attach_graph_service(service, path_manifest=_NoInvalidatedPathManifest())
            graph_service.build_graph()

            payload = service.repo_context_search(
                "where is DATABASE_URL used",
                graph_mode="expand",
                include_paths=["services/**"],
                top_k=3,
            ).model_dump()

            diagnostics = payload["diagnostics"]["graph_diagnostics"]
            self.assertEqual(payload["graph_mode_effective"], "off")
            self.assertIn("graph_filter_post_applied", {warning["code"] for warning in payload["warnings"]})
            self.assertIn("include_paths", diagnostics["graph_filter_post_families"])
            self.assertGreater(diagnostics["graph_candidates_before_filter"], diagnostics["graph_candidates_after_filter"])

    def test_explicit_expand_stale_graph_recommends_rebuild_graph(self) -> None:
        class StaleProvider:
            def graph_status(self, *, verify_source_revision: bool = True):
                return SimpleNamespace(
                    state="stale",
                    available=True,
                    expansion_allowed=False,
                    warning_codes=[],
                )

        with TemporaryDirectory() as tmp:
            runtime = _chunk(
                "runtime",
                "services/api/settings.py",
                "DATABASE_URL = os.environ['DATABASE_URL']",
            )
            service = self._service(tmp, [(runtime, 0.9)])
            service._graph_read_provider = StaleProvider()

            payload = service.repo_context_search(
                "where is DATABASE_URL used",
                graph_mode="expand",
                top_k=3,
            ).model_dump()

            self.assertEqual(payload["graph_mode_effective"], "off")
            self.assertIn("graph_stale", {warning["code"] for warning in payload["warnings"]})
            self.assertIn("rebuild_graph", {action["code"] for action in payload["recommended_next_actions"]})

    def test_explicit_expand_unknown_graph_invalidations_do_not_recommend_rebuild_graph(self) -> None:
        class UnknownInvalidationsProvider:
            def graph_status(self, *, verify_source_revision: bool = True):
                return SimpleNamespace(
                    state="stale",
                    available=True,
                    expansion_allowed=False,
                    warning_codes=["graph_invalidated_paths_unknown"],
                )

        with TemporaryDirectory() as tmp:
            runtime = _chunk(
                "runtime",
                "services/api/settings.py",
                "DATABASE_URL = os.environ['DATABASE_URL']",
            )
            service = self._service(tmp, [(runtime, 0.9)])
            service._graph_read_provider = UnknownInvalidationsProvider()

            payload = service.repo_context_search(
                "where is DATABASE_URL used",
                graph_mode="expand",
                top_k=3,
            ).model_dump()

            self.assertEqual(payload["graph_mode_effective"], "off")
            self.assertIn("graph_invalidated_paths_unknown", {warning["code"] for warning in payload["warnings"]})
            self.assertNotIn("rebuild_graph", {action["code"] for action in payload["recommended_next_actions"]})

    def test_explicit_expand_source_mismatch_recommends_update_graph_not_rebuild(self) -> None:
        class SourceMismatchProvider:
            def graph_status(self, *, verify_source_revision: bool = True):
                return SimpleNamespace(
                    state="stale",
                    available=True,
                    expansion_allowed=False,
                    warning_codes=[
                        "graph_source_index_contract_mismatch",
                        "graph_source_revision_hash_mismatch",
                    ],
                )

        with TemporaryDirectory() as tmp:
            runtime = _chunk(
                "runtime",
                "services/api/settings.py",
                "DATABASE_URL = os.environ['DATABASE_URL']",
            )
            service = self._service(tmp, [(runtime, 0.9)])
            service._graph_read_provider = SourceMismatchProvider()

            payload = service.repo_context_search(
                "where is DATABASE_URL used",
                graph_mode="expand",
                top_k=3,
            ).model_dump()

            action_codes = {action["code"] for action in payload["recommended_next_actions"]}
            self.assertIn("update_graph", action_codes)
            self.assertNotIn("rebuild_graph", action_codes)

    def test_explicit_expand_uses_degraded_graph_when_stale_but_allowed(self) -> None:
        with TemporaryDirectory() as tmp:
            runtime = _chunk(
                "runtime",
                "services/api/settings.py",
                "DATABASE_URL = os.environ['DATABASE_URL']",
            )
            deploy = _chunk(
                "deploy",
                "deploy/compose.yml",
                "DATABASE_URL: postgres://db",
                language="yaml",
                chunk_type="yaml_document",
            )
            service = self._service(tmp, [(runtime, 0.95), (deploy, 0.05)])
            graph_service = _attach_graph_service(service, path_manifest=_NoInvalidatedPathManifest())
            graph_service.build_graph()
            _mark_graph_degraded(service)
            _with_path_freshness(
                service,
                PathFreshnessSummary(
                    coverage_complete=True,
                    manifest_available=True,
                    identity_compatible=True,
                ),
            )

            payload = service.repo_context_search(
                "where is DATABASE_URL used",
                graph_mode="expand",
                top_k=5,
            ).model_dump()

            self.assertEqual(payload["graph_mode_effective"], "expand")
            self.assertTrue(payload["diagnostics"]["graph_used"])
            graph_diagnostics = payload["diagnostics"]["graph_diagnostics"]
            self.assertEqual(graph_diagnostics["status_state"], "stale")
            self.assertTrue(graph_diagnostics["expansion_allowed"])
            self.assertTrue(graph_diagnostics["degraded"])
            self.assertEqual(graph_diagnostics["seed_path_states"]["services/api/settings.py"], "ready")
            self.assertIn(
                "graph_source_index_contract_mismatch",
                {warning["code"] for warning in payload["warnings"]},
            )

    def test_degraded_graph_skips_when_stale_seed_path_overlaps(self) -> None:
        with TemporaryDirectory() as tmp:
            runtime = _chunk(
                "runtime",
                "services/api/settings.py",
                "DATABASE_URL = os.environ['DATABASE_URL']",
            )
            deploy = _chunk(
                "deploy",
                "deploy/compose.yml",
                "DATABASE_URL: postgres://db",
                language="yaml",
                chunk_type="yaml_document",
            )
            service = self._service(tmp, [(runtime, 0.95), (deploy, 0.05)])
            graph_service = _attach_graph_service(service, path_manifest=_NoInvalidatedPathManifest())
            graph_service.build_graph()
            _mark_graph_degraded(service)
            _with_path_freshness(
                service,
                PathFreshnessSummary(
                    coverage_complete=True,
                    manifest_available=True,
                    identity_compatible=True,
                    stale_indexed_paths_count=1,
                    stale_paths_preview=["services/api/settings.py"],
                ),
            )

            payload = service.repo_context_search(
                "where is DATABASE_URL used",
                graph_mode="expand",
                top_k=5,
            ).model_dump()

            diagnostics = payload["diagnostics"]["graph_diagnostics"]
            self.assertEqual(payload["graph_mode_effective"], "off")
            self.assertFalse(payload["diagnostics"]["graph_used"])
            self.assertEqual(diagnostics["skipped_reason"], "graph_stale_seed_path_overlap")
            self.assertTrue(diagnostics["degraded"])
            self.assertEqual(diagnostics["seed_path_states"]["services/api/settings.py"], "stale")
            self.assertIn("graph_stale_seed_path_overlap", {warning["code"] for warning in payload["warnings"]})
            self.assertFalse(any(result["evidence_paths"] for result in payload["results"]))

    def test_degraded_graph_blocks_when_invalidated_paths_are_unknown(self) -> None:
        with TemporaryDirectory() as tmp:
            runtime = _chunk(
                "runtime",
                "services/api/settings.py",
                "DATABASE_URL = os.environ['DATABASE_URL']",
            )
            deploy = _chunk(
                "deploy",
                "deploy/compose.yml",
                "DATABASE_URL: postgres://db",
                language="yaml",
                chunk_type="yaml_document",
            )
            service = self._service(tmp, [(runtime, 0.95), (deploy, 0.05)])
            graph_service = _attach_graph_service(service, path_manifest=_NoInvalidatedPathManifest())
            graph_service.build_graph()
            _mark_graph_degraded(service)
            _with_path_freshness(
                service,
                PathFreshnessSummary(
                    coverage_complete=True,
                    manifest_available=True,
                    identity_compatible=True,
                    stale_indexed_paths_count=2,
                    stale_paths_preview=["services/other.py"],
                ),
            )

            payload = service.repo_context_search(
                "where is DATABASE_URL used",
                graph_mode="expand",
                top_k=5,
            ).model_dump()

            diagnostics = payload["diagnostics"]["graph_diagnostics"]
            self.assertEqual(payload["graph_mode_effective"], "off")
            self.assertEqual(diagnostics["skipped_reason"], "graph_invalidated_paths_unknown")
            self.assertTrue(diagnostics["degraded"])
            self.assertIn("graph_invalidated_paths_unknown", {warning["code"] for warning in payload["warnings"]})
            self.assertFalse(any(result["evidence_paths"] for result in payload["results"]))

    def test_degraded_graph_uses_manifest_invalidations_after_graph_build(self) -> None:
        with TemporaryDirectory() as tmp:
            runtime = _chunk(
                "runtime",
                "services/api/settings.py",
                "DATABASE_URL = os.environ['DATABASE_URL']",
            )
            deploy = _chunk(
                "deploy",
                "deploy/compose.yml",
                "DATABASE_URL: postgres://db",
                language="yaml",
                chunk_type="yaml_document",
            )
            service = self._service(tmp, [(runtime, 0.95), (deploy, 0.05)])
            graph_service = _attach_graph_service(
                service,
                path_manifest=_InvalidatedPathManifest(["services/api/settings.py"]),
            )
            graph_service.build_graph()
            _mark_graph_degraded(service)
            _with_path_freshness(
                service,
                PathFreshnessSummary(
                    coverage_complete=True,
                    manifest_available=True,
                    identity_compatible=True,
                ),
            )

            payload = service.repo_context_search(
                "where is DATABASE_URL used",
                graph_mode="expand",
                top_k=5,
            ).model_dump()

            diagnostics = payload["diagnostics"]["graph_diagnostics"]
            self.assertEqual(payload["graph_mode_effective"], "off")
            self.assertFalse(payload["diagnostics"]["graph_used"])
            self.assertEqual(diagnostics["skipped_reason"], "graph_stale_seed_path_overlap")
            self.assertEqual(diagnostics["seed_path_states"]["services/api/settings.py"], "stale")
            self.assertIn("graph_worktree_changes", diagnostics["warning_codes"])
            self.assertIn("graph_stale_seed_path_overlap", {warning["code"] for warning in payload["warnings"]})
            self.assertNotIn("rebuild_graph", {action["code"] for action in payload["recommended_next_actions"]})

    def test_degraded_graph_filters_stale_expanded_paths(self) -> None:
        with TemporaryDirectory() as tmp:
            runtime = _chunk(
                "runtime",
                "services/api/settings.py",
                "DATABASE_URL = os.environ['DATABASE_URL']",
            )
            deploy = _chunk(
                "deploy",
                "deploy/compose.yml",
                "DATABASE_URL: postgres://db",
                language="yaml",
                chunk_type="yaml_document",
            )
            filler_chunks = [
                (
                    _chunk(
                        f"filler-{index}",
                        f"services/filler/{index}.py",
                        f"def filler_{index}(): return {index}",
                    ),
                    0.94 - (index / 1000),
                )
                for index in range(55)
            ]
            service = self._service(tmp, [(runtime, 0.99), *filler_chunks, (deploy, 0.01)])
            graph_service = _attach_graph_service(service, path_manifest=_NoInvalidatedPathManifest())
            graph_service.build_graph()
            _mark_graph_degraded(service)
            _with_path_freshness(
                service,
                PathFreshnessSummary(
                    coverage_complete=True,
                    manifest_available=True,
                    identity_compatible=True,
                    stale_indexed_paths_count=1,
                    stale_paths_preview=["deploy/compose.yml"],
                ),
            )

            payload = service.repo_context_search(
                "where is DATABASE_URL used",
                route="semantic",
                graph_mode="expand",
                top_k=5,
            ).model_dump()

            diagnostics = payload["diagnostics"]["graph_diagnostics"]
            self.assertEqual(payload["graph_mode_effective"], "off")
            self.assertEqual(diagnostics["skipped_reason"], "graph_stale_expanded_path_overlap")
            self.assertEqual(diagnostics["expanded_path_states"]["deploy/compose.yml"], "stale")
            self.assertIn("graph_stale_expanded_path_overlap", {warning["code"] for warning in payload["warnings"]})
            self.assertNotIn(
                "graph",
                {
                    branch
                    for result in payload["results"]
                    if result["relative_path"] == "deploy/compose.yml"
                    for branch in result["origin_branches"]
                },
            )

    def test_explicit_expand_non_ready_statuses_fail_closed(self) -> None:
        class StatusProvider:
            def __init__(self, state: str, *, available: bool = False) -> None:
                self._state = state
                self._available = available

            def graph_status(self, *, verify_source_revision: bool = True):
                return SimpleNamespace(
                    state=self._state,
                    available=self._available,
                    expansion_allowed=False,
                )

        cases = {
            "partial": ("rebuild_graph", True),
            "building": ("retry_with_graph_expand", False),
            "incompatible": ("rebuild_graph", False),
            "error": ("rebuild_graph", False),
        }
        for state, (expected_action, available) in cases.items():
            with self.subTest(state=state), TemporaryDirectory() as tmp:
                service = self._service(
                    tmp,
                    [(_chunk("runtime", "services/api/settings.py", "DATABASE_URL"), 0.9)],
                )
                service._graph_read_provider = StatusProvider(state, available=available)

                payload = service.repo_context_search(
                    "where is DATABASE_URL used",
                    graph_mode="expand",
                    top_k=3,
                ).model_dump()

                self.assertEqual(payload["graph_mode_effective"], "off")
                self.assertTrue(payload["results"])
                self.assertFalse(any(result["evidence_paths"] for result in payload["results"]))
                self.assertNotIn("graph", {branch for result in payload["results"] for branch in result["origin_branches"]})
                self.assertIn(f"graph_{state}", {warning["code"] for warning in payload["warnings"]})
                self.assertIn(expected_action, {action["code"] for action in payload["recommended_next_actions"]})

    def test_ready_graph_status_warnings_are_reported_in_search_response(self) -> None:
        with TemporaryDirectory() as tmp:
            service = self._service(
                tmp,
                [(_chunk("broken", "services/broken.py", "def broken(:\n    pass\n"), 0.9)],
            )
            graph_service = _attach_graph_service(service)
            graph_service.build_graph()

            payload = service.repo_context_search(
                "broken",
                graph_mode="expand",
                top_k=3,
            ).model_dump()

            warning_codes = {warning["code"] for warning in payload["warnings"]}
            diagnostics_codes = set(payload["diagnostics"]["graph_diagnostics"]["warning_codes"])
            self.assertIn("graph_extractor_python_ast_parse_failed", warning_codes)
            self.assertIn("graph_extractor_python_ast_parse_failed", diagnostics_codes)


if __name__ == "__main__":
    unittest.main()
