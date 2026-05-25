from __future__ import annotations

import unittest
from typing import get_args

from services.repo_semantic import contracts
from services.repo_semantic import models
from services.repo_semantic.contracts import backend as backend_contracts
from services.repo_semantic.contracts import common as common_contracts
from services.repo_semantic.contracts import graph as graph_contracts
from services.repo_semantic.contracts import registry as registry_contracts
from services.repo_semantic.contracts import repo_context as repo_context_contracts
from services.repo_semantic.contracts import search as search_contracts
from services.repo_semantic.contracts import status as status_contracts


FOCUSED_CONTRACT_EXPORTS = {
    common_contracts: [
        "BackendRole",
        "ChunkRecord",
        "ChunkScope",
        "CompatibilityState",
        "EvidenceReason",
        "FreshnessSeverity",
        "GraphEdgeType",
        "GraphMode",
        "GraphNodeType",
        "GraphState",
        "MatchType",
        "OriginBranch",
        "RepoContextRoute",
        "RepoStatus",
        "SearchMode",
        "SearchScope",
        "SnippetMode",
        "StatusSeverity",
    ],
    graph_contracts: [
        "GraphBuildResult",
        "GraphCounts",
        "GraphStatusResult",
        "GraphStatusSummary",
        "GraphUpdateResult",
    ],
    search_contracts: [
        "BranchDiagnostics",
        "ExactAnchorAnalysis",
        "ExactAnchorCandidate",
        "LineMatch",
        "PayloadCapabilities",
        "SearchDiagnostics",
        "SearchFilters",
        "SearchResponse",
        "SearchResult",
        "SearchWarning",
    ],
    repo_context_contracts: [
        "EvidencePath",
        "EvidencePathStep",
        "ExactAnchorUsage",
        "FileContextGroup",
        "FileDistributionEntry",
        "GraphBranchDiagnostics",
        "QueryUsage",
        "RecommendedAction",
        "RepoContextDiagnostics",
        "RepoContextResult",
        "RepoContextSearchResponse",
        "VerificationHint",
    ],
    status_contracts: [
        "BackendStatusSummary",
        "FreshnessPolicy",
        "GitWorktreeStatus",
        "IndexCollectionContract",
        "IndexCollectionStatus",
        "IndexContractStatus",
        "IndexCounts",
        "IndexFreshnessStatus",
        "IndexStatusResult",
        "LifecycleStatus",
        "PathFreshnessSummary",
        "PayloadIndexStatus",
        "ReadChunkResult",
        "RetrievalContractStatus",
        "RetrievalScopeStatus",
        "RuntimeStatusSummary",
        "StatusAction",
        "StatusAvailability",
        "StatusWarning",
        "WatcherStatusSummary",
    ],
    registry_contracts: [
        "RepoRegistryEntry",
    ],
    backend_contracts: [
        "BackendRegistryEntry",
        "BackendStatusResult",
    ],
}


class PublicModelContractTests(unittest.TestCase):
    def assert_field_order(self, model_class, expected: list[str]) -> None:
        self.assertEqual(list(model_class.model_fields), expected)

    def assert_default_factory(self, model_class, field_name: str) -> None:
        self.assertIsNotNone(model_class.model_fields[field_name].default_factory)

    def test_models_module_remains_public_import_facade(self) -> None:
        self.assertEqual(models.__all__, contracts.__all__)
        focused_names = [
            name
            for exported_names in FOCUSED_CONTRACT_EXPORTS.values()
            for name in exported_names
        ]
        self.assertEqual(set(focused_names), set(models.__all__))
        for module, exported_names in FOCUSED_CONTRACT_EXPORTS.items():
            for name in exported_names:
                self.assertIs(getattr(models, name), getattr(module, name), name)
                self.assertIs(getattr(contracts, name), getattr(module, name), name)

    def test_literal_alias_values_are_stable(self) -> None:
        self.assertEqual(set(get_args(models.SearchScope)), {"all", "code", "docs"})
        self.assertEqual(set(get_args(models.SearchMode)), {"semantic", "hybrid"})
        self.assertEqual(set(get_args(models.RepoContextRoute)), {"auto", "semantic", "hybrid", "exact_handoff"})
        self.assertEqual(set(get_args(models.GraphMode)), {"off", "auto", "expand"})
        self.assertEqual(
            set(get_args(models.GraphState)),
            {"missing", "ready", "partial", "stale", "incompatible", "building", "error"},
        )
        self.assertEqual(set(get_args(models.SnippetMode)), {"chunk_start", "query_centered"})
        self.assertEqual(set(get_args(models.OriginBranch)), {"dense", "sparse", "legacy_lexical", "graph"})
        self.assertEqual(
            set(get_args(models.GraphNodeType)),
            {
                "file",
                "chunk",
                "symbol",
                "doc_section",
                "config_item",
                "env_var",
                "policy_scope",
                "module",
                "route",
                "test_case",
            },
        )
        self.assertEqual(
            set(get_args(models.GraphEdgeType)),
            {
                "file_contains_chunk",
                "chunk_defines_symbol",
                "file_contains_doc_section",
                "chunk_mentions_path_or_symbol",
                "config_defines_env",
                "code_reads_env",
                "doc_references_symbol",
                "doc_references_config",
                "policy_applies_to_path",
                "file_defines_module",
                "module_imports_module",
                "file_imports_file",
                "route_defined_in_chunk",
                "route_handled_by_symbol",
                "test_targets_file",
                "test_targets_symbol",
                "doc_references_route",
                "doc_references_path",
                "doc_references_env",
            },
        )
        self.assertEqual(set(get_args(models.BackendRole)), {"embedding", "reranker", "colbert", "llm_helper"})

    def test_search_response_schema_defaults_are_stable(self) -> None:
        self.assert_field_order(
            models.SearchResponse,
            ["contract_version", "query", "mode", "results", "warnings", "diagnostics"],
        )
        fields = models.SearchResponse.model_fields
        self.assertEqual(fields["contract_version"].default, "search.v2")
        self.assertTrue(fields["query"].is_required())
        self.assertTrue(fields["mode"].is_required())
        self.assert_default_factory(models.SearchResponse, "results")
        self.assert_default_factory(models.SearchResponse, "warnings")
        self.assertIsNone(fields["diagnostics"].default)
        schema = models.SearchResponse.model_json_schema()
        self.assertEqual(schema["properties"]["contract_version"]["default"], "search.v2")
        self.assertIn("results", schema["properties"])

    def test_search_result_and_diagnostics_contracts_are_stable(self) -> None:
        self.assert_field_order(
            models.SearchResult,
            [
                "chunk_id",
                "repo_root",
                "scope",
                "relative_path",
                "language",
                "chunk_type",
                "start_line",
                "end_line",
                "line_range",
                "file_extension",
                "snippet",
                "snippet_start_line",
                "snippet_end_line",
                "symbol_path",
                "heading_path",
                "domain_tags",
                "score",
                "final_score",
                "match_type",
                "dense_score",
                "lexical_score",
                "matched_terms",
                "why_matched",
                "line_matches",
            ],
        )
        self.assertTrue(models.SearchResult.model_fields["chunk_id"].is_required())
        self.assertTrue(models.SearchResult.model_fields["score"].is_required())
        self.assert_default_factory(models.SearchResult, "domain_tags")
        self.assert_default_factory(models.SearchResult, "matched_terms")
        self.assert_default_factory(models.SearchResult, "why_matched")
        self.assert_default_factory(models.SearchResult, "line_matches")
        self.assert_field_order(
            models.SearchDiagnostics,
            [
                "candidate_limit",
                "candidates_scanned",
                "filtered_candidates",
                "final_results",
                "dense_best_effort",
                "lexical_cache_used",
                "retrieval_backend",
                "fusion_method",
                "rrf_k",
                "branch_weights",
                "branch_diagnostics",
                "sparse_available",
                "sparse_contract_compatible",
                "sparse_manifest_hash",
                "sparse_vocabulary_hash",
                "sparse_corpus_stats_hash",
                "sparse_stats_stale",
                "sparse_unavailable_codes",
                "exact_anchor_analysis",
                "filter_pushdown_supported",
                "filter_pushed_families",
                "filter_post_families",
                "filter_best_effort",
                "filter_payload_capabilities",
            ],
        )
        self.assertTrue(models.SearchDiagnostics.model_fields["final_results"].is_required())
        self.assertEqual(models.SearchDiagnostics.model_fields["dense_best_effort"].default, False)
        self.assertEqual(models.SearchDiagnostics.model_fields["lexical_cache_used"].default, False)
        self.assert_default_factory(models.SearchDiagnostics, "branch_weights")
        self.assert_default_factory(models.SearchDiagnostics, "branch_diagnostics")
        self.assert_default_factory(models.SearchDiagnostics, "sparse_unavailable_codes")
        self.assert_default_factory(models.SearchDiagnostics, "filter_pushed_families")
        self.assert_default_factory(models.SearchDiagnostics, "filter_post_families")
        self.assert_default_factory(models.SearchDiagnostics, "filter_payload_capabilities")

    def test_repo_context_response_schema_defaults_are_stable(self) -> None:
        self.assert_field_order(
            models.RepoContextSearchResponse,
            [
                "contract_version",
                "query",
                "queries_used",
                "route_requested",
                "route_used",
                "graph_mode_requested",
                "graph_mode_effective",
                "results",
                "file_groups",
                "warnings",
                "diagnostics",
                "recommended_next_actions",
            ],
        )
        fields = models.RepoContextSearchResponse.model_fields
        self.assertEqual(fields["contract_version"].default, "repo_context_search.v1")
        self.assertEqual(fields["graph_mode_effective"].default, "off")
        self.assertTrue(fields["query"].is_required())
        self.assertTrue(fields["queries_used"].is_required())
        self.assert_default_factory(models.RepoContextSearchResponse, "results")
        self.assert_default_factory(models.RepoContextSearchResponse, "file_groups")
        self.assert_default_factory(models.RepoContextSearchResponse, "warnings")
        self.assert_default_factory(models.RepoContextSearchResponse, "recommended_next_actions")
        self.assertIsNone(fields["diagnostics"].default)
        schema = models.RepoContextSearchResponse.model_json_schema()
        self.assertEqual(schema["properties"]["contract_version"]["default"], "repo_context_search.v1")
        self.assertIn("recommended_next_actions", schema["properties"])

    def test_repo_context_nested_models_are_stable(self) -> None:
        self.assert_field_order(
            models.QueryUsage,
            ["query", "role", "route_used", "weight", "result_count", "warning_codes"],
        )
        self.assertEqual(models.QueryUsage.model_fields["result_count"].default, 0)
        self.assert_default_factory(models.QueryUsage, "warning_codes")
        self.assert_field_order(
            models.RepoContextResult,
            [
                "chunk_id",
                "repo_root",
                "scope",
                "relative_path",
                "language",
                "chunk_type",
                "start_line",
                "end_line",
                "line_range",
                "file_extension",
                "snippet",
                "snippet_start_line",
                "snippet_end_line",
                "symbol_path",
                "heading_path",
                "domain_tags",
                "score",
                "final_score",
                "match_type",
                "dense_score",
                "lexical_score",
                "matched_terms",
                "why_matched",
                "line_matches",
                "final_rank",
                "origin_branches",
                "origin_queries",
                "branch_scores",
                "branch_ranks",
                "uncovered_terms",
                "evidence_paths",
                "verification",
            ],
        )
        self.assertTrue(models.RepoContextResult.model_fields["final_rank"].is_required())
        self.assertTrue(models.RepoContextResult.model_fields["final_score"].is_required())
        self.assert_default_factory(models.RepoContextResult, "origin_branches")
        self.assert_default_factory(models.RepoContextResult, "origin_queries")
        self.assert_default_factory(models.RepoContextResult, "branch_scores")
        self.assert_default_factory(models.RepoContextResult, "branch_ranks")
        self.assert_default_factory(models.RepoContextResult, "uncovered_terms")
        self.assert_default_factory(models.RepoContextResult, "evidence_paths")
        self.assert_default_factory(models.RepoContextResult, "verification")
        self.assert_field_order(
            models.EvidencePathStep,
            [
                "node_id",
                "node_type",
                "key",
                "relative_path",
                "chunk_id",
                "start_line",
                "end_line",
                "edge_type",
                "metadata",
            ],
        )
        self.assert_default_factory(models.EvidencePathStep, "metadata")
        self.assert_field_order(
            models.EvidencePath,
            ["reason", "seed_chunk_id", "seed_relative_path", "path", "confidence", "stale"],
        )
        self.assertTrue(models.EvidencePath.model_fields["reason"].is_required())
        self.assertEqual(models.EvidencePath.model_fields["stale"].default, False)
        self.assert_default_factory(models.EvidencePath, "path")
        self.assert_field_order(
            models.FileContextGroup,
            [
                "relative_path",
                "scope",
                "language",
                "best_rank",
                "best_score",
                "chunk_count",
                "chunk_ids",
                "line_ranges",
                "matched_terms",
                "origin_queries",
                "origin_branches",
                "recommended_action_codes",
            ],
        )
        self.assert_default_factory(models.FileContextGroup, "chunk_ids")
        self.assert_default_factory(models.FileContextGroup, "line_ranges")
        self.assert_default_factory(models.FileContextGroup, "matched_terms")
        self.assert_default_factory(models.FileContextGroup, "origin_queries")
        self.assert_default_factory(models.FileContextGroup, "origin_branches")
        self.assert_default_factory(models.FileContextGroup, "recommended_action_codes")
        self.assert_field_order(
            models.RepoContextDiagnostics,
            [
                "fusion_method",
                "rrf_k",
                "query_count",
                "per_query",
                "branch_diagnostics",
                "filter_pushed_families",
                "filter_post_families",
                "filter_best_effort",
                "exact_anchor_analysis",
                "exact_anchor_usages",
                "scope_distribution",
                "file_distribution",
                "matched_terms",
                "uncovered_terms",
                "sparse_available",
                "sparse_unavailable_codes",
                "graph_available",
                "graph_used",
                "graph_diagnostics",
                "rerank_used",
                "colbert_used",
            ],
        )
        self.assertTrue(models.RepoContextDiagnostics.model_fields["fusion_method"].is_required())
        self.assertTrue(models.RepoContextDiagnostics.model_fields["query_count"].is_required())
        self.assert_default_factory(models.RepoContextDiagnostics, "per_query")
        self.assert_default_factory(models.RepoContextDiagnostics, "branch_diagnostics")
        self.assert_default_factory(models.RepoContextDiagnostics, "exact_anchor_usages")
        self.assert_default_factory(models.RepoContextDiagnostics, "scope_distribution")
        self.assert_default_factory(models.RepoContextDiagnostics, "file_distribution")
        self.assert_default_factory(models.RepoContextDiagnostics, "matched_terms")
        self.assert_default_factory(models.RepoContextDiagnostics, "uncovered_terms")
        self.assert_default_factory(models.RepoContextDiagnostics, "sparse_unavailable_codes")
        self.assertEqual(models.RepoContextDiagnostics.model_fields["graph_available"].default, False)
        self.assertEqual(models.RepoContextDiagnostics.model_fields["graph_used"].default, False)
        self.assertIsNone(models.RepoContextDiagnostics.model_fields["graph_diagnostics"].default)
        self.assertEqual(models.RepoContextDiagnostics.model_fields["rerank_used"].default, False)
        self.assertEqual(models.RepoContextDiagnostics.model_fields["colbert_used"].default, False)
        self.assert_field_order(
            models.GraphBranchDiagnostics,
            [
                "requested_mode",
                "effective_mode",
                "available",
                "used",
                "status_state",
                "expansion_allowed",
                "skipped_reason",
                "seed_count",
                "bound_seed_count",
                "expanded_node_count",
                "graph_candidate_count",
                "final_graph_result_count",
                "graph_candidates_before_filter",
                "graph_candidates_after_filter",
                "graph_filter_post_families",
                "degraded",
                "seed_path_states",
                "expanded_path_states",
                "max_hops",
                "edge_types_used",
                "warning_codes",
            ],
        )
        self.assertTrue(models.GraphBranchDiagnostics.model_fields["requested_mode"].is_required())
        self.assertEqual(models.GraphBranchDiagnostics.model_fields["max_hops"].default, 1)
        self.assertEqual(models.GraphBranchDiagnostics.model_fields["degraded"].default, False)
        self.assert_default_factory(models.GraphBranchDiagnostics, "seed_path_states")
        self.assert_default_factory(models.GraphBranchDiagnostics, "expanded_path_states")
        self.assert_default_factory(models.GraphBranchDiagnostics, "edge_types_used")
        self.assert_default_factory(models.GraphBranchDiagnostics, "warning_codes")

    def test_recommended_action_contract_is_stable(self) -> None:
        self.assert_field_order(
            models.RecommendedAction,
            [
                "code",
                "severity",
                "title",
                "detail",
                "relative_path",
                "start_line",
                "end_line",
                "argv_hint",
                "cwd_hint",
                "authority",
                "tool_hint",
            ],
        )
        self.assertEqual(
            set(get_args(models.RecommendedAction.model_fields["code"].annotation)),
            {
                "read_file_range",
                "run_local_rg",
                "rebuild_index",
                "build_graph",
                "rebuild_graph",
                "update_graph",
                "retry_with_graph_expand",
                "retry_with_hybrid",
                "retry_with_semantic",
                "retry_with_path_filter",
                "retry_with_docs_scope",
                "retry_with_code_scope",
            },
        )
        self.assertEqual(set(get_args(models.RecommendedAction.model_fields["severity"].annotation)), {"info", "warning"})

    def test_status_and_registry_contract_fields_are_stable(self) -> None:
        self.assert_field_order(
            models.RetrievalContractStatus,
            [
                "dense_available",
                "sparse_available",
                "sparse_contract_compatible",
                "fusion_methods_supported",
                "payload_filter_pushdown_supported",
                "legacy_lexical_fallback_enabled",
                "unavailable_codes",
                "sparse_unavailable_codes",
                "sparse_manifest_hash",
                "sparse_vocabulary_hash",
                "sparse_corpus_stats_hash",
                "sparse_stats_stale",
                "expected_lexical_analyzer_version",
                "stored_lexical_analyzer_version",
                "expected_sparse_encoder_kind",
                "stored_sparse_encoder_kind",
                "scope_statuses",
            ],
        )
        self.assert_default_factory(models.RetrievalContractStatus, "fusion_methods_supported")
        self.assertEqual(models.RetrievalContractStatus.model_fields["legacy_lexical_fallback_enabled"].default, True)
        self.assert_field_order(
            models.RepoRegistryEntry,
            [
                "repo_root",
                "repo_key",
                "display_name",
                "status",
                "active",
                "index_profile",
                "include_globs",
                "doc_prefixes",
                "exclude_globs",
                "last_full_build_ts",
                "last_incremental_update_ts",
                "indexed_branch",
                "indexed_commit_hash",
                "last_error",
                "watch_enabled",
                "watch_running",
                "code_points_count",
                "docs_points_count",
                "created_at",
                "updated_at",
            ],
        )
        self.assert_field_order(
            models.BackendRegistryEntry,
            [
                "backend_id",
                "role",
                "backend_type",
                "transport",
                "endpoint",
                "managed_by_service",
                "autostart_policy",
                "health_path",
                "location_type",
                "device_type",
                "model_name",
                "config_blob",
                "created_at",
                "updated_at",
            ],
        )

    def test_graph_contract_fields_are_stable(self) -> None:
        self.assert_field_order(
            models.GraphStatusSummary,
            [
                "available",
                "state",
                "expansion_allowed",
                "schema_version",
                "built_commit",
                "built_at",
                "extractor_versions_hash",
                "file_coverage_ratio",
                "stale_path_count",
                "invalidated_paths_preview",
                "update_available",
                "update_required",
                "update_safe_auto_run",
                "update_blocked_reason",
                "update_path_count",
                "update_paths_preview",
                "rebuild_required",
                "warning_codes",
            ],
        )
        self.assertTrue(models.GraphStatusSummary.model_fields["available"].is_required())
        self.assertTrue(models.GraphStatusSummary.model_fields["state"].is_required())
        self.assertTrue(models.GraphStatusSummary.model_fields["expansion_allowed"].is_required())
        self.assert_default_factory(models.GraphStatusSummary, "invalidated_paths_preview")
        self.assert_default_factory(models.GraphStatusSummary, "update_paths_preview")
        self.assert_default_factory(models.GraphStatusSummary, "warning_codes")
        self.assert_field_order(
            models.GraphCounts,
            [
                "files",
                "nodes",
                "edges",
                "node_terms",
                "node_chunks",
                "node_type_counts",
                "edge_type_counts",
                "file_state_counts",
            ],
        )
        self.assertEqual(models.GraphCounts.model_fields["files"].default, 0)
        self.assert_default_factory(models.GraphCounts, "node_type_counts")
        self.assert_default_factory(models.GraphCounts, "edge_type_counts")
        self.assert_default_factory(models.GraphCounts, "file_state_counts")
        self.assert_field_order(
            models.GraphStatusResult,
            [
                "contract_version",
                "repo_root",
                "profile",
                "graph_path",
                "available",
                "state",
                "expansion_allowed",
                "schema_version",
                "built_commit",
                "built_at",
                "graph_contract_hash",
                "extractor_versions_hash",
                "source_index_contract",
                "file_coverage_ratio",
                "stale_path_count",
                "invalidated_paths_preview",
                "update_available",
                "update_required",
                "update_safe_auto_run",
                "update_blocked_reason",
                "update_path_count",
                "update_paths_preview",
                "rebuild_required",
                "counts",
                "warning_codes",
                "error_code",
                "error_detail",
            ],
        )
        self.assertEqual(models.GraphStatusResult.model_fields["contract_version"].default, "graph_status.v1")
        self.assertEqual(models.GraphStatusResult.model_fields["stale_path_count"].default, 0)
        self.assert_default_factory(models.GraphStatusResult, "counts")
        self.assert_default_factory(models.GraphStatusResult, "invalidated_paths_preview")
        self.assert_default_factory(models.GraphStatusResult, "update_paths_preview")
        self.assert_default_factory(models.GraphStatusResult, "warning_codes")
        self.assert_field_order(
            models.GraphBuildResult,
            [
                "contract_version",
                "repo_root",
                "profile",
                "graph_path",
                "built",
                "rebuilt",
                "state",
                "built_at",
                "built_commit",
                "files_count",
                "chunks_count",
                "nodes_count",
                "edges_count",
                "warning_codes",
                "status",
                "reason",
            ],
        )
        self.assertEqual(models.GraphBuildResult.model_fields["contract_version"].default, "graph_build.v1")
        self.assertTrue(models.GraphBuildResult.model_fields["built"].is_required())
        self.assertEqual(models.GraphBuildResult.model_fields["rebuilt"].default, False)
        self.assert_default_factory(models.GraphBuildResult, "warning_codes")
        self.assert_field_order(
            models.GraphUpdateResult,
            [
                "contract_version",
                "repo_root",
                "profile",
                "graph_path",
                "operation_id",
                "lease_id",
                "audit_id",
                "idempotency_key",
                "idempotent_retry",
                "allow_large_update",
                "operation_in_progress",
                "updated",
                "skipped",
                "skip_reason",
                "paths_requested",
                "paths_discovered",
                "paths_normalized",
                "paths_required",
                "reverse_dependent_paths_added",
                "paths_updated",
                "paths_deleted",
                "zero_chunk_paths",
                "paths_missing_from_index",
                "paths_skipped",
                "chunks_scanned_total",
                "chunks_for_updated_paths",
                "nodes_deleted",
                "edges_deleted",
                "nodes_upserted",
                "edges_upserted",
                "source_contract_updated",
                "source_contract_revision_before",
                "source_contract_revision_after",
                "graph_state_after",
                "expansion_allowed_after",
                "update_plan_hash",
                "metadata_only_update",
                "bounds_exceeded",
                "warning_codes",
                "recommended_actions",
                "status",
            ],
        )
        self.assertEqual(models.GraphUpdateResult.model_fields["contract_version"].default, "graph_update.v1")
        self.assertTrue(models.GraphUpdateResult.model_fields["updated"].is_required())
        self.assert_default_factory(models.GraphUpdateResult, "bounds_exceeded")
        self.assert_default_factory(models.GraphUpdateResult, "warning_codes")

    def test_nested_status_model_defaults_are_stable(self) -> None:
        self.assert_field_order(
            models.StatusAvailability,
            [
                "runtime_ready",
                "dependencies_ready",
                "search_available",
                "search_unavailable_codes",
                "host_action_required",
                "next_actions",
            ],
        )
        self.assertTrue(models.StatusAvailability.model_fields["runtime_ready"].is_required())
        self.assertTrue(models.StatusAvailability.model_fields["search_available"].is_required())
        self.assertIsNone(models.StatusAvailability.model_fields["dependencies_ready"].default)
        self.assertEqual(models.StatusAvailability.model_fields["host_action_required"].default, False)
        self.assert_default_factory(models.StatusAvailability, "search_unavailable_codes")
        self.assert_default_factory(models.StatusAvailability, "next_actions")
        self.assert_field_order(
            models.BackendStatusSummary,
            [
                "runtime_embedding_backend_id",
                "selected_embedding_backend_id",
                "backend_type",
                "model_name",
                "compatible",
                "switch_required",
                "healthy",
                "health_error",
                "managed_by_service",
                "autostart_policy",
                "location_type",
                "device_type",
                "action_plan_tool",
            ],
        )
        self.assertTrue(models.BackendStatusSummary.model_fields["compatible"].is_required())
        self.assertTrue(models.BackendStatusSummary.model_fields["switch_required"].is_required())
        self.assertEqual(models.BackendStatusSummary.model_fields["action_plan_tool"].default, "get_backend_action_plan")
        self.assert_field_order(
            models.IndexContractStatus,
            [
                "profile",
                "embedding_backend",
                "embedding_model",
                "schema_version",
                "query_template_hash",
                "document_prefix_hash",
                "compatibility",
                "blocking",
                "incompatibility_codes",
                "collections",
            ],
        )
        self.assert_default_factory(models.IndexContractStatus, "incompatibility_codes")
        self.assert_default_factory(models.IndexContractStatus, "collections")
        self.assert_field_order(
            models.RetrievalScopeStatus,
            [
                "scope",
                "collection_name",
                "state",
                "points_count",
                "dense_available",
                "dense_schema_kind",
                "sparse_available",
                "sparse_contract_compatible",
                "sparse_stats_stale",
                "sparse_manifest_hash",
                "sparse_vocabulary_hash",
                "sparse_corpus_stats_hash",
                "expected_lexical_analyzer_version",
                "stored_lexical_analyzer_version",
                "expected_sparse_encoder_kind",
                "stored_sparse_encoder_kind",
                "payload_indexes",
                "unavailable_codes",
            ],
        )
        self.assertEqual(models.RetrievalScopeStatus.model_fields["points_count"].default, 0)
        self.assertEqual(models.RetrievalScopeStatus.model_fields["dense_available"].default, False)
        self.assertEqual(models.RetrievalScopeStatus.model_fields["dense_schema_kind"].default, "unknown")
        self.assertEqual(models.RetrievalScopeStatus.model_fields["sparse_available"].default, False)
        self.assertEqual(models.RetrievalScopeStatus.model_fields["sparse_contract_compatible"].default, False)
        self.assert_default_factory(models.RetrievalScopeStatus, "payload_indexes")
        self.assert_default_factory(models.RetrievalScopeStatus, "unavailable_codes")
        self.assert_field_order(
            models.PayloadIndexStatus,
            ["scope", "collection_name", "field_name", "expected", "present", "warning_code"],
        )
        self.assertEqual(models.PayloadIndexStatus.model_fields["expected"].default, True)
        self.assertIsNone(models.PayloadIndexStatus.model_fields["present"].default)
        self.assert_field_order(
            models.FreshnessPolicy,
            [
                "search_allowed_when_stale",
                "stale_severity",
                "requires_rebuild",
                "exact_fallback_recommended",
            ],
        )
        self.assertEqual(models.FreshnessPolicy.model_fields["search_allowed_when_stale"].default, True)
        self.assertEqual(models.FreshnessPolicy.model_fields["stale_severity"].default, "none")
        self.assertEqual(models.FreshnessPolicy.model_fields["requires_rebuild"].default, False)
        self.assertEqual(models.FreshnessPolicy.model_fields["exact_fallback_recommended"].default, False)
        self.assert_field_order(
            models.WatcherStatusSummary,
            [
                "enabled",
                "running",
                "policy",
                "runtime_available",
                "autostart_policy",
                "start_required",
                "start_blocked_reason",
                "last_event_at",
                "last_error",
                "last_incremental_update_ts",
            ],
        )
        self.assertEqual(models.WatcherStatusSummary.model_fields["policy"].default, "single_active_repo")
        self.assertEqual(models.WatcherStatusSummary.model_fields["runtime_available"].default, False)
        self.assertEqual(models.WatcherStatusSummary.model_fields["autostart_policy"].default, "manual")
        self.assertEqual(models.WatcherStatusSummary.model_fields["start_required"].default, False)
        self.assertIsNone(models.WatcherStatusSummary.model_fields["start_blocked_reason"].default)
        self.assert_field_order(
            models.IndexFreshnessStatus,
            [
                "indexed_branch",
                "indexed_commit_hash",
                "current_branch",
                "current_head_commit",
                "primary_state",
                "states",
                "reason_codes",
                "head_mismatch",
                "indexable_worktree_changes",
                "untracked_indexable_files",
                "stale",
                "freshness_unknown",
                "stale_reasons",
                "host_side_hint",
                "policy",
            ],
        )
        self.assertEqual(models.IndexFreshnessStatus.model_fields["primary_state"].default, "fresh")
        self.assert_default_factory(models.IndexFreshnessStatus, "states")
        self.assert_default_factory(models.IndexFreshnessStatus, "reason_codes")
        self.assertEqual(models.IndexFreshnessStatus.model_fields["head_mismatch"].default, False)
        self.assertEqual(models.IndexFreshnessStatus.model_fields["stale"].default, False)
        self.assert_default_factory(models.IndexFreshnessStatus, "stale_reasons")
        self.assert_field_order(
            models.PathFreshnessSummary,
            [
                "coverage_complete",
                "coverage_error_code",
                "manifest_available",
                "manifest_path",
                "manifest_schema_version",
                "identity_compatible",
                "changed_indexable_paths_count",
                "untracked_indexable_paths_count",
                "missing_from_index_count",
                "stale_indexed_paths_count",
                "deleted_indexed_paths_count",
                "path_error_count",
                "stale_paths_preview",
                "deleted_paths_preview",
                "error_paths_preview",
            ],
        )
        self.assertFalse(models.PathFreshnessSummary.model_fields["coverage_complete"].default)
        self.assert_default_factory(models.PathFreshnessSummary, "stale_paths_preview")
        self.assert_field_order(
            models.LifecycleStatus,
            [
                "state",
                "action_in_progress",
                "operation_id",
                "phase",
                "started_at",
                "updated_at",
                "progress",
                "last_error",
                "last_error_at",
            ],
        )
        self.assertTrue(models.LifecycleStatus.model_fields["state"].is_required())
        self.assertTrue(models.LifecycleStatus.model_fields["action_in_progress"].is_required())
        self.assert_default_factory(models.LifecycleStatus, "progress")
        self.assertEqual(
            models.RetrievalContractStatus.model_fields["fusion_methods_supported"].default_factory(),
            ["weighted_rrf"],
        )

    def test_index_status_result_contract_version_and_fields_are_stable(self) -> None:
        self.assert_field_order(
            models.IndexStatusResult,
            [
                "contract_version",
                "status_kind",
                "repo_root",
                "mounted_repo_root",
                "active_repo_root",
                "runtime_switch_required",
                "repo_key",
                "repo_state",
                "active",
                "search_available",
                "reason_if_unavailable",
                "index_profile",
                "embedding_backend",
                "embedding_model",
                "embedding_backend_type",
                "embedding_backend_id",
                "selected_embedding_backend_id",
                "backend_switch_required",
                "runtime_switch_command_hint",
                "backend_switch_command_hint",
                "qdrant_url",
                "schema_version",
                "watch_enabled",
                "watch_running",
                "include_globs",
                "doc_prefixes",
                "last_full_build_ts",
                "last_incremental_update_ts",
                "active_branch",
                "head_commit_hash",
                "indexed_branch",
                "indexed_commit_hash",
                "worktree_dirty",
                "index_stale",
                "git",
                "freshness",
                "path_freshness",
                "availability",
                "runtime",
                "backend",
                "index_contract",
                "retrieval",
                "graph",
                "counts",
                "lifecycle",
                "watcher",
                "warnings",
                "collections",
            ],
        )
        fields = models.IndexStatusResult.model_fields
        self.assertEqual(fields["contract_version"].default, "index_status.v2")
        self.assertEqual(fields["status_kind"].default, "runtime_repo_status")
        self.assertTrue(fields["repo_root"].is_required())
        self.assertTrue(fields["mounted_repo_root"].is_required())
        self.assertTrue(fields["collections"].is_required())
        self.assertEqual(fields["runtime_switch_required"].default, False)
        self.assertEqual(fields["backend_switch_required"].default, False)
        self.assertEqual(fields["worktree_dirty"].default, False)
        self.assertEqual(fields["index_stale"].default, False)
        self.assertIsNone(fields["graph"].default)
        self.assertIsNone(fields["path_freshness"].default)
        self.assert_default_factory(models.IndexStatusResult, "include_globs")
        self.assert_default_factory(models.IndexStatusResult, "doc_prefixes")
        self.assert_default_factory(models.IndexStatusResult, "warnings")
        schema = models.IndexStatusResult.model_json_schema()
        self.assertEqual(schema["properties"]["contract_version"]["default"], "index_status.v2")
        self.assertIn("retrieval", schema["properties"])
        self.assertIn("graph", schema["properties"])


if __name__ == "__main__":
    unittest.main()
