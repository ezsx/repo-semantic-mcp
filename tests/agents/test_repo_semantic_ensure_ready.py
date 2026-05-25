import unittest
from unittest.mock import patch

from scripts.agents import repo_semantic_ensure_ready as ensure


def _status(**overrides):
    payload = {
        "repo_root": "C:/repo",
        "active_repo_root": "C:/repo",
        "index_profile": "cpu",
        "search_available": True,
        "index_stale": False,
        "embedding_backend_id": "embedding/cpu",
        "selected_embedding_backend_id": "embedding/cpu",
        "retrieval": {
            "dense_available": True,
            "sparse_available": True,
        },
        "watcher": {
            "enabled": True,
            "running": False,
            "start_required": False,
            "start_blocked_reason": None,
        },
        "graph": {
            "available": True,
            "expansion_allowed": True,
            "state": "ready",
        },
        "warnings": [],
    }
    payload.update(overrides)
    return payload


class RepoSemanticEnsureReadyTests(unittest.TestCase):
    def test_diagnose_mode_does_not_start_watcher(self) -> None:
        status = _status(
            watcher={
                "enabled": True,
                "running": False,
                "start_required": True,
                "start_blocked_reason": None,
            }
        )

        with patch.object(ensure, "_http_get_json", return_value=(True, {}, None)), patch.object(
            ensure,
            "_call_tool_payload",
            return_value=(True, status, None),
        ) as call_tool:
            result = ensure.ensure_ready(
                repo="C:/repo",
                mode="diagnose",
                start_watcher=False,
            )

        self.assertTrue(result["ok"])
        self.assertFalse(result["watcher_started"])
        self.assertNotIn("start_watcher", result["auto_recovery_attempted"])
        call_tool.assert_called_once()

    def test_safe_recover_starts_watcher_and_refreshes_status(self) -> None:
        first_status = _status(
            watcher={
                "enabled": True,
                "running": False,
                "start_required": True,
                "start_blocked_reason": None,
            },
            warnings=[{"code": "watcher_enabled_but_not_running"}],
        )
        refreshed_status = _status(
            watcher={
                "enabled": True,
                "running": True,
                "start_required": False,
                "start_blocked_reason": None,
            }
        )
        calls = [
            (True, first_status, None),
            (True, {"watch_running": True, "startup_reconcile": {"ran": True}}, None),
            (True, refreshed_status, None),
        ]

        with patch.object(ensure, "_http_get_json", return_value=(True, {}, None)), patch.object(
            ensure,
            "_call_tool_payload",
            side_effect=calls,
        ) as call_tool:
            result = ensure.ensure_ready(repo="C:/repo", mode="safe-recover")

        self.assertTrue(result["ok"])
        self.assertTrue(result["watcher_started"])
        self.assertTrue(result["startup_reconcile_ran"])
        self.assertIn("start_watcher", result["auto_recovery_attempted"])
        watcher_args = call_tool.call_args_list[1].kwargs["arguments"]
        self.assertEqual(watcher_args["repo_root"], "C:/repo")
        self.assertTrue(watcher_args["idempotency_key"].startswith("repo-semantic-safe-recover:start_watcher:v1:"))
        self.assertEqual(
            result["auto_recovery_idempotency_keys"]["start_watcher"],
            watcher_args["idempotency_key"],
        )

    def test_safe_recover_updates_graph_without_starting_watcher_when_index_is_fresh(self) -> None:
        first_status = _status(
            watcher={
                "enabled": True,
                "running": False,
                "start_required": False,
                "start_blocked_reason": None,
            },
            graph={
                "available": True,
                "expansion_allowed": False,
                "state": "stale",
                "update_required": True,
                "update_safe_auto_run": True,
                "update_blocked_reason": None,
                "warning_codes": ["graph_source_index_revision_mismatch"],
            },
        )
        graph_status = {
            "available": True,
            "expansion_allowed": True,
            "state": "ready",
            "update_required": False,
            "update_safe_auto_run": False,
            "warning_codes": [],
        }
        calls = [
            (True, first_status, None),
            (
                True,
                {
                    "contract_version": "graph_update.v1",
                    "updated": True,
                    "status": graph_status,
                },
                None,
            ),
        ]

        with patch.object(ensure, "_http_get_json", return_value=(True, {}, None)), patch.object(
            ensure,
            "_call_tool_payload",
            side_effect=calls,
        ) as call_tool:
            result = ensure.ensure_ready(
                repo="C:/repo",
                mode="safe-recover",
                start_watcher=False,
                require_graph=True,
            )

        self.assertTrue(result["ok"])
        self.assertFalse(result["watcher_started"])
        self.assertTrue(result["graph_update_attempted"])
        self.assertTrue(result["graph_update_succeeded"])
        self.assertIn("update_graph", result["auto_recovery_attempted"])
        graph_args = call_tool.call_args_list[1].kwargs["arguments"]
        self.assertEqual(graph_args["repo_root"], "C:/repo")
        self.assertEqual(graph_args["paths"], [])
        self.assertTrue(graph_args["idempotency_key"].startswith("repo-semantic-safe-recover:update_graph:v1:"))

    def test_safe_recover_graph_transport_failure_has_actionable_result(self) -> None:
        first_status = _status(
            graph={
                "available": True,
                "expansion_allowed": False,
                "state": "stale",
                "update_required": True,
                "update_safe_auto_run": True,
                "update_blocked_reason": None,
                "warning_codes": ["graph_source_index_revision_mismatch"],
            },
        )
        calls = [
            (True, first_status, None),
            (False, None, "transport closed"),
        ]

        with patch.object(ensure, "_http_get_json", return_value=(True, {}, None)), patch.object(
            ensure,
            "_call_tool_payload",
            side_effect=calls,
        ):
            result = ensure.ensure_ready(
                repo="C:/repo",
                mode="safe-recover",
                start_watcher=False,
                require_graph=True,
            )

        self.assertFalse(result["ok"])
        self.assertEqual(result["readiness"], "blocked")
        self.assertFalse(result["permission_required"])
        self.assertEqual(result["graph_update_skipped_reason"], "update_graph_transport_failed")
        self.assertEqual(result["next_actions"][0]["code"], "update_graph")

    def test_safe_recover_graph_bounds_skip_does_not_emit_conflicting_safe_action(self) -> None:
        first_status = _status(
            graph={
                "available": True,
                "expansion_allowed": False,
                "state": "stale",
                "update_required": True,
                "update_safe_auto_run": True,
                "update_blocked_reason": None,
                "warning_codes": ["graph_source_index_revision_mismatch"],
            },
        )
        permission_action = {
            "code": "update_graph",
            "requires_user_permission": True,
            "safe_auto_run": False,
            "tool_arguments": {"paths": [], "allow_large_update": True},
        }
        calls = [
            (True, first_status, None),
            (
                True,
                {
                    "contract_version": "graph_update.v1",
                    "updated": False,
                    "skipped": True,
                    "skip_reason": "graph_update_bounds_exceeded",
                    "bounds_exceeded": ["max_paths"],
                    "recommended_actions": [permission_action],
                    "status": first_status["graph"],
                },
                None,
            ),
        ]

        with patch.object(ensure, "_http_get_json", return_value=(True, {}, None)), patch.object(
            ensure,
            "_call_tool_payload",
            side_effect=calls,
        ):
            result = ensure.ensure_ready(
                repo="C:/repo",
                mode="safe-recover",
                start_watcher=False,
                require_graph=True,
            )

        self.assertFalse(result["ok"])
        self.assertEqual(result["readiness"], "needs_user_permission")
        self.assertEqual(result["next_actions"], [permission_action])
        self.assertEqual(result["permission_required_actions"], [permission_action])

    def test_safe_recover_uses_embedded_watcher_status_without_second_refresh(self) -> None:
        first_status = _status(
            watcher={
                "enabled": True,
                "running": False,
                "start_required": True,
                "start_blocked_reason": None,
            },
            warnings=[{"code": "watcher_enabled_but_not_running"}],
        )
        embedded_status = _status(
            watcher={
                "enabled": True,
                "running": True,
                "start_required": False,
                "start_blocked_reason": None,
            }
        )
        calls = [
            (True, first_status, None),
            (
                True,
                {
                    "watch_running": True,
                    "startup_reconcile": {"ran": True},
                    "status": embedded_status,
                },
                None,
            ),
        ]

        with patch.object(ensure, "_http_get_json", return_value=(True, {}, None)), patch.object(
            ensure,
            "_call_tool_payload",
            side_effect=calls,
        ) as call_tool:
            result = ensure.ensure_ready(repo="C:/repo", mode="safe-recover")

        self.assertTrue(result["ok"])
        self.assertTrue(result["watcher_started"])
        self.assertTrue(result["capabilities"]["watcher"])
        self.assertEqual(call_tool.call_count, 2)

    def test_safe_recover_uses_explicit_watcher_idempotency_key(self) -> None:
        first_status = _status(
            watcher={
                "enabled": True,
                "running": False,
                "start_required": True,
                "start_blocked_reason": None,
            }
        )
        refreshed_status = _status(
            watcher={
                "enabled": True,
                "running": True,
                "start_required": False,
                "start_blocked_reason": None,
            }
        )
        calls = [
            (True, first_status, None),
            (True, {"watch_running": True, "startup_reconcile": {"ran": True}}, None),
            (True, refreshed_status, None),
        ]

        with patch.object(ensure, "_http_get_json", return_value=(True, {}, None)), patch.object(
            ensure,
            "_call_tool_payload",
            side_effect=calls,
        ) as call_tool:
            result = ensure.ensure_ready(
                repo="C:/repo",
                mode="safe-recover",
                start_watcher_idempotency_key="agent-run-123",
            )

        self.assertTrue(result["ok"])
        self.assertEqual(
            call_tool.call_args_list[1].kwargs["arguments"]["idempotency_key"],
            "agent-run-123",
        )
        self.assertEqual(
            result["auto_recovery_idempotency_keys"]["start_watcher"],
            "agent-run-123",
        )

    def test_start_watcher_transport_error_is_confirmed_by_lightweight_healthz(self) -> None:
        first_status = _status(
            watcher={
                "enabled": True,
                "running": False,
                "start_required": True,
                "start_blocked_reason": None,
            },
            warnings=[{"code": "watcher_enabled_but_not_running"}],
        )
        calls = [
            (True, first_status, None),
            (False, None, "Transport closed"),
        ]
        health_calls = [
            (True, {}, None),
            (True, {}, None),
            (True, {"watch_running": True}, None),
        ]

        with patch.object(ensure, "_http_get_json", side_effect=health_calls), patch.object(
            ensure,
            "_call_tool_payload",
            side_effect=calls,
        ) as call_tool:
            result = ensure.ensure_ready(repo="C:/repo", mode="safe-recover")

        self.assertTrue(result["ok"])
        self.assertTrue(result["watcher_started"])
        self.assertTrue(result["watcher_start_confirmed_after_transport_error"])
        self.assertEqual(result["readiness"], "usable_degraded")
        self.assertEqual(call_tool.call_count, 2)

    def test_idempotent_watcher_retry_does_not_refresh_full_index_status(self) -> None:
        first_status = _status(
            watcher={
                "enabled": True,
                "running": False,
                "start_required": True,
                "start_blocked_reason": None,
            }
        )
        calls = [
            (True, first_status, None),
            (
                True,
                {
                    "watch_running": False,
                    "idempotent_retry": True,
                    "operation_in_progress": True,
                    "status": None,
                    "status_skipped_reason": "idempotent_retry_in_progress",
                    "startup_reconcile": {
                        "skipped": True,
                        "reason": "idempotent_retry_in_progress",
                    },
                },
                None,
            ),
        ]

        with patch.object(ensure, "_http_get_json", return_value=(True, {}, None)), patch.object(
            ensure,
            "_call_tool_payload",
            side_effect=calls,
        ) as call_tool:
            result = ensure.ensure_ready(repo="C:/repo", mode="safe-recover")

        self.assertTrue(result["ok"])
        self.assertTrue(result["watcher_start_in_progress"])
        self.assertEqual(result["readiness"], "usable_degraded")
        self.assertEqual(call_tool.call_count, 2)

    def test_repo_mismatch_requires_permission_without_watcher_start(self) -> None:
        status = _status(repo_root="C:/other", active_repo_root="C:/other")

        with patch.object(ensure, "_http_get_json", return_value=(True, {}, None)), patch.object(
            ensure,
            "_call_tool_payload",
            return_value=(True, status, None),
        ) as call_tool:
            result = ensure.ensure_ready(repo="C:/repo", mode="safe-recover")

        self.assertFalse(result["ok"])
        self.assertEqual(result["readiness"], "needs_user_permission")
        self.assertTrue(result["permission_required"])
        self.assertEqual(result["next_actions"][0]["code"], "runtime_repo_mismatch")
        call_tool.assert_called_once()

    def test_active_repo_mismatch_is_distinct_from_runtime_repo_mismatch(self) -> None:
        status = _status(repo_root="C:/repo", active_repo_root="C:/other")

        with patch.object(ensure, "_http_get_json", return_value=(True, {}, None)), patch.object(
            ensure,
            "_call_tool_payload",
            return_value=(True, status, None),
        ):
            result = ensure.ensure_ready(repo="C:/repo", mode="safe-recover")

        self.assertFalse(result["ok"])
        self.assertTrue(result["runtime_repo_matches"])
        self.assertFalse(result["active_repo_matches"])
        self.assertEqual(result["next_actions"][0]["code"], "active_repo_mismatch")

    def test_require_graph_returns_permission_action_when_graph_unavailable(self) -> None:
        status = _status(
            graph={
                "available": False,
                "expansion_allowed": False,
                "state": "missing",
                "warning_codes": ["graph_missing"],
            }
        )

        with patch.object(ensure, "_http_get_json", return_value=(True, {}, None)), patch.object(
            ensure,
            "_call_tool_payload",
            return_value=(True, status, None),
        ):
            result = ensure.ensure_ready(repo="C:/repo", require_graph=True)

        self.assertFalse(result["ok"])
        self.assertEqual(result["readiness"], "needs_user_permission")
        self.assertEqual(result["next_actions"][0]["code"], "build_graph")
        self.assertFalse(result["next_actions"][0]["safe_auto_run"])
        self.assertEqual(result["next_actions"][0]["reason_codes"], ["graph_missing"])

    def test_require_graph_recommends_update_graph_for_repairable_stale_graph(self) -> None:
        status = _status(
            graph={
                "available": True,
                "expansion_allowed": False,
                "state": "stale",
                "update_required": True,
                "update_safe_auto_run": True,
                "update_blocked_reason": None,
                "warning_codes": ["graph_source_index_revision_mismatch"],
            }
        )

        with patch.object(ensure, "_http_get_json", return_value=(True, {}, None)), patch.object(
            ensure,
            "_call_tool_payload",
            return_value=(True, status, None),
        ):
            result = ensure.ensure_ready(repo="C:/repo", require_graph=True)

        self.assertTrue(result["ok"])
        self.assertEqual(result["readiness"], "usable_degraded")
        self.assertEqual(result["next_actions"][0]["code"], "update_graph")
        self.assertFalse(result["next_actions"][0]["requires_user_permission"])
        self.assertTrue(result["next_actions"][0]["safe_auto_run"])

    def test_require_graph_unknown_invalidations_avoids_rebuild_guidance(self) -> None:
        status = _status(
            graph={
                "available": True,
                "expansion_allowed": False,
                "state": "stale",
                "warning_codes": ["graph_invalidated_paths_unknown"],
            }
        )

        with patch.object(ensure, "_http_get_json", return_value=(True, {}, None)), patch.object(
            ensure,
            "_call_tool_payload",
            return_value=(True, status, None),
        ):
            result = ensure.ensure_ready(repo="C:/repo", require_graph=True)

        self.assertTrue(result["ok"])
        self.assertEqual(result["readiness"], "usable_degraded")
        self.assertEqual(result["next_actions"][0]["code"], "graph_invalidated_paths_unknown")
        self.assertFalse(result["next_actions"][0]["expensive"])

    def test_http_fallback_failure_is_structured(self) -> None:
        with patch.object(ensure, "_http_get_json", return_value=(False, None, "down")), patch.object(
            ensure,
            "_call_tool_payload",
            return_value=(False, None, "transport closed"),
        ):
            result = ensure.ensure_ready(repo="C:/repo")

        self.assertFalse(result["ok"])
        self.assertEqual(result["readiness"], "blocked")
        self.assertEqual(result["blocking_codes"], ["http_fallback_unavailable"])
        self.assertTrue(result["permission_required"])
        self.assertEqual(result["exit_code"], 1)

    def test_bootstrap_readyz_window_is_polled_before_index_status(self) -> None:
        status = _status()
        health_calls = [
            (True, {}, None),
            (False, {"phase": "loading_runtime_status"}, "http_503"),
        ]

        with patch.object(ensure, "_http_get_json", side_effect=health_calls), patch.object(
            ensure,
            "_wait_for_http_readiness",
            return_value={"healthz": "ok", "readyz": "ok"},
        ) as wait_ready, patch.object(
            ensure,
            "_call_tool_payload",
            return_value=(True, status, None),
        ):
            result = ensure.ensure_ready(repo="C:/repo", timeout_sec=5)

        self.assertTrue(result["ok"])
        self.assertEqual(result["dependency_statuses"]["readyz"], "ok")
        wait_ready.assert_called_once_with("http://127.0.0.1:8011", timeout_sec=60)

    def test_start_backend_starts_existing_container_then_checks_status(self) -> None:
        status = _status()
        health_calls = [
            (False, None, "connection refused"),
            (False, None, "connection refused"),
            (True, {}, None),
            (True, {}, None),
        ]

        with patch.object(ensure, "_http_get_json", side_effect=health_calls), patch.object(
            ensure,
            "_start_existing_backend_container",
            return_value=(True, None),
        ) as start_container, patch.object(
            ensure,
            "_call_tool_payload",
            return_value=(True, status, None),
        ):
            result = ensure.ensure_ready(
                repo="C:/repo",
                mode="safe-recover",
                start_backend=True,
                backend_container_name="repo-semantic-mcp",
            )

        self.assertTrue(result["ok"])
        self.assertTrue(result["backend_started"])
        self.assertIn("start_backend_container", result["auto_recovery_attempted"])
        start_container.assert_called_once_with("repo-semantic-mcp", timeout_sec=60)

    def test_wsl_path_matches_windows_repo_root(self) -> None:
        status = _status(repo_root="C:/repo", active_repo_root="C:/repo")

        with patch.object(ensure, "_http_get_json", return_value=(True, {}, None)), patch.object(
            ensure,
            "_call_tool_payload",
            return_value=(True, status, None),
        ):
            result = ensure.ensure_ready(repo="/mnt/c/repo")

        self.assertTrue(result["ok"])
        self.assertTrue(result["repo_matches"])

    def test_watcher_start_failure_requires_permission(self) -> None:
        status = _status(
            watcher={
                "enabled": True,
                "running": False,
                "start_required": True,
                "start_blocked_reason": None,
            }
        )
        calls = [
            (True, status, None),
            (False, None, "boom"),
        ]

        with patch.object(ensure, "_http_get_json", return_value=(True, {}, None)), patch.object(
            ensure,
            "_call_tool_payload",
            side_effect=calls,
        ):
            result = ensure.ensure_ready(repo="C:/repo", mode="safe-recover")

        self.assertFalse(result["ok"])
        self.assertEqual(result["readiness"], "needs_user_permission")
        self.assertEqual(result["next_actions"][0]["code"], "start_watcher_failed")
        self.assertEqual(result["exit_code"], 3)

    def test_noop_startup_reconcile_counts_as_ran(self) -> None:
        first_status = _status(
            watcher={
                "enabled": True,
                "running": False,
                "start_required": True,
                "start_blocked_reason": None,
            }
        )
        refreshed_status = _status(
            watcher={
                "enabled": True,
                "running": True,
                "start_required": False,
                "start_blocked_reason": None,
            }
        )
        calls = [
            (True, first_status, None),
            (True, {"watch_running": True, "startup_reconcile": {}}, None),
            (True, refreshed_status, None),
        ]

        with patch.object(ensure, "_http_get_json", return_value=(True, {}, None)), patch.object(
            ensure,
            "_call_tool_payload",
            side_effect=calls,
        ):
            result = ensure.ensure_ready(repo="C:/repo", mode="safe-recover")

        self.assertTrue(result["ok"])
        self.assertTrue(result["startup_reconcile_ran"])

    def test_watcher_bounds_permission_payload_is_reported(self) -> None:
        first_status = _status(
            watcher={
                "enabled": True,
                "running": False,
                "start_required": True,
                "start_blocked_reason": None,
            }
        )
        action = {
            "code": "startup_reconcile_bounds_exceeded",
            "requires_user_permission": True,
        }
        calls = [
            (True, first_status, None),
            (
                True,
                {
                    "watch_running": False,
                    "startup_reconcile": {
                        "bounds_exceeded": ["max_reconcile_paths"],
                        "mutation_started": False,
                    },
                    "permission_required": True,
                    "recommended_actions": [action],
                },
                None,
            ),
            (True, first_status, None),
        ]

        with patch.object(ensure, "_http_get_json", return_value=(True, {}, None)), patch.object(
            ensure,
            "_call_tool_payload",
            side_effect=calls,
        ) as call_tool:
            result = ensure.ensure_ready(repo="C:/repo", mode="safe-recover")

        self.assertFalse(result["ok"])
        self.assertEqual(result["readiness"], "needs_user_permission")
        self.assertEqual(result["next_actions"], [action])
        self.assertEqual(result["permission_required_actions"], [action])
        self.assertEqual(call_tool.call_count, 2)

    def test_default_mode_is_read_only_diagnose(self) -> None:
        status = _status(
            watcher={
                "enabled": True,
                "running": False,
                "start_required": True,
                "start_blocked_reason": None,
            }
        )

        with patch.object(ensure, "_http_get_json", return_value=(True, {}, None)), patch.object(
            ensure,
            "_call_tool_payload",
            return_value=(True, status, None),
        ) as call_tool:
            result = ensure.ensure_ready(repo="C:/repo")

        self.assertTrue(result["ok"])
        self.assertFalse(result["watcher_started"])
        self.assertNotIn("start_watcher", result["auto_recovery_attempted"])
        call_tool.assert_called_once()

    def test_probe_failure_blocks_readiness(self) -> None:
        status = _status()
        calls = [
            (True, status, None),
            (False, None, "search down"),
        ]

        with patch.object(ensure, "_http_get_json", return_value=(True, {}, None)), patch.object(
            ensure,
            "_call_tool_payload",
            side_effect=calls,
        ):
            result = ensure.ensure_ready(repo="C:/repo", probe_query="connect flow")

        self.assertFalse(result["ok"])
        self.assertEqual(result["readiness"], "blocked")
        self.assertIn("repo_context_search_probe_failed", result["blocking_codes"])


if __name__ == "__main__":
    unittest.main()
