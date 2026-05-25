import unittest
from unittest.mock import patch

from scripts.agents import repo_semantic_reliability_smoke as smoke


def _status(**overrides):
    payload = {
        "repo_root": "C:/repo",
        "search_available": True,
        "reason_if_unavailable": None,
    }
    payload.update(overrides)
    return payload


def _ready(**overrides):
    payload = {
        "ok": True,
        "readiness": "ready",
        "repo_root": "C:/repo",
        "search_available": True,
        "freshness_state": "fresh",
        "graph_expansion_allowed": True,
        "permission_required": False,
        "permission_required_actions": [],
        "auto_recovery_attempted": [],
    }
    payload.update(overrides)
    return payload


def _envelope(result):
    return {
        "ok": True,
        "result": result,
        "error": None,
        "error_code": None,
        "retryable": False,
    }


class RepoSemanticReliabilitySmokeTests(unittest.TestCase):
    def test_diagnose_smoke_does_not_run_lifecycle_actions(self) -> None:
        calls = {
            "index_status": _envelope(_status()),
            "repo_context_search": _envelope({"results": [{"relative_path": "src/app.py"}]}),
            "graph_status": _envelope(
                {"state": "ready", "available": True, "expansion_allowed": True}
            ),
        }

        with patch.object(smoke.ensure, "ensure_ready", return_value=_ready()) as ensure_ready, patch.object(
            smoke,
            "_call_tool_result",
            side_effect=lambda tool_name, **_: calls[tool_name],
        ):
            result = smoke.run_smoke(repo="C:/repo")

        self.assertTrue(result["ok"])
        self.assertEqual(result["mode"], "diagnose")
        self.assertFalse(result["forbidden_lifecycle_detected"])
        self.assertEqual(result["lifecycle_operation_counters"]["rebuild_index"], 0)
        self.assertEqual(result["lifecycle_operation_counters"]["build_graph"], 0)
        ensure_ready.assert_called_once()
        self.assertEqual(ensure_ready.call_args.kwargs["mode"], "diagnose")
        self.assertFalse(ensure_ready.call_args.kwargs["start_watcher"])

    def test_allow_start_watcher_counts_only_safe_recovery_action(self) -> None:
        calls = {
            "index_status": _envelope(_status()),
            "repo_context_search": _envelope({"results": []}),
            "graph_status": _envelope(
                {"state": "ready", "available": True, "expansion_allowed": True}
            ),
        }

        with patch.object(
            smoke.ensure,
            "ensure_ready",
            return_value=_ready(auto_recovery_attempted=["start_watcher"]),
        ) as ensure_ready, patch.object(
            smoke,
            "_call_tool_result",
            side_effect=lambda tool_name, **_: calls[tool_name],
        ):
            result = smoke.run_smoke(repo="C:/repo", allow_start_watcher=True)

        self.assertTrue(result["ok"])
        self.assertEqual(result["mode"], "safe-recover")
        self.assertEqual(result["lifecycle_operation_counters"]["start_watcher"], 1)
        self.assertEqual(result["auto_recovery_attempted"], ["start_watcher"])
        self.assertFalse(result["forbidden_lifecycle_detected"])
        self.assertTrue(ensure_ready.call_args.kwargs["start_watcher"])

    def test_require_graph_allows_bounded_update_graph_recovery(self) -> None:
        calls = {
            "index_status": _envelope(_status()),
            "repo_context_search": _envelope({"results": []}),
            "graph_status": _envelope(
                {"state": "ready", "available": True, "expansion_allowed": True}
            ),
        }

        with patch.object(
            smoke.ensure,
            "ensure_ready",
            return_value=_ready(auto_recovery_attempted=["update_graph"]),
        ) as ensure_ready, patch.object(
            smoke,
            "_call_tool_result",
            side_effect=lambda tool_name, **_: calls[tool_name],
        ):
            result = smoke.run_smoke(repo="C:/repo", require_graph=True)

        policy = next(item for item in result["scenarios"] if item["name"] == "safe_recovery_action_policy")
        self.assertTrue(result["ok"])
        self.assertEqual(result["mode"], "safe-recover")
        self.assertEqual(result["lifecycle_operation_counters"]["update_graph"], 1)
        self.assertEqual(policy["data"]["allowed_recovery_actions"], ["update_graph"])
        self.assertEqual(policy["data"]["disallowed_actions"], [])
        self.assertEqual(
            ensure_ready.call_args.kwargs["mode"],
            "safe-recover",
        )
        self.assertTrue(ensure_ready.call_args.kwargs["require_graph"])

    def test_safe_recovery_policy_flags_disallowed_actions(self) -> None:
        calls = {
            "index_status": _envelope(_status()),
            "repo_context_search": _envelope({"results": []}),
            "graph_status": _envelope(
                {"state": "ready", "available": True, "expansion_allowed": True}
            ),
        }

        with patch.object(
            smoke.ensure,
            "ensure_ready",
            return_value=_ready(auto_recovery_attempted=["start_watcher", "mystery_repair"]),
        ), patch.object(
            smoke,
            "_call_tool_result",
            side_effect=lambda tool_name, **_: calls[tool_name],
        ):
            result = smoke.run_smoke(repo="C:/repo", allow_start_watcher=True)

        policy = next(item for item in result["scenarios"] if item["name"] == "safe_recovery_action_policy")
        self.assertFalse(result["ok"])
        self.assertEqual(policy["code"], "disallowed_safe_recovery_action")
        self.assertEqual(policy["data"]["allowed_recovery_actions"], ["start_watcher"])
        self.assertEqual(policy["data"]["disallowed_actions"], ["mystery_repair"])

    def test_safe_recovery_policy_rejects_unrequested_safe_action(self) -> None:
        calls = {
            "index_status": _envelope(_status()),
            "repo_context_search": _envelope({"results": []}),
            "graph_status": _envelope(
                {"state": "ready", "available": True, "expansion_allowed": True}
            ),
        }

        with patch.object(
            smoke.ensure,
            "ensure_ready",
            return_value=_ready(auto_recovery_attempted=["start_watcher"]),
        ), patch.object(
            smoke,
            "_call_tool_result",
            side_effect=lambda tool_name, **_: calls[tool_name],
        ):
            result = smoke.run_smoke(repo="C:/repo")

        policy = next(item for item in result["scenarios"] if item["name"] == "safe_recovery_action_policy")
        self.assertFalse(result["ok"])
        self.assertEqual(policy["code"], "disallowed_safe_recovery_action")
        self.assertEqual(policy["data"]["allowed_recovery_actions"], [])
        self.assertEqual(policy["data"]["disallowed_actions"], ["start_watcher"])

    def test_permission_required_without_actions_is_failure(self) -> None:
        calls = {
            "index_status": _envelope(_status()),
            "repo_context_search": _envelope({"results": []}),
            "graph_status": _envelope(
                {"state": "missing", "available": False, "expansion_allowed": False}
            ),
        }

        with patch.object(
            smoke.ensure,
            "ensure_ready",
            return_value=_ready(
                ok=False,
                readiness="needs_user_permission",
                permission_required=True,
                permission_required_actions=[],
            ),
        ), patch.object(
            smoke,
            "_call_tool_result",
            side_effect=lambda tool_name, **_: calls[tool_name],
        ):
            result = smoke.run_smoke(repo="C:/repo")

        scenario = next(
            item for item in result["scenarios"] if item["name"] == "permission_required_actions_present"
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["exit_code"], 3)
        self.assertEqual(scenario["code"], "permission_required_without_actions")

    def test_safe_recovery_diagnostics_include_watcher_retry_state(self) -> None:
        calls = {
            "index_status": _envelope(_status()),
            "repo_context_search": _envelope({"results": []}),
            "graph_status": _envelope(
                {"state": "ready", "available": True, "expansion_allowed": True}
            ),
        }

        with patch.object(
            smoke.ensure,
            "ensure_ready",
            return_value=_ready(
                readiness="usable_degraded",
                auto_recovery_attempted=["start_watcher"],
                auto_recovery_idempotency_keys={"start_watcher": "token=abc"},
                watcher_start_in_progress=True,
                watcher_start_confirmed_after_transport_error=False,
            ),
        ), patch.object(
            smoke,
            "_call_tool_result",
            side_effect=lambda tool_name, **_: calls[tool_name],
        ):
            result = smoke.run_smoke(repo="C:/repo", allow_start_watcher=True)

        ensure_scenario = next(item for item in result["scenarios"] if item["name"] == "ensure_ready")
        policy_scenario = next(item for item in result["scenarios"] if item["name"] == "safe_recovery_action_policy")
        self.assertTrue(result["ok"])
        self.assertTrue(ensure_scenario["data"]["watcher_start_in_progress"])
        self.assertEqual(
            policy_scenario["data"]["auto_recovery_idempotency_keys"]["start_watcher"],
            "token=<redacted>",
        )
        self.assertNotIn("token=abc", str(result))

    def test_repo_mismatch_is_machine_readable_failure(self) -> None:
        calls = {
            "index_status": _envelope(_status(repo_root="C:/other")),
            "repo_context_search": _envelope({"results": []}),
            "graph_status": _envelope(
                {"state": "ready", "available": True, "expansion_allowed": True}
            ),
        }

        with patch.object(smoke.ensure, "ensure_ready", return_value=_ready()), patch.object(
            smoke,
            "_call_tool_result",
            side_effect=lambda tool_name, **_: calls[tool_name],
        ):
            result = smoke.run_smoke(repo="C:/repo")

        repo_scenario = next(item for item in result["scenarios"] if item["name"] == "repo_root_match")
        self.assertFalse(result["ok"])
        self.assertEqual(result["exit_code"], 1)
        self.assertEqual(repo_scenario["code"], "repo_root_mismatch")

    def test_permission_required_readiness_uses_exit_code_three(self) -> None:
        calls = {
            "index_status": _envelope(_status()),
            "repo_context_search": _envelope({"results": []}),
            "graph_status": _envelope(
                {"state": "missing", "available": False, "expansion_allowed": False}
            ),
        }
        permission_action = {"code": "build_graph", "requires_user_permission": True}

        with patch.object(
            smoke.ensure,
            "ensure_ready",
            return_value=_ready(
                ok=False,
                readiness="needs_user_permission",
                permission_required=True,
                permission_required_actions=[permission_action],
            ),
        ), patch.object(
            smoke,
            "_call_tool_result",
            side_effect=lambda tool_name, **_: calls[tool_name],
        ):
            result = smoke.run_smoke(repo="C:/repo", require_graph=True)

        self.assertFalse(result["ok"])
        self.assertEqual(result["exit_code"], 3)
        self.assertTrue(result["permission_required"])
        self.assertEqual(result["permission_required_actions"], [permission_action])

    def test_search_probe_is_skipped_when_search_unavailable(self) -> None:
        calls = {
            "index_status": _envelope(_status(search_available=False)),
            "graph_status": _envelope(
                {"state": "ready", "available": True, "expansion_allowed": True}
            ),
        }

        with patch.object(smoke.ensure, "ensure_ready", return_value=_ready()), patch.object(
            smoke,
            "_call_tool_result",
            side_effect=lambda tool_name, **_: calls[tool_name],
        ):
            result = smoke.run_smoke(repo="C:/repo")

        probe = next(item for item in result["scenarios"] if item["name"] == "repo_context_search_probe")
        self.assertTrue(probe["skipped"])
        self.assertEqual(probe["code"], "skipped_search_unavailable")

    def test_search_probe_is_skipped_when_repo_mismatches(self) -> None:
        calls = {
            "index_status": _envelope(_status(repo_root="C:/other", search_available=True)),
            "graph_status": _envelope(
                {"state": "ready", "available": True, "expansion_allowed": True}
            ),
        }

        def call_tool(tool_name, **_):
            if tool_name == "repo_context_search":
                raise AssertionError("search probe must not run against a mismatched repo")
            return calls[tool_name]

        with patch.object(smoke.ensure, "ensure_ready", return_value=_ready()), patch.object(
            smoke,
            "_call_tool_result",
            side_effect=call_tool,
        ):
            result = smoke.run_smoke(repo="C:/repo")

        probe = next(item for item in result["scenarios"] if item["name"] == "repo_context_search_probe")
        search = next(item for item in result["scenarios"] if item["name"] == "search_available")
        self.assertFalse(result["ok"])
        self.assertFalse(search["ok"])
        self.assertEqual(search["code"], "repo_root_mismatch")
        self.assertTrue(probe["skipped"])
        self.assertEqual(probe["code"], "skipped_repo_mismatch")

    def test_redacts_sensitive_dict_keys_and_urls(self) -> None:
        calls = {
            "index_status": _envelope(_status()),
            "repo_context_search": _envelope({"results": []}),
            "graph_status": _envelope(
                {
                    "state": "ready",
                    "available": True,
                    "expansion_allowed": True,
                    "api_key": "sk-secret-value",
                }
            ),
        }
        permission_action = {
            "code": "build_graph",
            "requires_user_permission": True,
            "token": "secret-token-value",
        }

        with patch.object(
            smoke.ensure,
            "ensure_ready",
            return_value=_ready(
                ok=False,
                readiness="needs_user_permission",
                permission_required=True,
                permission_required_actions=[permission_action],
            ),
        ), patch.object(
            smoke,
            "_call_tool_result",
            side_effect=lambda tool_name, **_: calls[tool_name],
        ):
            result = smoke.run_smoke(
                repo="C:/repo",
                url="http://user:api_key=secret@example.local/mcp",
                require_graph=True,
            )

        dumped = str(result)
        self.assertNotIn("sk-secret-value", dumped)
        self.assertNotIn("secret-token-value", dumped)
        self.assertNotIn("api_key=secret", dumped)
        self.assertIn("<redacted>", dumped)

    def test_unexpected_exceptions_return_stable_payload(self) -> None:
        with patch.object(smoke.ensure, "ensure_ready", side_effect=RuntimeError("secret=boom")), patch.object(
            smoke,
            "_call_tool_result",
            side_effect=RuntimeError("token=tool-boom"),
        ):
            result = smoke.run_smoke(repo="C:/repo")

        ensure_scenario = next(item for item in result["scenarios"] if item["name"] == "ensure_ready")
        status_scenario = next(item for item in result["scenarios"] if item["name"] == "index_status_http_fallback")
        self.assertFalse(result["ok"])
        self.assertEqual(result["exit_code"], 1)
        self.assertEqual(ensure_scenario["code"], "blocked")
        self.assertEqual(status_scenario["code"], "unexpected_tool_call_exception")
        self.assertNotIn("secret=boom", str(result))
        self.assertNotIn("token=tool-boom", str(result))


if __name__ == "__main__":
    unittest.main()
