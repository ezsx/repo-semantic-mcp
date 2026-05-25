import json
import unittest
from unittest.mock import Mock, patch
import urllib.error

from scripts.runtime import repo_semantic_host_stdio_proxy as proxy
from scripts.agents import repo_semantic_call_tool


class HostStdioProxyTests(unittest.TestCase):
    def test_parse_sse_payload_returns_jsonrpc_message(self) -> None:
        raw = (
            "event: message\n"
            'data: {"jsonrpc":"2.0","id":1,"result":{"ok":true}}\n'
            "\n"
        )

        self.assertEqual(
            proxy._parse_sse_payload(raw),
            {"jsonrpc": "2.0", "id": 1, "result": {"ok": True}},
        )

    def test_notifications_are_kept_local(self) -> None:
        with patch.object(proxy, "_post_json_rpc") as post_json_rpc:
            response = proxy._handle_line(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "method": "notifications/initialized",
                        "params": {},
                    }
                )
            )

        self.assertIsNone(response)
        post_json_rpc.assert_not_called()

    def test_requests_are_forwarded_to_http_mcp(self) -> None:
        expected = {"jsonrpc": "2.0", "id": 3, "result": {"tools": []}}
        with patch.object(proxy, "_post_json_rpc", return_value=expected) as post_json_rpc:
            response = proxy._handle_line(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 3,
                        "method": "tools/list",
                        "params": {},
                    }
                )
            )

        self.assertEqual(response, expected)
        post_json_rpc.assert_called_once()

    def test_bad_json_returns_parse_error(self) -> None:
        response = proxy._handle_line("{bad json")

        self.assertEqual(response["jsonrpc"], "2.0")
        self.assertIsNone(response["id"])
        self.assertEqual(response["error"]["code"], -32700)


class RepoSemanticCallToolTests(unittest.TestCase):
    def test_call_tool_sse_parser_matches_proxy_parser(self) -> None:
        raw = (
            "event: message\n"
            'data: {"jsonrpc":"2.0","id":1,"result":{"content":[]}}\n'
            "\n"
        )

        self.assertEqual(
            repo_semantic_call_tool._parse_sse_payload(raw),
            {"jsonrpc": "2.0", "id": 1, "result": {"content": []}},
        )

    def test_extract_tool_text_decodes_json_text_content(self) -> None:
        response = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": '{"repo_root":"C:/repo","search_available":true}',
                    }
                ]
            },
        }

        self.assertEqual(
            repo_semantic_call_tool._extract_tool_text(response),
            {"repo_root": "C:/repo", "search_available": True},
        )

    def test_call_tool_result_wraps_transport_error(self) -> None:
        with patch.object(
            repo_semantic_call_tool,
            "call_tool",
            side_effect=urllib.error.URLError("down"),
        ):
            result = repo_semantic_call_tool.call_tool_result("index_status")

        self.assertFalse(result["ok"])
        self.assertEqual(result["transport"], "http_fallback")
        self.assertEqual(result["tool_name"], "index_status")
        self.assertEqual(result["error_code"], "backend_unavailable")
        self.assertTrue(result["retryable"])

    def test_call_tool_result_wraps_jsonrpc_error(self) -> None:
        response = {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32000, "message": "boom"},
        }
        with patch.object(repo_semantic_call_tool, "call_tool", return_value=response):
            result = repo_semantic_call_tool.call_tool_result("index_status")

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], response["error"])
        self.assertEqual(result["error_code"], "tool_error")
        self.assertFalse(result["retryable"])

    def test_call_tool_result_marks_bootstrap_tool_error_retryable(self) -> None:
        response = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "isError": True,
                "content": [
                    {
                        "type": "text",
                        "text": "Semantic MCP is still bootstrapping (phase 'loading_runtime_status')",
                    }
                ],
            },
        }
        with patch.object(repo_semantic_call_tool, "call_tool", return_value=response):
            result = repo_semantic_call_tool.call_tool_result("index_status")

        self.assertFalse(result["ok"])
        self.assertEqual(result["error_code"], "backend_bootstrapping")
        self.assertTrue(result["retryable"])

    def test_extract_tool_text_preserves_multiple_text_blocks(self) -> None:
        response = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [
                    {"type": "text", "text": '{"path":"a.py"}'},
                    {"type": "text", "text": '{"path":"b.py"}'},
                ]
            },
        }

        self.assertEqual(
            repo_semantic_call_tool._extract_tool_text(response),
            [{"path": "a.py"}, {"path": "b.py"}],
        )

    def test_http_proxy_does_not_retry_client_errors(self) -> None:
        error = urllib.error.HTTPError(
            "http://127.0.0.1:8011/mcp",
            404,
            "Not Found",
            {},
            None,
        )
        error.read = Mock(return_value=b'{"error":"missing"}')
        with patch.object(proxy, "_HTTP_OPENER") as opener:
            opener.open.side_effect = error

            response = proxy._post_json_rpc({"jsonrpc": "2.0", "id": 7, "method": "tools/list"})

        self.assertEqual(opener.open.call_count, 1)
        self.assertEqual(response["id"], 7)
        self.assertIn("HTTP 404", response["error"]["message"])
        self.assertIn("missing", response["error"]["message"])


if __name__ == "__main__":
    unittest.main()
