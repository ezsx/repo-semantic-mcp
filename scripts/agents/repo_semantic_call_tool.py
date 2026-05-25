"""Call repo-semantic-search MCP tools through the local HTTP endpoint.

This is a host-side fallback for Codex sessions where the custom MCP server did
not make it into the active toolset. It keeps agents on the repo-semantic-search
path instead of falling all the way back to plain grep. CLI output is a stable
ok/error envelope by default; pass --payload for the decoded tool payload only.

Examples:
    py -3.12 scripts/agents/repo_semantic_call_tool.py index_status
    py -3.12 scripts/agents/repo_semantic_call_tool.py repo_context_search --args-json "{\"query\":\"connect flow\",\"top_k\":5}"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any


DEFAULT_URL = (
    os.getenv("SEMANTIC_MCP_PROXY_URL")
    or os.getenv("SEMANTIC_MCP_URL")
    or "http://127.0.0.1:8011/mcp"
)
_HTTP_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _parse_sse_payload(raw: str) -> dict[str, Any] | None:
    data_lines: list[str] = []
    for line in raw.splitlines():
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
            continue
        if line == "" and data_lines:
            payload = "\n".join(data_lines)
            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                data_lines = []
    if data_lines:
        return json.loads("\n".join(data_lines))
    return None


def _post_json_rpc(url: str, message: dict[str, Any], timeout_sec: int) -> dict[str, Any]:
    data = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        },
    )
    with _HTTP_OPENER.open(request, timeout=timeout_sec) as response:
        body = response.read().decode("utf-8", errors="replace")
        content_type = response.headers.get("Content-Type", "")
    if "text/event-stream" in content_type:
        parsed = _parse_sse_payload(body)
        if parsed is None:
            raise RuntimeError("MCP HTTP response did not contain an SSE data payload")
        return parsed
    return json.loads(body)


def call_json_rpc(
    method: str,
    *,
    params: dict[str, Any] | None = None,
    url: str = DEFAULT_URL,
    timeout_sec: int = 900,
) -> dict[str, Any]:
    """Call the local repo-semantic-search HTTP MCP endpoint."""

    return _post_json_rpc(
        url,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params or {},
        },
        timeout_sec,
    )


def call_tool(
    tool_name: str,
    *,
    arguments: dict[str, Any] | None = None,
    url: str = DEFAULT_URL,
    timeout_sec: int = 900,
    raw: bool = False,
) -> Any:
    """Call one MCP tool and return its decoded text payload by default."""

    response = call_json_rpc(
        "tools/call",
        params={
            "name": tool_name,
            "arguments": arguments or {},
        },
        url=url,
        timeout_sec=timeout_sec,
    )
    return response if raw else _extract_tool_text(response)


def _error_code_for_exception(exc: Exception) -> tuple[str, bool]:
    if isinstance(exc, TimeoutError):
        return "timeout", True
    if isinstance(exc, urllib.error.HTTPError):
        if 500 <= exc.code < 600:
            return "backend_unavailable", True
        return "http_error", False
    if isinstance(exc, urllib.error.URLError):
        return "backend_unavailable", True
    if isinstance(exc, json.JSONDecodeError):
        return "invalid_json", False
    return "tool_call_failed", False


def _error_code_for_jsonrpc_error(error: Any) -> str:
    if not isinstance(error, dict):
        message = str(error).lower()
        if "bootstrapping" in message or "loading_runtime_status" in message:
            return "backend_bootstrapping"
        return "tool_error"
    message = str(error.get("message") or "").lower()
    data = error.get("data")
    code = data.get("code") if isinstance(data, dict) else None
    if isinstance(code, str) and code:
        return code
    if "bootstrapping" in message or "loading_runtime_status" in message:
        return "backend_bootstrapping"
    if "unknown tool" in message or "not found" in message or "not available" in message:
        return "tool_unavailable"
    if "permission" in message or "lifecycle" in message:
        return "permission_required"
    if "repo" in message and "mismatch" in message:
        return "repo_mismatch"
    if "invalid" in message and "json" in message:
        return "invalid_json_payload"
    return "tool_error"


def _jsonrpc_error_retryable(error: Any) -> bool:
    """Return whether a JSON-RPC/tool error is worth retrying soon."""

    if isinstance(error, dict):
        message = json.dumps(error, ensure_ascii=False).lower()
    else:
        message = str(error).lower()
    return "bootstrapping" in message or "loading_runtime_status" in message


def call_tool_result(
    tool_name: str,
    *,
    arguments: dict[str, Any] | None = None,
    url: str = DEFAULT_URL,
    timeout_sec: int = 900,
) -> dict[str, Any]:
    """Call one MCP tool and return a stable agent-facing envelope."""

    try:
        raw_response = call_tool(
            tool_name,
            arguments=arguments or {},
            url=url,
            timeout_sec=timeout_sec,
            raw=True,
        )
        payload = _extract_tool_text(raw_response)
    except Exception as exc:  # noqa: BLE001
        error_code, retryable = _error_code_for_exception(exc)
        return {
            "ok": False,
            "transport": "http_fallback",
            "url": url,
            "tool_name": tool_name,
            "result": None,
            "error": {"message": str(exc), "type": type(exc).__name__},
            "retryable": retryable,
            "error_code": error_code,
        }

    error = raw_response.get("error") if isinstance(raw_response, dict) else None
    is_error_result = bool(
        isinstance(raw_response, dict)
        and isinstance(raw_response.get("result"), dict)
        and raw_response["result"].get("isError")
    )
    ok = error is None and not is_error_result
    error_payload = error or (payload if is_error_result else None)
    error_code = None if ok else _error_code_for_jsonrpc_error(error_payload)
    return {
        "ok": ok,
        "transport": "http_fallback",
        "url": url,
        "tool_name": tool_name,
        "result": payload if ok else None,
        "error": error_payload,
        "retryable": False if ok else _jsonrpc_error_retryable(error_payload),
        "error_code": error_code,
    }


def list_tools(*, url: str = DEFAULT_URL, timeout_sec: int = 900, raw: bool = False) -> Any:
    """List MCP tools through the HTTP fallback endpoint."""

    response = call_json_rpc("tools/list", url=url, timeout_sec=timeout_sec)
    return response if raw else response.get("result", response)


def _extract_tool_text(response: dict[str, Any]) -> Any:
    if "error" in response:
        return response
    result = response.get("result")
    if not isinstance(result, dict):
        return response
    content = result.get("content")
    if not isinstance(content, list) or not content:
        return result
    text_items = [
        item.get("text")
        for item in content
        if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str)
    ]
    if not text_items:
        return result
    decoded_items: list[Any] = []
    for text in text_items:
        try:
            decoded_items.append(json.loads(text))
        except json.JSONDecodeError:
            decoded_items.append(text)
    return decoded_items[0] if len(decoded_items) == 1 else decoded_items


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tool_name")
    parser.add_argument("--args-json", default="{}", help="JSON object with MCP tool arguments.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print the raw JSON-RPC response instead of the tool text payload.",
    )
    parser.add_argument(
        "--payload",
        action="store_true",
        help="Print the decoded tool payload without the stable ok/error envelope.",
    )
    parser.add_argument(
        "--envelope",
        action="store_true",
        help="Deprecated no-op: the stable agent-facing envelope is the default.",
    )
    ns = parser.parse_args()

    try:
        arguments = json.loads(ns.args_json)
    except json.JSONDecodeError as exc:
        print(f"Invalid --args-json: {exc}", file=sys.stderr)
        return 2
    if not isinstance(arguments, dict):
        print("--args-json must decode to a JSON object", file=sys.stderr)
        return 2

    if not ns.raw and not ns.payload:
        envelope = call_tool_result(
            ns.tool_name,
            arguments=arguments,
            url=ns.url,
            timeout_sec=ns.timeout_sec,
        )
        print(json.dumps(envelope, ensure_ascii=False, indent=2))
        return 0 if envelope["ok"] else 1

    try:
        response = call_tool(
            ns.tool_name,
            arguments=arguments,
            url=ns.url,
            timeout_sec=ns.timeout_sec,
            raw=True,
        )
        payload = response if ns.raw else _extract_tool_text(response)
    except Exception as exc:  # noqa: BLE001
        error_code, retryable = _error_code_for_exception(exc)
        payload = {
            "ok": False,
            "transport": "http_fallback",
            "url": ns.url,
            "tool_name": ns.tool_name,
            "result": None,
            "error": {"message": str(exc), "type": type(exc).__name__},
            "retryable": retryable,
            "error_code": error_code,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if "error" not in response else 1


if __name__ == "__main__":
    raise SystemExit(main())
