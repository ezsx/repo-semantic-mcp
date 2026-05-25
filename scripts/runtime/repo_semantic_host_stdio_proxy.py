"""Host-side stdio proxy for the repo-semantic-search HTTP MCP endpoint.

The Codex app starts custom MCP servers as long-lived stdio processes. Running
that process through ``docker exec`` ties the chat transport to the lifetime of
the backend container: rebuilding or restarting the container kills the exec
process and the chat starts seeing "Transport closed".

This proxy stays on the host side and forwards newline-delimited MCP JSON-RPC
messages to the streamable HTTP endpoint. It intentionally uses only the Python
standard library so the Codex wrapper does not need the MCP SDK installed on the
host.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any


DEFAULT_PROXY_URL = "http://127.0.0.1:8011/mcp"
DEFAULT_REQUEST_TIMEOUT_SEC = 900
DEFAULT_RETRY_WINDOW_SEC = 120
DEFAULT_RETRY_SLEEP_SEC = 2
_HTTP_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _jlog(level: str, event: str, **fields: object) -> None:
    """Write structured logs to stderr so stdout stays MCP-clean."""

    payload = {
        "level": level,
        "event": event,
        "service": "repo-semantic-host-stdio-proxy",
        **fields,
    }
    print(json.dumps(payload, ensure_ascii=False, default=str), file=sys.stderr, flush=True)


def _proxy_url() -> str:
    """Return the configured HTTP MCP endpoint."""

    return os.getenv("SEMANTIC_MCP_PROXY_URL", DEFAULT_PROXY_URL).strip() or DEFAULT_PROXY_URL


def _request_timeout_sec() -> float:
    """Return per-request timeout for HTTP MCP calls."""

    raw = os.getenv("SEMANTIC_MCP_PROXY_REQUEST_TIMEOUT_SEC", "").strip()
    if not raw:
        return float(DEFAULT_REQUEST_TIMEOUT_SEC)
    try:
        return max(1.0, float(raw))
    except ValueError:
        return float(DEFAULT_REQUEST_TIMEOUT_SEC)


def _retry_window_sec() -> float:
    """Return how long to wait for transient backend restarts."""

    raw = os.getenv("SEMANTIC_MCP_PROXY_RETRY_WINDOW_SEC", "").strip()
    if not raw:
        return float(DEFAULT_RETRY_WINDOW_SEC)
    try:
        return max(0.0, float(raw))
    except ValueError:
        return float(DEFAULT_RETRY_WINDOW_SEC)


def _jsonrpc_error(request_id: object | None, code: int, message: str) -> dict[str, object]:
    """Build a JSON-RPC error response."""

    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": code,
            "message": message,
        },
    }


def _parse_sse_payload(raw: str) -> dict[str, Any] | None:
    """Parse the first JSON data event from a streamable HTTP SSE response."""

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
        try:
            return json.loads("\n".join(data_lines))
        except json.JSONDecodeError:
            return None
    return None


def _decode_http_response(response: urllib.response.addinfourl) -> dict[str, Any] | None:
    """Decode application/json or text/event-stream MCP response bodies."""

    body = response.read().decode("utf-8", errors="replace")
    content_type = response.headers.get("Content-Type", "")
    if "text/event-stream" in content_type:
        return _parse_sse_payload(body)
    if body.strip():
        return json.loads(body)
    return None


def _read_http_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return ""


def _post_json_rpc(message: dict[str, Any]) -> dict[str, Any] | None:
    """Forward one JSON-RPC message to the HTTP MCP endpoint with retry."""

    encoded = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        _proxy_url(),
        data=encoded,
        method="POST",
        headers={
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        },
    )
    deadline = time.monotonic() + _retry_window_sec()
    last_error: Exception | None = None
    while True:
        try:
            with _HTTP_OPENER.open(request, timeout=_request_timeout_sec()) as response:
                return _decode_http_response(response)
        except urllib.error.HTTPError as exc:
            body = _read_http_error_body(exc)
            detail = body.strip() or str(exc)
            if 400 <= exc.code < 500:
                return _jsonrpc_error(
                    message.get("id"),
                    -32000,
                    f"repo-semantic-search HTTP proxy returned HTTP {exc.code}: {detail}",
                )
            last_error = urllib.error.HTTPError(
                exc.url,
                exc.code,
                detail,
                exc.headers,
                None,
            )
            if time.monotonic() >= deadline:
                break
            time.sleep(DEFAULT_RETRY_SLEEP_SEC)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
            if time.monotonic() >= deadline:
                break
            time.sleep(DEFAULT_RETRY_SLEEP_SEC)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            break

    request_id = message.get("id")
    detail = str(last_error) if last_error is not None else "unknown HTTP MCP proxy error"
    return _jsonrpc_error(request_id, -32000, f"repo-semantic-search HTTP proxy unavailable: {detail}")


def _handle_line(line: str) -> dict[str, Any] | None:
    """Handle one newline-delimited MCP JSON-RPC input line."""

    try:
        message = json.loads(line)
    except json.JSONDecodeError as exc:
        return _jsonrpc_error(None, -32700, f"Parse error: {exc}")

    if not isinstance(message, dict):
        return _jsonrpc_error(None, -32600, "Invalid JSON-RPC message")

    # Notifications do not require a response. The remote FastMCP endpoint is
    # stateless for the notifications Codex sends during initialization, so
    # keeping them local avoids unnecessary 202/no-body handling.
    if "id" not in message:
        return None

    return _post_json_rpc(message)


def main() -> int:
    """Run newline-delimited JSON-RPC stdio proxy."""

    _jlog("info", "semantic_host_stdio_proxy_ready", remote_url=_proxy_url())
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        response = _handle_line(line)
        if response is None:
            continue
        sys.stdout.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
