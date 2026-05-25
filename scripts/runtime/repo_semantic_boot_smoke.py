"""Cold-boot acceptance smoke для repo-semantic-search."""

from __future__ import annotations

import argparse
import json
from typing import Any

import anyio
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client


def _extract_payload(result: Any) -> Any:
    """Извлечь structured payload из результата MCP tool-call."""

    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return structured
    texts: list[str] = []
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", None)
        if text:
            texts.append(text)
    if not texts:
        return None
    raw = "\n".join(texts)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


async def _run_smoke(base_url: str, query: str) -> dict[str, Any]:
    """Прогнать minimal product smoke через MCP tools."""

    async with streamablehttp_client(f"{base_url.rstrip('/')}/mcp") as (reader, writer, _):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            index_status = _extract_payload(await session.call_tool("index_status", {}))
            backend_status = _extract_payload(await session.call_tool("get_backend_status", {"role": "embedding"}))
            backend_ensure = _extract_payload(await session.call_tool("ensure_backend", {"role": "embedding"}))
            search_result = _extract_payload(
                await session.call_tool(
                    "semantic_search",
                    {
                        "query": query,
                        "top_k": 1,
                    },
                )
            )

    if not isinstance(index_status, dict):
        raise RuntimeError(f"Unexpected index_status payload: {index_status!r}")
    if not index_status.get("search_available"):
        raise RuntimeError(
            "index_status reports search_available=false: "
            f"{index_status.get('reason_if_unavailable')!r}"
        )

    if not isinstance(backend_status, dict):
        raise RuntimeError(f"Unexpected get_backend_status payload: {backend_status!r}")
    if backend_status.get("healthy") is not True:
        raise RuntimeError(
            "get_backend_status reports unhealthy backend: "
            f"{backend_status.get('health_error')!r}"
        )

    if not isinstance(backend_ensure, dict):
        raise RuntimeError(f"Unexpected ensure_backend payload: {backend_ensure!r}")
    if not backend_ensure.get("ensured"):
        raise RuntimeError(
            "ensure_backend did not return ensured=true: "
            f"{backend_ensure.get('operator_action_state')!r}"
        )

    if isinstance(search_result, str):
        parsed_search = json.loads(search_result)
    else:
        parsed_search = search_result
    if isinstance(parsed_search, dict):
        parsed_search = [parsed_search]
    if not isinstance(parsed_search, list):
        raise RuntimeError(f"Unexpected semantic_search payload: {parsed_search!r}")

    top_hit = parsed_search[0] if parsed_search else None
    return {
        "base_url": base_url,
        "index_status": {
            "repo_root": index_status.get("repo_root"),
            "active_repo_root": index_status.get("active_repo_root"),
            "repo_state": index_status.get("repo_state"),
            "search_available": index_status.get("search_available"),
            "backend_switch_required": index_status.get("backend_switch_required"),
        },
        "backend_status": {
            "backend_id": backend_status.get("entry", {}).get("backend_id"),
            "launch_mode": backend_status.get("launch_mode"),
            "preferred_profile": backend_status.get("preferred_profile"),
            "healthy": backend_status.get("healthy"),
            "runtime_switch_required": backend_status.get("runtime_switch_required"),
        },
        "ensure_backend": {
            "ensured": backend_ensure.get("ensured"),
            "operator_action_state": backend_ensure.get("operator_action_state"),
        },
        "semantic_search": {
            "query": query,
            "hits": len(parsed_search),
            "top_hit_path": top_hit.get("relative_path") if isinstance(top_hit, dict) else None,
        },
    }


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8011")
    parser.add_argument("--query", default="repo semantic backend registry")
    args = parser.parse_args()

    result = anyio.run(_run_smoke, args.base_url, args.query)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
