"""Прогон benchmark по semantic MCP для набора поисковых запросов."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import anyio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


def _parse_args() -> argparse.Namespace:
    """Разобрать аргументы CLI."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--mcp-url", default="http://127.0.0.1:8011/mcp")
    parser.add_argument("--queries-file", required=True)
    parser.add_argument("--scope", default="all", choices=("all", "code", "docs"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--tool", default="semantic_search", choices=("semantic_search", "hybrid_search"))
    parser.add_argument("--label", default="run")
    parser.add_argument("--wait-for-index-sec", type=int, default=0)
    return parser.parse_args()


def _status_ready(status_payload: dict) -> bool:
    """Проверить, что индекс не пуст и готов для benchmark."""

    return bool(
        status_payload.get("code_collection_count", 0) > 0
        or status_payload.get("docs_collection_count", 0) > 0
    )


async def _wait_for_index(
    session: ClientSession,
    timeout_sec: int,
) -> dict:
    """Дождаться непустого индекса или вернуть последний статус."""

    deadline = time.perf_counter() + timeout_sec
    last_status: dict | None = None
    last_log_ts = 0.0
    while True:
        result = await session.call_tool("index_status", {})
        last_status = json.loads(result.content[0].text)
        if _status_ready(last_status) or timeout_sec <= 0 or time.perf_counter() >= deadline:
            return last_status
        now = time.perf_counter()
        if now - last_log_ts >= 10:
            print(
                "[benchmark] waiting for index:"
                f" code={last_status.get('code_collection_count', 0)}"
                f" docs={last_status.get('docs_collection_count', 0)}",
                file=sys.stderr,
                flush=True,
            )
            last_log_ts = now
        await anyio.sleep(2)


async def _run_benchmark(args: argparse.Namespace) -> dict:
    """Выполнить benchmark against live MCP server."""

    queries = json.loads(Path(args.queries_file).read_text(encoding="utf-8"))
    report: dict = {"label": args.label, "status": None, "results": []}
    async with streamablehttp_client(args.mcp_url) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            report["status"] = await _wait_for_index(session, args.wait_for_index_sec)

            for query in queries:
                start = time.perf_counter()
                result = await session.call_tool(
                    args.tool,
                    {"query": query["query"], "top_k": args.top_k, "scope": args.scope},
                )
                elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
                rows = [json.loads(block.text) for block in result.content]
                top_paths = [row["relative_path"] for row in rows]
                expected_any = set(query.get("expected_any", []))
                report["results"].append(
                    {
                        "name": query["name"],
                        "elapsed_ms": elapsed_ms,
                        "hit_top_k": any(path in expected_any for path in top_paths),
                        "top_paths": top_paths,
                    }
                )
    return report


def main() -> None:
    """Точка входа CLI benchmark."""

    args = _parse_args()
    print(json.dumps(anyio.run(_run_benchmark, args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
