"""Тонкий stdio proxy к уже запущенному HTTP runtime repo-semantic-search."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import anyio
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.stdio import stdio_server

# Обеспечить импорт repo-owned модулей при запуске через docker exec.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.repo_semantic.logging import jlog


def _remote_url() -> str:
    """Собрать URL HTTP runtime, к которому подключается stdio proxy."""

    explicit = os.getenv("SEMANTIC_MCP_PROXY_URL", "").strip()
    if explicit:
        return explicit
    port = os.getenv("SEMANTIC_MCP_HTTP_PORT", "8011").strip() or "8011"
    return f"http://127.0.0.1:{port}/mcp"


async def _serve_proxy() -> None:
    """Поднять stdio proxy и пробросить MCP calls в HTTP runtime."""

    remote_url = _remote_url()
    async with streamablehttp_client(remote_url) as (read_stream, write_stream, _get_session_id):
        async with ClientSession(read_stream, write_stream) as remote_session:
            init_result = await remote_session.initialize()
            proxy_server = Server(
                name=init_result.serverInfo.name,
                version=init_result.serverInfo.version,
                instructions=init_result.instructions,
            )

            @proxy_server.list_tools()
            async def handle_list_tools():
                return await remote_session.list_tools()

            @proxy_server.call_tool(validate_input=False)
            async def handle_call_tool(name: str, arguments: dict | None):
                return await remote_session.call_tool(name, arguments or {})

            @proxy_server.list_resources()
            async def handle_list_resources():
                return await remote_session.list_resources()

            @proxy_server.read_resource()
            async def handle_read_resource(uri):
                return await remote_session.read_resource(uri)

            @proxy_server.list_resource_templates()
            async def handle_list_resource_templates():
                return await remote_session.list_resource_templates()

            @proxy_server.list_prompts()
            async def handle_list_prompts():
                return await remote_session.list_prompts()

            @proxy_server.get_prompt()
            async def handle_get_prompt(name: str, arguments: dict[str, str] | None):
                return await remote_session.get_prompt(name, arguments)

            jlog("info", "semantic_stdio_proxy_ready", remote_url=remote_url)
            async with stdio_server() as (local_read_stream, local_write_stream):
                await proxy_server.run(
                    local_read_stream,
                    local_write_stream,
                    proxy_server.create_initialization_options(
                        notification_options=NotificationOptions()
                    ),
                )


def main() -> None:
    """CLI entrypoint для stdio proxy."""

    anyio.run(_serve_proxy)


if __name__ == "__main__":
    main()
