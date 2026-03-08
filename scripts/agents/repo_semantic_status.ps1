param(
    [string]$ContainerName = "repo-semantic-mcp"
)

$ErrorActionPreference = "Stop"

$status = docker ps --filter "name=^$ContainerName$" --format "{{.Status}}"
if (-not $status) {
    throw "Container '$ContainerName' is not running."
}

$script = @'
import anyio
import json
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def main():
    async with streamablehttp_client("http://127.0.0.1:8011/mcp") as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool("index_status", {})
            if result.structuredContent is not None:
                print(json.dumps(result.structuredContent, ensure_ascii=False, indent=2))
                return
            if result.content:
                print(result.content[0].text)
                return
            print("null")

anyio.run(main)
'@

$script | docker exec -i $ContainerName python -
