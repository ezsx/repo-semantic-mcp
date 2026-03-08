#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${1:-repo-semantic-mcp}"

if ! docker ps --filter "name=^${CONTAINER_NAME}$" --format '{{.Status}}' | grep -q '^Up'; then
  echo "Container '${CONTAINER_NAME}' is not running." >&2
  exit 1
fi

docker exec -i "$CONTAINER_NAME" python - <<'PY'
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
PY
