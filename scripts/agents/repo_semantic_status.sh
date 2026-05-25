#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${1:-repo-semantic-mcp}"
BASE_URL="${2:-}"

if ! docker ps --filter "name=^${CONTAINER_NAME}$" --format '{{.Status}}' | grep -q '^Up'; then
  echo "Container '${CONTAINER_NAME}' is not running." >&2
  exit 1
fi

if [[ -z "$BASE_URL" ]]; then
  HTTP_PORT="$(docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$CONTAINER_NAME" 2>/dev/null \
    | awk -F= '$1 == "SEMANTIC_MCP_HTTP_PORT" { sub($1 "=",""); print; exit }')"
  HTTP_PORT="${HTTP_PORT:-8011}"
  BASE_URL="http://127.0.0.1:${HTTP_PORT}"
fi

python3 - "$BASE_URL" <<'PY'
import json
import sys
import urllib.request

base_url = sys.argv[1]
with urllib.request.urlopen(f"{base_url}/statusz", timeout=10) as response:
    payload = json.loads(response.read().decode("utf-8"))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
