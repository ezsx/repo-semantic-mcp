#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-repo-semantic-mcp}"
CONTAINER_START_TIMEOUT_SEC="${CONTAINER_START_TIMEOUT_SEC:-120}"
READY_TIMEOUT_SEC="${READY_TIMEOUT_SEC:-900}"

fail() {
  printf '%s\n' "$1" >&2
  exit 1
}

container_exists() {
  docker inspect "$1" >/dev/null 2>&1
}

ensure_container_running() {
  local name="$1"
  local timeout_sec="$2"
  local deadline
  deadline=$((SECONDS + timeout_sec))

  if ! container_exists "$name"; then
    return 1
  fi

  if [[ "$(docker inspect --format '{{.State.Running}}' "$name" 2>/dev/null)" == "true" ]]; then
    return 0
  fi

  docker start "$name" >/dev/null 2>&1 || true

  while (( SECONDS < deadline )); do
    if [[ "$(docker inspect --format '{{.State.Running}}' "$name" 2>/dev/null)" == "true" ]]; then
      return 0
    fi
    sleep 2
  done

  return 1
}

get_container_env_value() {
  local name="$1"
  local key="$2"

  docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$name" 2>/dev/null | \
    awk -F= -v key="$key" '$1 == key { print substr($0, index($0, "=") + 1); exit }'
}

wait_ready() {
  local base_url="$1"
  local timeout_sec="$2"
  local deadline
  deadline=$((SECONDS + timeout_sec))

  while (( SECONDS < deadline )); do
    if payload="$(curl --silent --show-error --noproxy '*' --fail "${base_url}/readyz" 2>/dev/null)" && [[ -n "$payload" ]]; then
      if python3 - <<'PY' "$payload"
import json
import sys

payload = json.loads(sys.argv[1])
sys.exit(0 if payload.get("ready") is True else 1)
PY
      then
        return 0
      fi
    fi
    sleep 2
  done

  fail "repo-semantic-search container did not become ready within ${timeout_sec} seconds."
}

for dependency in repo-semantic-qdrant repo-semantic-tei; do
  ensure_container_running "$dependency" 30 || true
done

if ! ensure_container_running "$CONTAINER_NAME" "$CONTAINER_START_TIMEOUT_SEC"; then
  fail "repo-semantic-search container '$CONTAINER_NAME' is missing or did not start. Start it explicitly before using the Codex wrapper."
fi

HTTP_PORT="$(get_container_env_value "$CONTAINER_NAME" SEMANTIC_MCP_HTTP_PORT)"
HTTP_PORT="${HTTP_PORT:-8011}"

wait_ready "http://127.0.0.1:${HTTP_PORT}" "$READY_TIMEOUT_SEC"

exec docker exec \
  -i \
  -e "SEMANTIC_MCP_PROXY_URL=http://127.0.0.1:${HTTP_PORT}/mcp" \
  "$CONTAINER_NAME" \
  python3 /repo/scripts/runtime/repo_semantic_stdio_proxy.py
