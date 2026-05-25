#!/usr/bin/env bash
set -euo pipefail

PROFILE="${PROFILE:-gpu}"
REPO_PATH="${REPO_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
QUERY="${QUERY:-repo semantic backend registry}"
BUILD_FLAG="${BUILD_FLAG:-0}"
CLEAN_FLAG="${CLEAN_FLAG:-0}"
SKIP_ENSURE="${SKIP_ENSURE:-0}"
CONTAINER_NAME="${CONTAINER_NAME:-repo-semantic-mcp}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_SMOKE_SCRIPT="$(cd "$SCRIPT_DIR/../runtime" && pwd)/repo_semantic_boot_smoke.py"

if [[ "$SKIP_ENSURE" != "1" ]]; then
  ENSURE_ARGS=( -Profile "$PROFILE" -TargetRepoPath "$REPO_PATH" )
  if [[ "$BUILD_FLAG" == "1" ]]; then
    ENSURE_ARGS+=( -Build )
  fi
  if [[ "$CLEAN_FLAG" == "1" ]]; then
    ENSURE_ARGS+=( -Clean )
  fi
  pwsh -NoLogo -NoProfile -File "$SCRIPT_DIR/ensure_repo_semantic_search.ps1" "${ENSURE_ARGS[@]}"
fi

HTTP_PORT="$(docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$CONTAINER_NAME" 2>/dev/null | awk -F= '/^SEMANTIC_MCP_HTTP_PORT=/{print $2; exit}')"
if [[ -z "$HTTP_PORT" ]]; then
  HTTP_PORT="8011"
fi

BASE_URL="http://127.0.0.1:${HTTP_PORT}"

docker exec -i "$CONTAINER_NAME" python3 - --base-url "$BASE_URL" --query "$QUERY" < "$RUNTIME_SMOKE_SCRIPT"
