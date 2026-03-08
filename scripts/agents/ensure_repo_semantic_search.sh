#!/usr/bin/env bash
set -euo pipefail

BUILD=0
CLEAN=0
PROFILE=""
ENV_FILE=""
TARGET_REPO_PATH=""
TIMEOUT_SEC=1800

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPLOY_DIR="$REPO_ROOT/deploy/repo-semantic-search"
COMPOSE_FILE="$DEPLOY_DIR/docker-compose.repo-semantic-search.yml"
COMPOSE_GPU_FILE="$DEPLOY_DIR/docker-compose.repo-semantic-search.gpu.yml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build)
      BUILD=1
      shift
      ;;
    --clean)
      CLEAN=1
      shift
      ;;
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --target-repo-path)
      TARGET_REPO_PATH="$2"
      shift 2
      ;;
    --timeout-sec)
      TIMEOUT_SEC="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$PROFILE" ]]; then
  PROFILE="cpu"
fi

resolve_existing_path() {
  for candidate in "$@"; do
    if [[ -n "$candidate" && -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

wait_docker_ready() {
  local start_ts
  start_ts="$(date +%s)"
  while true; do
    if docker version >/dev/null 2>&1; then
      return 0
    fi
    if (( "$(date +%s)" - start_ts >= 120 )); then
      echo "Docker daemon did not become ready within 120 seconds." >&2
      exit 1
    fi
    sleep 3
  done
}

test_semantic_mcp_protocol() {
  docker exec -i repo-semantic-mcp python - >/dev/null 2>&1 <<'PY'
import anyio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def main():
    async with streamablehttp_client("http://127.0.0.1:8011/mcp") as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            await session.list_tools()

anyio.run(main)
PY
}

if [[ -z "$ENV_FILE" ]]; then
  case "$PROFILE" in
    cpu)
      ENV_FILE="$(resolve_existing_path "$DEPLOY_DIR/.env" "$DEPLOY_DIR/.env.example" || true)"
      ;;
    gpu)
      ENV_FILE="$(resolve_existing_path "$DEPLOY_DIR/.env.gpu" "$DEPLOY_DIR/.env.gpu.example" "$DEPLOY_DIR/.env.gpu.qwen3" "$DEPLOY_DIR/.env.gpu.qwen3.example" || true)"
      ;;
    gpu-qwen3)
      ENV_FILE="$(resolve_existing_path "$DEPLOY_DIR/.env.gpu.qwen3" "$DEPLOY_DIR/.env.gpu" "$DEPLOY_DIR/.env.gpu.qwen3.example" "$DEPLOY_DIR/.env.gpu.example" || true)"
      ;;
    gpu-bge-m3)
      ENV_FILE="$(resolve_existing_path "$DEPLOY_DIR/.env.gpu.bge-m3" "$DEPLOY_DIR/.env.gpu.bge-m3.example" || true)"
      ;;
    *)
      echo "Unsupported profile: $PROFILE" >&2
      exit 1
      ;;
  esac
fi

wait_docker_ready

if [[ -n "$TARGET_REPO_PATH" ]]; then
  RESOLVED_TARGET_REPO="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$TARGET_REPO_PATH")"
  export SEMANTIC_MCP_TARGET_REPO_PATH="$RESOLVED_TARGET_REPO"
  export SEMANTIC_MCP_REPO_ROOT="/target_repo"
fi

COMPOSE_ARGS=(compose -f "$COMPOSE_FILE")
if [[ "$PROFILE" != "cpu" ]]; then
  COMPOSE_ARGS+=(-f "$COMPOSE_GPU_FILE")
fi
if [[ -n "$ENV_FILE" ]]; then
  COMPOSE_ARGS+=(--env-file "$ENV_FILE")
fi

echo "profile: $PROFILE"
if [[ -n "$ENV_FILE" ]]; then
  echo "env file: $ENV_FILE"
fi
if [[ -n "$TARGET_REPO_PATH" ]]; then
  echo "target repo: $RESOLVED_TARGET_REPO"
else
  echo "target repo: compose default"
fi

if [[ "$CLEAN" == "1" ]]; then
  docker "${COMPOSE_ARGS[@]}" down --remove-orphans
fi

UP_ARGS=("${COMPOSE_ARGS[@]}" up -d)
if [[ "$BUILD" == "1" ]]; then
  UP_ARGS+=(--build)
fi
docker "${UP_ARGS[@]}"

START_TS="$(date +%s)"
while true; do
  STATUS="$(docker ps --filter 'name=^repo-semantic-mcp$' --format '{{.Status}}')"
  if [[ "$STATUS" == Up* ]]; then
    if test_semantic_mcp_protocol; then
      echo "repo-semantic-search ready: MCP protocol is responding on http://127.0.0.1:8011/mcp"
      exit 0
    fi
  fi
  if (( "$(date +%s)" - START_TS >= TIMEOUT_SEC )); then
    echo "repo-semantic-search did not become ready within ${TIMEOUT_SEC} seconds." >&2
    exit 1
  fi
  sleep 5
done
