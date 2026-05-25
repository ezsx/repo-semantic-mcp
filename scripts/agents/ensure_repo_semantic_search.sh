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
  PROFILE="gpu"
fi

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

get_env_value() {
  local path="$1"
  local key="$2"
  [[ -n "$path" && -f "$path" ]] || return 1

  python3 - "$path" "$key" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
key = sys.argv[2]

for raw in path.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or "=" not in raw:
        continue
    current, value = raw.split("=", 1)
    if current.strip() == key:
        print(value.strip().strip("'\""))
        break
PY
}

get_merged_env_value() {
  local json_payload="$1"
  local key="$2"
  [[ -n "$json_payload" ]] || return 1

  python3 - "$key" "$json_payload" <<'PY'
import json
import sys

payload = json.loads(sys.argv[2]) if sys.argv[2] else {}
value = payload.get(sys.argv[1]) if isinstance(payload, dict) else None
if value is not None:
    print(value)
PY
}

get_container_env_value() {
  local container_name="$1"
  local key="$2"
  docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$container_name" 2>/dev/null \
    | awk -F= -v target="$key" '$1 == target { sub($1 "=",""); print; exit }'
}

resolve_backend_catalog() {
  local profile="$1"
  local env_file_path="${2:-}"
  local resolver="$REPO_ROOT/scripts/runtime/repo_semantic_backend_catalog.py"

  if [[ -n "$env_file_path" ]]; then
    python3 "$resolver" --profile "$profile" --env-file "$env_file_path"
  else
    python3 "$resolver" --profile "$profile"
  fi
}

json_get() {
  local json_payload="$1"
  local expression="$2"
  python3 - "$expression" "$json_payload" <<'PY'
import json
import sys

payload = json.loads(sys.argv[2])
value = payload
for part in sys.argv[1].split('.'):
    if not part:
        continue
    if isinstance(value, dict):
        value = value.get(part)
    else:
        value = None
        break
if isinstance(value, bool):
    print("true" if value else "false")
elif value is None:
    print("")
else:
    print(value)
PY
}

get_semantic_mcp_port() {
  local env_file_path="$1"
  local merged_env_json="${2:-}"
  local container_port
  local env_port

  container_port="$(get_container_env_value repo-semantic-mcp SEMANTIC_MCP_HTTP_PORT || true)"
  if [[ -n "$container_port" ]]; then
    printf '%s\n' "$container_port"
    return 0
  fi

  env_port="$(get_merged_env_value "$merged_env_json" "SEMANTIC_MCP_HTTP_PORT" || true)"
  if [[ -z "$env_port" ]]; then
    env_port="$(get_env_value "$env_file_path" "SEMANTIC_MCP_HTTP_PORT" || true)"
  fi
  if [[ -n "$env_port" ]]; then
    printf '%s\n' "$env_port"
    return 0
  fi

  printf '8011\n'
}

test_semantic_ready_route() {
  local base_url="$1"
  python3 - "$base_url" >/dev/null 2>&1 <<'PY'
import json
import sys
import urllib.request

base_url = sys.argv[1]
with urllib.request.urlopen(f"{base_url}/readyz", timeout=10) as response:
    payload = json.loads(response.read().decode("utf-8"))
    if not payload.get("ready"):
        raise SystemExit(1)
PY
}

get_logical_repo_root_for_activation() {
  local base_url="$1"
  local resolved_target_repo="${2:-}"

  if [[ -n "$resolved_target_repo" ]]; then
    printf '%s\n' "$resolved_target_repo"
    return 0
  fi

  local payload
  if payload="$(curl --silent --show-error --noproxy '*' --fail "${base_url}/statusz" 2>/dev/null)" && [[ -n "$payload" ]]; then
    python3 - "$payload" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
index_status = payload.get("index_status") or {}
print(index_status.get("repo_root") or payload.get("details", {}).get("repo_root") or "")
PY
    return 0
  fi

  return 1
}

get_runtime_embedding_backend_id() {
  local base_url="$1"
  local payload
  if payload="$(curl --silent --show-error --noproxy '*' --fail "${base_url}/statusz" 2>/dev/null)" && [[ -n "$payload" ]]; then
    python3 - "$payload" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
index_status = payload.get("index_status") or {}
print(index_status.get("embedding_backend_id") or "")
PY
    return 0
  fi

  return 1
}

activate_runtime_repo() {
  local repo_root="$1"
  local port="$2"

  [[ -n "$repo_root" ]] || return 0

  docker exec -i repo-semantic-mcp python3 - "$repo_root" "$port" <<'PY' >/dev/null
import anyio
import sys
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def main() -> None:
    repo_root = sys.argv[1]
    port = sys.argv[2]
    async with streamablehttp_client(f"http://127.0.0.1:{port}/mcp") as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            await session.call_tool("activate_repo", {"repo_root": repo_root})

anyio.run(main)
PY
}

select_runtime_embedding_backend() {
  local backend_id="$1"
  local port="$2"

  [[ -n "$backend_id" ]] || return 0

  docker exec -i repo-semantic-mcp python3 - "$backend_id" "$port" <<'PY' >/dev/null
import anyio
import sys
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def main() -> None:
    backend_id = sys.argv[1]
    port = sys.argv[2]
    async with streamablehttp_client(f"http://127.0.0.1:{port}/mcp") as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            await session.call_tool("set_role_backend", {"role": "embedding", "backend_id": backend_id})

anyio.run(main)
PY
}

ensure_managed_external_embedder() {
  local env_file_path="$1"
  local merged_env_json="${2:-}"
  local enabled

  enabled="$(get_merged_env_value "$merged_env_json" "SEMANTIC_MCP_WSL_EMBEDDER_ENABLED" || true)"
  if [[ -z "$enabled" ]]; then
    enabled="$(get_env_value "$env_file_path" "SEMANTIC_MCP_WSL_EMBEDDER_ENABLED" || true)"
  fi
  if [[ ! "$enabled" =~ ^(1|true|True|yes|on)$ ]]; then
    return 0
  fi

  echo "Managed WSL embedder auto-start is only implemented in the PowerShell helper." >&2
  echo "Run scripts/agents/ensure_repo_semantic_gpu_server.ps1 on Windows before using this profile." >&2
  exit 1
}

BACKEND_RESOLUTION="$(resolve_backend_catalog "$PROFILE" "$ENV_FILE")"
ENV_FILE="$(json_get "$BACKEND_RESOLUTION" "env_file")"
MERGED_ENV_JSON="$(python3 - "$BACKEND_RESOLUTION" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
print(json.dumps(payload.get("merged_env") or {}))
PY
)"
mapfile -t ENV_LAYERS < <(python3 - "$BACKEND_RESOLUTION" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
for item in payload.get("env_layers") or []:
    print(item)
PY
)
BACKEND_ID="$(json_get "$BACKEND_RESOLUTION" "backend.backend_id")"
LAUNCH_MODE="$(json_get "$BACKEND_RESOLUTION" "backend.config_blob.launch_mode")"
USE_GPU_COMPOSE="$(json_get "$BACKEND_RESOLUTION" "backend.config_blob.use_gpu_compose")"
BACKEND_ENDPOINT="$(json_get "$BACKEND_RESOLUTION" "backend.endpoint")"
mapfile -t COMPOSE_EXTRA_FILES < <(python3 - "$BACKEND_RESOLUTION" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
for item in payload.get("backend", {}).get("config_blob", {}).get("compose_extra_files") or []:
    print(item)
PY
)

if [[ -z "$ENV_FILE" ]]; then
  echo "No env file resolved for profile '$PROFILE'." >&2
  exit 1
fi
if [[ -z "$LAUNCH_MODE" ]]; then
  echo "Backend '$BACKEND_ID' does not define launch_mode in the backend catalog." >&2
  exit 1
fi

wait_docker_ready

if [[ -n "$TARGET_REPO_PATH" ]]; then
  RESOLVED_TARGET_REPO="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$TARGET_REPO_PATH")"
  export SEMANTIC_MCP_TARGET_REPO_PATH="$RESOLVED_TARGET_REPO"
  export SEMANTIC_MCP_REPO_ROOT="/target_repo"
  export SEMANTIC_MCP_LOGICAL_REPO_ROOT="$RESOLVED_TARGET_REPO"
fi

COMPOSE_ARGS=(compose -f "$COMPOSE_FILE")
if [[ "$USE_GPU_COMPOSE" == "true" ]]; then
  COMPOSE_ARGS+=(-f "$COMPOSE_GPU_FILE")
fi
for compose_extra_file in "${COMPOSE_EXTRA_FILES[@]}"; do
  [[ -z "$compose_extra_file" ]] && continue
  if [[ "$compose_extra_file" = /* ]]; then
    COMPOSE_ARGS+=(-f "$compose_extra_file")
  else
    COMPOSE_ARGS+=(-f "$DEPLOY_DIR/$compose_extra_file")
  fi
done
if [[ -n "$ENV_FILE" ]]; then
  :
fi
if [[ "${#ENV_LAYERS[@]}" -gt 0 ]]; then
  for env_layer in "${ENV_LAYERS[@]}"; do
    [[ -n "$env_layer" ]] && COMPOSE_ARGS+=(--env-file "$env_layer")
  done
elif [[ -n "$ENV_FILE" ]]; then
  COMPOSE_ARGS+=(--env-file "$ENV_FILE")
fi

if [[ "$LAUNCH_MODE" == "managed_host" ]]; then
  ensure_managed_external_embedder "$ENV_FILE" "$MERGED_ENV_JSON"
fi

echo "profile: $PROFILE"
if [[ -n "$ENV_FILE" ]]; then
  echo "env file: $ENV_FILE"
fi
if [[ "${#ENV_LAYERS[@]}" -gt 1 ]]; then
  echo "env layers: ${ENV_LAYERS[*]}"
fi
echo "backend id: $BACKEND_ID"
echo "launch mode: $LAUNCH_MODE"
if [[ -n "$BACKEND_ENDPOINT" ]]; then
  echo "backend endpoint: $BACKEND_ENDPOINT"
fi
if [[ -n "$TARGET_REPO_PATH" ]]; then
  echo "target repo: $RESOLVED_TARGET_REPO"
else
  echo "target repo: compose default"
fi

if [[ "$CLEAN" == "1" ]]; then
  docker "${COMPOSE_ARGS[@]}" down --remove-orphans
fi

case "$LAUNCH_MODE" in
  bundled_compose)
    UP_ARGS=("${COMPOSE_ARGS[@]}" up -d)
    if [[ "$BUILD" == "1" ]]; then
      UP_ARGS+=(--build)
    fi
    docker "${UP_ARGS[@]}"
    ;;
  managed_host|external_manual)
    if [[ "$BUILD" == "1" ]]; then
      docker "${COMPOSE_ARGS[@]}" build repo-semantic-mcp
    fi
    docker "${COMPOSE_ARGS[@]}" up -d qdrant
    docker "${COMPOSE_ARGS[@]}" up -d --no-deps repo-semantic-mcp
    ;;
  *)
    echo "Unsupported launch_mode '$LAUNCH_MODE' for backend '$BACKEND_ID'." >&2
    exit 1
    ;;
esac

START_TS="$(date +%s)"
while true; do
  STATUS="$(docker ps --filter 'name=^repo-semantic-mcp$' --format '{{.Status}}')"
  if [[ "$STATUS" == Up* ]]; then
    MCP_PORT="$(get_semantic_mcp_port "$ENV_FILE" "$MERGED_ENV_JSON")"
    BASE_URL="http://127.0.0.1:${MCP_PORT}"
    if test_semantic_ready_route "$BASE_URL"; then
      LOGICAL_REPO_ROOT="$(get_logical_repo_root_for_activation "$BASE_URL" "${RESOLVED_TARGET_REPO:-}" || true)"
      activate_runtime_repo "$LOGICAL_REPO_ROOT" "$MCP_PORT"
      RUNTIME_BACKEND_ID="$(get_runtime_embedding_backend_id "$BASE_URL" || true)"
      select_runtime_embedding_backend "$RUNTIME_BACKEND_ID" "$MCP_PORT"
      echo "repo-semantic-search ready: host-visible readiness is green on ${BASE_URL}/readyz"
      echo "mcp endpoint: ${BASE_URL}/mcp"
      exit 0
    fi
  fi
  if (( "$(date +%s)" - START_TS >= TIMEOUT_SEC )); then
    echo "repo-semantic-search did not become ready within ${TIMEOUT_SEC} seconds." >&2
    exit 1
  fi
  sleep 5
done
