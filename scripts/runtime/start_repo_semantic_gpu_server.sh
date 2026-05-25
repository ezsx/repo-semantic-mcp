#!/usr/bin/env bash
set -euo pipefail

SERVER_SCRIPT="${1:?server script path is required}"
VENV_ACTIVATE="${2:?venv activate path is required}"
MODEL_PATH="${3:?embedding model path is required}"
SERVER_PORT="${4:-8084}"
CUDA_VISIBLE_DEVICES_VALUE="${5:-0}"

source "$VENV_ACTIVATE"

exec env \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" \
  EMBEDDING_MODEL_PATH="$MODEL_PATH" \
  REPO_SEMANTIC_GPU_SERVER_PORT="$SERVER_PORT" \
  python3 "$SERVER_SCRIPT"
