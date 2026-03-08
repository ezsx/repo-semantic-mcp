#!/usr/bin/env bash
set -euo pipefail

REPO_PATH=""
PROFILE="gpu"
CLEAN=0
BUILD=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENSURE_SCRIPT="$SCRIPT_DIR/ensure_repo_semantic_search.sh"
STATUS_SCRIPT="$SCRIPT_DIR/repo_semantic_status.sh"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-path)
      REPO_PATH="$2"
      shift 2
      ;;
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --clean)
      CLEAN=1
      shift
      ;;
    --build)
      BUILD=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$REPO_PATH" ]]; then
  echo "--repo-path is required" >&2
  exit 1
fi

if [[ "$BUILD" == "0" ]]; then
  if ! docker image inspect repo-semantic-search-repo-semantic-mcp >/dev/null 2>&1; then
    BUILD=1
  fi
fi

ARGS=(--profile "$PROFILE" --target-repo-path "$REPO_PATH")
if [[ "$BUILD" == "1" ]]; then
  ARGS=(--build "${ARGS[@]}")
fi
if [[ "$CLEAN" == "1" ]]; then
  ARGS=(--clean "${ARGS[@]}")
fi

bash "$ENSURE_SCRIPT" "${ARGS[@]}"

echo
echo "repo-semantic-search status:"
bash "$STATUS_SCRIPT"
