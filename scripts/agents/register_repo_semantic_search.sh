#!/usr/bin/env bash
set -euo pipefail

URL="${1:-http://127.0.0.1:8011/mcp}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

python3 "$REPO_ROOT/scripts/agents/register_repo_semantic_search.py" --url "$URL"
