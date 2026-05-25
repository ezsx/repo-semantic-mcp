#!/usr/bin/env bash
set -euo pipefail

URL="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ -n "$URL" ]]; then
  python3 "$REPO_ROOT/scripts/agents/register_repo_semantic_search.py" --url "$URL"
else
  python3 "$REPO_ROOT/scripts/agents/register_repo_semantic_search.py"
fi
