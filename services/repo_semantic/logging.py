"""Минимальный структурированный логгер для semantic MCP."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone


def jlog(level: str, event: str, **kwargs) -> None:
    """Напечатать JSON-лог semantic MCP в stderr."""

    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "event": event,
        "service": "repo-semantic-mcp",
        **kwargs,
    }
    print(json.dumps(entry, ensure_ascii=False, default=str), file=sys.stderr)

