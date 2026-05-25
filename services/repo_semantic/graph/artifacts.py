"""Graph artifact path and contract helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from services.repo_semantic.config import SemanticMcpSettings
from services.repo_semantic.graph.schema import GRAPH_SCHEMA_VERSION


def graph_artifact_path(settings: SemanticMcpSettings) -> Path:
    """Return the per-repo/profile SQLite graph artifact path."""

    graph_dir = settings.registry_db_path.parent / "graphs"
    return graph_dir / f"{settings.repo_key_slug}_{settings.profile_slug}_graph_v{GRAPH_SCHEMA_VERSION}.sqlite3"


def stable_json_hash(payload: object) -> str:
    """Hash JSON-compatible data with deterministic formatting."""

    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
