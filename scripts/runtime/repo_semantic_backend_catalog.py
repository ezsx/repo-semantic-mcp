"""CLI для backend catalog resolution в helper/deploy orchestration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.repo_semantic.backend_catalog import (
    default_backend_id_for_profile,
    env_candidates_for_profile,
    get_backend_catalog_entry,
)


def _read_env_file(path: Path) -> dict[str, str]:
    """Прочитать простой dotenv-файл без shell expansion."""

    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        result[key.strip()] = value.strip().strip("'\"")
    return result


def _local_override_candidates(base_path: Path) -> list[Path]:
    """Построить список local override файлов для base env."""

    candidates: list[Path] = []
    if base_path.name.endswith(".example"):
        stripped = base_path.with_name(base_path.name[: -len(".example")])
        candidates.append(stripped.with_name(f"{stripped.name}.local"))
    else:
        candidates.append(base_path.with_name(f"{base_path.name}.local"))
    candidates.append(base_path.parent / ".env.local")
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _resolve_env_layers(repo_root: Path, profile: str, explicit_env_file: str | None) -> tuple[Path | None, list[str], list[Path]]:
    """Разрешить base env и optional local override слои для указанного profile."""

    deploy_dir = repo_root / "deploy" / "repo-semantic-search"
    if explicit_env_file:
        path = Path(explicit_env_file).expanduser().resolve()
        layers = [path]
        layers.extend(candidate for candidate in _local_override_candidates(path) if candidate.is_file())
        return path, [str(path)], layers

    candidates = [str(deploy_dir / name) for name in env_candidates_for_profile(profile)]
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            resolved = path.resolve()
            layers = [resolved]
            layers.extend(candidate for candidate in _local_override_candidates(resolved) if candidate.is_file())
            return resolved, candidates, layers
    return None, candidates, []


def _merge_env_layers(env_layers: list[Path]) -> dict[str, str]:
    """Собрать итоговый env dict из base env и local override layers."""

    merged: dict[str, str] = {}
    for path in env_layers:
        merged.update(_read_env_file(path))
    return merged


def _placeholder_keys(env_values: dict[str, str]) -> list[str]:
    """Найти env keys с placeholder-like значениями, требующими local override."""

    result: list[str] = []
    for key, value in env_values.items():
        normalized = value.strip().lower()
        if normalized.startswith("<set-") or normalized.startswith("<replace-"):
            result.append(key)
    return result


def resolve_backend(repo_root: Path, profile: str, explicit_env_file: str | None) -> dict[str, Any]:
    """Собрать orchestration resolution payload по profile/backend catalog."""

    env_path, env_candidates, env_layers = _resolve_env_layers(repo_root, profile, explicit_env_file)
    env_values = _merge_env_layers(env_layers)

    backend_id = env_values.get("SEMANTIC_MCP_EMBEDDING_BACKEND_ID") or default_backend_id_for_profile(profile)
    if not backend_id:
        raise RuntimeError(f"No backend mapping is configured for profile '{profile}'")

    catalog_entry = get_backend_catalog_entry(backend_id)
    if catalog_entry is None:
        raise RuntimeError(
            f"Unknown backend id '{backend_id}'. Set a known SEMANTIC_MCP_EMBEDDING_BACKEND_ID "
            "or extend services/repo_semantic/backend_catalog.py."
        )

    entry_payload = catalog_entry.to_registry_payload()
    if env_values.get("SEMANTIC_MCP_TEI_URL"):
        entry_payload["endpoint"] = env_values["SEMANTIC_MCP_TEI_URL"]

    placeholder_keys = _placeholder_keys(env_values)
    if (
        entry_payload["config_blob"].get("launch_mode") == "managed_host"
        and any(
            key in placeholder_keys
            for key in (
                "SEMANTIC_MCP_WSL_DISTRO",
                "SEMANTIC_MCP_WSL_VENV_PATH",
                "SEMANTIC_MCP_WSL_EMBEDDING_MODEL_PATH",
            )
        )
    ):
        raise RuntimeError(
            "Managed-host profile requires machine-local WSL values. "
            "Set them in deploy/repo-semantic-search/.env.gpu or .env.gpu.local."
        )

    return {
        "profile": profile,
        "env_file": str(env_path) if env_path is not None else None,
        "env_candidates": env_candidates,
        "env_layers": [str(path) for path in env_layers],
        "merged_env": env_values,
        "backend_id": backend_id,
        "backend": entry_payload,
    }


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--env-file")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    print(
        json.dumps(
            resolve_backend(repo_root, profile=args.profile, explicit_env_file=args.env_file),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
