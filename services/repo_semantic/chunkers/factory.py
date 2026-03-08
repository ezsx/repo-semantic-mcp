"""Dispatch logic for semantic indexing chunkers."""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path

from services.repo_semantic.chunkers.generic import chunk_generic_file
from services.repo_semantic.chunkers.markdown import chunk_markdown_file
from services.repo_semantic.chunkers.python import chunk_python_file
from services.repo_semantic.models import ChunkRecord, ChunkScope

DOC_PATH_PREFIXES = ("docs/", "agent_context/")
DOC_EXTENSIONS = {".md", ".txt", ".rst"}
TEXT_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
    ".rst",
    ".yml",
    ".yaml",
    ".json",
    ".toml",
    ".ini",
    ".cfg",
    ".sh",
    ".ps1",
    ".sql",
}


def classify_scope(relative_path: str) -> ChunkScope:
    """Определить, относится ли файл к code или docs коллекции."""

    lowered = relative_path.replace("\\", "/")
    if lowered.startswith(DOC_PATH_PREFIXES) or Path(lowered).suffix.lower() in DOC_EXTENSIONS:
        return "docs"
    return "code"


def derive_domain_tags(relative_path: str) -> list[str]:
    """Построить грубые доменные теги по path prefix."""

    normalized = relative_path.replace("\\", "/")
    tags: list[str] = []
    if "telegram-bot" in normalized or "telegram_bot" in normalized:
        tags.append("telegram_bot")
    if "flow-01" in normalized or "/user" in normalized or "user-api" in normalized:
        tags.append("flow01")
    if "maintenance" in normalized or "flow-02" in normalized:
        tags.append("maintenance")
    if "observability" in normalized:
        tags.append("observability")
    if normalized.startswith("docs/"):
        tags.append("docs")
    if normalized.startswith("agent_context/"):
        tags.append("agent_context")
    if normalized.startswith("tools/testing/load/reports/"):
        tags.append("load_reports")
    if not tags:
        tags.append("general")
    return tags


def should_index_path(relative_path: str, include_globs: list[str], exclude_globs: list[str]) -> bool:
    """Проверить, попадает ли путь в include/exclude правила."""

    normalized = relative_path.replace("\\", "/")
    included = any(fnmatch(normalized, pattern) for pattern in include_globs)
    if not included:
        return False
    return not any(fnmatch(normalized, pattern) for pattern in exclude_globs)


def is_text_like(file_path: Path) -> bool:
    """Отсечь бинарные и явно неиндексируемые файлы."""

    return file_path.suffix.lower() in TEXT_EXTENSIONS or file_path.name in {
        "AGENTS.md",
        "CLAUDE.md",
        "README",
        "README.md",
        "Dockerfile",
    }


def build_chunks_for_file(
    file_path: Path,
    repo_root: Path,
    include_globs: list[str],
    exclude_globs: list[str],
) -> list[ChunkRecord]:
    """Прочитать файл и разбить его на semantic chunks."""

    relative_path = file_path.relative_to(repo_root).as_posix()
    if not should_index_path(relative_path, include_globs, exclude_globs):
        return []
    if not is_text_like(file_path):
        return []

    text = file_path.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        return []

    scope = classify_scope(relative_path)
    domain_tags = derive_domain_tags(relative_path)
    source_mtime = file_path.stat().st_mtime
    suffix = file_path.suffix.lower()

    if suffix == ".py":
        return chunk_python_file(
            file_path=file_path,
            relative_path=relative_path,
            text=text,
            scope=scope,
            domain_tags=domain_tags,
            source_mtime=source_mtime,
        )
    if suffix in DOC_EXTENSIONS or file_path.name.endswith(".md"):
        return chunk_markdown_file(
            file_path=file_path,
            relative_path=relative_path,
            text=text,
            scope=scope,
            domain_tags=domain_tags,
            source_mtime=source_mtime,
        )
    return chunk_generic_file(
        file_path=file_path,
        relative_path=relative_path,
        text=text,
        scope=scope,
        domain_tags=domain_tags,
        source_mtime=source_mtime,
    )
