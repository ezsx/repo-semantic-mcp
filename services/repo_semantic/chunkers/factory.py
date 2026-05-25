"""Dispatch logic for semantic indexing chunkers."""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
import re

from services.repo_semantic.chunkers.generic import chunk_generic_file
from services.repo_semantic.chunkers.markdown import chunk_markdown_file
from services.repo_semantic.chunkers.python import chunk_python_file
from services.repo_semantic.chunkers.structured import chunk_json_file, chunk_toml_file, chunk_yaml_file
from services.repo_semantic.models import ChunkRecord, ChunkScope

# Fallback — используется только если doc_prefixes не передан явно
_DEFAULT_DOC_PATH_PREFIXES = ("docs/", "agent_context/")
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


def classify_scope(
    relative_path: str,
    doc_prefixes: tuple[str, ...] = _DEFAULT_DOC_PATH_PREFIXES,
) -> ChunkScope:
    """Определить, относится ли файл к code или docs коллекции.

    doc_prefixes передаётся из settings.effective_doc_prefixes и отражает
    реальную структуру текущего репозитория.
    """

    lowered = relative_path.replace("\\", "/")
    if lowered.startswith(doc_prefixes) or Path(lowered).suffix.lower() in DOC_EXTENSIONS:
        return "docs"
    return "code"


_INTERESTING_SUBDIRS: frozenset[str] = frozenset(
    {
        "api",
        "services",
        "adapters",
        "core",
        "schemas",
        "tests",
        "tools",
        "utils",
        "models",
        "migrations",
    }
)


def _normalized_tag_variants(value: str) -> list[str]:
    """Построить стабильные alias-варианты для path-derived domain tags.

    Это сохраняет backward compatibility для фильтров вроде `telegram_bot`,
    `maintenance_worker`, `flow01`, при этом не убирая текущие path-based теги.
    """

    lowered = value.strip().lower()
    if not lowered:
        return []

    normalized = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    variants: list[str] = []
    for candidate in (lowered, normalized):
        if candidate and candidate not in variants:
            variants.append(candidate)

    if normalized:
        collapsed = normalized.replace("_", "")
        if collapsed and collapsed not in variants:
            variants.append(collapsed)

    return variants


def derive_domain_tags(relative_path: str) -> list[str]:
    """Построить доменные теги по структуре пути.

    Используется при поиске через параметр domain_tags для фильтрации чанков.
    Теги строятся из реального пути без project-specific classifier'а:
    - path-based компоненты остаются как есть;
    - поверх них добавляются normalized aliases для backward compatibility
      (`telegram-bot` -> `telegram_bot`, `flow-01` -> `flow01`);
    - для "container" директорий (`apps`, `services`, `docs`, ...) добавляется
      и следующий path component, чтобы фильтрация не ломалась на общих префиксах.
    """

    normalized = relative_path.replace("\\", "/")
    parts = [p for p in normalized.split("/") if p]
    tags: list[str] = []
    raw_tags: list[str] = []

    if not parts:
        return ["general"]

    raw_tags.append(parts[0])

    if len(parts) >= 2:
        second = parts[1]
        if parts[0] in {"apps", "services", "docs", "doc", "src", "scripts", "deploy", "agent_context"}:
            raw_tags.append(second)
        elif second in _INTERESTING_SUBDIRS:
            raw_tags.append(second)

    for raw_tag in raw_tags:
        for candidate in _normalized_tag_variants(raw_tag):
            if candidate not in tags:
                tags.append(candidate)

    return tags or ["general"]


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
    doc_prefixes: tuple[str, ...] = _DEFAULT_DOC_PATH_PREFIXES,
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

    return build_chunks_for_text(
        file_path=file_path,
        repo_root=repo_root,
        text=text,
        source_mtime=file_path.stat().st_mtime,
        doc_prefixes=doc_prefixes,
    )


def build_chunks_for_text(
    file_path: Path,
    repo_root: Path,
    text: str,
    source_mtime: float,
    doc_prefixes: tuple[str, ...] = _DEFAULT_DOC_PATH_PREFIXES,
) -> list[ChunkRecord]:
    """Split an already-read text snapshot into semantic chunks."""

    relative_path = file_path.relative_to(repo_root).as_posix()
    if not text.strip():
        return []

    scope = classify_scope(relative_path, doc_prefixes)
    domain_tags = derive_domain_tags(relative_path)
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
    if suffix == ".json":
        chunks = chunk_json_file(
            file_path=file_path,
            relative_path=relative_path,
            text=text,
            scope=scope,
            domain_tags=domain_tags,
            source_mtime=source_mtime,
        )
        return chunks or chunk_generic_file(
            file_path=file_path,
            relative_path=relative_path,
            text=text,
            scope=scope,
            domain_tags=domain_tags,
            source_mtime=source_mtime,
        )
    if suffix in {".yml", ".yaml"}:
        chunks = chunk_yaml_file(
            file_path=file_path,
            relative_path=relative_path,
            text=text,
            scope=scope,
            domain_tags=domain_tags,
            source_mtime=source_mtime,
        )
        return chunks or chunk_generic_file(
            file_path=file_path,
            relative_path=relative_path,
            text=text,
            scope=scope,
            domain_tags=domain_tags,
            source_mtime=source_mtime,
        )
    if suffix == ".toml":
        chunks = chunk_toml_file(
            file_path=file_path,
            relative_path=relative_path,
            text=text,
            scope=scope,
            domain_tags=domain_tags,
            source_mtime=source_mtime,
        )
        return chunks or chunk_generic_file(
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
