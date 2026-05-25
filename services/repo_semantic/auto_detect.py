"""Auto-detection of repository structure for smart glob generation.

When SEMANTIC_MCP_INCLUDE_GLOBS is set to ["auto"] (the default), the indexer
scans the repo root and builds include globs from the actual directory layout.
This way, switching to a new repo or adding new top-level dirs just works.
"""

from __future__ import annotations

from pathlib import Path

# Directories that contain source code → code corpus
_SOURCE_DIRS: frozenset[str] = frozenset(
    {
        "src",
        "app",
        "apps",
        "services",
        "libs",
        "lib",
        "packages",
        "pkg",
        "plugins",
        "modules",
        "core",
        "api",
        "backend",
        "frontend",
        "server",
        "client",
        "internal",
        "cmd",
    }
)

# Directories that contain tests → code corpus
_TEST_DIRS: frozenset[str] = frozenset(
    {
        "tests",
        "test",
        "testing",
        "spec",
        "specs",
        "__tests__",
        "e2e",
        "benchmarks",
        "benchmark",
    }
)

# Directories for scripts, tooling, deploy config → code corpus
_SCRIPT_DIRS: frozenset[str] = frozenset(
    {
        "scripts",
        "tools",
        "bin",
        "utils",
        "deploy",
        "infra",
        "infrastructure",
        "k8s",
        "kubernetes",
        "ci",
    }
)

# Directories for data, fixtures, evaluation sets → code corpus (JSON/CSV etc.)
_DATA_DIRS: frozenset[str] = frozenset(
    {
        "data",
        "datasets",
        "dataset",
        "fixtures",
        "examples",
        "samples",
        "resources",
    }
)

# Directories that contain documentation → docs corpus
_DOC_DIRS: frozenset[str] = frozenset(
    {
        "docs",
        "doc",
        "documentation",
        "wiki",
        "pages",
        "agent_context",
        "guides",
        "handbook",
    }
)

# All known directories (union of all groups)
_ALL_KNOWN_DIRS: frozenset[str] = _SOURCE_DIRS | _TEST_DIRS | _SCRIPT_DIRS | _DOC_DIRS | _DATA_DIRS

# Root-level files that are always included
_ROOT_FILES: frozenset[str] = frozenset(
    {
        "AGENTS.md",
        "CLAUDE.md",
        "STRUCTURE.md",
        "README.md",
        "README.rst",
        "CONTRIBUTING.md",
        "CONTRIBUTING.rst",
        "CHANGELOG.md",
        "CHANGELOG.rst",
    }
)

_TEXT_FILE_SUFFIXES: frozenset[str] = frozenset(
    {
        ".py",
        ".md",
        ".rst",
        ".txt",
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
)

_ROOT_TEXT_FILE_NAMES: frozenset[str] = frozenset(
    {
        "README",
        "Dockerfile",
        "Makefile",
    }
)

_DOC_LIKE_SUFFIXES: frozenset[str] = frozenset(
    {
        ".md",
        ".rst",
        ".txt",
        ".yaml",
        ".yml",
        ".json",
        ".toml",
    }
)

_CODE_LIKE_SUFFIXES: frozenset[str] = frozenset(
    {
        ".py",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".go",
        ".rs",
        ".java",
        ".kt",
        ".c",
        ".cc",
        ".cpp",
        ".cs",
        ".php",
        ".rb",
        ".swift",
        ".scala",
    }
)

# Doc path prefixes used by classify_scope; extended by detect_doc_prefixes
DEFAULT_DOC_PREFIXES: tuple[str, ...] = (
    "docs/",
    "doc/",
    "documentation/",
    "wiki/",
    "agent_context/",
)


def detect_repo_globs(repo_root: Path) -> list[str]:
    """Scan repo root and return include_globs based on detected directory layout.

    Returns globs only for directories and root files that actually exist in
    the given repo. This avoids matching nothing when the repo uses a different
    layout than the hardcoded defaults.
    """

    globs: list[str] = []

    for item in repo_root.iterdir():
        name_lower = item.name.lower()
        if item.is_dir() and name_lower in _ALL_KNOWN_DIRS:
            globs.append(f"{item.name}/**")
        elif item.is_file() and item.name in _ROOT_FILES:
            globs.append(item.name)

    # Safe fallback: unknown top-level directories and text-like root files
    # should still be indexed. This avoids empty or partial indexes for repos
    # that do not follow the curated directory naming list.
    for item in repo_root.iterdir():
        if item.name.startswith("."):
            continue
        if item.is_dir():
            globs.append(f"{item.name}/**")
            continue
        if item.is_file() and (
            item.name in _ROOT_FILES
            or item.name in _ROOT_TEXT_FILE_NAMES
            or item.suffix.lower() in _TEXT_FILE_SUFFIXES
        ):
            globs.append(item.name)

    return sorted(set(globs))


def _is_doc_like_dir(dir_path: Path, *, max_files: int = 200) -> bool:
    """Определить, похож ли top-level каталог на docs-корпус.

    Для нестандартных layout'ов считаем каталог doc-like, если внутри есть
    хотя бы один doc/structured файл и нет явных source-code файлов.
    """

    scanned = 0
    saw_doc_like = False

    for path in dir_path.rglob("*"):
        if scanned >= max_files:
            break
        if not path.is_file():
            continue
        scanned += 1
        suffix = path.suffix.lower()
        if suffix in _CODE_LIKE_SUFFIXES:
            return False
        if (
            suffix in _DOC_LIKE_SUFFIXES
            or path.name in _ROOT_FILES
            or path.name in _ROOT_TEXT_FILE_NAMES
        ):
            saw_doc_like = True

    return saw_doc_like


def detect_doc_prefixes(repo_root: Path) -> tuple[str, ...]:
    """Return doc path prefixes for classify_scope, extended from actual repo dirs.

    Starts from DEFAULT_DOC_PREFIXES and adds any doc-like dirs that actually
    exist in the repo (e.g. if the repo uses "documentation/" instead of "docs/").
    """

    found: list[str] = list(DEFAULT_DOC_PREFIXES)

    for item in repo_root.iterdir():
        if not item.is_dir():
            continue
        if item.name.lower() in _DOC_DIRS or _is_doc_like_dir(item):
            prefix = f"{item.name}/"
            if prefix not in found:
                found.append(prefix)

    return tuple(found)
