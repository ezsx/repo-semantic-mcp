"""Code-aware lexical analyzer used by sparse retrieval and snippets."""

from __future__ import annotations

import re

from services.repo_semantic.lexical.types import SparseToken

TOKEN_RE = re.compile(
    r"""
    "(?:[^"\\]|\\.){1,240}"
    |
    --[A-Za-z0-9][A-Za-z0-9_-]*
    |[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.{}:-]+)+
    |/[A-Za-z0-9_./{}:-]+
    |[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+
    |[A-Za-z_][A-Za-z0-9_]*
    |[0-9]+
    """,
    re.VERBOSE,
)
ENV_VAR_RE = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$")
CLI_FLAG_RE = re.compile(r"^--[A-Za-z0-9][A-Za-z0-9_-]*$")
FILE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,12}$")
SQL_IDENTIFIER_RE = re.compile(r"^[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)+$")
CAMEL_RE = re.compile(r"[a-z][A-Z]|[A-Z][a-z]")
KNOWN_FILE_EXTENSIONS = {
    "c",
    "cc",
    "cfg",
    "conf",
    "cpp",
    "cs",
    "css",
    "env",
    "go",
    "h",
    "hpp",
    "html",
    "ini",
    "java",
    "js",
    "json",
    "jsx",
    "kt",
    "lock",
    "md",
    "py",
    "rb",
    "rs",
    "scss",
    "sh",
    "sql",
    "toml",
    "ts",
    "tsx",
    "txt",
    "xml",
    "yaml",
    "yml",
}


def _split_case(value: str) -> list[str]:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    spaced = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", spaced)
    return [part for part in re.split(r"[^A-Za-z0-9]+", spaced) if part]


def _split_separators(value: str) -> list[str]:
    return [part for part in re.split(r"[_\-.]+", value) if part]


def analyze_sparse_text(text: str) -> list[SparseToken]:
    """Return exact-preserving code-aware lexical tokens.

    The vocabulary key is canonical and lower-case. The original surface is kept
    for explanations and future exact-search guidance.
    """

    tokens: list[SparseToken] = []

    def add(
        bucket: list[SparseToken],
        seen_local: set[str],
        surface: str,
        canonical: str,
        token_type: str,
    ) -> None:
        canonical = canonical.strip().lower()
        if not canonical:
            return
        if canonical in seen_local:
            return
        seen_local.add(canonical)
        bucket.append(SparseToken(surface=surface, canonical=canonical, token_type=token_type))

    def emit(bucket: list[SparseToken]) -> None:
        tokens.extend(bucket)

    def add_parts(bucket: list[SparseToken], seen_local: set[str], surface: str, value: str, token_type: str) -> None:
        for part in _split_separators(value):
            add(bucket, seen_local, surface, part, token_type)
            case_parts = _split_case(part)
            if len(case_parts) > 1 or (case_parts and case_parts[0] != part):
                for case_part in case_parts:
                    add(bucket, seen_local, part, case_part, "case_part")

    for match in TOKEN_RE.finditer(text):
        raw_surface = match.group(0).strip()
        if not raw_surface:
            continue
        bucket: list[SparseToken] = []
        seen_local: set[str] = set()

        if raw_surface.startswith('"') and raw_surface.endswith('"'):
            surface = raw_surface[1:-1].replace('\\"', '"').strip()
            if not surface:
                continue
            add(bucket, seen_local, surface, surface, "quoted_literal")
            for part in re.split(r"[^A-Za-z0-9]+", surface):
                if part:
                    add(bucket, seen_local, part, part, "part")
            emit(bucket)
            continue

        surface = raw_surface

        if CLI_FLAG_RE.match(surface):
            body = surface[2:]
            add(bucket, seen_local, surface, surface, "cli_flag")
            add(bucket, seen_local, surface, body, "cli_flag")
            for part in _split_separators(body):
                add(bucket, seen_local, surface, part, "part")
            emit(bucket)
            continue

        if "/" in surface:
            is_route = surface.startswith("/")
            token_type = "route" if is_route else "path"
            add(bucket, seen_local, surface, surface, token_type)
            if is_route:
                add(bucket, seen_local, surface, surface.strip("/"), "route")
            for segment in [part for part in re.split(r"[/{}:]+", surface.strip("/")) if part]:
                segment_type = "route_part" if is_route else "path_segment"
                add(bucket, seen_local, segment, segment, segment_type)
                if "." in segment and not segment.startswith("."):
                    base, extension = segment.rsplit(".", 1)
                    add(bucket, seen_local, base, base, "path_segment")
                    add_parts(bucket, seen_local, base, base, "part")
                    add(bucket, seen_local, extension, extension, "file_extension")
                    continue
                add_parts(bucket, seen_local, segment, segment, "part" if not is_route else "route_part")
            emit(bucket)
            continue

        if ENV_VAR_RE.match(surface):
            add(bucket, seen_local, surface, surface, "env_var")
            for part in surface.split("_"):
                if part:
                    add(bucket, seen_local, surface, part, "part")
            emit(bucket)
            continue

        if "." in surface:
            exact_type = "sql_identifier" if SQL_IDENTIFIER_RE.match(surface) else "exact"
            add(bucket, seen_local, surface, surface, exact_type)
            for part in surface.split("."):
                if part:
                    add(bucket, seen_local, part, part, "dotted_part")
                    if CAMEL_RE.search(part):
                        for case_part in _split_case(part):
                            add(bucket, seen_local, part, case_part, "case_part")
            emit(bucket)
            continue

        add(bucket, seen_local, surface, surface, "exact")
        if "_" in surface or "-" in surface:
            for part in _split_separators(surface):
                add(bucket, seen_local, surface, part, "part")
                if CAMEL_RE.search(part):
                    for case_part in _split_case(part):
                        add(bucket, seen_local, part, case_part, "case_part")
        elif CAMEL_RE.search(surface):
            for part in _split_case(surface):
                add(bucket, seen_local, surface, part, "case_part")
        emit(bucket)

    return tokens


def sparse_terms(text: str) -> list[str]:
    """Return canonical sparse terms with multiplicity preserved."""

    return [token.canonical for token in analyze_sparse_text(text)]


def sparse_unique_terms(text: str) -> list[str]:
    """Return canonical sparse terms in stable first-seen order."""

    return list(dict.fromkeys(sparse_terms(text)))
