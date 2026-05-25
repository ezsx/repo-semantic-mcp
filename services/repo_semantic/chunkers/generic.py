"""Generic chunking for configs, scripts and plain text."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from services.repo_semantic.models import ChunkRecord

MAX_CHARS_PER_CHUNK = 2400
MAX_LINES_PER_CHUNK = 80


def _language_from_suffix(path: Path) -> str:
    """Определить язык/тип файла по расширению."""

    suffix = path.suffix.lower()
    return {
        ".py": "python",
        ".md": "markdown",
        ".yml": "yaml",
        ".yaml": "yaml",
        ".json": "json",
        ".toml": "toml",
        ".sh": "bash",
        ".ps1": "powershell",
        ".sql": "sql",
        ".txt": "text",
    }.get(suffix, "text")


def chunk_generic_file(
    file_path: Path,
    relative_path: str,
    text: str,
    scope: str,
    domain_tags: list[str],
    source_mtime: float,
) -> list[ChunkRecord]:
    """Разбить файл на простые текстовые окна.

    Этот chunker служит fallback для конфигов, shell-скриптов и любых файлов,
    для которых у нас пока нет более сильного структурного парсера.
    """

    lines = text.splitlines()
    if not lines:
        return []

    records: list[ChunkRecord] = []
    start = 0
    part = 1
    language = _language_from_suffix(file_path)

    while start < len(lines):
        end = min(start + MAX_LINES_PER_CHUNK, len(lines))
        chunk_lines = lines[start:end]
        chunk_text = "\n".join(chunk_lines).strip()
        while len(chunk_text) > MAX_CHARS_PER_CHUNK and end > start + 5:
            end -= 5
            chunk_lines = lines[start:end]
            chunk_text = "\n".join(chunk_lines).strip()

        if chunk_text:
            locator = f"generic:{part}:{start + 1}:{end}"
            point_id = sha256(
                f"{relative_path}|{locator}|{chunk_text}".encode("utf-8")
            ).hexdigest()
            records.append(
                ChunkRecord(
                    point_id=point_id,
                    scope=scope,
                    relative_path=relative_path,
                    language=language,
                    chunk_type="generic_window",
                    text=chunk_text,
                    start_line=start + 1,
                    end_line=end,
                    content_hash=sha256(chunk_text.encode("utf-8")).hexdigest(),
                    source_mtime=source_mtime,
                    domain_tags=domain_tags.copy(),
                    extra={"locator": locator},
                )
            )

        start = end
        part += 1

    return records

