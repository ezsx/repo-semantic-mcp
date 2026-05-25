"""Structured chunkers for JSON, YAML and TOML files.

Вместо нарезки по N строк эти чанкеры разбивают файлы по логическим границам:
- JSON: каждый элемент массива или ключ верхнего уровня — отдельный чанк
- YAML: каждый top-level ключ (блок) — отдельный чанк
- TOML: каждая секция [header] — отдельный чанк

Fallback на generic chunker, если парсинг провалился.
"""

from __future__ import annotations

import json
import re
from hashlib import sha256
from pathlib import Path

from services.repo_semantic.models import ChunkRecord


def _make_chunk(
    *,
    relative_path: str,
    scope: str,
    domain_tags: list[str],
    source_mtime: float,
    language: str,
    chunk_type: str,
    text: str,
    start_line: int,
    end_line: int,
    symbol_path: str | None = None,
    locator: str,
) -> ChunkRecord:
    """Собрать ChunkRecord с правильным point_id."""

    point_id = sha256(f"{relative_path}|{locator}|{text}".encode("utf-8")).hexdigest()
    return ChunkRecord(
        point_id=point_id,
        scope=scope,
        relative_path=relative_path,
        language=language,
        chunk_type=chunk_type,
        text=text,
        start_line=start_line,
        end_line=end_line,
        content_hash=sha256(text.encode("utf-8")).hexdigest(),
        source_mtime=source_mtime,
        symbol_path=symbol_path,
        domain_tags=domain_tags.copy(),
        extra={"locator": locator},
    )


def chunk_json_file(
    file_path: Path,
    relative_path: str,
    text: str,
    scope: str,
    domain_tags: list[str],
    source_mtime: float,
) -> list[ChunkRecord]:
    """Разбить JSON-файл на чанки по логическим единицам.

    - Массив объектов: каждый элемент — чанк (с порядковым номером в prefixе)
    - Словарь: каждый ключ верхнего уровня — чанк
    - Всё остальное: один чанк на весь файл
    """

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return []  # factory.py упадёт на generic

    records: list[ChunkRecord] = []

    if isinstance(data, list):
        for i, item in enumerate(data):
            item_text = json.dumps(item, ensure_ascii=False, indent=2)
            if not item_text.strip():
                continue
            locator = f"json:item:{i}"
            # Грубая оценка строк: каждый элемент занимает ~len(lines)
            start_line = i * 4 + 1
            end_line = start_line + item_text.count("\n")
            symbol = str(item.get("id") or item.get("name") or item.get("key") or i) if isinstance(item, dict) else str(i)
            records.append(
                _make_chunk(
                    relative_path=relative_path,
                    scope=scope,
                    domain_tags=domain_tags,
                    source_mtime=source_mtime,
                    language="json",
                    chunk_type="json_array_item",
                    text=item_text,
                    start_line=start_line,
                    end_line=end_line,
                    symbol_path=symbol,
                    locator=locator,
                )
            )

    elif isinstance(data, dict):
        lines = text.splitlines()
        for key, value in data.items():
            value_text = json.dumps({key: value}, ensure_ascii=False, indent=2)
            chunk_text = value_text
            if not chunk_text.strip():
                continue
            # Найти строку с этим ключом в исходном тексте
            key_pattern = json.dumps(key)  # с кавычками
            start_line = next(
                (i + 1 for i, l in enumerate(lines) if key_pattern in l or f'"{key}"' in l),
                1,
            )
            end_line = start_line + chunk_text.count("\n")
            locator = f"json:key:{key}"
            records.append(
                _make_chunk(
                    relative_path=relative_path,
                    scope=scope,
                    domain_tags=domain_tags,
                    source_mtime=source_mtime,
                    language="json",
                    chunk_type="json_dict_key",
                    text=chunk_text,
                    start_line=start_line,
                    end_line=end_line,
                    symbol_path=key,
                    locator=locator,
                )
            )

    else:
        # Скалярное значение — весь файл как один чанк
        chunk_text = text.strip()
        records.append(
            _make_chunk(
                relative_path=relative_path,
                scope=scope,
                domain_tags=domain_tags,
                source_mtime=source_mtime,
                language="json",
                chunk_type="json_file",
                text=chunk_text,
                start_line=1,
                end_line=text.count("\n") + 1,
                locator="json:file",
            )
        )

    return records


# ---------------------------------------------------------------------------
# YAML
# ---------------------------------------------------------------------------

# Строка, начинающаяся с не-пробела и не-комментария и содержащая ':'
_YAML_TOP_KEY = re.compile(r"^([A-Za-z_\-\d][^:\n]*):")


def chunk_yaml_file(
    file_path: Path,
    relative_path: str,
    text: str,
    scope: str,
    domain_tags: list[str],
    source_mtime: float,
) -> list[ChunkRecord]:
    """Разбить YAML-файл на чанки по top-level ключам.

    Каждый блок, начинающийся с ключа на уровне отступа 0, — отдельный чанк.
    Заголовочные комментарии и директивы --- включаются в первый блок.
    """

    lines = text.splitlines()
    sections: list[tuple[str, int, int]] = []  # (key, start_line_0based, end_line_0based)
    current_key: str = "__preamble__"
    current_start: int = 0

    for i, line in enumerate(lines):
        if line.startswith("#") or not line.strip():
            continue
        m = _YAML_TOP_KEY.match(line)
        if m:
            sections.append((current_key, current_start, i - 1))
            current_key = m.group(1).strip()
            current_start = i

    sections.append((current_key, current_start, len(lines) - 1))

    records: list[ChunkRecord] = []
    for key, start, end in sections:
        block_lines = lines[start : end + 1]
        chunk_text = "\n".join(block_lines).strip()
        if not chunk_text:
            continue
        locator = f"yaml:key:{key}:{start + 1}"
        records.append(
            _make_chunk(
                relative_path=relative_path,
                scope=scope,
                domain_tags=domain_tags,
                source_mtime=source_mtime,
                language="yaml",
                chunk_type="yaml_section",
                text=chunk_text,
                start_line=start + 1,
                end_line=end + 1,
                symbol_path=key,
                locator=locator,
            )
        )

    return records


# ---------------------------------------------------------------------------
# TOML
# ---------------------------------------------------------------------------

_TOML_SECTION = re.compile(r"^\[([^\]]+)\]")


def chunk_toml_file(
    file_path: Path,
    relative_path: str,
    text: str,
    scope: str,
    domain_tags: list[str],
    source_mtime: float,
) -> list[ChunkRecord]:
    """Разбить TOML-файл на чанки по секциям [header].

    Контент до первой секции (глобальные переменные) — отдельный чанк.
    """

    lines = text.splitlines()
    sections: list[tuple[str, int, int]] = []
    current_key: str = "__root__"
    current_start: int = 0

    for i, line in enumerate(lines):
        m = _TOML_SECTION.match(line)
        if m:
            sections.append((current_key, current_start, i - 1))
            current_key = m.group(1).strip()
            current_start = i

    sections.append((current_key, current_start, len(lines) - 1))

    records: list[ChunkRecord] = []
    for key, start, end in sections:
        block_lines = lines[start : end + 1]
        chunk_text = "\n".join(block_lines).strip()
        if not chunk_text:
            continue
        locator = f"toml:section:{key}:{start + 1}"
        records.append(
            _make_chunk(
                relative_path=relative_path,
                scope=scope,
                domain_tags=domain_tags,
                source_mtime=source_mtime,
                language="toml",
                chunk_type="toml_section",
                text=chunk_text,
                start_line=start + 1,
                end_line=end + 1,
                symbol_path=key,
                locator=locator,
            )
        )

    return records
