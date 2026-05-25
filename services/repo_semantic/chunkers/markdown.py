"""Markdown chunking by heading sections."""

from __future__ import annotations

import re
from hashlib import sha256
from pathlib import Path

from services.repo_semantic.models import ChunkRecord

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
MAX_SECTION_CHARS = 2600


def _split_large_markdown_section(text: str) -> list[str]:
    """Разбить длинную markdown-секцию на разумные подчанки."""

    if len(text) <= MAX_SECTION_CHARS:
        return [text]

    parts: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in text.splitlines():
        current.append(line)
        current_len += len(line) + 1
        if current_len >= MAX_SECTION_CHARS and line.strip() == "":
            parts.append("\n".join(current).strip())
            current = []
            current_len = 0

    if current:
        parts.append("\n".join(current).strip())
    return [part for part in parts if part]


def chunk_markdown_file(
    file_path: Path,
    relative_path: str,
    text: str,
    scope: str,
    domain_tags: list[str],
    source_mtime: float,
) -> list[ChunkRecord]:
    """Разбить markdown файл на секции по heading path."""

    lines = text.splitlines()
    if not lines:
        return []

    headings: list[tuple[int, int, str]] = []
    for index, line in enumerate(lines, start=1):
        match = HEADING_RE.match(line)
        if match:
            headings.append((index, len(match.group(1)), match.group(2).strip()))

    if not headings:
        from services.repo_semantic.chunkers.generic import chunk_generic_file

        return chunk_generic_file(
            file_path=file_path,
            relative_path=relative_path,
            text=text,
            scope=scope,
            domain_tags=domain_tags,
            source_mtime=source_mtime,
        )

    records: list[ChunkRecord] = []
    stack: list[tuple[int, str]] = []

    for position, (line_no, level, title) in enumerate(headings):
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        section_start = line_no
        section_end = (
            headings[position + 1][0] - 1 if position + 1 < len(headings) else len(lines)
        )
        section_text = "\n".join(lines[section_start - 1 : section_end]).strip()
        if not section_text:
            continue

        heading_path = " / ".join(item[1] for item in stack)
        parts = _split_large_markdown_section(section_text)
        for part_index, part_text in enumerate(parts, start=1):
            locator = f"{heading_path}:{part_index}:{section_start}:{section_end}"
            point_id = sha256(
                f"{relative_path}|{locator}|{part_text}".encode("utf-8")
            ).hexdigest()
            records.append(
                ChunkRecord(
                    point_id=point_id,
                    scope=scope,
                    relative_path=relative_path,
                    language="markdown",
                    chunk_type="markdown_section",
                    text=part_text,
                    start_line=section_start,
                    end_line=section_end,
                    content_hash=sha256(part_text.encode("utf-8")).hexdigest(),
                    source_mtime=source_mtime,
                    heading_path=heading_path,
                    domain_tags=domain_tags.copy(),
                    extra={"locator": locator},
                )
            )

    return records

