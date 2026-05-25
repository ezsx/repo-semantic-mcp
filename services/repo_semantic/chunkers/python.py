"""Python symbol-aware chunking via builtin AST."""

from __future__ import annotations

import ast
from hashlib import sha256
from pathlib import Path

from services.repo_semantic.models import ChunkRecord


def _slice_lines(lines: list[str], start: int, end: int) -> str:
    """Безопасно вырезать диапазон строк из файла."""

    return "\n".join(lines[start - 1 : end]).strip()


def _meaningful(text: str) -> bool:
    """Понять, содержит ли блок полезный контент."""

    return any(line.strip() for line in text.splitlines())


def chunk_python_file(
    file_path: Path,
    relative_path: str,
    text: str,
    scope: str,
    domain_tags: list[str],
    source_mtime: float,
) -> list[ChunkRecord]:
    """Разбить Python файл на module preamble, функции, классы и методы."""

    lines = text.splitlines()
    if not lines:
        return []

    try:
        tree = ast.parse(text)
    except SyntaxError:
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
    top_level = [
        node
        for node in tree.body
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        )
    ]

    if top_level:
        first_start = min(node.lineno for node in top_level)
        preamble_text = _slice_lines(lines, 1, first_start - 1)
        if _meaningful(preamble_text):
            locator = f"module_preamble:1:{first_start - 1}"
            point_id = sha256(
                f"{relative_path}|{locator}|{preamble_text}".encode("utf-8")
            ).hexdigest()
            records.append(
                ChunkRecord(
                    point_id=point_id,
                    scope=scope,
                    relative_path=relative_path,
                    language="python",
                    chunk_type="module_preamble",
                    text=preamble_text,
                    start_line=1,
                    end_line=first_start - 1,
                    content_hash=sha256(preamble_text.encode("utf-8")).hexdigest(),
                    source_mtime=source_mtime,
                    domain_tags=domain_tags.copy(),
                    extra={"locator": locator},
                )
            )

    for node in top_level:
        end_lineno = getattr(node, "end_lineno", node.lineno)
        node_text = _slice_lines(lines, node.lineno, end_lineno)
        if not node_text:
            continue

        if isinstance(node, ast.ClassDef):
            summary_end = min(
                end_lineno,
                node.lineno + 20,
                getattr(node.body[0], "lineno", end_lineno) - 1 if node.body else end_lineno,
            )
            summary_text = _slice_lines(lines, node.lineno, max(summary_end, node.lineno))
            summary_locator = f"class:{node.name}:summary:{node.lineno}:{summary_end}"
            records.append(
                ChunkRecord(
                    point_id=sha256(
                        f"{relative_path}|{summary_locator}|{summary_text}".encode("utf-8")
                    ).hexdigest(),
                    scope=scope,
                    relative_path=relative_path,
                    language="python",
                    chunk_type="class_summary",
                    text=summary_text,
                    start_line=node.lineno,
                    end_line=max(summary_end, node.lineno),
                    content_hash=sha256(summary_text.encode("utf-8")).hexdigest(),
                    source_mtime=source_mtime,
                    symbol_path=node.name,
                    domain_tags=domain_tags.copy(),
                    extra={"locator": summary_locator},
                )
            )

            for child in node.body:
                if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                child_end = getattr(child, "end_lineno", child.lineno)
                child_text = _slice_lines(lines, child.lineno, child_end)
                if not child_text:
                    continue
                symbol_path = f"{node.name}/{child.name}"
                locator = f"method:{symbol_path}:{child.lineno}:{child_end}"
                records.append(
                    ChunkRecord(
                        point_id=sha256(
                            f"{relative_path}|{locator}|{child_text}".encode("utf-8")
                        ).hexdigest(),
                        scope=scope,
                        relative_path=relative_path,
                        language="python",
                        chunk_type="python_method",
                        text=child_text,
                        start_line=child.lineno,
                        end_line=child_end,
                        content_hash=sha256(child_text.encode("utf-8")).hexdigest(),
                        source_mtime=source_mtime,
                        symbol_path=symbol_path,
                        domain_tags=domain_tags.copy(),
                        extra={"locator": locator},
                    )
                )
            continue

        symbol_name = node.name
        kind = "python_async_function" if isinstance(node, ast.AsyncFunctionDef) else "python_function"
        locator = f"function:{symbol_name}:{node.lineno}:{end_lineno}"
        records.append(
            ChunkRecord(
                point_id=sha256(
                    f"{relative_path}|{locator}|{node_text}".encode("utf-8")
                ).hexdigest(),
                scope=scope,
                relative_path=relative_path,
                language="python",
                chunk_type=kind,
                text=node_text,
                start_line=node.lineno,
                end_line=end_lineno,
                content_hash=sha256(node_text.encode("utf-8")).hexdigest(),
                source_mtime=source_mtime,
                symbol_path=symbol_name,
                domain_tags=domain_tags.copy(),
                extra={"locator": locator},
            )
        )

    if records:
        return records

    from services.repo_semantic.chunkers.generic import chunk_generic_file

    return chunk_generic_file(
        file_path=file_path,
        relative_path=relative_path,
        text=text,
        scope=scope,
        domain_tags=domain_tags,
        source_mtime=source_mtime,
    )

