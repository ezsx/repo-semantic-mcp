"""Test-case to target graph extraction."""

from __future__ import annotations

import ast
from pathlib import PurePosixPath

from services.repo_semantic.graph.extractors.python_imports import _absolute_import_from, _resolve_module
from services.repo_semantic.graph.extractors.types import (
    GraphExtractionContext,
    GraphExtractionRows,
    confidence_value,
    edge_id,
    file_node_id,
    normalize_path,
    payload_json,
    source_text_for_chunks,
    symbol_node_id,
    term_rows,
    test_case_node_id as _test_case_node_id,
)

EXTRACTOR_NAME = "test_targets"
EXTRACTOR_VERSION = "test_targets.v1"
MAX_TEST_CASES_PER_FILE = 200


def extract(context: GraphExtractionContext) -> GraphExtractionRows:
    rows = GraphExtractionRows()
    for relative_path, chunks in sorted(context.chunks_by_path.items()):
        if not _is_test_path(relative_path):
            continue
        text = source_text_for_chunks(chunks)
        try:
            tree = ast.parse(text)
        except SyntaxError:
            rows.warning_codes.append("test_target_ast_parse_failed")
            continue
        imports = _repo_imports(
            tree,
            context,
            context.module_by_path.get(relative_path, ""),
            current_is_package=relative_path.endswith("/__init__.py"),
        )
        target_by_name = {
            imported_name: (target_path, symbol_name)
            for imported_name, target_path, symbol_name in imports
            if imported_name
        }
        test_nodes = _test_nodes(tree)[:MAX_TEST_CASES_PER_FILE]
        if len(test_nodes) >= MAX_TEST_CASES_PER_FILE:
            rows.warning_codes.append("test_target_cap_reached")
        naming_target = _naming_target(relative_path, context)
        for test_name, line, ast_node in test_nodes:
            test_chunk = context.chunk_for_line(relative_path, line)
            if test_chunk is None:
                continue
            test_node = _test_case_node_id(relative_path, test_name)
            rows.nodes.append(
                {
                    "node_id": test_node,
                    "node_type": "test_case",
                    "key": f"{relative_path}::{test_name}",
                    "relative_path": relative_path,
                    "chunk_id": test_chunk.point_id,
                    "start_line": test_chunk.start_line,
                    "end_line": test_chunk.end_line,
                    "payload_json": payload_json(
                        {
                            "test_framework": "pytest",
                            "target_inference": "import",
                            "confidence": "high",
                        }
                    ),
                }
            )
            rows.node_terms.extend(term_rows(node_id_value=test_node, text=test_name, source=EXTRACTOR_NAME))
            rows.node_chunks.append(
                {
                    "node_id": test_node,
                    "chunk_id": test_chunk.point_id,
                    "relation": "primary",
                    "weight": 1.0,
                }
            )
            referenced_names = _referenced_names(ast_node)
            emitted_target = False
            for imported_name, (target_path, symbol_name) in target_by_name.items():
                if imported_name not in referenced_names:
                    continue
                target_file = file_node_id(target_path)
                rows.edges.append(_edge(test_node, target_file, "test_targets_file", relative_path, "high", "import"))
                emitted_target = True
                if symbol_name and (target_path, symbol_name) in context.known_symbols:
                    target_symbol = symbol_node_id(target_path, symbol_name)
                    rows.edges.append(
                        _edge(test_node, target_symbol, "test_targets_symbol", relative_path, "high", "import")
                    )
            if not emitted_target and naming_target:
                rows.edges.append(
                    _edge(test_node, file_node_id(naming_target), "test_targets_file", relative_path, "medium", "naming")
                )
    return rows


def _edge(
    source_node_id: str,
    target_node_id: str,
    edge_type: str,
    relative_path: str,
    confidence: str,
    source: str,
) -> dict[str, object]:
    return {
        "edge_id": edge_id(source_node_id, target_node_id, edge_type),
        "source_node_id": source_node_id,
        "target_node_id": target_node_id,
        "edge_type": edge_type,
        "confidence": confidence_value(confidence),
        "extractor": EXTRACTOR_NAME,
        "relative_path": relative_path,
        "payload_json": payload_json(
            {
                "confidence": confidence,
                "source": source,
                "extractor_version": EXTRACTOR_VERSION,
            }
        ),
    }


def _is_test_path(relative_path: str) -> bool:
    path = normalize_path(relative_path)
    name = PurePosixPath(path).name
    return path.startswith("tests/") or name.startswith("test_") or name.endswith("_test.py")


def _repo_imports(
    tree: ast.AST,
    context: GraphExtractionContext,
    current_module: str,
    *,
    current_is_package: bool = False,
) -> list[tuple[str, str, str | None]]:
    imports: list[tuple[str, str, str | None]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                target_module = _resolve_module(alias.name, context.path_by_module)
                if target_module:
                    imports.append((alias.asname or alias.name.rsplit(".", 1)[-1], context.path_by_module[target_module], None))
        elif isinstance(node, ast.ImportFrom):
            base = _absolute_import_from(node, current_module, current_is_package=current_is_package)
            if not base:
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                imported_name = alias.asname or alias.name
                submodule_path = context.path_by_module.get(f"{base}.{alias.name}")
                if submodule_path is not None:
                    imports.append((imported_name, submodule_path, None))
                    continue
                if node.module:
                    target_module = _resolve_module(base, context.path_by_module)
                    if target_module is not None:
                        imports.append((imported_name, context.path_by_module[target_module], alias.name))
    return imports


def _test_nodes(tree: ast.AST) -> list[tuple[str, int, ast.AST]]:
    nodes: list[tuple[str, int, ast.AST]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            nodes.append((node.name, node.lineno, node))
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            nodes.append((node.name, node.lineno, node))
    return sorted(nodes, key=lambda item: (item[1], item[0]))


def _referenced_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            names.add(child.id)
        elif isinstance(child, ast.Attribute):
            names.add(child.attr)
    return names


def _naming_target(relative_path: str, context: GraphExtractionContext) -> str | None:
    name = PurePosixPath(relative_path).name
    stem = name[:-3] if name.endswith(".py") else name
    for prefix in ("test_",):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
    if stem.endswith("_test"):
        stem = stem[: -len("_test")]
    if not stem:
        return None
    candidates = [
        path
        for path in context.known_files
        if path.endswith(f"/{stem}.py") and path != normalize_path(relative_path) and not _is_test_path(path)
    ]
    return candidates[0] if len(candidates) == 1 else None
