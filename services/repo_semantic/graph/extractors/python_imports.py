"""Python module and import graph extraction."""

from __future__ import annotations

import ast

from services.repo_semantic.graph.extractors.types import (
    GraphExtractionContext,
    GraphExtractionRows,
    confidence_value,
    edge_id,
    file_node_id,
    module_node_id,
    payload_json,
    source_text_for_chunks,
    term_rows,
)

EXTRACTOR_NAME = "python_imports"
EXTRACTOR_VERSION = "python_imports.v1"
MAX_IMPORTS_PER_FILE = 200


def extract(context: GraphExtractionContext) -> GraphExtractionRows:
    rows = GraphExtractionRows()
    for relative_path, chunks in sorted(context.chunks_by_path.items()):
        module_name = context.module_by_path.get(relative_path)
        if module_name is None:
            continue
        first_chunk = chunks[0]
        module_node = module_node_id(module_name)
        file_node = file_node_id(relative_path)
        rows.nodes.append(
            {
                "node_id": module_node,
                "node_type": "module",
                "key": module_name,
                "relative_path": relative_path,
                "chunk_id": first_chunk.point_id,
                "start_line": first_chunk.start_line,
                "end_line": first_chunk.end_line,
                "payload_json": payload_json(
                    {
                        "language": "python",
                        "relative_path": relative_path,
                        "module_kind": "package" if relative_path.endswith("__init__.py") else "module",
                        "import_root": module_name.split(".", 1)[0],
                    }
                ),
            }
        )
        rows.node_terms.extend(term_rows(node_id_value=module_node, text=module_name, source=EXTRACTOR_NAME))
        rows.node_chunks.append(
            {
                "node_id": module_node,
                "chunk_id": first_chunk.point_id,
                "relation": "primary",
                "weight": 0.9,
            }
        )
        rows.edges.append(
            {
                "edge_id": edge_id(file_node, module_node, "file_defines_module"),
                "source_node_id": file_node,
                "target_node_id": module_node,
                "edge_type": "file_defines_module",
                "confidence": 1.0,
                "extractor": EXTRACTOR_NAME,
                "relative_path": relative_path,
                "payload_json": payload_json(
                    {
                        "confidence": "high",
                        "source": "ast",
                        "extractor_version": EXTRACTOR_VERSION,
                    }
                ),
            }
        )
        text = source_text_for_chunks(chunks)
        try:
            tree = ast.parse(text)
        except SyntaxError:
            rows.warning_codes.append("python_ast_parse_failed")
            continue
        current_is_package = relative_path.endswith("/__init__.py")
        for imported_module in _imports_for_tree(
            tree,
            module_name,
            current_is_package=current_is_package,
        )[:MAX_IMPORTS_PER_FILE]:
            target_module = _resolve_module(imported_module, context.path_by_module)
            if target_module is None:
                continue
            target_path = context.path_by_module[target_module]
            if target_module == module_name or target_path == relative_path:
                continue
            target_module_node = module_node_id(target_module)
            target_file_node = file_node_id(target_path)
            rows.edges.append(
                {
                    "edge_id": edge_id(module_node, target_module_node, "module_imports_module"),
                    "source_node_id": module_node,
                    "target_node_id": target_module_node,
                    "edge_type": "module_imports_module",
                    "confidence": confidence_value("high"),
                    "extractor": EXTRACTOR_NAME,
                    "relative_path": relative_path,
                    "payload_json": payload_json(
                        {
                            "confidence": "high",
                            "source": "ast",
                            "extractor_version": EXTRACTOR_VERSION,
                            "surface": imported_module,
                            "target_resolution": "resolved",
                        }
                    ),
                }
            )
            rows.edges.append(
                {
                    "edge_id": edge_id(file_node, target_file_node, "file_imports_file"),
                    "source_node_id": file_node,
                    "target_node_id": target_file_node,
                    "edge_type": "file_imports_file",
                    "confidence": confidence_value("high"),
                    "extractor": EXTRACTOR_NAME,
                    "relative_path": relative_path,
                    "payload_json": payload_json(
                        {
                            "confidence": "high",
                            "source": "ast",
                            "extractor_version": EXTRACTOR_VERSION,
                            "surface": imported_module,
                            "target_resolution": "resolved",
                        }
                    ),
                }
            )
    return rows


def _imports_for_tree(tree: ast.AST, current_module: str, *, current_is_package: bool = False) -> list[str]:
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            base = _absolute_import_from(node, current_module, current_is_package=current_is_package)
            if base:
                if node.module:
                    imports.append(base)
                for alias in node.names:
                    if alias.name != "*":
                        imports.append(f"{base}.{alias.name}")
    return list(dict.fromkeys(imports))


def _absolute_import_from(
    node: ast.ImportFrom,
    current_module: str,
    *,
    current_is_package: bool = False,
) -> str | None:
    module = node.module or ""
    if node.level <= 0:
        return module or None
    package_parts = current_module.split(".") if current_is_package else current_module.split(".")[:-1]
    if node.level > 1:
        package_parts = package_parts[: -(node.level - 1)] if node.level - 1 <= len(package_parts) else []
    parts = [*package_parts]
    if module:
        parts.extend(module.split("."))
    return ".".join(part for part in parts if part) or None


def _resolve_module(imported_module: str, path_by_module: dict[str, str]) -> str | None:
    candidate = imported_module
    while candidate:
        if candidate in path_by_module:
            return candidate
        if "." not in candidate:
            return None
        candidate = candidate.rsplit(".", 1)[0]
    return None
