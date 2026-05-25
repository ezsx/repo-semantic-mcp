"""Python route extraction for common web and bot handler patterns."""

from __future__ import annotations

import ast

from services.repo_semantic.graph.extractors.types import (
    GraphExtractionContext,
    GraphExtractionRows,
    chunk_node_id,
    confidence_value,
    edge_id,
    normalize_route,
    payload_json,
    route_key,
    route_node_id,
    route_term_rows,
    source_text_for_chunks,
    symbol_node_id,
    term_rows,
)

EXTRACTOR_NAME = "python_routes"
EXTRACTOR_VERSION = "python_routes.v1"
MAX_ROUTES_PER_FILE = 120
HTTP_METHODS = {"get", "post", "put", "patch", "delete", "options", "head"}


def extract(context: GraphExtractionContext) -> GraphExtractionRows:
    rows = GraphExtractionRows()
    route_nodes_by_route: dict[str, list[str]] = {}
    for relative_path, chunks in sorted(context.chunks_by_path.items()):
        if not relative_path.endswith(".py"):
            continue
        text = source_text_for_chunks(chunks)
        try:
            tree = ast.parse(text)
        except SyntaxError:
            rows.warning_codes.append("python_route_ast_parse_failed")
            continue
        emitted = 0
        function_nodes = sorted(
            [
                node
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ],
            key=lambda item: (item.lineno, item.name),
        )
        functions_by_name: dict[str, ast.FunctionDef | ast.AsyncFunctionDef | None] = {}
        for node in function_nodes:
            functions_by_name[node.name] = node if node.name not in functions_by_name else None
        for node in function_nodes:
            for route in _routes_for_function(node):
                if emitted >= MAX_ROUTES_PER_FILE:
                    rows.warning_codes.append("python_route_cap_reached")
                    break
                if _emit_route(
                    rows=rows,
                    route_nodes_by_route=route_nodes_by_route,
                    context=context,
                    relative_path=relative_path,
                    handler_name=node.name,
                    handler_symbol_path=_handler_symbol_path(context, relative_path, node.lineno, node.name),
                    handler_line=node.lineno,
                    route=route,
                ):
                    emitted += 1
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for route in _routes_from_registration_call(node):
                if emitted >= MAX_ROUTES_PER_FILE:
                    rows.warning_codes.append("python_route_cap_reached")
                    break
                handler_name = route.get("handler_name")
                handler_node = functions_by_name.get(handler_name or "")
                resolved_handler_name = handler_name if handler_node is not None else None
                handler_line = handler_node.lineno if handler_node is not None else getattr(node, "lineno", None)
                if _emit_route(
                    rows=rows,
                    route_nodes_by_route=route_nodes_by_route,
                    context=context,
                    relative_path=relative_path,
                    handler_name=resolved_handler_name,
                    handler_symbol_path=_handler_symbol_path(
                        context,
                        relative_path,
                        handler_line,
                        resolved_handler_name,
                    ),
                    handler_line=handler_line,
                    route=route,
                ):
                    emitted += 1
    context.route_nodes_by_route = {
        route: list(dict.fromkeys(node_ids))
        for route, node_ids in route_nodes_by_route.items()
    }
    return rows


def _emit_route(
    *,
    rows: GraphExtractionRows,
    route_nodes_by_route: dict[str, list[str]],
    context: GraphExtractionContext,
    relative_path: str,
    handler_name: str | None,
    handler_symbol_path: str | None,
    handler_line: int | None,
    route: dict[str, str | None],
) -> bool:
    handler_chunk = context.chunk_for_line(relative_path, handler_line)
    if handler_chunk is None:
        return False
    surface = route["surface"]
    if surface is None:
        return False
    normalized = normalize_route(surface)
    method = route["method"]
    route_node = route_node_id("http", method, normalized)
    route_nodes_by_route.setdefault(normalized, []).append(route_node)
    chunk_node = chunk_node_id(handler_chunk.point_id)
    rows.nodes.append(
        {
            "node_id": route_node,
            "node_type": "route",
            "key": route_key(route_kind="http", method=method, normalized_route=normalized),
            "relative_path": relative_path,
            "chunk_id": handler_chunk.point_id,
            "start_line": handler_chunk.start_line,
            "end_line": handler_chunk.end_line,
            "payload_json": payload_json(
                {
                    "route_kind": "http",
                    "method": method,
                    "framework": route["framework"],
                    "normalized_route": normalized,
                    "surface": surface,
                    "confidence": "high",
                }
            ),
        }
    )
    rows.node_terms.extend(
        route_term_rows(
            node_id_value=route_node,
            surface=surface,
            normalized_route=normalized,
            method=method,
            source=EXTRACTOR_NAME,
        )
    )
    rows.node_chunks.append(
        {
            "node_id": route_node,
            "chunk_id": handler_chunk.point_id,
            "relation": "primary",
            "weight": 1.0,
        }
    )
    rows.edges.append(
        {
            "edge_id": edge_id(route_node, chunk_node, "route_defined_in_chunk"),
            "source_node_id": route_node,
            "target_node_id": chunk_node,
            "edge_type": "route_defined_in_chunk",
            "confidence": confidence_value("high"),
            "extractor": EXTRACTOR_NAME,
            "relative_path": relative_path,
            "payload_json": _route_payload(route=route, normalized=normalized),
        }
    )
    symbol_key = handler_symbol_path or handler_name
    if symbol_key:
        symbol_node = symbol_node_id(relative_path, symbol_key)
        rows.nodes.append(
            {
                "node_id": symbol_node,
                "node_type": "symbol",
                "key": symbol_key,
                "relative_path": relative_path,
                "chunk_id": handler_chunk.point_id,
                "start_line": handler_chunk.start_line,
                "end_line": handler_chunk.end_line,
                "payload_json": payload_json({"language": "python", "source": EXTRACTOR_NAME}),
            }
        )
        rows.node_terms.extend(term_rows(node_id_value=symbol_node, text=symbol_key, source="symbol"))
        rows.node_chunks.append(
            {
                "node_id": symbol_node,
                "chunk_id": handler_chunk.point_id,
                "relation": "primary",
                "weight": 1.0,
            }
        )
        rows.edges.append(
            {
                "edge_id": edge_id(route_node, symbol_node, "route_handled_by_symbol"),
                "source_node_id": route_node,
                "target_node_id": symbol_node,
                "edge_type": "route_handled_by_symbol",
                "confidence": confidence_value("high"),
                "extractor": EXTRACTOR_NAME,
                "relative_path": relative_path,
                "payload_json": _route_payload(route=route, normalized=normalized),
            }
        )
    return True


def _handler_symbol_path(
    context: GraphExtractionContext,
    relative_path: str,
    handler_line: int | None,
    fallback_name: str | None,
) -> str | None:
    if fallback_name is None:
        return None
    chunk = context.chunk_for_line(relative_path, handler_line)
    if chunk is not None and chunk.symbol_path:
        return chunk.symbol_path
    return fallback_name


def _route_payload(*, route: dict[str, str | None], normalized: str) -> str:
    return payload_json(
        {
            "confidence": "high",
            "source": "ast",
            "extractor_version": EXTRACTOR_VERSION,
            "surface": route["surface"],
            "normalized": normalized,
            "method": route["method"],
            "framework": route["framework"],
        }
    )


def _routes_for_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[dict[str, str | None]]:
    routes: list[dict[str, str | None]] = []
    for decorator in node.decorator_list:
        routes.extend(_routes_from_decorator_call(decorator))
    return routes


def _routes_from_decorator_call(node: ast.AST) -> list[dict[str, str | None]]:
    if not isinstance(node, ast.Call):
        return []
    attr = node.func.attr.lower() if isinstance(node.func, ast.Attribute) else ""
    if attr in HTTP_METHODS and node.args:
        surface = _literal_string(node.args[0])
        if surface:
            return [{"surface": surface, "method": attr.upper(), "framework": "fastapi"}]
    if attr == "route" and node.args:
        surface = _literal_string(node.args[0])
        if surface:
            return [
                {"surface": surface, "method": method, "framework": "flask"}
                for method in _methods_from_keywords(node.keywords)
            ]
    return []


def _routes_from_registration_call(node: ast.Call) -> list[dict[str, str | None]]:
    attr = node.func.attr.lower() if isinstance(node.func, ast.Attribute) else ""
    name = node.func.id if isinstance(node.func, ast.Name) else ""
    if attr == "add_api_route" and len(node.args) >= 2:
        surface = _literal_string(node.args[0])
        handler_name = _handler_name(node.args[1])
        if surface and handler_name:
            return [
                {
                    "surface": surface,
                    "method": method,
                    "framework": "fastapi",
                    "handler_name": handler_name,
                }
                for method in _methods_from_keywords(node.keywords)
            ]
    if attr == "add_url_rule" and node.args:
        surface = _literal_string(node.args[0])
        handler_name = _keyword_handler(node.keywords, "view_func")
        if handler_name is None and len(node.args) >= 3:
            handler_name = _handler_name(node.args[2])
        if surface and handler_name:
            return [
                {
                    "surface": surface,
                    "method": method,
                    "framework": "flask",
                    "handler_name": handler_name,
                }
                for method in _methods_from_keywords(node.keywords)
            ]
    if name in {"path", "re_path"} and len(node.args) >= 2:
        surface = _literal_string(node.args[0])
        handler_name = _handler_name(node.args[1])
        if surface and handler_name:
            return [
                {
                    "surface": surface,
                    "method": None,
                    "framework": "django",
                    "handler_name": handler_name,
                }
            ]
    return []


def _literal_string(node: ast.AST) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _methods_from_keywords(keywords: list[ast.keyword]) -> list[str | None]:
    for keyword in keywords:
        if keyword.arg != "methods":
            continue
        value = keyword.value
        if isinstance(value, (ast.List, ast.Tuple)):
            methods: list[str] = []
            for item in value.elts:
                method = _literal_string(item)
                if method:
                    methods.append(method.upper())
            return methods or [None]
    return [None]


def _handler_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _keyword_handler(keywords: list[ast.keyword], name: str) -> str | None:
    for keyword in keywords:
        if keyword.arg == name:
            return _handler_name(keyword.value)
    return None
