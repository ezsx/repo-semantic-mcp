from __future__ import annotations

from dataclasses import dataclass
import sys
import types


class FakeFastMCP:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.tool_registrations: list[dict[str, object]] = []
        self.resource_registrations: list[dict[str, object]] = []
        self.custom_route_registrations: list[dict[str, object]] = []

    def tool(self, *args, **kwargs):
        def decorator(function):
            self.tool_registrations.append(
                {"name": function.__name__, "args": args, "kwargs": kwargs}
            )
            return function

        return decorator

    def resource(self, *args, **kwargs):
        def decorator(function):
            self.resource_registrations.append(
                {"name": function.__name__, "args": args, "kwargs": kwargs}
            )
            return function

        return decorator

    def custom_route(self, *args, **kwargs):
        def decorator(function):
            self.custom_route_registrations.append(
                {"name": function.__name__, "args": args, "kwargs": kwargs}
            )
            return function

        return decorator


@dataclass(slots=True)
class McpStubHandle:
    previous_modules: dict[str, types.ModuleType | None]

    def restore(self) -> None:
        for name, module in self.previous_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def install_mcp_stub() -> McpStubHandle:
    module_names = ["mcp", "mcp.server", "mcp.server.fastmcp"]
    handle = McpStubHandle(
        previous_modules={name: sys.modules.get(name) for name in module_names}
    )
    mcp_module = types.ModuleType("mcp")
    server_module = types.ModuleType("mcp.server")
    fastmcp_module = types.ModuleType("mcp.server.fastmcp")
    fastmcp_module.FastMCP = FakeFastMCP
    sys.modules["mcp"] = mcp_module
    sys.modules["mcp.server"] = server_module
    sys.modules["mcp.server.fastmcp"] = fastmcp_module
    return handle
