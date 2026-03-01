# MCP Server Layer - Interface layer for external communication
"""MCP Server层提供JSON-RPC 2.0协议处理和工具暴露。"""

from mcp_server.server import main, run_stdio_server

__all__ = ["main", "run_stdio_server"]
