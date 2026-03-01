# MCP Server Layer - Interface layer for external communication
"""MCP Server层提供JSON-RPC 2.0协议处理和工具暴露。"""

from mcp_server.server import main, run_stdio_server
from mcp_server.protocol_handler import (
    ProtocolHandler,
    ToolDefinition,
    JSONRPCErrorCodes,
    create_mcp_server,
    get_protocol_handler,
)
from mcp_server.tools import (
    QueryKnowledgeHubTool,
    get_tool_definition,
    get_tool_instance,
    query_knowledge_hub_handler,
    register_tool,
)

__all__ = [
    "main",
    "run_stdio_server",
    "ProtocolHandler",
    "ToolDefinition",
    "JSONRPCErrorCodes",
    "create_mcp_server",
    "get_protocol_handler",
    "QueryKnowledgeHubTool",
    "get_tool_definition",
    "get_tool_instance",
    "query_knowledge_hub_handler",
    "register_tool",
]
