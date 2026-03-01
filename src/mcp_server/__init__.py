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
    GetDocumentSummaryTool,
    ListCollectionsTool,
    QueryKnowledgeHubTool,
    get_document_summary_tool_definition,
    get_document_summary_tool_instance,
    get_list_collections_tool_definition,
    get_list_collections_tool_instance,
    get_query_knowledge_hub_tool_definition,
    get_query_knowledge_hub_tool_instance,
    get_document_summary_handler,
    list_collections_handler,
    query_knowledge_hub_handler,
    register_document_summary_tool,
    register_list_collections_tool,
    register_query_knowledge_hub_tool,
)

__all__ = [
    "main",
    "run_stdio_server",
    "ProtocolHandler",
    "ToolDefinition",
    "JSONRPCErrorCodes",
    "create_mcp_server",
    "get_protocol_handler",
    "GetDocumentSummaryTool",
    "ListCollectionsTool",
    "QueryKnowledgeHubTool",
    "get_document_summary_tool_definition",
    "get_document_summary_tool_instance",
    "get_list_collections_tool_definition",
    "get_list_collections_tool_instance",
    "get_query_knowledge_hub_tool_definition",
    "get_query_knowledge_hub_tool_instance",
    "get_document_summary_handler",
    "list_collections_handler",
    "query_knowledge_hub_handler",
    "register_document_summary_tool",
    "register_list_collections_tool",
    "register_query_knowledge_hub_tool",
]
