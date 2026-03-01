# MCP Server Tools

from mcp_server.tools.get_document_summary import (
    GetDocumentSummaryTool,
    get_tool_definition as get_document_summary_tool_definition,
    get_tool_instance as get_document_summary_tool_instance,
    get_document_summary_handler,
    register_tool as register_document_summary_tool,
)
from mcp_server.tools.list_collections import (
    ListCollectionsTool,
    get_tool_definition as get_list_collections_tool_definition,
    get_tool_instance as get_list_collections_tool_instance,
    list_collections_handler,
    register_tool as register_list_collections_tool,
)
from mcp_server.tools.query_knowledge_hub import (
    QueryKnowledgeHubTool,
    get_tool_definition as get_query_knowledge_hub_tool_definition,
    get_tool_instance as get_query_knowledge_hub_tool_instance,
    query_knowledge_hub_handler,
    register_tool as register_query_knowledge_hub_tool,
)

__all__ = [
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
