"""MCP Tool: get_document_summary

This tool retrieves document summary information (title, summary, tags)
by querying the knowledge base with a document ID.

Usage via MCP:
    Tool name: get_document_summary
    Input schema:
        - doc_id (string, required): The document ID to query
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from mcp import types

from core.response.response_builder import MCPToolResponse
from core.settings import Settings, load_settings

if TYPE_CHECKING:
    import chromadb
    from chromadb.config import Settings as ChromaSettings

logger = logging.getLogger(__name__)


# Tool metadata
TOOL_NAME = "get_document_summary"
TOOL_DESCRIPTION = """Get document summary by document ID.

This tool retrieves structured information about a document including:
- title: Document title
- summary: Document summary/description
- tags: Document tags/categories
- chunk_count: Number of chunks in the document

Parameters:
- doc_id: The unique document ID (required)
- collection_name: The collection to search in (optional). If not specified, searches all collections.
"""

TOOL_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "doc_id": {
            "type": "string",
            "description": "The unique document ID to query.",
        },
        "collection_name": {
            "type": "string",
            "description": "The collection to search in. If not specified, searches all collections.",
        },
    },
    "required": ["doc_id"],
}


@dataclass
class DocumentSummary:
    """Document summary information.

    Attributes:
        doc_id: Document unique identifier
        title: Document title
        summary: Document summary/description
        tags: List of document tags
        chunk_count: Number of chunks in the document
        source_path: Source file path (if available)
    """

    doc_id: str
    title: str = ""
    summary: str = ""
    tags: list[str] = None
    chunk_count: int = 0
    source_path: str = ""

    def __post_init__(self) -> None:
        if self.tags is None:
            self.tags = []


class GetDocumentSummaryTool:
    """MCP Tool for retrieving document summary by ID.

    This class queries ChromaDB to find all chunks belonging to a document
    and extracts the document-level metadata (title, summary, tags).

    Design Principles:
    - Fail-fast: Invalid doc_id returns clear error message
    - Efficient: Single query to get all document chunks
    - Graceful: Missing metadata fields return empty strings/lists
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        persist_directory: Optional[str] = None,
    ) -> None:
        """Initialize GetDocumentSummaryTool.

        Args:
            settings: Application settings. If None, loaded from default path.
            persist_directory: Override persist directory. If None, uses settings.
        """
        self._settings = settings
        self._persist_directory = persist_directory

    @property
    def settings(self) -> Settings:
        """Get settings, loading if necessary."""
        if self._settings is None:
            self._settings = load_settings()
        return self._settings

    @property
    def persist_directory(self) -> str:
        """Get persist directory."""
        if self._persist_directory is None:
            vs_config = getattr(self.settings, "vector_store", None)
            if vs_config:
                self._persist_directory = getattr(vs_config, "persist_directory", None)
            if not self._persist_directory:
                self._persist_directory = "./data/db/chroma"
        return self._persist_directory

    async def execute(
        self,
        doc_id: str,
        collection_name: Optional[str] = None,
    ) -> MCPToolResponse:
        """Execute the get_document_summary tool.

        Args:
            doc_id: The document ID to query.
            collection_name: Optional collection to search in. If None, searches all collections.

        Returns:
            MCPToolResponse with document summary.
        """
        logger.info(f"Executing get_document_summary: doc_id={doc_id}, collection={collection_name}")

        # Validate input
        if not doc_id or not doc_id.strip():
            return self._build_error_response("doc_id cannot be empty")

        try:
            # Get document info from ChromaDB
            doc_summary = self._get_document_summary(doc_id.strip(), collection_name)

            if doc_summary is None:
                return self._build_not_found_response(doc_id, collection_name)

            # Build success response
            response = self._build_response(doc_summary)

            logger.info(f"get_document_summary completed: doc_id={doc_id}, chunks={doc_summary.chunk_count}")
            return response

        except Exception as e:
            logger.exception(f"get_document_summary failed: {e}")
            return self._build_error_response(str(e))

    def _get_document_summary(
        self,
        doc_id: str,
        collection_name: Optional[str] = None,
    ) -> Optional[DocumentSummary]:
        """Get document summary from ChromaDB.

        Args:
            doc_id: Document ID to query.
            collection_name: Optional collection to search in. If None, searches all collections.

        Returns:
            DocumentSummary if found, None if not found.
        """
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        # Connect to ChromaDB
        client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=False,
            ),
        )

        # Determine which collections to search
        if collection_name:
            # Search only the specified collection
            collections = [client.get_or_create_collection(name=collection_name)]
        else:
            # Search all collections
            collections = client.list_collections()

        # Search across collections
        for collection in collections:
            try:
                # Query by doc_id in metadata
                results = collection.get(
                    where={"doc_id": doc_id},
                    limit=1,  # We only need the first chunk for metadata
                )

                if results and results.get("ids") and len(results["ids"]) > 0:
                    # Found the document in this collection
                    metadatas = results.get("metadatas", [])
                    metadata = metadatas[0] if metadatas else {}

                    # Get all chunks count
                    all_results = collection.get(
                        where={"doc_id": doc_id},
                    )
                    chunk_count = len(all_results.get("ids", [])) if all_results else 0

                    return DocumentSummary(
                        doc_id=doc_id,
                        title=metadata.get("title", ""),
                        summary=metadata.get("summary", ""),
                        tags=metadata.get("tags", []),
                        chunk_count=chunk_count,
                        source_path=metadata.get("source_path", ""),
                    )
            except Exception as e:
                logger.warning(f"Failed to query collection {collection.name}: {e}")
                continue

        # Document not found in any collection
        return None

    def _build_response(
        self,
        doc_summary: DocumentSummary,
    ) -> MCPToolResponse:
        """Build response with document summary.

        Args:
            doc_summary: Document summary data.

        Returns:
            MCPToolResponse with formatted content.
        """
        lines = []

        # Header
        lines.append(f"## 文档摘要\n")
        lines.append(f"**文档ID:** `{doc_summary.doc_id}`\n")

        # Document info
        if doc_summary.title:
            lines.append(f"### {doc_summary.title}\n")
        else:
            lines.append(f"### 无标题文档\n")

        # Summary
        lines.append("**摘要:**")
        if doc_summary.summary:
            lines.append(f"{doc_summary.summary}\n")
        else:
            lines.append("_暂无摘要_\n")

        # Tags
        lines.append("**标签:**")
        if doc_summary.tags:
            tags_str = ", ".join(f"`{tag}`" for tag in doc_summary.tags)
            lines.append(f"{tags_str}\n")
        else:
            lines.append("_暂无标签_\n")

        # Stats
        lines.append("---")
        lines.append(f"**Chunk数:** {doc_summary.chunk_count}")
        if doc_summary.source_path:
            lines.append(f"**源文件:** `{doc_summary.source_path}`")

        content = "\n".join(lines)

        return MCPToolResponse(
            content=content,
            citations=[],
            metadata={
                "doc_id": doc_summary.doc_id,
                "title": doc_summary.title,
                "summary": doc_summary.summary,
                "tags": doc_summary.tags,
                "chunk_count": doc_summary.chunk_count,
                "source_path": doc_summary.source_path,
            },
            is_empty=False,
        )

    def _build_not_found_response(
        self,
        doc_id: str,
        collection_name: Optional[str] = None,
    ) -> MCPToolResponse:
        """Build response for not found document.

        Args:
            doc_id: The document ID that was not found.
            collection_name: The collection that was searched (if specified).

        Returns:
            MCPToolResponse indicating document not found.
        """
        content = f"## 文档未找到\n\n"
        content += f"**文档ID:** `{doc_id}`\n\n"
        if collection_name:
            content += f"**搜索集合:** `{collection_name}`\n\n"
        content += "未找到该文档的信息。\n\n"
        content += "**可能的原因:**\n"
        content += "- 文档ID不正确\n"
        content += "- 文档尚未被摄入到知识库\n"
        content += "- 文档已被删除\n"
        if collection_name:
            content += f"- 文档不在集合 `{collection_name}` 中\n"

        return MCPToolResponse(
            content=content,
            citations=[],
            metadata={"doc_id": doc_id, "collection_name": collection_name, "error": "not_found"},
            is_empty=True,
        )

    def _build_error_response(self, error_message: str) -> MCPToolResponse:
        """Build error response.

        Args:
            error_message: Error description.

        Returns:
            MCPToolResponse indicating error.
        """
        content = "## 获取文档摘要失败\n\n"
        content += f"**错误信息:** {error_message}\n\n"
        content += "请检查:\n"
        content += "- ChromaDB 服务是否正常运行\n"
        content += "- 文档ID格式是否正确\n"

        return MCPToolResponse(
            content=content,
            citations=[],
            metadata={"error": error_message},
            is_empty=True,
        )


# Module-level tool instance (lazy-initialized)
_tool_instance: Optional[GetDocumentSummaryTool] = None


def get_tool_instance(settings: Optional[Settings] = None) -> GetDocumentSummaryTool:
    """Get or create the tool instance.

    Args:
        settings: Optional settings to use for initialization.

    Returns:
        GetDocumentSummaryTool instance.
    """
    global _tool_instance
    if _tool_instance is None:
        _tool_instance = GetDocumentSummaryTool(settings=settings)
    return _tool_instance


async def get_document_summary_handler(
    arguments: dict[str, Any],
) -> types.CallToolResult:
    """Handler function for MCP tool registration.

    This function is registered with the ProtocolHandler and called
    when the MCP client invokes the get_document_summary tool.

    Args:
        arguments: Tool arguments containing doc_id and optional collection_name.

    Returns:
        MCP CallToolResult with content blocks.
    """
    doc_id = arguments.get("doc_id", "")
    collection_name = arguments.get("collection_name")

    tool = get_tool_instance()

    try:
        response = await tool.execute(
            doc_id=doc_id,
            collection_name=collection_name,
        )

        # Convert to MCP content blocks
        content_blocks = [
            types.TextContent(
                type="text",
                text=response.content,
            )
        ]

        return types.CallToolResult(
            content=content_blocks,
            isError=response.is_empty and "error" in response.metadata,
        )

    except Exception as e:
        logger.exception(f"get_document_summary handler error: {e}")
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=f"内部错误: 获取文档摘要失败",
                )
            ],
            isError=True,
        )


def register_tool(protocol_handler) -> None:
    """Register get_document_summary tool with the protocol handler.

    Args:
        protocol_handler: ProtocolHandler instance to register with.
    """
    protocol_handler.register_tool(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        input_schema=TOOL_INPUT_SCHEMA,
        handler=get_document_summary_handler,
    )
    logger.info(f"Registered MCP tool: {TOOL_NAME}")


def get_tool_definition() -> dict[str, Any]:
    """Get the tool definition for registration.

    Returns:
        Dictionary with name, description, and input_schema
    """
    return {
        "name": TOOL_NAME,
        "description": TOOL_DESCRIPTION,
        "input_schema": TOOL_INPUT_SCHEMA,
    }
