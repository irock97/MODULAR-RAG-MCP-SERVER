"""MCP Tool: list_collections

This tool lists available document collections with their statistics.

Usage via MCP:
    Tool name: list_collections
    Input schema:
        - include_stats (boolean, optional): Include detailed statistics for each collection
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from mcp import types

from core.response.response_builder import MCPToolResponse
from core.settings import Settings, load_settings

if TYPE_CHECKING:
    import chromadb
    from chromadb.config import Settings as ChromaSettings

logger = logging.getLogger(__name__)


# Tool metadata
TOOL_NAME = "list_collections"
TOOL_DESCRIPTION = """List available document collections.

This tool returns a list of all document collections that have been created
in the knowledge base, along with optional statistics like document count
and chunk count.

Parameters:
- include_stats: Whether to include detailed statistics for each collection
"""

TOOL_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "include_stats": {
            "type": "boolean",
            "description": "Include detailed statistics for each collection.",
            "default": True,
        },
    },
    "required": [],
}


@dataclass
class CollectionInfo:
    """Information about a collection.

    Attributes:
        name: Collection name
        chunk_count: Number of text chunks
    """

    name: str
    chunk_count: int = 0


class ListCollectionsTool:
    """MCP Tool for listing available collections.

    This class retrieves all collections from the ChromaDB persist directory
    and returns their names along with optional statistics.

    Design Principles:
    - Configurable: Statistics inclusion is optional
    - Error resilient: Graceful handling when ChromaDB is unavailable
    - Fast: Minimal API calls when stats not needed
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        persist_directory: Optional[str] = None,
    ) -> None:
        """Initialize ListCollectionsTool.

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
        include_stats: bool = True,
    ) -> MCPToolResponse:
        """Execute the list_collections tool.

        Args:
            include_stats: Whether to include collection statistics.

        Returns:
            MCPToolResponse with collection list and optional stats.
        """
        logger.info(f"Executing list_collections: include_stats={include_stats}")

        try:
            # Get collections
            collections = self._get_collections(include_stats)

            if not collections:
                return self._build_empty_response()

            # Build response
            response = self._build_response(collections, include_stats)

            logger.info(f"list_collections completed: {len(collections)} collections")
            return response

        except Exception as e:
            logger.exception(f"list_collections failed: {e}")
            return self._build_error_response(str(e))

    def _get_collections(
        self,
        include_stats: bool,
    ) -> list[CollectionInfo]:
        """Get list of collections with optional statistics.

        Args:
            include_stats: Whether to get statistics.

        Returns:
            List of CollectionInfo objects.
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

        # Get all collections
        chroma_collections = client.list_collections()

        collections = []
        for chroma_col in chroma_collections:
            col_info = CollectionInfo(name=chroma_col.name)

            if include_stats:
                try:
                    # Get count from the collection
                    col_info.chunk_count = chroma_col.count()
                except Exception as e:
                    logger.warning(f"Failed to get stats for collection {chroma_col.name}: {e}")
                    col_info.chunk_count = 0

            collections.append(col_info)

        # Sort by name
        collections.sort(key=lambda c: c.name)

        return collections

    def _build_response(
        self,
        collections: list[CollectionInfo],
        include_stats: bool,
    ) -> MCPToolResponse:
        """Build response with collection list.

        Args:
            collections: List of collections.
            include_stats: Whether stats were included.

        Returns:
            MCPToolResponse with formatted content.
        """
        lines = []

        # Header
        lines.append(f"## 可用文档集合\n")
        lines.append(f"共找到 {len(collections)} 个集合:\n")

        # Collection list
        for i, col in enumerate(collections, 1):
            lines.append(f"### {i}. {col.name}")

            if include_stats:
                lines.append(f"- **Chunk数:** {col.chunk_count}")

            lines.append("")

        # Summary
        total_chunks = sum(c.chunk_count for c in collections)

        lines.append("---")
        lines.append(f"**总计:** {len(collections)} 个集合, {total_chunks} Chunks")

        content = "\n".join(lines)

        return MCPToolResponse(
            content=content,
            citations=[],
            metadata={
                "collection_count": len(collections),
                "total_chunks": total_chunks,
                "collections": [
                    {
                        "name": c.name,
                        "chunk_count": c.chunk_count,
                    }
                    for c in collections
                ],
            },
            is_empty=False,
        )

    def _build_empty_response(self) -> MCPToolResponse:
        """Build response for empty collection list.

        Returns:
            MCPToolResponse indicating no collections found.
        """
        content = "## 可用文档集合\n\n"
        content += "未找到任何文档集合。\n\n"
        content += "**提示:** 请先运行文档摄入脚本以创建集合。\n"

        return MCPToolResponse(
            content=content,
            citations=[],
            metadata={"collection_count": 0},
            is_empty=True,
        )

    def _build_error_response(self, error_message: str) -> MCPToolResponse:
        """Build error response.

        Args:
            error_message: Error description.

        Returns:
            MCPToolResponse indicating error.
        """
        content = "## 获取集合列表失败\n\n"
        content += f"**错误信息:** {error_message}\n\n"
        content += "请检查:\n"
        content += "- ChromaDB 服务是否正常运行\n"
        content += "- 数据目录是否存在\n"

        return MCPToolResponse(
            content=content,
            citations=[],
            metadata={"error": error_message},
            is_empty=True,
        )


# Module-level tool instance (lazy-initialized)
_tool_instance: Optional[ListCollectionsTool] = None


def get_tool_instance(settings: Optional[Settings] = None) -> ListCollectionsTool:
    """Get or create the tool instance.

    Args:
        settings: Optional settings to use for initialization.

    Returns:
        ListCollectionsTool instance.
    """
    global _tool_instance
    if _tool_instance is None:
        _tool_instance = ListCollectionsTool(settings=settings)
    return _tool_instance


async def list_collections_handler(
    arguments: dict[str, Any],
) -> types.CallToolResult:
    """Handler function for MCP tool registration.

    This function is registered with the ProtocolHandler and called
    when the MCP client invokes the list_collections tool.

    Args:
        arguments: Tool arguments containing include_stats.

    Returns:
        MCP CallToolResult with content blocks.
    """
    include_stats = arguments.get("include_stats", True)

    tool = get_tool_instance()

    try:
        response = await tool.execute(include_stats=include_stats)

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
        logger.exception(f"list_collections handler error: {e}")
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=f"内部错误: 获取集合列表失败",
                )
            ],
            isError=True,
        )


def register_tool(protocol_handler) -> None:
    """Register list_collections tool with the protocol handler.

    Args:
        protocol_handler: ProtocolHandler instance to register with.
    """
    protocol_handler.register_tool(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        input_schema=TOOL_INPUT_SCHEMA,
        handler=list_collections_handler,
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
