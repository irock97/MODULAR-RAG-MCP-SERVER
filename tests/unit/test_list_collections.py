"""Unit tests for list_collections Tool.

These tests verify the list_collections functionality.
"""

import pytest
from unittest.mock import MagicMock, patch

from mcp_server.tools.list_collections import (
    CollectionInfo,
    ListCollectionsTool,
    TOOL_NAME,
    TOOL_INPUT_SCHEMA,
    get_tool_definition,
)


class TestCollectionInfo:
    """Tests for CollectionInfo dataclass."""

    def test_collection_info_creation(self):
        """Test creating a CollectionInfo instance."""
        col = CollectionInfo(name="test_collection", chunk_count=50)
        assert col.name == "test_collection"
        assert col.chunk_count == 50

    def test_collection_info_defaults(self):
        """Test CollectionInfo with default values."""
        col = CollectionInfo(name="test")
        assert col.name == "test"
        assert col.chunk_count == 0


class TestListCollectionsTool:
    """Tests for ListCollectionsTool."""

    def test_tool_name(self):
        """Test tool name is correct."""
        assert TOOL_NAME == "list_collections"

    def test_tool_input_schema(self):
        """Test tool input schema structure."""
        assert "type" in TOOL_INPUT_SCHEMA
        assert "properties" in TOOL_INPUT_SCHEMA
        assert "include_stats" in TOOL_INPUT_SCHEMA["properties"]

    def test_tool_creation(self):
        """Test creating a ListCollectionsTool instance."""
        tool = ListCollectionsTool()
        assert tool is not None

    def test_tool_creation_with_settings(self):
        """Test creating with custom settings."""
        mock_settings = MagicMock()
        tool = ListCollectionsTool(settings=mock_settings)
        assert tool._settings is mock_settings

    @pytest.mark.asyncio
    async def test_execute_empty_collections(self):
        """Test execute with no collections."""
        tool = ListCollectionsTool()

        with patch.object(tool, "_get_collections") as mock_get:
            mock_get.return_value = []

            response = await tool.execute(include_stats=True)

            assert response.is_empty is True
            assert "未找到" in response.content or "No" in response.content

    @pytest.mark.asyncio
    async def test_execute_with_collections(self):
        """Test execute with collections."""
        tool = ListCollectionsTool()

        collections = [
            CollectionInfo(name="collection1", chunk_count=20),
            CollectionInfo(name="collection2", chunk_count=50),
        ]

        with patch.object(tool, "_get_collections") as mock_get:
            mock_get.return_value = collections

            response = await tool.execute(include_stats=True)

            assert response.is_empty is False
            assert "collection1" in response.content
            assert "collection2" in response.content
            assert len(response.metadata["collections"]) == 2

    @pytest.mark.asyncio
    async def test_execute_without_stats(self):
        """Test execute without statistics."""
        tool = ListCollectionsTool()

        collections = [
            CollectionInfo(name="collection1"),
            CollectionInfo(name="collection2"),
        ]

        with patch.object(tool, "_get_collections") as mock_get:
            mock_get.return_value = collections

            response = await tool.execute(include_stats=False)

            assert response.is_empty is False
            assert "collection1" in response.content
            assert "文档数" not in response.content


class TestToolRegistration:
    """Tests for tool registration."""

    def test_get_tool_definition(self):
        """Test get_tool_definition returns correct structure."""
        definition = get_tool_definition()

        assert "name" in definition
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["name"] == "list_collections"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
