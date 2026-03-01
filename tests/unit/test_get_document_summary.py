"""Unit tests for get_document_summary Tool.

These tests verify the get_document_summary functionality.
"""

import pytest
from unittest.mock import MagicMock, patch

from mcp_server.tools.get_document_summary import (
    DocumentSummary,
    GetDocumentSummaryTool,
    TOOL_NAME,
    TOOL_INPUT_SCHEMA,
    get_tool_definition,
)


class TestDocumentSummary:
    """Tests for DocumentSummary dataclass."""

    def test_document_summary_creation(self):
        """Test creating a DocumentSummary instance."""
        doc = DocumentSummary(
            doc_id="test_doc_123",
            title="Test Document",
            summary="This is a test summary",
            tags=["tag1", "tag2"],
            chunk_count=5,
        )
        assert doc.doc_id == "test_doc_123"
        assert doc.title == "Test Document"
        assert doc.summary == "This is a test summary"
        assert doc.tags == ["tag1", "tag2"]
        assert doc.chunk_count == 5

    def test_document_summary_defaults(self):
        """Test DocumentSummary with default values."""
        doc = DocumentSummary(doc_id="test")
        assert doc.doc_id == "test"
        assert doc.title == ""
        assert doc.summary == ""
        assert doc.tags == []
        assert doc.chunk_count == 0
        assert doc.source_path == ""


class TestGetDocumentSummaryTool:
    """Tests for GetDocumentSummaryTool."""

    def test_tool_name(self):
        """Test tool name is correct."""
        assert TOOL_NAME == "get_document_summary"

    def test_tool_input_schema(self):
        """Test tool input schema structure."""
        assert "type" in TOOL_INPUT_SCHEMA
        assert "properties" in TOOL_INPUT_SCHEMA
        assert "doc_id" in TOOL_INPUT_SCHEMA["properties"]
        assert "required" in TOOL_INPUT_SCHEMA
        assert "doc_id" in TOOL_INPUT_SCHEMA["required"]

    def test_tool_creation(self):
        """Test creating a GetDocumentSummaryTool instance."""
        tool = GetDocumentSummaryTool()
        assert tool is not None

    def test_tool_creation_with_settings(self):
        """Test creating with custom settings."""
        mock_settings = MagicMock()
        tool = GetDocumentSummaryTool(settings=mock_settings)
        assert tool._settings is mock_settings

    @pytest.mark.asyncio
    async def test_execute_empty_doc_id(self):
        """Test execute with empty doc_id."""
        tool = GetDocumentSummaryTool()

        response = await tool.execute(doc_id="")

        assert response.is_empty is True
        assert "error" in response.metadata
        assert "cannot be empty" in response.content.lower() or "错误" in response.content

    @pytest.mark.asyncio
    async def test_execute_whitespace_doc_id(self):
        """Test execute with whitespace doc_id."""
        tool = GetDocumentSummaryTool()

        response = await tool.execute(doc_id="   ")

        assert response.is_empty is True
        assert "error" in response.metadata

    @pytest.mark.asyncio
    async def test_execute_not_found(self):
        """Test execute with non-existent doc_id."""
        tool = GetDocumentSummaryTool()

        with patch.object(tool, "_get_document_summary") as mock_get:
            mock_get.return_value = None

            response = await tool.execute(doc_id="non_existent_doc")

            assert response.is_empty is True
            assert "未找到" in response.content or "not found" in response.content.lower()

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test execute with valid doc_id."""
        tool = GetDocumentSummaryTool()

        doc_summary = DocumentSummary(
            doc_id="abc123",
            title="Test Document",
            summary="This is a test summary",
            tags=["python", "testing"],
            chunk_count=10,
            source_path="/docs/test.pdf",
        )

        with patch.object(tool, "_get_document_summary") as mock_get:
            mock_get.return_value = doc_summary

            response = await tool.execute(doc_id="abc123")

            assert response.is_empty is False
            assert "Test Document" in response.content
            assert "This is a test summary" in response.content
            assert "python" in response.content
            assert "testing" in response.content
            assert response.metadata["chunk_count"] == 10

    @pytest.mark.asyncio
    async def test_execute_no_title(self):
        """Test execute with document that has no title."""
        tool = GetDocumentSummaryTool()

        doc_summary = DocumentSummary(
            doc_id="abc123",
            title="",
            summary="Some summary",
            tags=[],
            chunk_count=3,
        )

        with patch.object(tool, "_get_document_summary") as mock_get:
            mock_get.return_value = doc_summary

            response = await tool.execute(doc_id="abc123")

            assert response.is_empty is False
            assert "无标题文档" in response.content
            assert "Some summary" in response.content

    @pytest.mark.asyncio
    async def test_execute_no_tags(self):
        """Test execute with document that has no tags."""
        tool = GetDocumentSummaryTool()

        doc_summary = DocumentSummary(
            doc_id="abc123",
            title="Test",
            summary="Summary",
            tags=[],
            chunk_count=2,
        )

        with patch.object(tool, "_get_document_summary") as mock_get:
            mock_get.return_value = doc_summary

            response = await tool.execute(doc_id="abc123")

            assert response.is_empty is False
            assert "暂无标签" in response.content

    @pytest.mark.asyncio
    async def test_execute_no_summary(self):
        """Test execute with document that has no summary."""
        tool = GetDocumentSummaryTool()

        doc_summary = DocumentSummary(
            doc_id="abc123",
            title="Test",
            summary="",
            tags=["tag1"],
            chunk_count=1,
        )

        with patch.object(tool, "_get_document_summary") as mock_get:
            mock_get.return_value = doc_summary

            response = await tool.execute(doc_id="abc123")

            assert response.is_empty is False
            assert "暂无摘要" in response.content

    @pytest.mark.asyncio
    async def test_execute_with_collection_name(self):
        """Test execute with specific collection name."""
        tool = GetDocumentSummaryTool()

        doc_summary = DocumentSummary(
            doc_id="abc123",
            title="Test Document",
            summary="Test summary",
            tags=["tag1"],
            chunk_count=5,
        )

        with patch.object(tool, "_get_document_summary") as mock_get:
            mock_get.return_value = doc_summary

            response = await tool.execute(doc_id="abc123", collection_name="my_collection")

            assert response.is_empty is False
            # Verify collection_name was passed to _get_document_summary
            mock_get.assert_called_once_with("abc123", "my_collection")

    @pytest.mark.asyncio
    async def test_execute_not_found_with_collection(self):
        """Test execute with non-existent doc_id in specific collection."""
        tool = GetDocumentSummaryTool()

        with patch.object(tool, "_get_document_summary") as mock_get:
            mock_get.return_value = None

            response = await tool.execute(doc_id="non_existent", collection_name="test_collection")

            assert response.is_empty is True
            # Verify collection_name was passed
            mock_get.assert_called_once_with("non_existent", "test_collection")
            # Check that the response mentions the collection
            assert "test_collection" in response.content


class TestToolRegistration:
    """Tests for tool registration."""

    def test_get_tool_definition(self):
        """Test get_tool_definition returns correct structure."""
        definition = get_tool_definition()

        assert "name" in definition
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["name"] == "get_document_summary"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
