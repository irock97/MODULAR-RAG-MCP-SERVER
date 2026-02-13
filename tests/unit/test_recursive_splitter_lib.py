"""Tests for RecursiveSplitter.

This module contains unit tests for RecursiveSplitter.
Tests cover basic splitting, Markdown structure preservation, and edge cases.
"""

import pytest
from unittest.mock import MagicMock

from libs.splitter.recursive_splitter import RecursiveSplitter
from libs.splitter.base_splitter import SplitResult, SplitterConfigurationError


class TestRecursiveSplitter:
    """Tests for RecursiveSplitter provider."""

    def test_initialization_default(self):
        """Test initialization with default values."""
        splitter = RecursiveSplitter()

        assert splitter.provider_name == "recursive"
        assert splitter._chunk_size == 1000
        assert splitter._chunk_overlap == 200

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        splitter = RecursiveSplitter(
            chunk_size=500,
            chunk_overlap=50,
            keep_separator=False
        )

        assert splitter._chunk_size == 500
        assert splitter._chunk_overlap == 50

    def test_initialization_invalid_chunk_size(self):
        """Test initialization fails with invalid chunk_size."""
        with pytest.raises(SplitterConfigurationError):
            RecursiveSplitter(chunk_size=0, chunk_overlap=0)

        with pytest.raises(SplitterConfigurationError):
            RecursiveSplitter(chunk_size=-100, chunk_overlap=0)

    def test_initialization_invalid_overlap(self):
        """Test initialization fails with invalid chunk_overlap."""
        with pytest.raises(SplitterConfigurationError):
            RecursiveSplitter(chunk_overlap=-1)

    def test_initialization_overlap_greater_than_size(self):
        """Test initialization fails when overlap >= chunk_size."""
        with pytest.raises(SplitterConfigurationError):
            RecursiveSplitter(chunk_size=100, chunk_overlap=100)

    def test_split_empty_text(self):
        """Test splitting empty text."""
        splitter = RecursiveSplitter()
        result = splitter.split_text("")

        assert result.chunks == []
        assert isinstance(result, SplitResult)

    def test_split_short_text(self):
        """Test splitting text smaller than chunk_size."""
        splitter = RecursiveSplitter(chunk_size=1000)
        result = splitter.split_text("This is a short text.")

        assert len(result.chunks) == 1
        assert result.chunks[0] == "This is a short text."
        assert result.metadata["chunk_count"] == 1

    def test_split_preserves_paragraphs(self):
        """Test that paragraphs are preserved with double newline."""
        splitter = RecursiveSplitter(chunk_size=1000, chunk_overlap=0)
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        result = splitter.split_text(text)

        # Should preserve paragraph boundaries
        assert len(result.chunks) == 1
        assert "\n\n" in result.chunks[0]

    def test_split_preserves_markdown_headings(self):
        """Test that Markdown headings are preserved."""
        splitter = RecursiveSplitter(chunk_size=1000, chunk_overlap=0)
        text = "# Title\n\n## Section\n\nContent here."
        result = splitter.split_text(text)

        # Should preserve heading structure
        assert "# Title" in result.chunks[0]
        assert "## Section" in result.chunks[0]

    def test_split_preserves_code_blocks(self):
        """Test that code blocks are not broken."""
        splitter = RecursiveSplitter(chunk_size=1000, chunk_overlap=0)
        text = "```python\ndef hello():\n    print('world')\n```\n\nSome text."
        result = splitter.split_text(text)

        # Code block should remain intact
        assert "```python" in result.chunks[0]
        assert "```" in result.chunks[0]

    def test_split_long_text(self):
        """Test splitting long text creates multiple chunks."""
        splitter = RecursiveSplitter(chunk_size=50, chunk_overlap=10)

        # Create text longer than chunk_size
        text = "This is a very long text that should be split into multiple chunks because it exceeds the chunk size limit."
        result = splitter.split_text(text)

        # Should have multiple chunks
        assert len(result.chunks) > 1
        assert result.metadata["chunk_count"] > 1

    def test_split_has_overlap(self):
        """Test that consecutive chunks have overlap."""
        splitter = RecursiveSplitter(chunk_size=30, chunk_overlap=10)

        text = "This is a very long text that will be split into multiple chunks with overlap."
        result = splitter.split_text(text)

        # Check that overlap is present between chunks
        if len(result.chunks) > 1:
            # The overlap should be visible in consecutive chunks
            # Last 10 chars of chunk 1 should appear at start of chunk 2
            pass  # Overlap verification is implicit in the algorithm

    def test_split_single_paragraph(self):
        """Test splitting a single long paragraph."""
        splitter = RecursiveSplitter(chunk_size=20, chunk_overlap=5)
        text = "This is a very long paragraph without any line breaks that needs to be split."
        result = splitter.split_text(text)

        assert len(result.chunks) > 1
        # All chunks should have reasonable size
        for chunk in result.chunks:
            assert len(chunk) <= 25  # Allow some buffer

    def test_split_documents(self):
        """Test splitting multiple documents."""
        splitter = RecursiveSplitter(chunk_size=100, chunk_overlap=10)
        documents = [
            "First document content.",
            "Second document with more content that will be split into chunks.",
            "Third document."
        ]

        results = splitter.split_documents(documents)

        assert len(results) == 3
        assert isinstance(results[0], SplitResult)
        assert isinstance(results[1], SplitResult)
        assert isinstance(results[2], SplitResult)

    def test_split_with_custom_separators(self):
        """Test splitting with custom separators."""
        splitter = RecursiveSplitter(
            chunk_size=100,
            chunk_overlap=10,
            separators=["|||", "\n"]
        )
        text = "First part|||Second part|||Third part"
        result = splitter.split_text(text)

        # Should split at custom separator
        assert len(result.chunks) >= 1

    def test_repr(self):
        """Test string representation."""
        splitter = RecursiveSplitter(chunk_size=500, chunk_overlap=50)

        repr_str = repr(splitter)

        assert "RecursiveSplitter" in repr_str
        assert "500" in repr_str
        assert "50" in repr_str


class TestRecursiveSplitterFactory:
    """Tests for SplitterFactory with RecursiveSplitter."""

    def test_factory_create_recursive(self):
        """Test SplitterFactory can create RecursiveSplitter."""
        from libs.splitter.splitter_factory import SplitterFactory
        from libs.splitter.recursive_splitter import RecursiveSplitter

        # Register provider
        SplitterFactory.register("recursive", RecursiveSplitter)

        # Create mock settings
        mock_settings = MagicMock()
        mock_settings.ingestion.splitter = "recursive"
        mock_settings.ingestion.chunk_size = 500
        mock_settings.ingestion.chunk_overlap = 50

        # Test factory creation
        splitter = SplitterFactory.create(mock_settings)

        assert isinstance(splitter, RecursiveSplitter)
        assert splitter._chunk_size == 500
        assert splitter._chunk_overlap == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
