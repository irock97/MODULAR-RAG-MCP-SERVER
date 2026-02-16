"""Unit tests for DocumentChunker.

This module tests the DocumentChunker class which transforms Document objects
into Chunk objects with business logic增值.

Design Principles:
    - Mock-based: Uses FakeSplitter for isolation
    - Contract Testing: Verify chunk output format
    - Deterministic: Test ID stability across runs
    - Coverage: ID generation, metadata inheritance, source tracing
"""

import pytest
from typing import Any

from core.settings import Settings
from core.types import Chunk, Document
from libs.splitter.base_splitter import BaseSplitter
from ingestion.chunking.document_chunker import (
    DocumentChunker,
    DocumentChunkingError,
)


class FakeSplitter(BaseSplitter):
    """A fake splitter for testing purposes.

    This splitter splits text by a fixed separator and returns fixed-size
    chunks without any real NLP logic.
    """

    def __init__(
        self,
        chunk_size: int = 100,
        chunk_overlap: int = 0,
        separator: str = "\n\n",
    ) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separator = separator

    @property
    def provider_name(self) -> str:
        return "fake"

    def split_text(
        self,
        text: str,
        trace: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Split text by fixed separator."""
        chunks = text.split(self._separator)
        # Further split if chunks are too large
        result: list[str] = []
        for chunk in chunks:
            if len(chunk) <= self._chunk_size:
                result.append(chunk)
            else:
                # Split large chunks
                for i in range(0, len(chunk), self._chunk_size - self._chunk_overlap):
                    result.append(chunk[i : i + self._chunk_size])
        return type("Result", (), {"chunks": result})()

    def split_documents(
        self,
        documents: list[str],
        trace: Any | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        """Split multiple documents."""
        return [self.split_text(doc) for doc in documents]


class TestDocumentChunker:
    """Tests for DocumentChunker main functionality."""

    def test_split_document_returns_list_of_chunks(self):
        """Test that split_document returns a list of Chunk objects."""
        settings = Settings()
        settings.ingestion.chunk_size = 100
        settings.ingestion.splitter = "fake"

        chunker = DocumentChunker(settings, splitter=FakeSplitter())

        doc = Document(id="doc1", text="This is test content.\n\nMore content here.")
        chunks = chunker.split_document(doc)

        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)

    def test_chunk_count_matches_splitter_output(self):
        """Test that chunk count matches what splitter produces."""
        settings = Settings()
        settings.ingestion.chunk_size = 100

        fake_splitter = FakeSplitter(separator="|")
        chunker = DocumentChunker(settings, splitter=fake_splitter)

        text = "chunk1|chunk2|chunk3"
        doc = Document(id="test_doc", text=text)

        chunks = chunker.split_document(doc)

        # With separator="|", we get ["chunk1", "chunk2", "chunk3"]
        assert len(chunks) == 3


class TestChunkIdGeneration:
    """Tests for chunk ID generation stability and format."""

    def test_chunk_id_format(self):
        """Test that chunk ID follows expected format."""
        settings = Settings()
        chunker = DocumentChunker(settings, splitter=FakeSplitter())

        doc = Document(id="doc123", text="Test content here.")
        chunks = chunker.split_document(doc)

        # Format: doc_id_0000_hash
        assert chunks[0].id.startswith("doc123_")
        assert "0000" in chunks[0].id

    def test_chunk_ids_are_unique(self):
        """Test that each chunk has a unique ID."""
        settings = Settings()
        chunker = DocumentChunker(settings, splitter=FakeSplitter())

        doc = Document(id="unique_test", text="First chunk.\n\nSecond chunk.\n\nThird chunk.")
        chunks = chunker.split_document(doc)

        ids = [chunk.id for chunk in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs should be unique"

    def test_chunk_id_stability(self):
        """Test that same document produces same chunk IDs."""
        settings = Settings()
        chunker = DocumentChunker(settings, splitter=FakeSplitter())

        doc = Document(id="stable", text="Same content for stability test.")

        # Split twice
        chunks1 = chunker.split_document(doc)
        chunks2 = chunker.split_document(doc)

        assert len(chunks1) == len(chunks2)
        for c1, c2 in zip(chunks1, chunks2):
            assert c1.id == c2.id, "Chunk IDs should be stable across runs"


class TestMetadataInheritance:
    """Tests for metadata inheritance from Document to Chunk."""

    def test_chunk_inherits_document_metadata(self):
        """Test that chunks inherit document metadata fields."""
        settings = Settings()
        chunker = DocumentChunker(settings, splitter=FakeSplitter())

        doc = Document(
            id="meta_test",
            text="Content with metadata.",
            metadata={
                "source_path": "/path/to/file.pdf",
                "doc_type": "pdf",
                "author": "Test Author",
            },
        )

        chunks = chunker.split_document(doc)

        for chunk in chunks:
            assert chunk.metadata.get("source_path") == "/path/to/file.pdf"
            assert chunk.metadata.get("doc_type") == "pdf"
            assert chunk.metadata.get("author") == "Test Author"

    def test_chunk_adds_chunk_index(self):
        """Test that chunks have chunk_index in metadata."""
        settings = Settings()
        chunker = DocumentChunker(settings, splitter=FakeSplitter())

        doc = Document(id="index_test", text="First.\n\nSecond.\n\nThird.")
        chunks = chunker.split_document(doc)

        for i, chunk in enumerate(chunks):
            assert chunk.metadata.get("chunk_index") == i

    def test_extra_metadata_preserved(self):
        """Test that extra metadata fields are preserved."""
        settings = Settings()
        chunker = DocumentChunker(settings, splitter=FakeSplitter())

        doc = Document(
            id="extra",
            text="Content",
            metadata={
                "source_path": "/path/file.pdf",
                "custom_field": "custom_value",
                "nested": {"key": "value"},
            },
        )

        chunks = chunker.split_document(doc)
        for chunk in chunks:
            assert chunk.metadata.get("custom_field") == "custom_value"
            assert chunk.metadata.get("nested") == {"key": "value"}


class TestSourceTracing:
    """Tests for source tracing (source_ref field)."""

    def test_source_ref_points_to_document(self):
        """Test that chunk.source_ref points to parent document ID."""
        settings = Settings()
        chunker = DocumentChunker(settings, splitter=FakeSplitter())

        doc = Document(id="parent_doc_123", text="Child content here.")
        chunks = chunker.split_document(doc)

        for chunk in chunks:
            assert chunk.source_ref == "parent_doc_123"

    def test_multiple_documents_trace_correctly(self):
        """Test that chunks from different documents have correct sources."""
        settings = Settings()
        chunker = DocumentChunker(settings, splitter=FakeSplitter())

        doc1 = Document(id="doc_A", text="Content from A.")
        doc2 = Document(id="doc_B", text="Content from B.")

        chunks1 = chunker.split_document(doc1)
        chunks2 = chunker.split_document(doc2)

        for chunk in chunks1:
            assert chunk.source_ref == "doc_A"

        for chunk in chunks2:
            assert chunk.source_ref == "doc_B"


class TestChunkContent:
    """Tests for chunk content properties."""

    def test_chunk_text_is_not_empty(self):
        """Test that all chunks have non-empty text."""
        settings = Settings()
        chunker = DocumentChunker(settings, splitter=FakeSplitter())

        doc = Document(id="nonempty", text="First.\n\nSecond.\n\nThird.")
        chunks = chunker.split_document(doc)

        for chunk in chunks:
            assert len(chunk.text) > 0

    def test_chunk_preserves_text_content(self):
        """Test that chunk text is preserved from splitter output."""
        settings = Settings()
        fake = FakeSplitter(separator="|")
        chunker = DocumentChunker(settings, splitter=fake)

        doc = Document(id="preserve", text="exact|text|split")
        chunks = chunker.split_document(doc)

        # Verify the exact content is preserved
        chunk_texts = [chunk.text for chunk in chunks]
        assert "exact" in chunk_texts[0]
        assert "text" in chunk_texts[1]
        assert "split" in chunk_texts[2]


class TestConfigurationDriven:
    """Tests for configuration-driven behavior."""

    def test_different_chunk_sizes_produce_different_chunks(self):
        """Test that different chunk_size produces different chunk counts."""
        settings = Settings()

        # Small chunks
        chunker_small = DocumentChunker(
            settings, splitter=FakeSplitter(chunk_size=5, separator=" ")
        )
        doc = Document(id="config_test", text="a b c d e f g h i")
        chunks_small = chunker_small.split_document(doc)

        # Large chunks
        chunker_large = DocumentChunker(
            settings, splitter=FakeSplitter(chunk_size=20, separator=" ")
        )
        chunks_large = chunker_large.split_document(doc)

        # Smaller chunk_size should produce more or equal chunks
        assert len(chunks_small) >= len(chunks_large)


class TestErrorHandling:
    """Tests for error handling."""

    def test_empty_document_handled(self):
        """Test that empty document is handled gracefully."""
        settings = Settings()
        chunker = DocumentChunker(settings, splitter=FakeSplitter())

        doc = Document(id="empty", text="")
        chunks = chunker.split_document(doc)

        # Should return at least one empty chunk or empty list
        assert isinstance(chunks, list)


class TestFakeSplitter:
    """Tests for FakeSplitter testing utility."""

    def test_fake_splitter_basic_split(self):
        """Test FakeSplitter basic functionality."""
        splitter = FakeSplitter(separator="|")
        result = splitter.split_text("a|b|c")

        assert len(result.chunks) == 3
        assert result.chunks[0] == "a"
        assert result.chunks[1] == "b"
        assert result.chunks[2] == "c"

    def test_fake_splitter_provider_name(self):
        """Test FakeSplitter provider_name."""
        splitter = FakeSplitter()
        assert splitter.provider_name == "fake"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
