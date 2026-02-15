"""Unit tests for core data types.

This module tests the Document, Chunk, and ChunkRecord types
for serialization, field access, and contract compliance.

Design Principles:
    - Serialization tests: Verify to_dict/from_dict roundtrip
    - Field stability: Assert required fields exist
    - Metadata contract: source_path is required minimum
"""

import pytest

from core.types import Document, Chunk, ChunkRecord


class TestDocument:
    """Tests for Document type."""

    def test_initialization_minimal(self):
        """Test Document with minimal required fields."""
        doc = Document(id="doc1", text="Hello world")

        assert doc.id == "doc1"
        assert doc.text == "Hello world"
        assert doc.metadata == {}

    def test_initialization_with_metadata(self):
        """Test Document with full metadata."""
        metadata = {"source_path": "/docs/test.pdf", "title": "Test Doc"}
        doc = Document(id="doc1", text="Content", metadata=metadata)

        assert doc.metadata["source_path"] == "/docs/test.pdf"
        assert doc.metadata["title"] == "Test Doc"

    def test_to_dict(self):
        """Test Document serialization to dict."""
        metadata = {"source_path": "/test.pdf"}
        doc = Document(id="id1", text="Text", metadata=metadata)

        result = doc.to_dict()

        assert result["id"] == "id1"
        assert result["text"] == "Text"
        assert result["metadata"] == metadata

    def test_from_dict(self):
        """Test Document deserialization from dict."""
        data = {
            "id": "doc2",
            "text": "Some text",
            "metadata": {"source_path": "/path/file.pdf"}
        }

        doc = Document.from_dict(data)

        assert doc.id == "doc2"
        assert doc.text == "Some text"
        assert doc.metadata["source_path"] == "/path/file.pdf"

    def test_roundtrip_serialization(self):
        """Test dict -> Document -> dict roundtrip preserves data."""
        original = Document(
            id="roundtrip",
            text="Content for testing",
            metadata={"source_path": "/test.pdf", "extra": "value"}
        )

        restored = Document.from_dict(original.to_dict())

        assert restored.id == original.id
        assert restored.text == original.text
        assert restored.metadata == original.metadata


class TestChunk:
    """Tests for Chunk type."""

    def test_initialization_minimal(self):
        """Test Chunk with minimal required fields."""
        chunk = Chunk(id="chunk1", text="Chunk content")

        assert chunk.id == "chunk1"
        assert chunk.text == "Chunk content"
        assert chunk.start_offset is None
        assert chunk.end_offset is None
        assert chunk.source_ref is None

    def test_initialization_full(self):
        """Test Chunk with all fields."""
        metadata = {"page": 1}
        chunk = Chunk(
            id="chunk2",
            text="Full chunk",
            metadata=metadata,
            start_offset=100,
            end_offset=200,
            source_ref="doc1"
        )

        assert chunk.metadata["page"] == 1
        assert chunk.start_offset == 100
        assert chunk.end_offset == 200
        assert chunk.source_ref == "doc1"

    def test_to_dict(self):
        """Test Chunk serialization."""
        chunk = Chunk(
            id="c1",
            text="Text",
            metadata={"source_path": "/test.pdf"},
            start_offset=0,
            end_offset=100,
            source_ref="doc1"
        )

        result = chunk.to_dict()

        assert result["id"] == "c1"
        assert result["text"] == "Text"
        assert result["start_offset"] == 0
        assert result["end_offset"] == 100
        assert result["source_ref"] == "doc1"

    def test_from_dict(self):
        """Test Chunk deserialization."""
        data = {
            "id": "c2",
            "text": "Some chunk",
            "metadata": {"source_path": "/doc.pdf"},
            "start_offset": 50,
            "end_offset": 150,
            "source_ref": "parent_doc"
        }

        chunk = Chunk.from_dict(data)

        assert chunk.id == "c2"
        assert chunk.start_offset == 50
        assert chunk.source_ref == "parent_doc"

    def test_roundtrip_serialization(self):
        """Test Chunk roundtrip preserves all fields."""
        original = Chunk(
            id="round",
            text="Roundtrip content",
            metadata={"source_path": "/test.pdf", "page": 5},
            start_offset=500,
            end_offset=1000,
            source_ref="doc_ref"
        )

        restored = Chunk.from_dict(original.to_dict())

        assert restored.id == original.id
        assert restored.text == original.text
        assert restored.metadata == original.metadata
        assert restored.start_offset == original.start_offset
        assert restored.end_offset == original.end_offset
        assert restored.source_ref == original.source_ref


class TestChunkRecord:
    """Tests for ChunkRecord type."""

    def test_initialization_minimal(self):
        """Test ChunkRecord with minimal fields."""
        record = ChunkRecord(id="rec1", text="Record content")

        assert record.id == "rec1"
        assert record.text == "Record content"
        assert record.dense_vector is None
        assert record.sparse_vector is None

    def test_initialization_with_vectors(self):
        """Test ChunkRecord with vector data."""
        dense = [0.1, 0.2, 0.3]
        sparse = {"term1": 0.5, "term2": 0.8}

        record = ChunkRecord(
            id="vec_rec",
            text="With vectors",
            dense_vector=dense,
            sparse_vector=sparse
        )

        assert record.dense_vector == [0.1, 0.2, 0.3]
        assert record.sparse_vector["term1"] == 0.5

    def test_to_dict(self):
        """Test ChunkRecord serialization."""
        record = ChunkRecord(
            id="r1",
            text="Content",
            dense_vector=[0.1, 0.2],
            sparse_vector={"word": 0.9}
        )

        result = record.to_dict()

        assert result["id"] == "r1"
        assert result["dense_vector"] == [0.1, 0.2]
        assert result["sparse_vector"]["word"] == 0.9

    def test_from_dict(self):
        """Test ChunkRecord deserialization."""
        data = {
            "id": "r2",
            "text": "Deserialized",
            "metadata": {"source": "test"},
            "dense_vector": [0.5, 0.6],
            "sparse_vector": None
        }

        record = ChunkRecord.from_dict(data)

        assert record.id == "r2"
        assert record.dense_vector == [0.5, 0.6]
        assert record.sparse_vector is None

    def test_roundtrip_serialization(self):
        """Test ChunkRecord roundtrip with vectors."""
        original = ChunkRecord(
            id="vec_round",
            text="Vector content",
            metadata={"source_path": "/test.pdf"},
            dense_vector=[0.1, 0.2, 0.3, 0.4],
            sparse_vector={"important": 1.0, "relevant": 0.8}
        )

        restored = ChunkRecord.from_dict(original.to_dict())

        assert restored.id == original.id
        assert restored.text == original.text
        assert restored.dense_vector == original.dense_vector
        assert restored.sparse_vector == original.sparse_vector


class TestMetadataContract:
    """Tests for metadata contract compliance."""

    def test_document_requires_source_path_in_metadata(self):
        """Verify Document metadata can contain source_path."""
        # source_path is the minimum required field in contract
        doc = Document(
            id="contract_test",
            text="Testing metadata contract",
            metadata={"source_path": "/required/path.pdf"}
        )

        assert "source_path" in doc.metadata
        assert doc.metadata["source_path"] == "/required/path.pdf"

    def test_metadata_allows_extra_fields(self):
        """Verify metadata supports incremental extension."""
        doc = Document(
            id="extensible",
            text="Content",
            metadata={
                "source_path": "/path.pdf",
                "author": "Test Author",      # Extra field
                "created_at": "2024-01-01",   # Extra field
            }
        )

        # Should not raise, extra fields allowed
        assert doc.metadata["author"] == "Test Author"
        assert doc.metadata["created_at"] == "2024-01-01"

    def test_chunk_metadata_inheritance(self):
        """Verify Chunk can inherit and extend Document metadata."""
        doc_metadata = {"source_path": "/source.pdf", "title": "Source"}
        chunk = Chunk(
            id="child",
            text="Child content",
            metadata=doc_metadata,
            source_ref="parent_id"
        )

        assert chunk.metadata["source_path"] == "/source.pdf"
        assert chunk.metadata["title"] == "Source"


class TestImmutability:
    """Tests for immutability of core types."""

    def test_document_equality(self):
        """Verify Document equality is based on all fields."""
        doc1 = Document(id="d1", text="Text 1", metadata={"key": "val"})
        doc2 = Document(id="d1", text="Text 1", metadata={"key": "val"})
        doc3 = Document(id="d2", text="Text 2", metadata={"key": "val"})

        assert doc1 == doc2
        assert doc1 != doc3

    def test_chunk_equality(self):
        """Verify Chunk equality is based on all fields."""
        chunk1 = Chunk(id="c1", text="Text", metadata={})
        chunk2 = Chunk(id="c1", text="Text", metadata={})
        chunk3 = Chunk(id="c2", text="Text", metadata={})

        assert chunk1 == chunk2
        assert chunk1 != chunk3

    def test_document_id_access_is_readonly(self):
        """Verify Document fields are read-only after creation."""
        doc = Document(id="readonly", text="Content", metadata={})

        # Can read
        assert doc.id == "readonly"
        assert doc.text == "Content"

        # Cannot modify (frozen dataclass)
        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
            doc.id = "modified"

    def test_chunk_offsets_readonly(self):
        """Verify Chunk offsets are read-only after creation."""
        chunk = Chunk(
            id="c1",
            text="Content",
            start_offset=0,
            end_offset=100
        )

        assert chunk.start_offset == 0

        # Cannot modify (frozen dataclass)
        with pytest.raises(Exception):
            chunk.start_offset = 99


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
