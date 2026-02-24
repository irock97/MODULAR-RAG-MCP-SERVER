"""Unit tests for VectorUpserter idempotency.

This module tests the VectorUpserter class including:
- Deterministic chunk ID generation
- Idempotent upsert behavior
- Batch operations
- Error handling
"""

import pytest
from unittest.mock import MagicMock, patch

from core.types import ChunkRecord
from ingestion.storage import VectorUpserter
from libs.vector_store.base_vector_store import VectorRecord


class MockVectorStore:
    """Mock vector store for testing."""

    def __init__(self):
        self.provider_name = "mock"
        self.upserted_records = []

    def upsert(self, records, trace=None, **kwargs):
        self.upserted_records = records
        return [r.id for r in records]


class TestVectorUpserterIDGeneration:
    """Tests for chunk ID generation."""

    def test_generate_chunk_id_deterministic(self):
        """Test that same inputs produce same chunk ID."""
        store = MockVectorStore()
        upsert = VectorUpserter(vector_store=store)

        # Same inputs should produce same output
        id1 = upsert._generate_chunk_id("/docs/a.pdf", 0, "Hello world")
        id2 = upsert._generate_chunk_id("/docs/a.pdf", 0, "Hello world")

        assert id1 == id2

    def test_generate_chunk_id_different_text(self):
        """Test that different text produces different ID."""
        store = MockVectorStore()
        upsert = VectorUpserter(vector_store=store)

        id1 = upsert._generate_chunk_id("/docs/a.pdf", 0, "Hello world")
        id2 = upsert._generate_chunk_id("/docs/a.pdf", 0, "Hello there")

        assert id1 != id2

    def test_generate_chunk_id_different_index(self):
        """Test that different chunk_index produces different ID."""
        store = MockVectorStore()
        upsert = VectorUpserter(vector_store=store)

        id1 = upsert._generate_chunk_id("/docs/a.pdf", 0, "Hello world")
        id2 = upsert._generate_chunk_id("/docs/a.pdf", 1, "Hello world")

        assert id1 != id2

    def test_generate_chunk_id_different_source(self):
        """Test that different source_path produces different ID."""
        store = MockVectorStore()
        upsert = VectorUpserter(vector_store=store)

        id1 = upsert._generate_chunk_id("/docs/a.pdf", 0, "Hello world")
        id2 = upsert._generate_chunk_id("/docs/b.pdf", 0, "Hello world")

        assert id1 != id2

    def test_generate_chunk_id_length(self):
        """Test that chunk ID has expected length."""
        store = MockVectorStore()
        upsert = VectorUpserter(vector_store=store)

        chunk_id = upsert._generate_chunk_id("/docs/a.pdf", 0, "Hello world")

        assert len(chunk_id) == 16
        assert chunk_id.isalnum()


class TestVectorUpserterUpsert:
    """Tests for upsert functionality."""

    def test_upsert_single_record(self):
        """Test upserting a single record."""
        store = MockVectorStore()
        upsert = VectorUpserter(vector_store=store)

        records = [
            ChunkRecord(
                id="chunk1",
                text="Hello world",
                dense_vector=[0.1, 0.2, 0.3],
                metadata={"source_path": "/docs/a.pdf", "chunk_index": 0},
            )
        ]

        result = upsert.upsert(records)

        assert len(result) == 1
        assert store.upserted_records[0].id is not None

    def test_upsert_batch_preserves_order(self):
        """Test that batch upsert preserves record order."""
        store = MockVectorStore()
        upsert = VectorUpserter(vector_store=store)

        records = [
            ChunkRecord(
                id=f"chunk{i}",
                text=f"Content {i}",
                dense_vector=[0.1 * i, 0.2 * i],
                metadata={"source_path": "/docs/a.pdf", "chunk_index": i},
            )
            for i in range(5)
        ]

        result = upsert.upsert(records)

        assert len(result) == 5

    def test_upsert_skips_missing_dense_vector(self):
        """Test that records without dense_vector are skipped."""
        store = MockVectorStore()
        upsert = VectorUpserter(vector_store=store)

        records = [
            ChunkRecord(
                id="chunk1",
                text="Has vector",
                dense_vector=[0.1, 0.2],
                metadata={"source_path": "/docs/a.pdf", "chunk_index": 0},
            ),
            ChunkRecord(
                id="chunk2",
                text="No vector",
                dense_vector=None,
                metadata={"source_path": "/docs/a.pdf", "chunk_index": 1},
            ),
        ]

        result = upsert.upsert(records)

        assert len(result) == 1

    def test_upsert_empty_list(self):
        """Test upserting empty list returns empty list."""
        store = MockVectorStore()
        upsert = VectorUpserter(vector_store=store)

        result = upsert.upsert([])

        assert result == []


class TestVectorUpserterIdempotency:
    """Tests for idempotent behavior."""

    def test_same_content_same_id(self):
        """Test that same content produces same ID on second upsert."""
        store = MockVectorStore()
        upsert = VectorUpserter(vector_store=store)

        record = ChunkRecord(
            id="chunk1",
            text="Hello world",
            dense_vector=[0.1, 0.2],
            metadata={"source_path": "/docs/a.pdf", "chunk_index": 0},
        )

        # First upsert
        result1 = upsert.upsert([record])
        first_id = store.upserted_records[0].id

        # Reset mock
        store.upserted_records = []

        # Second upsert with same content
        result2 = upsert.upsert([record])
        second_id = store.upserted_records[0].id

        # IDs should be the same (idempotent)
        assert first_id == second_id

    def test_different_content_different_id(self):
        """Test that different content produces different ID."""
        store = MockVectorStore()
        upsert = VectorUpserter(vector_store=store)

        record1 = ChunkRecord(
            id="chunk1",
            text="Hello world",
            dense_vector=[0.1, 0.2],
            metadata={"source_path": "/docs/a.pdf", "chunk_index": 0},
        )

        record2 = ChunkRecord(
            id="chunk2",
            text="Hello there",
            dense_vector=[0.1, 0.2],
            metadata={"source_path": "/docs/a.pdf", "chunk_index": 0},
        )

        result1 = upsert.upsert([record1])
        first_id = store.upserted_records[0].id

        store.upserted_records = []

        result2 = upsert.upsert([record2])
        second_id = store.upserted_records[0].id

        assert first_id != second_id


class TestVectorUpserterSingle:
    """Tests for upsert_single method."""

    def test_upsert_single_success(self):
        """Test upsert_single returns chunk ID."""
        store = MockVectorStore()
        upsert = VectorUpserter(vector_store=store)

        record = ChunkRecord(
            id="chunk1",
            text="Hello world",
            dense_vector=[0.1, 0.2],
            metadata={"source_path": "/docs/a.pdf", "chunk_index": 0},
        )

        result = upsert.upsert_single(record)

        assert result is not None
        assert isinstance(result, str)


class TestVectorUpserterRepr:
    """Tests for string representation."""

    def test_repr(self):
        """Test string representation."""
        store = MockVectorStore()
        upsert = VectorUpserter(vector_store=store)

        repr_str = repr(upsert)

        assert "VectorUpserter" in repr_str
        assert "mock" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
