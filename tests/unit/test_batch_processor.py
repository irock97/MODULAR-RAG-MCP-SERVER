"""Unit tests for BatchProcessor.

This module tests the BatchProcessor class that orchestrates batched
encoding operations for dense and sparse embedding.

Design Principles:
    - Mock-based: Uses mocks for encoders
    - Contract Testing: Verify ChunkRecord output format
    - Deterministic: Test batch splitting and order preservation
    - Coverage: Empty input, single batch, multiple batches, order stability
"""

import pytest
from unittest.mock import MagicMock

from core.types import Chunk, ChunkRecord
from ingestion.embedding import BatchProcessor, DenseEncoder, SparseEncoder


class TestBatchProcessor:
    """Tests for BatchProcessor main functionality."""

    def test_batch_split_5_chunks_batch_size_2(self):
        """Test that 5 chunks split into 3 batches with batch_size=2.

        Acceptance criterion: batch_size=2 时对 5 chunks 分成 3 批
        """
        processor = BatchProcessor(batch_size=2)

        chunks = [
            Chunk(id="c0", text="Chunk 0"),
            Chunk(id="c1", text="Chunk 1"),
            Chunk(id="c2", text="Chunk 2"),
            Chunk(id="c3", text="Chunk 3"),
            Chunk(id="c4", text="Chunk 4"),
        ]

        batches = processor._split_into_batches(chunks)

        assert len(batches) == 3
        assert len(batches[0]) == 2  # [c0, c1]
        assert len(batches[1]) == 2  # [c2, c3]
        assert len(batches[2]) == 1  # [c4]

    def test_order_preservation(self):
        """Test that chunk order is preserved through batch processing.

        Acceptance criterion: 顺序稳定
        """
        # Create mock encoders
        dense_encoder = MagicMock(spec=DenseEncoder)
        sparse_encoder = MagicMock(spec=SparseEncoder)

        def mock_encode(cks, trace=None):
            return [
                ChunkRecord(id=ck.id, text=ck.text, dense_vector=[1.0] * 10)
                for ck in cks
            ]

        dense_encoder.encode.side_effect = mock_encode
        sparse_encoder.encode.side_effect = mock_encode

        processor = BatchProcessor(
            batch_size=2,
            dense_encoder=dense_encoder,
            sparse_encoder=sparse_encoder
        )

        chunks = [
            Chunk(id="c0", text="First"),
            Chunk(id="c1", text="Second"),
            Chunk(id="c2", text="Third"),
            Chunk(id="c3", text="Fourth"),
        ]

        records = processor.process(chunks)

        # Verify order is preserved
        assert len(records) == 4
        assert records[0].id == "c0"
        assert records[1].id == "c1"
        assert records[2].id == "c2"
        assert records[3].id == "c3"

    def test_empty_chunks(self):
        """Test processing empty chunk list returns empty list."""
        processor = BatchProcessor(batch_size=2)

        records = processor.process([])

        assert records == []

    def test_batch_size_property(self):
        """Test batch_size property returns configured value."""
        processor = BatchProcessor(batch_size=16)

        assert processor.batch_size == 16

    def test_batch_size_validation(self):
        """Test that invalid batch_size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            BatchProcessor(batch_size=0)

        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            BatchProcessor(batch_size=-1)

    def test_single_batch(self):
        """Test processing chunks that fit in a single batch."""
        dense_encoder = MagicMock(spec=DenseEncoder)
        dense_encoder.encode.return_value = [
            ChunkRecord(id=ck.id, text=ck.text, dense_vector=[1.0])
            for ck in [Chunk(id=f"c{i}", text=f"Content {i}") for i in range(5)]
        ]

        processor = BatchProcessor(batch_size=10, dense_encoder=dense_encoder)

        chunks = [Chunk(id=f"c{i}", text=f"Content {i}") for i in range(5)]

        records = processor.process(chunks)

        assert len(records) == 5
        # Single batch means encode called once with all 5 chunks
        assert dense_encoder.encode.call_count == 1

    def test_repr(self):
        """Test string representation."""
        processor = BatchProcessor(
            batch_size=32,
            progress_callback=lambda *_: None
        )

        repr_str = repr(processor)

        assert "BatchProcessor" in repr_str
        assert "batch_size=32" in repr_str


class TestBatchProcessorWithMocks:
    """Tests for BatchProcessor using mocked encoders."""

    def test_dense_only_constructor(self):
        """Test processing with dense encoder configured in constructor."""
        dense_encoder = MagicMock(spec=DenseEncoder)
        dense_encoder.encode.return_value = [
            ChunkRecord(id="c0", text="Hello", dense_vector=[0.1, 0.2]),
            ChunkRecord(id="c1", text="World", dense_vector=[0.3, 0.4]),
        ]

        processor = BatchProcessor(batch_size=2, dense_encoder=dense_encoder)

        chunks = [
            Chunk(id="c0", text="Hello"),
            Chunk(id="c1", text="World"),
        ]

        records = processor.process(chunks)

        assert len(records) == 2
        assert records[0].dense_vector == [0.1, 0.2]
        assert records[1].dense_vector == [0.3, 0.4]
        # Sparse should be None when only dense encoder configured
        assert records[0].sparse_vector is None

    def test_sparse_only_constructor(self):
        """Test processing with sparse encoder configured in constructor."""
        sparse_encoder = MagicMock(spec=SparseEncoder)
        sparse_encoder.encode.return_value = [
            ChunkRecord(id="c0", text="Hello world", sparse_vector={"hello": 1.0}),
            ChunkRecord(id="c1", text="Test document", sparse_vector={"test": 1.0}),
        ]

        processor = BatchProcessor(batch_size=2, sparse_encoder=sparse_encoder)

        chunks = [
            Chunk(id="c0", text="Hello world"),
            Chunk(id="c1", text="Test document"),
        ]

        records = processor.process(chunks)

        assert len(records) == 2
        assert records[0].sparse_vector == {"hello": 1.0}
        assert records[1].sparse_vector == {"test": 1.0}
        # Dense should be None when only sparse encoder configured
        assert records[0].dense_vector is None

    def test_dual_encoding_constructor(self):
        """Test processing with both encoders configured in constructor."""
        dense_encoder = MagicMock(spec=DenseEncoder)
        dense_encoder.encode.return_value = [
            ChunkRecord(id="c0", text="Hello world", dense_vector=[0.1]),
            ChunkRecord(id="c1", text="Test doc", dense_vector=[0.2]),
        ]

        sparse_encoder = MagicMock(spec=SparseEncoder)
        sparse_encoder.encode.return_value = [
            ChunkRecord(id="c0", text="Hello world", sparse_vector={"hello": 1.0}),
            ChunkRecord(id="c1", text="Test doc", sparse_vector={"test": 1.0}),
        ]

        processor = BatchProcessor(
            batch_size=2,
            dense_encoder=dense_encoder,
            sparse_encoder=sparse_encoder
        )

        chunks = [
            Chunk(id="c0", text="Hello world"),
            Chunk(id="c1", text="Test doc"),
        ]

        records = processor.process(chunks)

        assert len(records) == 2
        # Both vectors should be present
        assert records[0].dense_vector == [0.1]
        assert records[0].sparse_vector == {"hello": 1.0}
        assert records[1].dense_vector == [0.2]
        assert records[1].sparse_vector == {"test": 1.0}

    def test_encoder_methods_called_correctly(self):
        """Test that encoders are called with correct arguments."""
        dense_encoder = MagicMock(spec=DenseEncoder)
        dense_encoder.encode.return_value = [
            ChunkRecord(id=ck.id, text=ck.text, dense_vector=[1.0])
            for ck in [Chunk(id=f"c{i}", text=f"Text {i}") for i in range(3)]
        ]

        processor = BatchProcessor(batch_size=2, dense_encoder=dense_encoder)

        chunks = [
            Chunk(id="c0", text="Zero"),
            Chunk(id="c1", text="One"),
            Chunk(id="c2", text="Two"),
        ]

        processor.process(chunks)

        # With 3 chunks and batch_size=2, we expect 2 calls
        assert dense_encoder.encode.call_count == 2


class TestBatchProcessorEdgeCases:
    """Tests for edge cases."""

    def test_batch_size_one(self):
        """Test batch_size=1 processes each chunk individually."""
        dense_encoder = MagicMock(spec=DenseEncoder)
        dense_encoder.encode.return_value = [
            ChunkRecord(id=ck.id, text=ck.text, dense_vector=[1.0])
            for ck in [Chunk(id=f"c{i}", text=f"Text {i}") for i in range(3)]
        ]

        processor = BatchProcessor(batch_size=1, dense_encoder=dense_encoder)

        chunks = [
            Chunk(id="c0", text="Zero"),
            Chunk(id="c1", text="One"),
            Chunk(id="c2", text="Two"),
        ]

        records = processor.process(chunks)

        # Each chunk should be its own batch
        assert dense_encoder.encode.call_count == 3
        assert len(records) == 3

    def test_large_batch(self):
        """Test processing a large number of chunks."""
        dense_encoder = MagicMock(spec=DenseEncoder)
        dense_encoder.encode.return_value = [
            ChunkRecord(id=ck.id, text=ck.text, dense_vector=[1.0])
            for ck in [Chunk(id=f"c{i}", text=f"Content {i}") for i in range(50)]
        ]

        processor = BatchProcessor(batch_size=100, dense_encoder=dense_encoder)

        chunks = [Chunk(id=f"c{i}", text=f"Content {i}") for i in range(50)]

        records = processor.process(chunks)

        # All 50 chunks in single batch
        assert dense_encoder.encode.call_count == 1
        assert len(records) == 50

    def test_no_encoders(self):
        """Test processing with no encoders configured."""
        processor = BatchProcessor(batch_size=2)

        chunks = [
            Chunk(id="c0", text="Hello"),
            Chunk(id="c1", text="World"),
        ]

        records = processor.process(chunks)

        # Should return empty records (no vectors)
        assert len(records) == 2
        assert records[0].dense_vector is None
        assert records[0].sparse_vector is None


class TestBatchProcessorTiming:
    """Tests for batch timing functionality."""

    def test_encoding_duration_in_metadata(self):
        """Test that encoding duration is recorded in metadata."""
        dense_encoder = MagicMock(spec=DenseEncoder)
        dense_encoder.encode.return_value = [
            ChunkRecord(id=ck.id, text=ck.text, dense_vector=[1.0])
            for ck in [Chunk(id=f"c{i}", text=f"Text {i}") for i in range(3)]
        ]

        processor = BatchProcessor(batch_size=2, dense_encoder=dense_encoder)

        chunks = [Chunk(id=f"c{i}", text=f"Text {i}") for i in range(3)]

        records = processor.process(chunks)

        # Each record should have _encoding_duration in metadata
        for record in records:
            assert "_encoding_duration" in record.metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
