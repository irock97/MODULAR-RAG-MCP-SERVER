"""Unit tests for DenseEncoder.

This module tests the DenseEncoder class that generates dense vector
representations for document chunks using an embedding model.

Design Principles:
    - Mock-based: Uses mock embedding for isolation
    - Contract Testing: Verify ChunkRecord output format
    - Deterministic: Test vector preservation across runs
    - Coverage: Empty input, single chunk, batch encoding, metadata preservation
"""

from unittest.mock import MagicMock

import pytest

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import Chunk, ChunkRecord
from libs.embedding.base_embedding import BaseEmbedding, EmbeddingResult
from ingestion.embedding import DenseEncoder


class MockEmbedding(BaseEmbedding):
    """A mock embedding for testing purposes.

    This embedding returns fixed-size vectors with predictable values
    based on the input text hash.
    """

    def __init__(
        self,
        provider_name: str = "mock",
        dimensions: int = 3,
    ) -> None:
        self._provider_name = provider_name
        self._dimensions = dimensions

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(
        self,
        texts: list[str],
        trace: TraceContext | None = None,
        **kwargs,
    ) -> EmbeddingResult:
        """Return mock vectors based on text length."""
        vectors = []
        for i, text in enumerate(texts):
            # Create predictable mock vectors
            vector = [float(len(text) + i), float(i), float(i * 2)]
            # Pad to dimensions
            while len(vector) < self._dimensions:
                vector.append(0.0)
            vectors.append(vector[: self._dimensions])
        return EmbeddingResult(vectors=vectors, usage={"prompt_tokens": len(texts) * 10})

    def embed_single(
        self,
        text: str,
        trace: TraceContext | None = None,
        **kwargs,
    ) -> list[float]:
        """Return mock vector for single text."""
        return self.embed([text], trace=trace).vectors[0]


class TestDenseEncoder:
    """Tests for DenseEncoder main functionality."""

    def test_encode_multiple_chunks(self):
        """Test encoding multiple chunks returns ChunkRecords with vectors."""
        embedding = MockEmbedding(provider_name="mock", dimensions=3)
        encoder = DenseEncoder(embedding)

        chunks = [
            Chunk(id="chunk1", text="Hello world"),
            Chunk(id="chunk2", text="This is a test"),
            Chunk(id="chunk3", text="Third chunk here"),
        ]

        records = encoder.encode(chunks)

        assert len(records) == 3
        assert all(isinstance(record, ChunkRecord) for record in records)

    def test_encode_returns_correct_chunk_ids(self):
        """Test that encoded records preserve chunk IDs."""
        embedding = MockEmbedding()
        encoder = DenseEncoder(embedding)

        chunks = [
            Chunk(id="doc_0001", text="Content A"),
            Chunk(id="doc_0002", text="Content B"),
        ]

        records = encoder.encode(chunks)

        assert records[0].id == "doc_0001"
        assert records[1].id == "doc_0002"

    def test_encode_returns_dense_vectors(self):
        """Test that encoded records contain dense vectors."""
        embedding = MockEmbedding(provider_name="mock", dimensions=4)
        encoder = DenseEncoder(embedding)

        chunks = [Chunk(id="test1", text="Hello world")]

        records = encoder.encode(chunks)

        assert len(records) == 1
        assert records[0].dense_vector is not None
        assert len(records[0].dense_vector) == 4

    def test_encode_preserves_text(self):
        """Test that encoded records preserve original text."""
        embedding = MockEmbedding()
        encoder = DenseEncoder(embedding)

        chunks = [Chunk(id="test1", text="Original text content")]

        records = encoder.encode(chunks)

        assert records[0].text == "Original text content"

    def test_encode_preserves_metadata(self):
        """Test that encoded records preserve chunk metadata."""
        embedding = MockEmbedding()
        encoder = DenseEncoder(embedding)

        chunks = [
            Chunk(
                id="test1",
                text="Content",
                metadata={
                    "source_path": "/path/to/file.pdf",
                    "chunk_index": 0,
                    "custom_field": "custom_value",
                },
            )
        ]

        records = encoder.encode(chunks)

        assert records[0].metadata["source_path"] == "/path/to/file.pdf"
        assert records[0].metadata["chunk_index"] == 0
        assert records[0].metadata["custom_field"] == "custom_value"

    def test_encode_single_chunk(self):
        """Test encoding a single chunk."""
        embedding = MockEmbedding(provider_name="mock", dimensions=3)
        encoder = DenseEncoder(embedding)

        chunk = Chunk(id="single", text="Only one chunk")

        records = encoder.encode([chunk])

        assert len(records) == 1
        assert records[0].id == "single"
        assert records[0].dense_vector is not None

    def test_encode_empty_list(self):
        """Test encoding empty list returns empty list."""
        embedding = MockEmbedding()
        encoder = DenseEncoder(embedding)

        records = encoder.encode([])

        assert records == []

    def test_encode_empty_text_chunk(self):
        """Test encoding chunk with empty text."""
        embedding = MockEmbedding(provider_name="mock", dimensions=3)
        encoder = DenseEncoder(embedding)

        chunks = [Chunk(id="empty", text="")]

        # Should still return a record (empty vector)
        records = encoder.encode(chunks)

        assert len(records) == 1
        assert records[0].id == "empty"


class TestDenseEncoderSingle:
    """Tests for DenseEncoder.encode_single method."""

    def test_encode_single_chunk_method(self):
        """Test encode_single method returns ChunkRecord."""
        embedding = MockEmbedding(provider_name="mock", dimensions=3)
        encoder = DenseEncoder(embedding)

        chunk = Chunk(id="single_test", text="Single chunk content")

        record = encoder.encode_single(chunk)

        assert isinstance(record, ChunkRecord)
        assert record.id == "single_test"
        assert record.text == "Single chunk content"
        assert record.dense_vector is not None

    def test_encode_single_empty_text(self):
        """Test encode_single with empty text returns empty vector."""
        embedding = MockEmbedding(provider_name="mock", dimensions=3)
        encoder = DenseEncoder(embedding)

        chunk = Chunk(id="empty", text="")

        record = encoder.encode_single(chunk)

        assert record.dense_vector == []


class TestDenseEncoderProperties:
    """Tests for DenseEncoder properties."""

    def test_provider_name_property(self):
        """Test provider_name property returns embedding provider."""
        embedding = MockEmbedding(provider_name="test_provider")
        encoder = DenseEncoder(embedding)

        assert encoder.provider_name == "test_provider"

    def test_dimensions_property(self):
        """Test dimensions property returns embedding dimensions."""
        embedding = MockEmbedding(dimensions=512)
        encoder = DenseEncoder(embedding)

        assert encoder.dimensions == 512

    def test_embedding_property(self):
        """Test embedding property returns the embedding instance."""
        embedding = MockEmbedding()
        encoder = DenseEncoder(embedding)

        assert encoder.embedding is embedding

    def test_repr(self):
        """Test string representation."""
        embedding = MockEmbedding(provider_name="mock", dimensions=256)
        encoder = DenseEncoder(embedding, batch_size=50)

        repr_str = repr(encoder)

        assert "DenseEncoder" in repr_str
        assert "mock" in repr_str
        assert "256" in repr_str
        assert "50" in repr_str


class TestDenseEncoderWithDifferentProviders:
    """Tests for DenseEncoder with different embedding providers."""

    def test_different_provider_dimensions(self):
        """Test encoder respects different provider dimensions."""
        embedding_1536 = MockEmbedding(provider_name="openai", dimensions=1536)
        embedding_1024 = MockEmbedding(provider_name="qwen", dimensions=1024)

        encoder_1536 = DenseEncoder(embedding_1536)
        encoder_1024 = DenseEncoder(embedding_1024)

        assert encoder_1536.dimensions == 1536
        assert encoder_1024.dimensions == 1024


class TestDenseEncoderBatchProcessing:
    """Tests for batch processing behavior."""

    def test_batch_encoding_order(self):
        """Test that batch encoding preserves chunk order."""
        embedding = MockEmbedding(provider_name="mock", dimensions=3)
        encoder = DenseEncoder(embedding)

        chunks = [
            Chunk(id="first", text="First content"),
            Chunk(id="second", text="Second content"),
            Chunk(id="third", text="Third content"),
        ]

        records = encoder.encode(chunks)

        assert records[0].id == "first"
        assert records[1].id == "second"
        assert records[2].id == "third"

    def test_large_batch(self):
        """Test encoding a large batch of chunks."""
        embedding = MockEmbedding(provider_name="mock", dimensions=5)
        encoder = DenseEncoder(embedding, batch_size=100)

        chunks = [Chunk(id=f"chunk_{i}", text=f"Content {i}") for i in range(50)]

        records = encoder.encode(chunks)

        assert len(records) == 50
        assert all(record.dense_vector is not None for record in records)

    def test_small_batch_size(self):
        """Test encoder with custom batch size."""
        embedding = MockEmbedding(provider_name="mock", dimensions=3)
        encoder = DenseEncoder(embedding, batch_size=2)

        chunks = [Chunk(id=f"c{i}", text=f"Text {i}") for i in range(4)]

        records = encoder.encode(chunks)

        assert len(records) == 4

    def test_actual_batch_processing(self):
        """Test that encoder actually uses batch_size for splitting."""
        from unittest.mock import MagicMock

        embedding = MagicMock()
        embedding.provider_name = "mock"
        embedding.dimensions = 3

        # Return vectors based on number of texts in batch
        def mock_embed(texts, trace=None, **kwargs):
            vectors = [[float(i), float(i + 1), float(i + 2)] for i in range(len(texts))]
            return EmbeddingResult(vectors=vectors, usage={"prompt_tokens": len(texts) * 10})

        embedding.embed.side_effect = mock_embed

        encoder = DenseEncoder(embedding, batch_size=2)

        chunks = [
            Chunk(id="c0", text="Text 0"),
            Chunk(id="c1", text="Text 1"),
            Chunk(id="c2", text="Text 2"),
            Chunk(id="c3", text="Text 3"),
        ]

        records = encoder.encode(chunks)

        # With batch_size=2, 4 chunks should result in 2 API calls
        assert embedding.embed.call_count == 2

        # Each call should receive 2 texts
        first_call_texts = embedding.embed.call_args_list[0][0][0]
        second_call_texts = embedding.embed.call_args_list[1][0][0]
        assert len(first_call_texts) == 2
        assert len(second_call_texts) == 2

        assert len(records) == 4


class TestDenseEncoderIntegration:
    """Integration tests for DenseEncoder with real-like scenarios."""

    def test_document_chunks_to_records(self):
        """Test converting document chunks to chunk records using real config."""
        from core.settings import load_settings
        from libs.embedding.embedding_factory import EmbeddingFactory

        # Load settings from config
        settings = load_settings("config/settings.yaml")

        # Create embedding instance from settings
        embedding = EmbeddingFactory.create(settings)

        encoder = DenseEncoder(embedding)

        # Simulate chunks from document chunking
        chunks = [
            Chunk(
                id="doc_a_0000",
                text="Introduction to the topic.",
                metadata={"source_path": "/docs/intro.pdf", "chunk_index": 0},
                start_offset=0,
                end_offset=30,
                source_ref="doc_a",
            ),
            Chunk(
                id="doc_a_0001",
                text="Detailed explanation of concepts.",
                metadata={"source_path": "/docs/intro.pdf", "chunk_index": 1},
                start_offset=31,
                end_offset=70,
                source_ref="doc_a",
            ),
        ]

        records = encoder.encode(chunks)

        assert len(records) == 2
        # Check first record
        assert records[0].id == "doc_a_0000"
        assert records[0].dense_vector is not None
        assert len(records[0].dense_vector) == embedding.dimensions
        assert records[0].metadata["source_path"] == "/docs/intro.pdf"
        assert records[0].sparse_vector is None

        # Check second record
        assert records[1].id == "doc_a_0001"
        assert records[1].dense_vector is not None
        assert records[1].metadata["source_path"] == "/docs/intro.pdf"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
