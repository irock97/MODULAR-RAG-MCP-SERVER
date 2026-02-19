"""Unit tests for SparseEncoder.

This module tests the SparseEncoder class that generates sparse vector
representations for document chunks using TF (term frequency) weighting.

Design Principles:
    - Mock-free: Uses real text processing
    - Contract Testing: Verify ChunkRecord output format
    - Deterministic: Test term extraction and weighting
    - Coverage: Empty input, single chunk, batch encoding, metadata preservation
"""

import pytest

from core.settings import Settings
from core.types import Chunk, ChunkRecord
from ingestion.embedding import SparseEncoder


class TestSparseEncoder:
    """Tests for SparseEncoder main functionality."""

    def test_encode_multiple_chunks(self):
        """Test encoding multiple chunks returns ChunkRecords with vectors."""
        encoder = SparseEncoder()

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
        encoder = SparseEncoder()

        chunks = [
            Chunk(id="doc_0001", text="Content A"),
            Chunk(id="doc_0002", text="Content B"),
        ]

        records = encoder.encode(chunks)

        assert records[0].id == "doc_0001"
        assert records[1].id == "doc_0002"

    def test_encode_returns_sparse_vectors(self):
        """Test that encoded records contain sparse vectors."""
        encoder = SparseEncoder()

        chunks = [Chunk(id="test1", text="Hello world programming")]

        records = encoder.encode(chunks)

        assert len(records) == 1
        assert records[0].sparse_vector is not None
        assert len(records[0].sparse_vector) > 0

    def test_encode_preserves_text(self):
        """Test that encoded records preserve original text."""
        encoder = SparseEncoder()

        chunks = [Chunk(id="test1", text="Original text content")]

        records = encoder.encode(chunks)

        assert records[0].text == "Original text content"

    def test_encode_preserves_metadata(self):
        """Test that encoded records preserve chunk metadata."""
        encoder = SparseEncoder()

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
        encoder = SparseEncoder()

        chunk = Chunk(id="single", text="Only one chunk")

        records = encoder.encode([chunk])

        assert len(records) == 1
        assert records[0].id == "single"
        assert records[0].sparse_vector is not None

    def test_encode_empty_list(self):
        """Test encoding empty list returns empty list."""
        encoder = SparseEncoder()

        records = encoder.encode([])

        assert records == []

    def test_encode_empty_text_chunk(self):
        """Test encoding chunk with empty text returns empty vector."""
        encoder = SparseEncoder()

        chunks = [Chunk(id="empty", text="")]

        records = encoder.encode(chunks)

        assert len(records) == 1
        assert records[0].sparse_vector == {}


class TestSparseEncoderTokenization:
    """Tests for tokenization behavior."""

    def test_tokenize_lowercase_conversion(self):
        """Test that tokens are converted to lowercase."""
        encoder = SparseEncoder()

        tokens = encoder._tokenize("Hello WORLD hello")

        assert "hello" in tokens
        # "hello" appears twice (case insensitive), so both become lowercase
        assert tokens.count("hello") == 2
        # No uppercase tokens should exist
        assert all(t.islower() for t in tokens)

    def test_tokenize_removes_punctuation(self):
        """Test that punctuation is removed."""
        encoder = SparseEncoder()

        tokens = encoder._tokenize("Hello, world! How are you?")

        assert "," not in tokens
        assert "!" not in tokens
        assert "hello" in tokens
        assert "world" in tokens

    def test_tokenize_filters_stop_words(self):
        """Test that stop words are filtered out."""
        encoder = SparseEncoder()

        tokens = encoder._tokenize("The quick brown fox is running")

        assert "the" not in tokens
        assert "is" not in tokens
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens
        assert "running" in tokens

    def test_tokenize_filters_short_terms(self):
        """Test that terms shorter than min_length are filtered."""
        encoder = SparseEncoder(min_term_length=3)

        tokens = encoder._tokenize("a bc def ghij klmno")

        assert "a" not in tokens
        assert "bc" not in tokens
        assert "def" in tokens
        assert "ghij" in tokens
        assert "klmno" in tokens


class TestSparseEncoderTF:
    """Tests for TF (Term Frequency) computation."""

    def test_tf_normalization(self):
        """Test that term frequencies are normalized by max frequency."""
        encoder = SparseEncoder()

        chunks = [Chunk(id="test", text="apple apple banana cherry")]
        records = encoder.encode(chunks)

        vector = records[0].sparse_vector

        # "apple" appears twice (max freq = 2), so TF = 1.0
        # "banana" and "cherry" appear once, so TF = 0.5
        assert vector["apple"] > vector["banana"]
        assert vector["apple"] == 1.0
        assert vector["banana"] == vector["cherry"]

    def test_vector_sorted_by_weight(self):
        """Test that sparse vectors are sorted by weight."""
        encoder = SparseEncoder()

        chunks = [Chunk(id="test", text="apple apple banana cherry date")]
        records = encoder.encode(chunks)

        vector = records[0].sparse_vector
        weights = list(vector.values())

        # Should be sorted descending by weight
        for i in range(len(weights) - 1):
            assert weights[i] >= weights[i + 1]

    def test_tf_values_are_normalized(self):
        """Test that TF values are normalized between 0 and 1."""
        encoder = SparseEncoder()

        chunks = [Chunk(id="test", text="word word word other other")]
        records = encoder.encode(chunks)

        vector = records[0].sparse_vector

        # "word" appears 3 times, "other" appears 2 times
        # Max freq = 3, so "word" = 1.0, "other" = 2/3 ≈ 0.67
        assert vector["word"] == 1.0
        assert 0 < vector["other"] < 1.0


class TestSparseEncoderProperties:
    """Tests for SparseEncoder properties."""

    def test_stop_words_property(self):
        """Test stop_words property returns the configured set."""
        custom_stop_words = frozenset({"custom", "stop", "words"})
        encoder = SparseEncoder(stop_words=custom_stop_words)

        assert encoder.stop_words == custom_stop_words

    def test_repr(self):
        """Test string representation."""
        encoder = SparseEncoder(min_term_length=3)

        repr_str = repr(encoder)

        assert "SparseEncoder" in repr_str
        assert "min_term_length=3" in repr_str


class TestSparseEncoderSingle:
    """Tests for SparseEncoder.encode_single method."""

    def test_encode_single_chunk_method(self):
        """Test encode_single method returns ChunkRecord."""
        encoder = SparseEncoder()

        chunk = Chunk(id="single_test", text="Single chunk content")

        record = encoder.encode_single(chunk)

        assert isinstance(record, ChunkRecord)
        assert record.id == "single_test"
        assert record.text == "Single chunk content"
        assert record.sparse_vector is not None

    def test_encode_single_empty_text(self):
        """Test encode_single with empty text returns empty vector."""
        encoder = SparseEncoder()

        chunk = Chunk(id="empty", text="")

        record = encoder.encode_single(chunk)

        assert record.sparse_vector == {}

    def test_encode_single_preserves_metadata(self):
        """Test encode_single preserves chunk metadata."""
        encoder = SparseEncoder()

        chunk = Chunk(
            id="test",
            text="Some content",
            metadata={"custom": "value"}
        )

        record = encoder.encode_single(chunk)

        assert record.metadata["custom"] == "value"


class TestSparseEncoderBatchProcessing:
    """Tests for batch processing behavior."""

    def test_batch_encoding_order(self):
        """Test that batch encoding preserves chunk order."""
        encoder = SparseEncoder()

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
        encoder = SparseEncoder()

        chunks = [Chunk(id=f"chunk_{i}", text=f"Content {i}") for i in range(50)]

        records = encoder.encode(chunks)

        assert len(records) == 50
        assert all(record.sparse_vector is not None for record in records)


class TestSparseEncoderIntegration:
    """Integration tests for SparseEncoder with real-like scenarios."""

    def test_document_chunks_to_records(self):
        """Test converting document chunks to chunk records."""
        encoder = SparseEncoder()

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
        assert records[0].sparse_vector is not None
        assert records[0].metadata["source_path"] == "/docs/intro.pdf"
        assert records[0].dense_vector is None

        # Check second record
        assert records[1].id == "doc_a_0001"
        assert records[1].sparse_vector is not None
        assert records[1].metadata["source_path"] == "/docs/intro.pdf"

    def test_code_snippet_encoding(self):
        """Test encoding code snippet chunks."""
        encoder = SparseEncoder()

        chunks = [
            Chunk(
                id="code_0000",
                text="def selection_sort(nums): pass",
            ),
        ]

        records = encoder.encode(chunks)

        assert len(records) == 1
        vector = records[0].sparse_vector
        # Should extract meaningful terms from code
        assert "selection_sort" in vector or "def" in vector

    def test_technical_document_encoding(self):
        """Test encoding technical document with special terminology."""
        encoder = SparseEncoder()

        chunks = [
            Chunk(
                id="tech_0000",
                text="The algorithm has time complexity analysis.",
            ),
        ]

        records = encoder.encode(chunks)

        assert len(records) == 1
        vector = records[0].sparse_vector
        # Should extract terms like "algorithm", "time", "complexity"
        assert any(term in vector for term in ["algorithm", "time", "complexity", "analysis"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
