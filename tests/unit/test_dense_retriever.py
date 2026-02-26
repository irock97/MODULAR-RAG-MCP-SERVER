"""Unit tests for DenseRetriever."""

import pytest
from unittest.mock import Mock, MagicMock

from core.query_engine.dense_retriever import DenseRetriever
from core.types import RetrievalResult


class TestDenseRetriever:
    """Tests for DenseRetriever."""

    def test_retrieve_basic(self):
        """Test basic retrieval."""
        # Mock embedding
        mock_embedding = Mock()
        mock_embedding.provider_name = "mock"
        mock_embedding.embed.return_value = [[0.1, 0.2, 0.3]]

        # Mock vector store
        mock_vector_store = Mock()
        mock_vector_store.provider_name = "mock"
        mock_vector_store.query.return_value = [
            {
                "id": "chunk1",
                "score": 0.9,
                "text": "Text 1",
                "metadata": {"source": "doc1"},
            },
            {
                "id": "chunk2",
                "score": 0.8,
                "text": "Text 2",
                "metadata": {"source": "doc2"},
            },
        ]

        # Create retriever with mocks
        retriever = DenseRetriever(
            settings=Mock(),
            embedding_client=mock_embedding,
            vector_store=mock_vector_store,
        )

        # Retrieve
        results = retriever.retrieve("test query", top_k=5)

        # Verify
        assert len(results) == 2
        assert results[0].chunk_id == "chunk1"
        assert results[0].score == 0.9
        assert results[0].text == "Text 1"
        assert results[1].chunk_id == "chunk2"
        assert results[1].score == 0.8
        assert results[1].text == "Text 2"

        # Verify embedding called
        mock_embedding.embed.assert_called_once_with(["test query"], trace=None)

        # Verify vector store called
        mock_vector_store.query.assert_called_once()
        call_args = mock_vector_store.query.call_args
        assert call_args.kwargs["top_k"] == 5

    def test_retrieve_with_filters(self):
        """Test retrieval with metadata filters."""
        # Mock embedding
        mock_embedding = Mock()
        mock_embedding.provider_name = "mock"
        mock_embedding.embed.return_value = [[0.1, 0.2]]

        # Mock vector store
        mock_vector_store = Mock()
        mock_vector_store.provider_name = "mock"
        mock_vector_store.query.return_value = [
            {
                "id": "chunk1",
                "score": 0.9,
                "text": "Tech text",
                "metadata": {"collection": "tech"},
            },
        ]

        # Create retriever
        retriever = DenseRetriever(
            settings=Mock(),
            embedding_client=mock_embedding,
            vector_store=mock_vector_store,
        )

        # Retrieve with filters
        results = retriever.retrieve("test", filters={"collection": "tech"})

        # Verify filters passed
        call_args = mock_vector_store.query.call_args
        assert call_args.kwargs["filters"] == {"collection": "tech"}

        assert len(results) == 1
        assert results[0].metadata["collection"] == "tech"

    def test_retrieve_empty_query_raises(self):
        """Test that empty query raises ValueError."""
        retriever = DenseRetriever(
            settings=Mock(),
            embedding_client=Mock(),
            vector_store=Mock(),
        )

        with pytest.raises(ValueError, match="cannot be empty"):
            retriever.retrieve("")

        with pytest.raises(ValueError, match="cannot be empty"):
            retriever.retrieve("   ")

    def test_retrieve_with_trace(self):
        """Test retrieval with trace context."""
        from core.trace.trace_context import TraceContext

        # Mock embedding
        mock_embedding = Mock()
        mock_embedding.provider_name = "mock"
        mock_embedding.embed.return_value = [[0.1, 0.2]]

        # Mock vector store
        mock_vector_store = Mock()
        mock_vector_store.provider_name = "mock"
        mock_vector_store.query.return_value = [
            {
                "id": "chunk1",
                "score": 0.9,
                "text": "Text",
                "metadata": {},
            },
        ]

        # Create retriever
        retriever = DenseRetriever(
            settings=Mock(),
            embedding_client=mock_embedding,
            vector_store=mock_vector_store,
        )

        # Retrieve with trace
        trace = TraceContext()
        results = retriever.retrieve("test", trace=trace)

        # Verify trace recorded stages
        assert "dense_embedding" in trace.get_all_stages()
        assert "dense_retrieval" in trace.get_all_stages()

    def test_retrieve_empty_results(self):
        """Test retrieval with no results."""
        # Mock embedding
        mock_embedding = Mock()
        mock_embedding.provider_name = "mock"
        mock_embedding.embed.return_value = [[0.1, 0.2]]

        # Mock vector store with empty results
        mock_vector_store = Mock()
        mock_vector_store.provider_name = "mock"
        mock_vector_store.query.return_value = []

        # Create retriever
        retriever = DenseRetriever(
            settings=Mock(),
            embedding_client=mock_embedding,
            vector_store=mock_vector_store,
        )

        # Retrieve
        results = retriever.retrieve("test")

        assert results == []

    def test_retrieve_uses_default_top_k(self):
        """Test retrieval uses default_top_k when not specified."""
        # Mock embedding
        mock_embedding = Mock()
        mock_embedding.provider_name = "mock"
        mock_embedding.embed.return_value = [[0.1, 0.2]]

        # Mock vector store
        mock_vector_store = Mock()
        mock_vector_store.provider_name = "mock"
        mock_vector_store.query.return_value = []

        # Mock settings with no retrieval config
        mock_settings = Mock()
        mock_settings.retrieval = None

        # Create retriever with custom default_top_k
        retriever = DenseRetriever(
            settings=mock_settings,
            embedding_client=mock_embedding,
            vector_store=mock_vector_store,
            default_top_k=20,
        )

        # Retrieve without specifying top_k
        retriever.retrieve("test")

        # Verify default_top_k used
        call_args = mock_vector_store.query.call_args
        assert call_args.kwargs["top_k"] == 20


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""

    def test_creation(self):
        """Test RetrievalResult creation."""
        result = RetrievalResult(
            chunk_id="chunk1",
            score=0.95,
            text="Sample text",
            metadata={"source": "doc1"},
        )

        assert result.chunk_id == "chunk1"
        assert result.score == 0.95
        assert result.text == "Sample text"
        assert result.metadata == {"source": "doc1"}

    def test_to_dict(self):
        """Test to_dict conversion."""
        result = RetrievalResult(
            chunk_id="chunk1",
            score=0.95,
            text="Sample text",
            metadata={"source": "doc1"},
        )

        d = result.to_dict()
        assert d["chunk_id"] == "chunk1"
        assert d["score"] == 0.95
        assert d["text"] == "Sample text"
        assert d["metadata"] == {"source": "doc1"}

    def test_from_dict(self):
        """Test from_dict creation."""
        data = {
            "chunk_id": "chunk1",
            "score": 0.95,
            "text": "Sample text",
            "metadata": {"source": "doc1"},
        }

        result = RetrievalResult.from_dict(data)
        assert result.chunk_id == "chunk1"
        assert result.score == 0.95
        assert result.text == "Sample text"
        assert result.metadata == {"source": "doc1"}
