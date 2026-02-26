"""Unit tests for SparseRetriever."""

import pytest
from unittest.mock import Mock, MagicMock

from core.query_engine.sparse_retriever import SparseRetriever
from core.types import RetrievalResult


class TestSparseRetriever:
    """Tests for SparseRetriever."""

    def test_retrieve_basic(self):
        """Test basic retrieval."""
        # Mock BM25 indexer
        mock_bm25 = Mock()
        mock_bm25.search.return_value = [
            ("chunk1", 0.9),
            ("chunk2", 0.8),
        ]

        # Mock vector store
        mock_vector_store = Mock()
        mock_vector_store.get_by_ids.return_value = [
            {
                "id": "chunk1",
                "text": "Text 1",
                "metadata": {"source": "doc1"},
            },
            {
                "id": "chunk2",
                "text": "Text 2",
                "metadata": {"source": "doc2"},
            },
        ]

        # Create retriever with mocks
        retriever = SparseRetriever(
            bm25_indexer=mock_bm25,
            vector_store=mock_vector_store,
        )

        # Retrieve
        results = retriever.retrieve(keywords=["test", "query"], top_k=5)

        # Verify
        assert len(results) == 2
        assert results[0].chunk_id == "chunk1"
        assert results[0].score == 0.9
        assert results[0].text == "Text 1"
        assert results[1].chunk_id == "chunk2"
        assert results[1].score == 0.8
        assert results[1].text == "Text 2"

        # Verify BM25 search called
        mock_bm25.search.assert_called_once_with(
            query_terms=["test", "query"],
            top_k=5,
            trace=None
        )

        # Verify vector store called
        mock_vector_store.get_by_ids.assert_called_once()
        call_args = mock_vector_store.get_by_ids.call_args
        assert "chunk1" in call_args.args[0]
        assert "chunk2" in call_args.args[0]

    def test_retrieve_empty_keywords_raises(self):
        """Test that empty keywords raises ValueError."""
        retriever = SparseRetriever(
            bm25_indexer=Mock(),
            vector_store=Mock(),
        )

        with pytest.raises(ValueError, match="cannot be empty"):
            retriever.retrieve(keywords=[])

    def test_retrieve_with_trace(self):
        """Test retrieval with trace context."""
        from core.trace.trace_context import TraceContext

        # Mock BM25 indexer
        mock_bm25 = Mock()
        mock_bm25.search.return_value = [("chunk1", 0.9)]

        # Mock vector store
        mock_vector_store = Mock()
        mock_vector_store.get_by_ids.return_value = [
            {
                "id": "chunk1",
                "text": "Text",
                "metadata": {},
            },
        ]

        # Create retriever
        retriever = SparseRetriever(
            bm25_indexer=mock_bm25,
            vector_store=mock_vector_store,
        )

        # Retrieve with trace
        trace = TraceContext()
        results = retriever.retrieve(keywords=["test"], trace=trace)

        # Verify trace recorded stages
        assert "sparse_retrieval_start" in trace.get_all_stages()
        assert "sparse_retrieval_complete" in trace.get_all_stages()

    def test_retrieve_empty_bm25_results(self):
        """Test retrieval with no BM25 results."""
        # Mock BM25 indexer returning empty
        mock_bm25 = Mock()
        mock_bm25.search.return_value = []

        # Mock vector store
        mock_vector_store = Mock()

        # Create retriever
        retriever = SparseRetriever(
            bm25_indexer=mock_bm25,
            vector_store=mock_vector_store,
        )

        # Retrieve
        results = retriever.retrieve(keywords=["test"])

        assert results == []
        # Vector store should not be called
        mock_vector_store.get_by_ids.assert_not_called()

    def test_retrieve_uses_default_top_k(self):
        """Test retrieval uses default_top_k when not specified."""
        # Mock BM25 indexer
        mock_bm25 = Mock()
        mock_bm25.search.return_value = []

        # Mock vector store
        mock_vector_store = Mock()

        # Create retriever with custom default_top_k
        retriever = SparseRetriever(
            bm25_indexer=mock_bm25,
            vector_store=mock_vector_store,
            default_top_k=20,
        )

        # Retrieve without specifying top_k
        retriever.retrieve(keywords=["test"])

        # Verify default_top_k used
        call_args = mock_bm25.search.call_args
        assert call_args.kwargs["top_k"] == 20

    def test_retrieve_vector_store_failure(self):
        """Test retrieval when vector store fails."""
        # Mock BM25 indexer
        mock_bm25 = Mock()
        mock_bm25.search.return_value = [
            ("chunk1", 0.9),
        ]

        # Mock vector store that fails
        mock_vector_store = Mock()
        mock_vector_store.get_by_ids.side_effect = Exception("Connection error")

        # Create retriever
        retriever = SparseRetriever(
            bm25_indexer=mock_bm25,
            vector_store=mock_vector_store,
        )

        # Should return results with empty text
        results = retriever.retrieve(keywords=["test"])

        # Should still return the chunk with BM25 score but empty text
        assert len(results) == 1
        assert results[0].chunk_id == "chunk1"
        assert results[0].score == 0.9
        assert results[0].text == ""


class TestSparseRetrieverDependencyValidation:
    """Tests for dependency validation."""

    def test_missing_bm25_with_settings(self):
        """Test that missing BM25 indexer is auto-created when settings provided."""
        # When settings is provided, BM25 indexer is auto-created
        mock_settings = Mock()
        mock_settings.bm25_index_dir = "data/db/bm25"
        mock_settings.retrieval = Mock()
        mock_settings.retrieval.sparse_top_k = 10

        retriever = SparseRetriever(
            settings=mock_settings,
            vector_store=Mock(),
        )

        # Should raise, BM25 indexer will be auto-created ( notmay be empty)
        results = retriever.retrieve(keywords=["test"])
        assert results == []

    def test_missing_vector_store_raises(self):
        """Test that missing vector store raises RuntimeError."""
        retriever = SparseRetriever(bm25_indexer=Mock())

        with pytest.raises(RuntimeError, match="vector_store"):
            retriever.retrieve(keywords=["test"])

    def test_invalid_keywords_type_raises(self):
        """Test that invalid keywords type raises ValueError."""
        retriever = SparseRetriever(
            bm25_indexer=Mock(),
            vector_store=Mock(),
        )

        # Should raise when keywords is not a list
        with pytest.raises((ValueError, TypeError)):
            retriever.retrieve(keywords=123)  # type: ignore
