"""Unit tests for HybridSearch."""

import pytest
from unittest.mock import Mock, MagicMock

from core.query_engine import (
    HybridSearch,
    HybridSearchConfig,
    QueryProcessor,
    DenseRetriever,
    SparseRetriever,
    RRFFusion,
)
from core.types import RetrievalResult


class TestHybridSearch:
    """Tests for HybridSearch."""

    def test_search_basic(self):
        """Test basic hybrid search."""
        # Mock query processor
        mock_processor = Mock()
        mock_processed = Mock()
        mock_processed.keywords = ["test"]
        mock_processed.filters = {}
        mock_processor.process.return_value = mock_processed

        # Mock dense retriever
        mock_dense = Mock()
        mock_dense.retrieve.return_value = [
            RetrievalResult(chunk_id="doc1", score=0.9, text="Dense 1"),
            RetrievalResult(chunk_id="doc2", score=0.8, text="Dense 2"),
        ]

        # Mock sparse retriever
        mock_sparse = Mock()
        mock_sparse.retrieve.return_value = [
            RetrievalResult(chunk_id="doc2", score=0.95, text="Sparse 2"),
            RetrievalResult(chunk_id="doc3", score=0.7, text="Sparse 3"),
        ]

        # Create hybrid search
        hybrid = HybridSearch(
            query_processor=mock_processor,
            dense_retriever=mock_dense,
            sparse_retriever=mock_sparse,
        )

        # Search
        results = hybrid.search("test query", top_k=5)

        # Verify results
        assert len(results) > 0
        # Should have results from both sources
        result_ids = [r.chunk_id for r in results]
        assert "doc1" in result_ids or "doc2" in result_ids or "doc3" in result_ids

    def test_search_with_empty_query_raises(self):
        """Test that empty query raises ValueError."""
        hybrid = HybridSearch()

        with pytest.raises(ValueError, match="Query cannot be empty"):
            hybrid.search("")

    def test_search_no_retrievers_returns_empty(self):
        """Test that missing retrievers returns empty results."""
        hybrid = HybridSearch()

        # Should return empty results instead of raising
        results = hybrid.search("test query")
        assert results == []

    def test_search_parallel_disabled(self):
        """Test search with parallel disabled."""
        # Mock query processor
        mock_processor = Mock()
        mock_processed = Mock()
        mock_processed.keywords = ["test"]
        mock_processed.filters = {}
        mock_processor.process.return_value = mock_processed

        # Mock retrievers
        mock_dense = Mock()
        mock_dense.retrieve.return_value = [
            RetrievalResult(chunk_id="doc1", score=0.9, text="Dense 1"),
        ]

        mock_sparse = Mock()
        mock_sparse.retrieve.return_value = [
            RetrievalResult(chunk_id="doc2", score=0.8, text="Sparse 1"),
        ]

        # Create with parallel disabled
        config = HybridSearchConfig(parallel_retrieval=False)
        hybrid = HybridSearch(
            query_processor=mock_processor,
            dense_retriever=mock_dense,
            sparse_retriever=mock_sparse,
            config=config,
        )

        # Search
        results = hybrid.search("test query", top_k=5)

        # Both retrievers should be called sequentially
        mock_dense.retrieve.assert_called_once()
        mock_sparse.retrieve.assert_called_once()

    def test_search_with_filters(self):
        """Test search with filters."""
        # Mock query processor
        mock_processor = Mock()
        mock_processed = Mock()
        mock_processed.keywords = ["test"]
        mock_processed.filters = {"category": "A"}
        mock_processor.process.return_value = mock_processed

        # Mock retrievers
        mock_dense = Mock()
        mock_dense.retrieve.return_value = []

        mock_sparse = Mock()
        mock_sparse.retrieve.return_value = []

        hybrid = HybridSearch(
            query_processor=mock_processor,
            dense_retriever=mock_dense,
            sparse_retriever=mock_sparse,
        )

        # Search with additional filters
        results = hybrid.search("test query", filters={"year": "2024"})

        # Verify filters were merged
        call_args = mock_dense.retrieve.call_args
        assert call_args.kwargs.get("filters", {}).get("category") == "A"
        assert call_args.kwargs.get("filters", {}).get("year") == "2024"

    def test_search_dense_only(self):
        """Test search with only dense retriever."""
        # Mock query processor
        mock_processor = Mock()
        mock_processed = Mock()
        mock_processed.keywords = ["test"]
        mock_processed.filters = {}
        mock_processor.process.return_value = mock_processed

        # Mock dense retriever only
        mock_dense = Mock()
        mock_dense.retrieve.return_value = [
            RetrievalResult(chunk_id="doc1", score=0.9, text="Dense 1"),
        ]

        hybrid = HybridSearch(
            query_processor=mock_processor,
            dense_retriever=mock_dense,
            sparse_retriever=None,
        )

        results = hybrid.search("test query", top_k=5)

        # Should still return results from dense
        assert len(results) > 0

    def test_search_sparse_only(self):
        """Test search with only sparse retriever."""
        # Mock query processor
        mock_processor = Mock()
        mock_processed = Mock()
        mock_processed.keywords = ["test"]
        mock_processed.filters = {}
        mock_processor.process.return_value = mock_processed

        # Mock sparse retriever only
        mock_sparse = Mock()
        mock_sparse.retrieve.return_value = [
            RetrievalResult(chunk_id="doc1", score=0.9, text="Sparse 1"),
        ]

        hybrid = HybridSearch(
            query_processor=mock_processor,
            dense_retriever=None,
            sparse_retriever=mock_sparse,
        )

        results = hybrid.search("test query", top_k=5)

        # Should still return results from sparse
        assert len(results) > 0

    def test_search_with_trace(self):
        """Test search with trace context."""
        from core.trace.trace_context import TraceContext

        # Mock query processor
        mock_processor = Mock()
        mock_processed = Mock()
        mock_processed.keywords = ["test"]
        mock_processed.filters = {}
        mock_processor.process.return_value = mock_processed

        # Mock retrievers
        mock_dense = Mock()
        mock_dense.retrieve.return_value = []

        mock_sparse = Mock()
        mock_sparse.retrieve.return_value = []

        hybrid = HybridSearch(
            query_processor=mock_processor,
            dense_retriever=mock_dense,
            sparse_retriever=mock_sparse,
        )

        # Search with trace
        trace = TraceContext()
        results = hybrid.search("test query", trace=trace)

        # Verify trace recorded stages
        stages = trace.get_all_stages()
        assert "hybrid_search_start" in stages
        assert "query_processed" in stages
        assert "hybrid_search_complete" in stages

    def test_search_with_custom_config(self):
        """Test search with custom config."""
        # Mock query processor
        mock_processor = Mock()
        mock_processed = Mock()
        mock_processed.keywords = ["test"]
        mock_processed.filters = {}
        mock_processor.process.return_value = mock_processed

        # Mock retrievers
        mock_dense = Mock()
        mock_dense.retrieve.return_value = [
            RetrievalResult(chunk_id="doc1", score=0.9, text="Dense 1"),
        ]

        mock_sparse = Mock()
        mock_sparse.retrieve.return_value = [
            RetrievalResult(chunk_id="doc2", score=0.8, text="Sparse 1"),
        ]

        # Create with custom config
        config = HybridSearchConfig(
            dense_top_k=30,
            sparse_top_k=40,
            fusion_top_k=15,
            enable_dense=True,
            enable_sparse=True,
            parallel_retrieval=True,
            metadata_filter_post=False,
        )
        hybrid = HybridSearch(
            query_processor=mock_processor,
            dense_retriever=mock_dense,
            sparse_retriever=mock_sparse,
            config=config,
        )

        results = hybrid.search("test query", top_k=5)

        # Verify config was used
        assert hybrid.config.dense_top_k == 30
        assert hybrid.config.sparse_top_k == 40
        assert hybrid.config.fusion_top_k == 15
        assert hybrid.config.metadata_filter_post is False


class TestHybridSearchConfig:
    """Tests for HybridSearchConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = HybridSearchConfig()

        assert config.dense_top_k == 20
        assert config.sparse_top_k == 20
        assert config.fusion_top_k == 10
        assert config.enable_dense is True
        assert config.enable_sparse is True
        assert config.parallel_retrieval is True
        assert config.metadata_filter_post is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = HybridSearchConfig(
            dense_top_k=30,
            sparse_top_k=40,
            fusion_top_k=15,
            enable_dense=True,
            enable_sparse=False,
            parallel_retrieval=False,
            metadata_filter_post=False,
        )

        assert config.dense_top_k == 30
        assert config.sparse_top_k == 40
        assert config.fusion_top_k == 15
        assert config.enable_dense is True
        assert config.enable_sparse is False
        assert config.parallel_retrieval is False
        assert config.metadata_filter_post is False


class TestHybridSearchEdgeCases:
    """Tests for edge cases."""

    def test_both_retrievers_fail(self):
        """Test when both retrievers fail."""
        # Mock query processor
        mock_processor = Mock()
        mock_processed = Mock()
        mock_processed.keywords = ["test"]
        mock_processed.filters = {}
        mock_processor.process.return_value = mock_processed

        # Mock retrievers that fail
        mock_dense = Mock()
        mock_dense.retrieve.side_effect = Exception("Dense error")

        mock_sparse = Mock()
        mock_sparse.retrieve.side_effect = Exception("Sparse error")

        hybrid = HybridSearch(
            query_processor=mock_processor,
            dense_retriever=mock_dense,
            sparse_retriever=mock_sparse,
        )

        # Should return empty results
        results = hybrid.search("test query", top_k=5)
        assert results == []

    def test_one_retriever_fails(self):
        """Test when one retriever fails."""
        # Mock query processor
        mock_processor = Mock()
        mock_processed = Mock()
        mock_processed.keywords = ["test"]
        mock_processed.filters = {}
        mock_processor.process.return_value = mock_processed

        # Dense fails, sparse succeeds
        mock_dense = Mock()
        mock_dense.retrieve.side_effect = Exception("Dense error")

        mock_sparse = Mock()
        mock_sparse.retrieve.return_value = [
            RetrievalResult(chunk_id="doc1", score=0.9, text="Sparse 1"),
        ]

        hybrid = HybridSearch(
            query_processor=mock_processor,
            dense_retriever=mock_dense,
            sparse_retriever=mock_sparse,
        )

        # Should return results from sparse
        results = hybrid.search("test query", top_k=5)
        assert len(results) > 0
        assert results[0].chunk_id == "doc1"

    def test_no_results_from_dense(self):
        """Test when dense returns no results."""
        # Mock query processor
        mock_processor = Mock()
        mock_processed = Mock()
        mock_processed.keywords = ["test"]
        mock_processed.filters = {}
        mock_processor.process.return_value = mock_processed

        # Dense returns empty
        mock_dense = Mock()
        mock_dense.retrieve.return_value = []

        # Sparse returns results
        mock_sparse = Mock()
        mock_sparse.retrieve.return_value = [
            RetrievalResult(chunk_id="doc1", score=0.9, text="Sparse 1"),
        ]

        hybrid = HybridSearch(
            query_processor=mock_processor,
            dense_retriever=mock_dense,
            sparse_retriever=mock_sparse,
        )

        results = hybrid.search("test query", top_k=5)

        # Should return results from sparse
        assert len(results) > 0
