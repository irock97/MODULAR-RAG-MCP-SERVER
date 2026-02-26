"""Unit tests for Reranker with fallback behavior."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

from core.query_engine.reranker import (
    CoreReranker,
    Reranker,
    RerankerConfig,
    RerankerResult,
    create_reranker,
)
from core.types import RetrievalResult


class TestReranker:
    """Tests for Reranker class."""

    def test_rerank_disabled_returns_original(self):
        """Test that disabled reranker returns original results."""
        config = RerankerConfig(enabled=False)
        reranker = Reranker(config=config)

        candidates = [
            RetrievalResult(chunk_id="doc1", score=0.9, text="Text 1"),
            RetrievalResult(chunk_id="doc2", score=0.8, text="Text 2"),
        ]

        result = reranker.rerank("test query", candidates, top_k=5)

        assert result.used_fallback is False
        assert result.reranker_type == "none"
        assert len(result.results) == 2

    def test_rerank_with_none_backend(self):
        """Test that None backend returns original results."""
        reranker = Reranker(reranker_backend=None)

        candidates = [
            RetrievalResult(chunk_id="doc1", score=0.9, text="Text 1"),
        ]

        result = reranker.rerank("test query", candidates)

        assert result.used_fallback is False
        assert len(result.results) == 1

    def test_rerank_empty_candidates(self):
        """Test reranking with empty candidates."""
        mock_backend = Mock()
        mock_backend.provider_name = "mock"

        reranker = Reranker(reranker_backend=mock_backend)

        result = reranker.rerank("test query", [])

        assert result.used_fallback is False
        assert result.results == []
        mock_backend.rerank.assert_not_called()

    def test_rerank_success(self):
        """Test successful reranking."""
        # Mock reranker backend
        mock_backend = Mock()
        mock_backend.provider_name = "cross_encoder"

        # Mock rerank result
        mock_rerank_result = Mock()
        mock_rerank_result.ids = ["doc2", "doc1"]
        mock_rerank_result.scores = [0.95, 0.85]

        mock_backend.rerank.return_value = mock_rerank_result

        # Pass explicit config with enabled=True
        reranker = Reranker(
            reranker_backend=mock_backend,
            config=RerankerConfig(enabled=True),
        )

        candidates = [
            RetrievalResult(chunk_id="doc1", score=0.9, text="Text 1"),
            RetrievalResult(chunk_id="doc2", score=0.8, text="Text 2"),
        ]

        result = reranker.rerank("test query", candidates, top_k=5)

        assert result.used_fallback is False
        assert result.reranker_type == "cross_encoder"
        assert len(result.results) == 2
        assert result.results[0].chunk_id == "doc2"  # Re-ranked
        mock_backend.rerank.assert_called_once()

    def test_rerank_fallback_on_exception(self):
        """Test fallback when reranker raises exception."""
        mock_backend = Mock()
        mock_backend.provider_name = "cross_encoder"
        mock_backend.rerank.side_effect = Exception("Reranker error")

        config = RerankerConfig(enabled=True, fallback_to_fusion=True)
        reranker = Reranker(reranker_backend=mock_backend, config=config)

        candidates = [
            RetrievalResult(chunk_id="doc1", score=0.9, text="Text 1"),
            RetrievalResult(chunk_id="doc2", score=0.8, text="Text 2"),
        ]

        result = reranker.rerank("test query", candidates)

        assert result.used_fallback is True
        assert result.fallback_reason == "Reranker error"
        assert result.reranker_type == "cross_encoder"
        # Should return original order (fallback)
        assert result.results[0].chunk_id == "doc1"

    def test_rerank_top_k_limit(self):
        """Test top_k limit is applied."""
        mock_backend = Mock()
        mock_backend.provider_name = "cross_encoder"

        mock_rerank_result = Mock()
        mock_rerank_result.ids = ["doc3", "doc2", "doc1"]
        mock_rerank_result.scores = [0.95, 0.85, 0.75]

        mock_backend.rerank.return_value = mock_rerank_result

        reranker = Reranker(reranker_backend=mock_backend)

        candidates = [
            RetrievalResult(chunk_id="doc1", score=0.9, text="Text 1"),
            RetrievalResult(chunk_id="doc2", score=0.8, text="Text 2"),
            RetrievalResult(chunk_id="doc3", score=0.7, text="Text 3"),
            RetrievalResult(chunk_id="doc4", score=0.6, text="Text 4"),
        ]

        result = reranker.rerank("test query", candidates, top_k=2)

        assert len(result.results) == 2

    def test_rerank_with_trace(self):
        """Test reranking with trace context."""
        from core.trace.trace_context import TraceContext

        mock_backend = Mock()
        mock_backend.provider_name = "cross_encoder"

        mock_rerank_result = Mock()
        mock_rerank_result.ids = ["doc1", "doc2"]
        mock_rerank_result.scores = [0.9, 0.8]

        mock_backend.rerank.return_value = mock_rerank_result

        # Pass explicit config with enabled=True
        reranker = Reranker(
            reranker_backend=mock_backend,
            config=RerankerConfig(enabled=True),
        )

        candidates = [
            RetrievalResult(chunk_id="doc1", score=0.9, text="Text 1"),
            RetrievalResult(chunk_id="doc2", score=0.8, text="Text 2"),
        ]

        trace = TraceContext()
        result = reranker.rerank("test query", candidates, trace=trace)

        stages = trace.get_all_stages()
        assert "rerank_start" in stages
        assert "rerank_complete" in stages

    def test_rerank_fallback_with_trace(self):
        """Test fallback behavior is traced."""
        from core.trace.trace_context import TraceContext

        mock_backend = Mock()
        mock_backend.provider_name = "cross_encoder"
        mock_backend.rerank.side_effect = Exception("Error")

        # Pass explicit config with enabled=True
        reranker = Reranker(
            reranker_backend=mock_backend,
            config=RerankerConfig(enabled=True),
        )

        candidates = [
            RetrievalResult(chunk_id="doc1", score=0.9, text="Text 1"),
            RetrievalResult(chunk_id="doc2", score=0.8, text="Text 2"),
        ]

        trace = TraceContext()
        result = reranker.rerank("test query", candidates, trace=trace)

        stages = trace.get_all_stages()
        assert "rerank_fallback" in stages


class TestRerankerConfig:
    """Tests for RerankerConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = RerankerConfig()

        assert config.enabled is True
        assert config.top_k == 5
        assert config.fallback_to_fusion is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = RerankerConfig(
            enabled=False,
            top_k=5,
            fallback_to_fusion=False,
        )

        assert config.enabled is False
        assert config.top_k == 5
        assert config.fallback_to_fusion is False


class TestRerankerResult:
    """Tests for RerankerResult."""

    def test_reranker_result_creation(self):
        """Test RerankerResult creation."""
        results = [
            RetrievalResult(chunk_id="doc1", score=0.9, text="Text"),
        ]

        result = RerankerResult(
            results=results,
            used_fallback=False,
            reranker_type="cross_encoder",
        )

        assert result.results == results
        assert result.used_fallback is False
        assert result.reranker_type == "cross_encoder"
        assert result.fallback_reason is None


class TestCreateReranker:
    """Tests for create_reranker factory function."""

    def test_create_reranker_basic(self):
        """Test basic reranker creation."""
        mock_backend = Mock()
        mock_backend.provider_name = "mock"

        config = RerankerConfig(enabled=True, top_k=5)
        reranker = CoreReranker(
            reranker_backend=mock_backend,
            config=config,
        )

        assert reranker.config.enabled is True
        assert reranker.config.top_k == 5
        assert reranker.reranker_backend == mock_backend

    def test_create_reranker_disabled(self):
        """Test creating disabled reranker."""
        config = RerankerConfig(enabled=False)
        reranker = CoreReranker(config=config)

        assert reranker.config.enabled is False

    def test_create_reranker_with_settings(self):
        """Test creating reranker from settings."""
        from core.query_engine.reranker import CoreReranker
        mock_rerank_settings = Mock()
        mock_rerank_settings.enabled = True
        mock_rerank_settings.top_k = 5

        mock_settings = Mock()
        mock_settings.rerank = mock_rerank_settings

        mock_backend = Mock()
        mock_backend.provider_name = "cross_encoder"

        # Mock the factory
        with patch(
            "libs.reranker.reranker_factory.RerankerFactory"
        ) as mock_factory:
            mock_factory.create.return_value = mock_backend

            reranker = create_reranker(settings=mock_settings)

            assert reranker.reranker_backend == mock_backend


class TestRerankerEdgeCases:
    """Tests for edge cases."""

    def test_rerank_preserves_metadata(self):
        """Test that metadata is preserved after reranking."""
        mock_backend = Mock()
        mock_backend.provider_name = "cross_encoder"

        mock_rerank_result = Mock()
        mock_rerank_result.ids = ["doc1"]
        mock_rerank_result.scores = [0.9]

        mock_backend.rerank.return_value = mock_rerank_result

        reranker = Reranker(reranker_backend=mock_backend)

        candidates = [
            RetrievalResult(
                chunk_id="doc1",
                score=0.9,
                text="Text",
                metadata={"source": "test", "year": 2024},
            ),
        ]

        result = reranker.rerank("test query", candidates)

        assert result.results[0].metadata["source"] == "test"
        assert result.results[0].metadata["year"] == 2024

    def test_rerank_missing_chunk_in_original(self):
        """Test handling when reranked ID not in original results."""
        mock_backend = Mock()
        mock_backend.provider_name = "cross_encoder"

        # Reranker returns ID not in original results
        mock_rerank_result = Mock()
        mock_rerank_result.ids = ["doc_unknown", "doc1", "doc2"]
        mock_rerank_result.scores = [0.9, 0.8, 0.7]

        mock_backend.rerank.return_value = mock_rerank_result

        # Pass explicit config with enabled=True
        reranker = Reranker(
            reranker_backend=mock_backend,
            config=RerankerConfig(enabled=True),
        )

        candidates = [
            RetrievalResult(chunk_id="doc1", score=0.9, text="Text 1"),
            RetrievalResult(chunk_id="doc2", score=0.8, text="Text 2"),
        ]

        result = reranker.rerank("test query", candidates)

        # Should handle missing ID gracefully (doc_unknown has empty text)
        assert len(result.results) == 3
        # First result should be the unknown one with empty text
        assert result.results[0].chunk_id == "doc_unknown"
        assert result.results[0].text == ""

    def test_rerank_empty_ids_from_backend(self):
        """Test handling when backend returns empty IDs."""
        mock_backend = Mock()
        mock_backend.provider_name = "cross_encoder"

        mock_rerank_result = Mock()
        mock_rerank_result.ids = []
        mock_rerank_result.scores = []

        mock_backend.rerank.return_value = mock_rerank_result

        # Pass explicit config with enabled=True
        reranker = Reranker(
            reranker_backend=mock_backend,
            config=RerankerConfig(enabled=True),
        )

        candidates = [
            RetrievalResult(chunk_id="doc1", score=0.9, text="Text 1"),
            RetrievalResult(chunk_id="doc2", score=0.8, text="Text 2"),
        ]

        result = reranker.rerank("test query", candidates)

        assert result.results == []
