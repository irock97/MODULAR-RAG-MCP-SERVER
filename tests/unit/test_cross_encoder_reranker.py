"""Unit tests for CrossEncoderReranker.

This module contains unit tests for CrossEncoderReranker that use mocked
scorer responses to ensure deterministic behavior.

Design Principles:
    - Unit tests: Use mock scorer for deterministic testing
    - Coverage: Test initialization, reranking, threshold, and error cases
    - Mock-based: No actual Cross-Encoder model loading in unit tests
"""

import pytest
from unittest.mock import MagicMock

from libs.reranker.cross_encoder_reranker import CrossEncoderReranker
from libs.reranker.base_reranker import Candidate, RerankResult
from libs.reranker.base_reranker import RerankerConfigurationError


def mock_scorer(query: str, texts: list[str]) -> list[float]:
    """Mock scorer that scores based on keyword matching.

    Returns higher scores for texts containing 'relevant' keyword.
    """
    results = []
    for i, text in enumerate(texts):
        base_score = 0.5 + (i * 0.1)  # Different base scores
        if "relevant" in text.lower():
            base_score += 0.5
        if "query" in text.lower():
            base_score += 0.3
        results.append(min(base_score, 1.0))  # Cap at 1.0
    return results


class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker provider."""

    def test_initialization_default(self):
        """Test initialization with default values."""
        reranker = CrossEncoderReranker()

        assert reranker.provider_name == "cross_encoder"
        assert reranker._model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert reranker._top_k == 10
        assert reranker._timeout == 30.0

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        reranker = CrossEncoderReranker(
            model="custom/model",
            top_k=5,
            batch_size=16,
            timeout=60.0
        )

        assert reranker._model == "custom/model"
        assert reranker._top_k == 5
        assert reranker._batch_size == 16
        assert reranker._timeout == 60.0

    def test_initialization_with_mock_scorer(self):
        """Test initialization with mock scorer."""
        reranker = CrossEncoderReranker(scorer=mock_scorer)

        assert reranker._scorer == mock_scorer
        assert reranker._cross_encoder is None

    def test_rerank_empty_candidates(self):
        """Test reranking with empty candidates returns empty result."""
        reranker = CrossEncoderReranker()

        result = reranker.rerank(query="test", candidates=[])

        assert result.ids == []
        assert result.scores == []
        assert result.metadata["reranked"] is False

    def test_rerank_success(self):
        """Test successful reranking with mock scorer."""
        reranker = CrossEncoderReranker(scorer=mock_scorer)

        candidates = [
            Candidate(id="doc1", content="This is a relevant document", score=0.8),
            Candidate(id="doc2", content="This is not relevant", score=0.6),
            Candidate(id="doc3", content="Another relevant document about query", score=0.7),
        ]

        result = reranker.rerank(query="test query", candidates=candidates)

        # Should return all candidates
        assert len(result.ids) == 3
        # Check that reranking happened (order may vary based on mock scorer)
        assert set(result.ids) == {"doc1", "doc2", "doc3"}
        # Top result should have highest score
        assert result.scores[0] >= result.scores[1]
        assert result.scores[1] >= result.scores[2]

        assert result.metadata["reranked"] is True
        assert result.metadata["model"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_rerank_top_k_limit(self):
        """Test that top_k limits results."""
        reranker = CrossEncoderReranker(scorer=mock_scorer, top_k=2)

        candidates = [
            Candidate(id=f"doc{i}", content=f"Document {i}") for i in range(5)
        ]

        result = reranker.rerank(query="test", candidates=candidates)

        assert len(result.ids) == 2

    def test_rerank_with_kwargs_override(self):
        """Test that kwargs override constructor parameters."""
        reranker = CrossEncoderReranker(scorer=mock_scorer, top_k=10)

        candidates = [
            Candidate(id=f"doc{i}", content=f"Document {i}") for i in range(5)
        ]

        # Override top_k to 2
        result = reranker.rerank(query="test", candidates=candidates, top_k=2)

        assert len(result.ids) == 2

    def test_rerank_with_threshold(self):
        """Test reranking with score threshold."""
        reranker = CrossEncoderReranker(scorer=mock_scorer)

        candidates = [
            Candidate(id="doc1", content="This is relevant content", score=0.8),
            Candidate(id="doc2", content="Not relevant content", score=0.3),
            Candidate(id="doc3", content="Very relevant content about query", score=0.9),
        ]

        result = reranker.rerank_with_threshold(
            query="test",
            candidates=candidates,
            score_threshold=0.6
        )

        # Should only include docs with score >= 0.6
        assert len(result.ids) <= 3
        assert all(s >= 0.6 for s in result.scores)
        assert result.metadata["score_threshold"] == 0.6


class TestCrossEncoderRerankerFactory:
    """Tests for RerankerFactory with CrossEncoderReranker."""

    def test_factory_create_cross_encoder(self):
        """Test that RerankerFactory can create CrossEncoderReranker."""
        from libs.reranker.reranker_factory import RerankerFactory
        from libs.reranker.cross_encoder_reranker import CrossEncoderReranker

        # Register provider
        RerankerFactory.register("cross_encoder", CrossEncoderReranker)

        # Create mock settings
        mock_settings = MagicMock()
        mock_settings.rerank.enabled = True
        mock_settings.rerank.provider = "cross_encoder"
        mock_settings.rerank.model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        mock_settings.rerank.top_k = 10

        # Create reranker
        reranker = RerankerFactory.create(mock_settings)

        assert isinstance(reranker, CrossEncoderReranker)
        assert reranker.provider_name == "cross_encoder"


class TestCrossEncoderRerankerRepr:
    """Tests for CrossEncoderReranker string representation."""

    def test_repr(self):
        """Test __repr__ method."""
        reranker = CrossEncoderReranker(model="custom/model", top_k=5)

        repr_str = repr(reranker)

        assert "CrossEncoderReranker" in repr_str
        assert "cross_encoder" in repr_str
        assert "custom/model" in repr_str
        assert "top_k=5" in repr_str


class TestCrossEncoderRerankerScoring:
    """Tests for Cross-Encoder scoring behavior."""

    def test_scores_are_floats(self):
        """Test that scores are returned as floats."""
        reranker = CrossEncoderReranker(scorer=mock_scorer)

        candidates = [
            Candidate(id="doc1", content="Relevant content"),
            Candidate(id="doc2", content="Not relevant"),
        ]

        result = reranker.rerank(query="test", candidates=candidates)

        assert all(isinstance(s, float) for s in result.scores)

    def test_candidates_preserve_order_on_tie(self):
        """Test that candidates maintain relative order on score ties."""
        def tie_scorer(query: str, texts: list[str]) -> list[float]:
            # All same score
            return [0.5] * len(texts)

        reranker = CrossEncoderReranker(scorer=tie_scorer)

        candidates = [
            Candidate(id="doc1", content="Content 1"),
            Candidate(id="doc2", content="Content 2"),
            Candidate(id="doc3", content="Content 3"),
        ]

        result = reranker.rerank(query="test", candidates=candidates)

        # Should maintain original order on tie
        assert result.ids == ["doc1", "doc2", "doc3"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
