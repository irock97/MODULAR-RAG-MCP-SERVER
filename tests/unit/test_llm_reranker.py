"""Unit tests for LLMReranker.

This module contains unit tests for LLMReranker that use mocked LLM
responses to ensure deterministic behavior.

Design Principles:
    - Unit tests: Use mock LLM for deterministic testing
    - Coverage: Test initialization, reranking, parsing, and error cases
    - Mock-based: No actual LLM calls in unit tests
"""

import pytest
from unittest.mock import MagicMock, patch

from libs.reranker.llm_reranker import LLMReranker
from libs.reranker.base_reranker import Candidate, RerankResult
from libs.reranker.base_reranker import RerankerConfigurationError


class MockLLM:
    """Mock LLM for testing."""

    def __init__(self, response_content: str):
        self.response_content = response_content
        self.provider_name = "mock"

    def complete(self, prompt: str, **kwargs) -> str:
        return MagicMock(content=self.response_content)


class TestLLMReranker:
    """Tests for LLMReranker provider."""

    def test_initialization_with_llm(self):
        """Test initialization with an LLM."""
        mock_llm = MagicMock()
        mock_llm.provider_name = "mock"

        reranker = LLMReranker(llm=mock_llm)

        assert reranker.provider_name == "llm"
        assert reranker._llm == mock_llm
        assert reranker._top_k == 10
        assert reranker._score_threshold == 0.0

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        reranker = LLMReranker(
            top_k=5,
            score_threshold=0.5
        )

        assert reranker._top_k == 5
        assert reranker._score_threshold == 0.5

    def test_initialization_with_custom_prompt(self):
        """Test initialization with custom prompt template."""
        custom_prompt = "Custom prompt: Score {query} vs {candidates}"
        reranker = LLMReranker(prompt_template=custom_prompt)

        assert custom_prompt in reranker._prompt_template

    def test_initialization_without_llm(self):
        """Test that initialization without LLM is allowed but will fail on use."""
        reranker = LLMReranker()

        assert reranker._llm is None
        assert reranker.provider_name == "llm"

    def test_rerank_empty_candidates(self):
        """Test reranking with empty candidates returns empty result."""
        reranker = LLMReranker()

        result = reranker.rerank(query="test", candidates=[])

        assert result.ids == []
        assert result.scores == []
        assert result.metadata["reranked"] is False

    def test_rerank_without_llm_raises_error(self):
        """Test that reranking without LLM raises configuration error."""
        reranker = LLMReranker()

        with pytest.raises(RerankerConfigurationError) as exc_info:
            reranker.rerank(query="test", candidates=[Candidate(id="1", content="test")])

        assert "LLM is not configured" in str(exc_info.value)

    def test_rerank_success(self):
        """Test successful reranking with mock LLM response."""
        mock_response = """```json
[
  {"passage_id": "doc2", "score": 3, "reasoning": "Direct answer"},
  {"passage_id": "doc1", "score": 2, "reasoning": "Partial answer"},
  {"passage_id": "doc3", "score": 1, "reasoning": "Partial match"}
]
```"""

        mock_llm = MockLLM(response_content=mock_response)
        # Use score_threshold=0 to avoid filtering
        reranker = LLMReranker(llm=mock_llm, score_threshold=0.0)

        candidates = [
            Candidate(id="doc1", content="Content 1", score=0.8),
            Candidate(id="doc2", content="Content 2", score=0.6),
            Candidate(id="doc3", content="Content 3", score=0.7),
        ]

        result = reranker.rerank(query="test query", candidates=candidates)

        # Should return in score order (doc2=1.0, doc1=0.67, doc3=0.33)
        assert len(result.ids) == 3
        assert result.ids[0] == "doc2"  # Highest score first (3.0 -> 1.0)
        # Scores are normalized: doc2=1.0, doc1=0.67, doc3=0.33
        assert len(result.scores) == 3
        assert result.scores[0] == 1.0  # doc2
        assert result.metadata["reranked"] is True
        assert result.metadata["total_scored"] == 3

    def test_rerank_score_normalization(self):
        """Test that scores are normalized to 0-1 range."""
        mock_response = """```json
[
  {"passage_id": "doc1", "score": 2, "reasoning": "Partial answer"},
  {"passage_id": "doc2", "score": 3, "reasoning": "Full answer"}
]
```"""

        mock_llm = MockLLM(response_content=mock_response)
        # Use score_threshold=0 to avoid filtering
        reranker = LLMReranker(llm=mock_llm, score_threshold=0.0)

        candidates = [
            Candidate(id="doc1", content="Content 1"),
            Candidate(id="doc2", content="Content 2"),
        ]

        result = reranker.rerank(query="test", candidates=candidates)

        # Score 2 -> 2/3 = 0.67, Score 3 -> 1.0
        assert 0.66 <= result.scores[1] <= 0.68  # doc1 (score 2) should be second
        assert result.scores[0] == 1.0  # doc2 (score 3) should be first

    def test_rerank_score_threshold_filtering(self):
        """Test that score threshold filters results."""
        mock_response = """```json
[
  {"passage_id": "doc1", "score": 3, "reasoning": "Highly relevant"},
  {"passage_id": "doc2", "score": 0, "reasoning": "Not relevant"},
  {"passage_id": "doc3", "score": 1, "reasoning": "Marginally relevant"}
]
```"""

        mock_llm = MockLLM(response_content=mock_response)
        reranker = LLMReranker(llm=mock_llm, score_threshold=0.5)

        candidates = [
            Candidate(id="doc1", content="Content 1"),
            Candidate(id="doc2", content="Content 2"),
            Candidate(id="doc3", content="Content 3"),
        ]

        result = reranker.rerank(query="test", candidates=candidates)

        # Should only include doc1 (score 1.0) and doc3 (score 0.33 >= 0.5)
        # doc2 has score 0 which is < 0.5
        assert len(result.ids) == 2
        assert "doc1" in result.ids
        assert "doc3" in result.ids
        assert "doc2" not in result.ids

    def test_rerank_top_k_limit(self):
        """Test that top_k limits results."""
        mock_response = """```json
[
  {"passage_id": "doc1", "score": 3, "reasoning": "First"},
  {"passage_id": "doc2", "score": 3, "reasoning": "Second"},
  {"passage_id": "doc3", "score": 3, "reasoning": "Third"},
  {"passage_id": "doc4", "score": 3, "reasoning": "Fourth"},
  {"passage_id": "doc5", "score": 3, "reasoning": "Fifth"}
]
```"""

        mock_llm = MockLLM(response_content=mock_response)
        reranker = LLMReranker(llm=mock_llm, top_k=3)

        candidates = [
            Candidate(id=f"doc{i}", content=f"Content {i}") for i in range(1, 6)
        ]

        result = reranker.rerank(query="test", candidates=candidates)

        assert len(result.ids) == 3
        assert result.ids == ["doc1", "doc2", "doc3"]

    def test_rerank_invalid_json_response(self):
        """Test handling of invalid JSON response."""
        mock_response = "This is not JSON at all"
        mock_llm = MockLLM(response_content=mock_response)
        reranker = LLMReranker(llm=mock_llm)

        candidates = [Candidate(id="doc1", content="test")]

        with pytest.raises(RerankerConfigurationError):
            reranker.rerank(query="test", candidates=candidates)

    def test_rerank_fallback_on_no_valid_scores(self):
        """Test fallback when no valid scores are parsed."""
        # Response with non-matching IDs
        mock_response = """```json
[
  {"passage_id": "unknown_id", "score": 3, "reasoning": "Test"}
]
```"""

        mock_llm = MockLLM(response_content=mock_response)
        # Use score_threshold=0 to avoid filtering
        reranker = LLMReranker(llm=mock_llm, score_threshold=0.0)

        candidates = [Candidate(id="doc1", content="test content")]

        # Should not raise, but fallback to original order with 0 scores
        result = reranker.rerank(query="test", candidates=candidates)

        assert len(result.ids) == 1
        assert result.ids[0] == "doc1"
        assert result.scores[0] == 0.0

    def test_rerank_json_without_code_fences(self):
        """Test parsing JSON without code fences."""
        mock_response = """[
  {"passage_id": "doc1", "score": 3, "reasoning": "Test"},
  {"passage_id": "doc2", "score": 2, "reasoning": "Test"}
]"""

        mock_llm = MockLLM(response_content=mock_response)
        # Use score_threshold=0 to avoid filtering
        reranker = LLMReranker(llm=mock_llm, score_threshold=0.0)

        candidates = [
            Candidate(id="doc1", content="test"),
            Candidate(id="doc2", content="test"),
        ]

        result = reranker.rerank(query="test", candidates=candidates)

        assert len(result.ids) == 2
        assert result.ids[0] == "doc1"

    def test_rerank_with_kwargs_override(self):
        """Test that kwargs override constructor parameters."""
        mock_response = """```json
[
  {"passage_id": "doc1", "score": 3, "reasoning": "Test"}
]
```"""

        mock_llm = MockLLM(response_content=mock_response)
        reranker = LLMReranker(llm=mock_llm, top_k=10)

        candidates = [Candidate(id="doc1", content="test")]

        # Override top_k to 1
        result = reranker.rerank(query="test", candidates=candidates, top_k=1)

        assert len(result.ids) == 1


class TestLLMRerankerFactory:
    """Tests for RerankerFactory with LLMReranker."""

    def test_factory_create_llm_reranker(self):
        """Test that RerankerFactory can create LLMReranker."""
        from libs.reranker.reranker_factory import RerankerFactory
        from libs.reranker.llm_reranker import LLMReranker

        # Register provider
        RerankerFactory.register("llm", LLMReranker)

        # Create mock settings
        mock_settings = MagicMock()
        mock_settings.rerank.enabled = True
        mock_settings.rerank.provider = "llm"
        mock_settings.rerank.model = "gpt-4"
        mock_settings.rerank.top_k = 10

        # Create reranker
        reranker = RerankerFactory.create(mock_settings)

        assert isinstance(reranker, LLMReranker)
        assert reranker.provider_name == "llm"

    def test_factory_unregistered_raises_error(self):
        """Test that unregistered provider raises error."""
        from libs.reranker.reranker_factory import RerankerFactory

        mock_settings = MagicMock()
        mock_settings.rerank.enabled = True
        mock_settings.rerank.provider = "unknown"

        with pytest.raises(Exception):
            RerankerFactory.create(mock_settings)


class TestLLMRerankerRepr:
    """Tests for LLMReranker string representation."""

    def test_repr(self):
        """Test __repr__ method."""
        reranker = LLMReranker(top_k=5, score_threshold=0.5)

        repr_str = repr(reranker)

        assert "LLMReranker" in repr_str
        assert "llm" in repr_str
        assert "top_k=5" in repr_str
        assert "score_threshold=0.5" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
