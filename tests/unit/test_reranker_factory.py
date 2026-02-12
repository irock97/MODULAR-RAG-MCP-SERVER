"""Tests for Reranker Factory and BaseReranker.

These tests verify:
1. BaseReranker interface is correctly defined
2. RerankerFactory dynamic provider registration
3. FakeReranker works as expected for testing
4. NoneReranker provides pass-through behavior
"""

from typing import Any

import pytest

from libs.reranker.base_reranker import (
    BaseReranker,
    Candidate,
    RerankResult,
    NoneReranker,
    RerankerError,
    UnknownRerankerProviderError,
    RerankerConfigurationError,
)
from libs.reranker.reranker_factory import RerankerFactory
from core.settings import Settings, RerankConfig


class FakeReranker(BaseReranker):
    """Fake reranker for testing.

    This implementation returns deterministic results for testing
    without making actual API calls.
    """

    def __init__(
        self,
        response_rank: list[str] | None = None,
        **kwargs: Any
    ) -> None:
        """Initialize the Fake Reranker.

        Args:
            response_rank: Predefined ranking order (IDs)
            **kwargs: Extra arguments (ignored for compatibility)
        """
        self._response_rank = response_rank or ["id1", "id2", "id3"]
        self.call_count = 0

    @property
    def provider_name(self) -> str:
        return "fake"

    def rerank(
        self,
        query: str,
        candidates: list[Candidate],
        trace: Any = None,
        **kwargs: Any
    ) -> RerankResult:
        """Return fake reranked result."""
        self.call_count += 1

        # Return predefined ranking or original order
        if self._response_rank:
            ids = self._response_rank
            scores = [0.9 - i * 0.1 for i in range(len(ids))]
        else:
            ids = [c.id for c in candidates]
            scores = [c.score or 0.5 for c in candidates]

        return RerankResult(
            ids=ids,
            scores=scores,
            metadata={"reranked": True}
        )


class TestCandidate:
    """Test Candidate dataclass."""

    def test_create_candidate(self) -> None:
        """Test creating a basic candidate."""
        candidate = Candidate(id="doc1", content="Hello world", score=0.8)
        assert candidate.id == "doc1"
        assert candidate.content == "Hello world"
        assert candidate.score == 0.8

    def test_create_candidate_without_score(self) -> None:
        """Test creating a candidate without score."""
        candidate = Candidate(id="doc1", content="Hello")
        assert candidate.score is None


class TestRerankResult:
    """Test RerankResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating a basic result."""
        result = RerankResult(ids=["id1", "id2"], scores=[0.9, 0.8])
        assert result.ids == ["id1", "id2"]
        assert result.scores == [0.9, 0.8]
        assert result.metadata is None

    def test_create_result_with_metadata(self) -> None:
        """Test creating a result with metadata."""
        result = RerankResult(
            ids=["id1"],
            scores=[0.9],
            metadata={"reranked": True}
        )
        assert result.metadata["reranked"] is True


class TestNoneReranker:
    """Test NoneReranker (pass-through behavior)."""

    def test_provider_name(self) -> None:
        """Test that NoneReranker returns correct provider name."""
        reranker = NoneReranker()
        assert reranker.provider_name == "none"

    def test_maintains_original_order(self) -> None:
        """Test that NoneReranker maintains original order."""
        reranker = NoneReranker()
        candidates = [
            Candidate(id="doc3", content="Third"),
            Candidate(id="doc1", content="First"),
            Candidate(id="doc2", content="Second"),
        ]

        result = reranker.rerank("test query", candidates)

        # Should maintain original order
        assert result.ids == ["doc3", "doc1", "doc2"]

    def test_preserves_scores(self) -> None:
        """Test that NoneReranker preserves original scores."""
        reranker = NoneReranker()
        candidates = [
            Candidate(id="doc1", content="First", score=0.5),
            Candidate(id="doc2", content="Second", score=0.7),
        ]

        result = reranker.rerank("test query", candidates)

        # Should preserve original scores
        assert result.scores == [0.5, 0.7]

    def test_metadata_not_reranked(self) -> None:
        """Test that metadata indicates no reranking occurred."""
        reranker = NoneReranker()
        candidates = [Candidate(id="doc1", content="Test")]

        result = reranker.rerank("query", candidates)

        assert result.metadata["reranked"] is False

    def test_handles_none_scores(self) -> None:
        """Test handling candidates without scores."""
        reranker = NoneReranker()
        candidates = [
            Candidate(id="doc1", content="First"),
            Candidate(id="doc2", content="Second"),
        ]

        result = reranker.rerank("query", candidates)

        # Should use 0.0 for None scores
        assert result.scores == [0.0, 0.0]


class TestFakeReranker:
    """Test FakeReranker implementation."""

    def test_provider_name(self) -> None:
        """Test that FakeReranker returns correct provider name."""
        fake = FakeReranker()
        assert fake.provider_name == "fake"

    def test_rerank_returns_result(self) -> None:
        """Test that rerank returns RerankResult."""
        fake = FakeReranker()
        candidates = [
            Candidate(id="id1", content="Doc 1"),
            Candidate(id="id2", content="Doc 2"),
        ]

        result = fake.rerank("test query", candidates)

        assert isinstance(result, RerankResult)
        assert len(result.ids) == 3

    def test_rerank_call_count(self) -> None:
        """Test that call count is tracked."""
        fake = FakeReranker()
        assert fake.call_count == 0

        fake.rerank("query", [Candidate(id="id1", content="Test")])
        assert fake.call_count == 1

    def test_rerank_with_custom_ranking(self) -> None:
        """Test rerank with custom predefined ranking."""
        fake = FakeReranker(response_rank=["a", "b", "c"])

        result = fake.rerank("query", [
            Candidate(id="x", content="X"),
            Candidate(id="y", content="Y"),
        ])

        assert result.ids == ["a", "b", "c"]


class TestRerankerFactoryRegistration:
    """Test RerankerFactory dynamic provider registration."""

    def setup_method(self) -> None:
        """Clear registry before each test."""
        RerankerFactory.clear()

    def teardown_method(self) -> None:
        """Clear registry after each test."""
        RerankerFactory.clear()

    def test_no_providers_registered_by_default(self) -> None:
        """Test that no providers are registered by default."""
        assert RerankerFactory.get_provider_names() == []

    def test_register_provider(self) -> None:
        """Test registering a new provider."""

        class CustomReranker(BaseReranker):
            def __init__(self, **kwargs: Any) -> None:
                pass

            @property
            def provider_name(self) -> str:
                return "custom"

            def rerank(self, query, candidates, **kwargs):
                return RerankResult(ids=[], scores=[])

        RerankerFactory.register("custom", CustomReranker)

        assert "custom" in RerankerFactory.get_provider_names()
        assert RerankerFactory.has_provider("custom")

    def test_register_fake(self) -> None:
        """Test registering FakeReranker."""
        RerankerFactory.register("fake", FakeReranker)

        assert "fake" in RerankerFactory.get_provider_names()
        assert RerankerFactory.has_provider("fake")

    def test_unregister_provider(self) -> None:
        """Test unregistering a provider."""
        RerankerFactory.register("test", FakeReranker)
        assert RerankerFactory.has_provider("test")

        result = RerankerFactory.unregister("test")
        assert result is True
        assert not RerankerFactory.has_provider("test")

    def test_unregister_nonexistent(self) -> None:
        """Test unregistering a provider that doesn't exist."""
        result = RerankerFactory.unregister("nonexistent")
        assert result is False

    def test_provider_case_insensitive(self) -> None:
        """Test that provider names are case-insensitive."""

        class TestProvider(BaseReranker):
            def __init__(self, **kwargs: Any) -> None:
                pass

            @property
            def provider_name(self) -> str:
                return "test"

            def rerank(self, query, candidates, **kwargs):
                return RerankResult(ids=[], scores=[])

        RerankerFactory.register("TestProvider", TestProvider)
        assert RerankerFactory.has_provider("testprovider")
        assert RerankerFactory.has_provider("TESTPROVIDER")
        assert RerankerFactory.has_provider("TestProvider")

    def test_clear_all_providers(self) -> None:
        """Test clearing all providers."""
        RerankerFactory.register("a", FakeReranker)
        RerankerFactory.register("b", FakeReranker)
        RerankerFactory.register("c", FakeReranker)

        assert len(RerankerFactory.get_provider_names()) == 3

        RerankerFactory.clear()

        assert RerankerFactory.get_provider_names() == []


class TestRerankerFactoryCreate:
    """Test RerankerFactory.create() method."""

    def setup_method(self) -> None:
        """Clear and register fake before each test."""
        RerankerFactory.clear()
        RerankerFactory.register("fake", FakeReranker)

    def teardown_method(self) -> None:
        """Clear after each test."""
        RerankerFactory.clear()

    def test_create_fake_provider(self) -> None:
        """Test creating a FakeReranker instance."""
        settings = Settings(
            rerank=RerankConfig(enabled=True, provider="fake", model="fake-model")
        )

        reranker = RerankerFactory.create(settings)

        assert isinstance(reranker, FakeReranker)
        assert reranker.provider_name == "fake"

    def test_create_unknown_provider(self) -> None:
        """Test that unknown provider raises error."""
        settings = Settings(
            rerank=RerankConfig(enabled=True, provider="unknown-reranker")
        )

        with pytest.raises(UnknownRerankerProviderError) as exc_info:
            RerankerFactory.create(settings)

        assert "unknown-reranker" in str(exc_info.value)

    def test_create_with_kwargs_override(self) -> None:
        """Test that kwargs override settings."""
        settings = Settings(
            rerank=RerankConfig(enabled=True, provider="fake", model="original")
        )

        reranker = RerankerFactory.create(settings, model="override")

        assert isinstance(reranker, FakeReranker)

    def test_reranking_disabled_returns_none_reranker(self) -> None:
        """Test that disabled reranking returns NoneReranker."""
        settings = Settings(
            rerank=RerankConfig(enabled=False, provider="fake")
        )

        reranker = RerankerFactory.create(settings)

        assert isinstance(reranker, NoneReranker)
        assert reranker.provider_name == "none"

    def test_provider_none_returns_none_reranker(self) -> None:
        """Test that provider='none' returns NoneReranker."""
        settings = Settings(
            rerank=RerankConfig(enabled=True, provider="none")
        )

        reranker = RerankerFactory.create(settings)

        assert isinstance(reranker, NoneReranker)
        assert reranker.provider_name == "none"


class TestRerankerFactoryDynamicRegistration:
    """Test dynamic provider registration workflow."""

    def setup_method(self) -> None:
        """Clear before each test."""
        RerankerFactory.clear()

    def teardown_method(self) -> None:
        """Clear after each test."""
        RerankerFactory.clear()

    def test_register_and_create_custom_provider(self) -> None:
        """Test registering and creating a custom provider."""

        class CrossEncoderReranker(BaseReranker):
            def __init__(
                self,
                model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                **kwargs: Any
            ) -> None:
                self._model = model

            @property
            def provider_name(self) -> str:
                return "cross_encoder"

            def rerank(
                self,
                query: str,
                candidates: list[Candidate],
                **kwargs: Any
            ) -> RerankResult:
                # Simple mock: reverse order for verification
                ids = [c.id for c in reversed(candidates)]
                scores = [c.score or 0.5 for c in reversed(candidates)]
                return RerankResult(ids=ids, scores=scores)

        # Register the provider
        RerankerFactory.register("cross_encoder", CrossEncoderReranker)

        # Create an instance
        settings = Settings(
            rerank=RerankConfig(
                enabled=True,
                provider="cross_encoder",
                model="cross-encoder/ms-marco-TinyBERT-L-2"
            )
        )

        reranker = RerankerFactory.create(settings)

        assert isinstance(reranker, CrossEncoderReranker)
        assert reranker.provider_name == "cross_encoder"
        assert reranker._model == "cross-encoder/ms-marco-TinyBERT-L-2"

        # Verify the reranker works
        candidates = [
            Candidate(id="first", content="First", score=0.8),
            Candidate(id="second", content="Second", score=0.7),
        ]
        result = reranker.rerank("test query", candidates)
        assert result.ids == ["second", "first"]

    def test_multiple_providers(self) -> None:
        """Test registering multiple providers."""

        class CrossEncoderReranker(BaseReranker):
            def __init__(self, **kwargs: Any) -> None:
                pass

            @property
            def provider_name(self) -> str:
                return "cross_encoder"

            def rerank(self, query, candidates, **kwargs):
                return RerankResult(ids=[], scores=[])

        class LLMReranker(BaseReranker):
            def __init__(self, **kwargs: Any) -> None:
                pass

            @property
            def provider_name(self) -> str:
                return "llm"

            def rerank(self, query, candidates, **kwargs):
                return RerankResult(ids=[], scores=[])

        class CohereReranker(BaseReranker):
            def __init__(self, **kwargs: Any) -> None:
                pass

            @property
            def provider_name(self) -> str:
                return "cohere"

            def rerank(self, query, candidates, **kwargs):
                return RerankResult(ids=[], scores=[])

        # Register all providers
        RerankerFactory.register("cross_encoder", CrossEncoderReranker)
        RerankerFactory.register("llm", LLMReranker)
        RerankerFactory.register("cohere", CohereReranker)

        # Verify all are registered
        assert len(RerankerFactory.get_provider_names()) == 3
        assert "cross_encoder" in RerankerFactory.get_provider_names()
        assert "llm" in RerankerFactory.get_provider_names()
        assert "cohere" in RerankerFactory.get_provider_names()


class TestBaseRerankerInterface:
    """Test that BaseReranker is properly abstract."""

    def test_cannot_instantiate_base(self) -> None:
        """Test that BaseReranker cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseReranker()

    def test_subclass_must_implement_rerank(self) -> None:
        """Test that subclasses must implement rerank."""

        class IncompleteReranker(BaseReranker):
            @property
            def provider_name(self) -> str:
                return "incomplete"

            # Missing rerank() implementation

        with pytest.raises(TypeError):
            IncompleteReranker()


class TestRerankerErrors:
    """Test reranker error classes."""

    def test_reranker_error_basic(self) -> None:
        """Test basic RerankerError."""
        error = RerankerError("Test error")
        assert str(error) == "Test error"
        assert error.provider is None
        assert error.code is None

    def test_reranker_error_with_details(self) -> None:
        """Test RerankerError with provider and code."""
        error = RerankerError(
            "API error",
            provider="cross_encoder",
            code=429,
            details={"retry_after": 60}
        )
        assert error.provider == "cross_encoder"
        assert error.code == 429
        assert error.details["retry_after"] == 60

    def test_unknown_provider_error(self) -> None:
        """Test UnknownRerankerProviderError."""
        error = UnknownRerankerProviderError(
            "Unknown provider: test",
            provider="test"
        )
        assert error.provider == "test"
