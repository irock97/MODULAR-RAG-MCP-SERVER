"""Tests for CustomEvaluator and EvaluatorFactory.

These tests verify:
1. CustomEvaluator computes metrics correctly
2. EvaluatorFactory dynamic provider registration
3. Batch evaluation functionality
"""

from typing import Any

import pytest

from libs.evaluator.base_evaluator import (
    BaseEvaluator,
    EvaluationResult,
    EvaluatorError,
    UnknownEvaluatorProviderError,
    EvaluatorConfigurationError,
)
from libs.evaluator.evaluator_factory import EvaluatorFactory
from libs.evaluator.custom_evaluator import CustomEvaluator
from core.settings import Settings, EvaluationConfig


class FakeEvaluator(BaseEvaluator):
    """Fake evaluator for testing.

    This implementation returns deterministic results for testing
    without making actual computations.
    """

    def __init__(
        self,
        response_metrics: dict[str, float] | None = None,
        **kwargs: Any
    ) -> None:
        """Initialize the Fake Evaluator.

        Args:
            response_metrics: Predefined metrics to return
            **kwargs: Extra arguments (ignored for compatibility)
        """
        self._response_metrics = response_metrics or {"hit_rate": 1.0, "mrr": 0.5}
        self.call_count = 0

    @property
    def provider_name(self) -> str:
        return "fake"

    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        trace: Any = None,
        **kwargs: Any
    ) -> EvaluationResult:
        """Return fake evaluation result."""
        self.call_count += 1
        return EvaluationResult(
            metrics=self._response_metrics.copy(),
            details={"query": query}
        )


class TestEvaluationResult:
    """Test EvaluationResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating a basic result."""
        result = EvaluationResult(metrics={"hit_rate": 1.0})
        assert result.metrics["hit_rate"] == 1.0
        assert result.details is None

    def test_create_result_with_details(self) -> None:
        """Test creating a result with details."""
        result = EvaluationResult(
            metrics={"mrr": 0.5},
            details={"query": "test query"}
        )
        assert result.metrics["mrr"] == 0.5
        assert result.details["query"] == "test query"


class TestCustomEvaluator:
    """Test CustomEvaluator implementation."""

    def test_provider_name(self) -> None:
        """Test that CustomEvaluator returns correct provider name."""
        evaluator = CustomEvaluator()
        assert evaluator.provider_name == "custom"

    def test_evaluate_returns_result(self) -> None:
        """Test that evaluate returns an EvaluationResult."""
        evaluator = CustomEvaluator()
        result = evaluator.evaluate(
            query="test query",
            retrieved_ids=["doc1", "doc2", "doc3"],
            golden_ids=["doc1"]
        )
        assert isinstance(result, EvaluationResult)
        assert "hit_rate" in result.metrics
        assert "mrr" in result.metrics

    def test_hit_rate_with_hit(self) -> None:
        """Test hit rate when relevant doc is retrieved."""
        evaluator = CustomEvaluator(metrics=["hit_rate"])
        result = evaluator.evaluate(
            query="test",
            retrieved_ids=["doc1", "doc2", "doc3"],
            golden_ids=["doc2"]
        )
        assert result.metrics["hit_rate"] == 1.0

    def test_hit_rate_without_hit(self) -> None:
        """Test hit rate when no relevant doc is retrieved."""
        evaluator = CustomEvaluator(metrics=["hit_rate"])
        result = evaluator.evaluate(
            query="test",
            retrieved_ids=["doc1", "doc2"],
            golden_ids=["doc3", "doc4"]
        )
        assert result.metrics["hit_rate"] == 0.0

    def test_mrr_first_hit(self) -> None:
        """Test MRR when first retrieved doc is relevant."""
        evaluator = CustomEvaluator(metrics=["mrr"])
        result = evaluator.evaluate(
            query="test",
            retrieved_ids=["doc1", "doc2", "doc3"],
            golden_ids=["doc1"]
        )
        assert result.metrics["mrr"] == 1.0  # 1/1 = 1.0

    def test_mrr_second_hit(self) -> None:
        """Test MRR when second retrieved doc is relevant."""
        evaluator = CustomEvaluator(metrics=["mrr"])
        result = evaluator.evaluate(
            query="test",
            retrieved_ids=["doc1", "doc2", "doc3"],
            golden_ids=["doc2"]
        )
        assert result.metrics["mrr"] == 0.5  # 1/2 = 0.5

    def test_mrr_no_hit(self) -> None:
        """Test MRR when no relevant doc is retrieved."""
        evaluator = CustomEvaluator(metrics=["mrr"])
        result = evaluator.evaluate(
            query="test",
            retrieved_ids=["doc1", "doc2"],
            golden_ids=["doc3"]
        )
        assert result.metrics["mrr"] == 0.0

    def test_precision_at_k(self) -> None:
        """Test precision at k."""
        evaluator = CustomEvaluator(metrics=["precision"], k_values=[5])
        result = evaluator.evaluate(
            query="test",
            retrieved_ids=["d1", "d2", "d3", "d4", "d5"],
            golden_ids=["d1", "d3"]  # 2 out of 5 are relevant
        )
        assert result.metrics["precision@5"] == 0.4  # 2/5

    def test_recall_at_k(self) -> None:
        """Test recall at k."""
        evaluator = CustomEvaluator(metrics=["recall"], k_values=[5])
        result = evaluator.evaluate(
            query="test",
            retrieved_ids=["d1", "d2", "d3"],
            golden_ids=["d1", "d2", "d3", "d4"]  # 4 relevant, 3 retrieved
        )
        assert result.metrics["recall@5"] == 0.75  # 3/4

    def test_evaluate_call_count(self) -> None:
        """Test that call count is tracked."""
        evaluator = CustomEvaluator()
        assert evaluator.call_count == 0

        evaluator.evaluate("query", ["id1"], ["id1"])
        assert evaluator.call_count == 1

    def test_details_contain_metadata(self) -> None:
        """Test that details contain query and count info."""
        evaluator = CustomEvaluator()
        result = evaluator.evaluate(
            query="test query",
            retrieved_ids=["d1", "d2"],
            golden_ids=["d1"]
        )
        assert result.details is not None
        assert result.details["query_length"] == len("test query")
        assert result.details["retrieved_count"] == 2
        assert result.details["golden_count"] == 1


class TestEvaluatorFactoryRegistration:
    """Test EvaluatorFactory dynamic provider registration."""

    def setup_method(self) -> None:
        """Clear registry before each test."""
        EvaluatorFactory.clear()

    def teardown_method(self) -> None:
        """Clear registry after each test."""
        EvaluatorFactory.clear()

    def test_no_providers_registered_by_default(self) -> None:
        """Test that no providers are registered by default."""
        assert EvaluatorFactory.get_provider_names() == []

    def test_register_provider(self) -> None:
        """Test registering a new provider."""

        class CustomProvider(BaseEvaluator):
            def __init__(self, **kwargs: Any) -> None:
                pass

            @property
            def provider_name(self) -> str:
                return "custom"

            def evaluate(self, query, retrieved_ids, golden_ids, **kwargs):
                return EvaluationResult(metrics={})

        EvaluatorFactory.register("custom", CustomProvider)

        assert "custom" in EvaluatorFactory.get_provider_names()
        assert EvaluatorFactory.has_provider("custom")

    def test_register_fake(self) -> None:
        """Test registering FakeEvaluator."""
        EvaluatorFactory.register("fake", FakeEvaluator)

        assert "fake" in EvaluatorFactory.get_provider_names()
        assert EvaluatorFactory.has_provider("fake")

    def test_unregister_provider(self) -> None:
        """Test unregistering a provider."""
        EvaluatorFactory.register("test", FakeEvaluator)
        assert EvaluatorFactory.has_provider("test")

        result = EvaluatorFactory.unregister("test")
        assert result is True
        assert not EvaluatorFactory.has_provider("test")

    def test_unregister_nonexistent(self) -> None:
        """Test unregistering a provider that doesn't exist."""
        result = EvaluatorFactory.unregister("nonexistent")
        assert result is False

    def test_provider_case_insensitive(self) -> None:
        """Test that provider names are case-insensitive."""

        class TestProvider(BaseEvaluator):
            def __init__(self, **kwargs: Any) -> None:
                pass

            @property
            def provider_name(self) -> str:
                return "test"

            def evaluate(self, query, retrieved_ids, golden_ids, **kwargs):
                return EvaluationResult(metrics={})

        EvaluatorFactory.register("TestProvider", TestProvider)
        assert EvaluatorFactory.has_provider("testprovider")
        assert EvaluatorFactory.has_provider("TESTPROVIDER")
        assert EvaluatorFactory.has_provider("TestProvider")

    def test_clear_all_providers(self) -> None:
        """Test clearing all providers."""
        EvaluatorFactory.register("a", FakeEvaluator)
        EvaluatorFactory.register("b", FakeEvaluator)
        EvaluatorFactory.register("c", FakeEvaluator)

        assert len(EvaluatorFactory.get_provider_names()) == 3

        EvaluatorFactory.clear()

        assert EvaluatorFactory.get_provider_names() == []


class TestEvaluatorFactoryCreate:
    """Test EvaluatorFactory.create() method."""

    def setup_method(self) -> None:
        """Clear and register fake before each test."""
        EvaluatorFactory.clear()
        EvaluatorFactory.register("custom", CustomEvaluator)
        EvaluatorFactory.register("fake", FakeEvaluator)

    def teardown_method(self) -> None:
        """Clear after each test."""
        EvaluatorFactory.clear()

    def test_create_custom_provider(self) -> None:
        """Test creating a CustomEvaluator instance."""
        settings = Settings(
            evaluation=EvaluationConfig(provider="custom")
        )

        evaluator = EvaluatorFactory.create(settings)

        assert isinstance(evaluator, CustomEvaluator)
        assert evaluator.provider_name == "custom"

    def test_create_unknown_provider(self) -> None:
        """Test that unknown provider raises error."""
        settings = Settings(
            evaluation=EvaluationConfig(provider="unknown-evaluator")
        )

        with pytest.raises(UnknownEvaluatorProviderError) as exc_info:
            EvaluatorFactory.create(settings)

        assert "unknown-evaluator" in str(exc_info.value)

    def test_create_with_kwargs_override(self) -> None:
        """Test that kwargs override settings."""
        settings = Settings(
            evaluation=EvaluationConfig(provider="custom")
        )

        evaluator = EvaluatorFactory.create(
            settings,
            metrics=["hit_rate", "mrr", "precision"]
        )

        assert isinstance(evaluator, CustomEvaluator)
        assert "precision" in evaluator._metrics

    def test_create_missing_provider_config(self) -> None:
        """Test that missing provider config raises error."""
        settings = Settings(
            evaluation=EvaluationConfig(provider=None)
        )

        with pytest.raises(EvaluatorConfigurationError):
            EvaluatorFactory.create(settings)


class TestEvaluatorFactoryDynamicRegistration:
    """Test dynamic provider registration workflow."""

    def setup_method(self) -> None:
        """Clear before each test."""
        EvaluatorFactory.clear()

    def teardown_method(self) -> None:
        """Clear after each test."""
        EvaluatorFactory.clear()

    def test_register_and_create_custom_provider(self) -> None:
        """Test registering and creating a custom provider."""

        class RagasEvaluator(BaseEvaluator):
            def __init__(self, **kwargs: Any) -> None:
                pass

            @property
            def provider_name(self) -> str:
                return "ragas"

            def evaluate(self, query, retrieved_ids, golden_ids, **kwargs):
                return EvaluationResult(metrics={"ragas_score": 0.9})

        # Register the provider
        EvaluatorFactory.register("ragas", RagasEvaluator)

        # Create an instance
        settings = Settings(
            evaluation=EvaluationConfig(provider="ragas")
        )

        evaluator = EvaluatorFactory.create(settings)

        assert isinstance(evaluator, RagasEvaluator)
        assert evaluator.provider_name == "ragas"

        # Verify the evaluator works
        result = evaluator.evaluate("test", ["d1"], ["d1"])
        assert result.metrics["ragas_score"] == 0.9

    def test_multiple_providers(self) -> None:
        """Test registering multiple providers."""

        class CustomProvider(BaseEvaluator):
            def __init__(self, **kwargs: Any) -> None:
                pass

            @property
            def provider_name(self) -> str:
                return "custom"

            def evaluate(self, query, retrieved_ids, golden_ids, **kwargs):
                return EvaluationResult(metrics={})

        class RagasEvaluator(BaseEvaluator):
            def __init__(self, **kwargs: Any) -> None:
                pass

            @property
            def provider_name(self) -> str:
                return "ragas"

            def evaluate(self, query, retrieved_ids, golden_ids, **kwargs):
                return EvaluationResult(metrics={})

        class DeepEvalProvider(BaseEvaluator):
            def __init__(self, **kwargs: Any) -> None:
                pass

            @property
            def provider_name(self) -> str:
                return "deepeval"

            def evaluate(self, query, retrieved_ids, golden_ids, **kwargs):
                return EvaluationResult(metrics={})

        # Register all providers
        EvaluatorFactory.register("custom", CustomProvider)
        EvaluatorFactory.register("ragas", RagasEvaluator)
        EvaluatorFactory.register("deepeval", DeepEvalProvider)

        # Verify all are registered
        assert len(EvaluatorFactory.get_provider_names()) == 3
        assert "custom" in EvaluatorFactory.get_provider_names()
        assert "ragas" in EvaluatorFactory.get_provider_names()
        assert "deepeval" in EvaluatorFactory.get_provider_names()


class TestBaseEvaluatorInterface:
    """Test that BaseEvaluator is properly abstract."""

    def test_cannot_instantiate_base(self) -> None:
        """Test that BaseEvaluator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseEvaluator()

    def test_subclass_must_implement_evaluate(self) -> None:
        """Test that subclasses must implement evaluate."""

        class IncompleteEvaluator(BaseEvaluator):
            @property
            def provider_name(self) -> str:
                return "incomplete"

            # Missing evaluate() implementation

        with pytest.raises(TypeError):
            IncompleteEvaluator()


class TestEvaluatorErrors:
    """Test evaluator error classes."""

    def test_evaluator_error_basic(self) -> None:
        """Test basic EvaluatorError."""
        error = EvaluatorError("Test error")
        assert str(error) == "Test error"
        assert error.provider is None
        assert error.code is None

    def test_evaluator_error_with_details(self) -> None:
        """Test EvaluatorError with provider and code."""
        error = EvaluatorError(
            "API error",
            provider="custom",
            code=429,
            details={"retry_after": 60}
        )
        assert error.provider == "custom"
        assert error.code == 429
        assert error.details["retry_after"] == 60

    def test_unknown_provider_error(self) -> None:
        """Test UnknownEvaluatorProviderError."""
        error = UnknownEvaluatorProviderError(
            "Unknown provider: test",
            provider="test"
        )
        assert error.provider == "test"
