"""Abstract base class for evaluators.

This module defines the BaseEvaluator interface for computing
retrieval quality metrics (hit_rate, mrr, precision, recall, etc.).

Design Principles:
    - Pluggable: All evaluators implement this interface
    - Type Safe: Full type hints for all methods
    - Observable: trace parameter for tracing integration
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from observability.logger import TraceContext


@dataclass
class EvaluationResult:
    """Result from an evaluation operation.

    Attributes:
        metrics: Dictionary of metric names to values
        details: Additional details about the evaluation
    """
    metrics: dict[str, float]
    details: dict[str, Any] | None = None


class BaseEvaluator(ABC):
    """Abstract base class for retrieval evaluators.

    All evaluators (custom, ragas, deepeval, etc.) must
    inherit from this class and implement the evaluate() method.

    Example:
        >>> class CustomEvaluator(BaseEvaluator):
        ...     def evaluate(self, query, retrieved_ids, golden_ids):
        ...         # Implementation here
        ...         pass
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this provider.

        Returns:
            Provider identifier (e.g., 'custom', 'ragas')
        """
        ...

    @abstractmethod
    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        trace: TraceContext | None = None,
        **kwargs: Any
    ) -> EvaluationResult:
        """Evaluate retrieval quality for a single query.

        Args:
            query: The original query string
            retrieved_ids: List of IDs returned by the retriever
            golden_ids: List of IDs that are relevant to the query
            trace: Tracing context for observability
            **kwargs: Additional provider-specific arguments

        Returns:
            EvaluationResult containing metrics (hit_rate, mrr, etc.)

        Raises:
            EvaluatorError: If evaluation fails
        """
        ...

    def evaluate_batch(
        self,
        queries: list[str],
        retrieved_ids_list: list[list[str]],
        golden_ids_list: list[list[str]],
        trace: TraceContext | None = None,
        **kwargs: Any
    ) -> list[EvaluationResult]:
        """Evaluate retrieval quality for multiple queries.

        Args:
            queries: List of query strings
            retrieved_ids_list: List of retrieved ID lists
            golden_ids_list: List of golden ID lists
            trace: Tracing context for observability
            **kwargs: Additional arguments

        Returns:
            List of EvaluationResult for each query

        Raises:
            EvaluatorError: If any evaluation fails
        """
        results = []
        for query, retrieved_ids, golden_ids in zip(
            queries, retrieved_ids_list, golden_ids_list
        ):
            result = self.evaluate(query, retrieved_ids, golden_ids, trace=trace)
            results.append(result)
        return results


class EvaluatorError(Exception):
    """Base exception for evaluator-related errors."""

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        code: int | None = None,
        details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.code = code
        self.details = details or {}


class UnknownEvaluatorProviderError(EvaluatorError):
    """Raised when an unknown evaluator provider is specified."""

    pass


class EvaluatorConfigurationError(EvaluatorError):
    """Raised when evaluator configuration is invalid."""

    pass
