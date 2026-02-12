"""Abstract base class for Rerankers.

This module defines the BaseReranker interface that all reranking
implementations must follow. This enables pluggable reranking strategies.

Design Principles:
    - Pluggable: All providers implement this interface
    - Type Safe: Full type hints for all methods
    - Fallback: NoneReranker provides pass-through behavior
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from observability.logger import TraceContext


@dataclass
class RerankResult:
    """Result from a reranking operation.

    Attributes:
        ids: List of candidate IDs in ranked order
        scores: Relevance scores for each candidate
        metadata: Additional information about the rerank
    """

    ids: list[str]
    scores: list[float]
    metadata: dict[str, Any] | None = None


@dataclass
class Candidate:
    """A candidate for reranking.

    Attributes:
        id: Unique identifier for the candidate
        content: Text content of the candidate
        score: Original relevance score (from retrieval)
    """

    id: str
    content: str
    score: float | None = None


class BaseReranker(ABC):
    """Abstract base class for reranking providers.

    All reranking implementations (LLM, Cross-Encoder, None, etc.) must
    inherit from this class and implement the rerank() method.

    Example:
        >>> class CrossEncoderReranker(BaseReranker):
        ...     def rerank(self, query, candidates):
        ...         # Implementation here
        ...         pass
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this provider.

        Returns:
            Provider identifier (e.g., 'cross_encoder', 'llm', 'none')
        """
        ...

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[Candidate],
        trace: TraceContext | None = None,
        **kwargs: Any
    ) -> RerankResult:
        """Rerank candidates based on query relevance.

        Args:
            query: The search query
            candidates: List of candidates to rerank
            trace: Tracing context for observability
            **kwargs: Additional provider-specific arguments
                - top_k: Number of top results to return

        Returns:
            RerankResult with candidates in ranked order

        Raises:
            RerankerError: If reranking fails
        """
        ...

    def rerank_with_scores(
        self,
        query: str,
        candidates: list[Candidate],
        trace: TraceContext | None = None,
        **kwargs: Any
    ) -> RerankResult:
        """Rerank with original scores preserved.

        Args:
            query: The search query
            candidates: List of candidates to rerank
            trace: Tracing context for observability
            **kwargs: Additional arguments

        Returns:
            RerankResult with original scores in metadata
        """
        result = self.rerank(query, candidates, trace=trace, **kwargs)
        original_scores = [c.score for c in candidates]
        if result.metadata is None:
            result.metadata = {}
        result.metadata["original_scores"] = original_scores
        return result


class NoneReranker(BaseReranker):
    """Pass-through reranker that maintains original order.

    This is the default fallback when reranking is disabled.
    It simply returns candidates in their original order without modification.
    """

    @property
    def provider_name(self) -> str:
        return "none"

    def rerank(
        self,
        query: str,
        candidates: list[Candidate],
        trace: TraceContext | None = None,
        **kwargs: Any
    ) -> RerankResult:
        """Return candidates in original order (no reranking)."""
        ids = [c.id for c in candidates]
        scores = [c.score or 0.0 for c in candidates]
        return RerankResult(
            ids=ids,
            scores=scores,
            metadata={"reranked": False}
        )


class RerankerError(Exception):
    """Base exception for reranker-related errors."""

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


class UnknownRerankerProviderError(RerankerError):
    """Raised when an unknown reranker provider is specified."""

    pass


class RerankerConfigurationError(RerankerError):
    """Raised when reranker configuration is invalid."""

    pass
