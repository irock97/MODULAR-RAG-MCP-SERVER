"""Abstract base class for Embedding providers.

This module defines the BaseEmbedding interface that all embedding
implementations must follow. This enables pluggable embedding providers.

Design Principles:
    - Pluggable: All providers implement this interface
    - Type Safe: Full type hints for all methods
"""

from abc import ABC, abstractmethod
from typing import Any


class EmbeddingResult:
    """Result from an embedding operation.

    Attributes:
        vectors: List of embedding vectors
        usage: Token usage information (if available)
    """

    def __init__(
        self,
        vectors: list[list[float]],
        usage: dict[str, int] | None = None
    ) -> None:
        """Initialize EmbeddingResult.

        Args:
            vectors: List of embedding vectors
            usage: Optional token usage information
        """
        self.vectors = vectors
        self.usage = usage

    def __repr__(self) -> str:
        return f"EmbeddingResult(vectors={len(self.vectors)}, usage={self.usage})"


class BaseEmbedding(ABC):
    """Abstract base class for embedding providers.

    All embedding implementations (OpenAI, Local, etc.) must
    inherit from this class and implement the embed() method.

    Example:
        >>> class OpenAIEmbedding(BaseEmbedding):
        ...     def embed(self, texts: list[str]) -> EmbeddingResult:
        ...         # Implementation here
        ...         pass
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this provider.

        Returns:
            Provider identifier (e.g., 'openai', 'local')
        """
        ...

    @abstractmethod
    def embed(
        self,
        texts: list[str],
        **kwargs: Any
    ) -> EmbeddingResult:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed
            **kwargs: Additional provider-specific arguments
                - dimensions: Request specific embedding dimensions
                - trace: Tracing context for observability

        Returns:
            EmbeddingResult containing list of vectors

        Raises:
            EmbeddingError: If embedding fails
        """
        ...

    @abstractmethod
    def embed_single(self, text: str, **kwargs: Any) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Single text string to embed
            **kwargs: Additional arguments

        Returns:
            Single embedding vector

        Raises:
            EmbeddingError: If embedding fails
        """
        ...


class EmbeddingError(Exception):
    """Base exception for embedding-related errors."""

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


class UnknownEmbeddingProviderError(EmbeddingError):
    """Raised when an unknown embedding provider is specified."""

    pass


class EmbeddingConfigurationError(EmbeddingError):
    """Raised when embedding configuration is invalid."""

    pass
