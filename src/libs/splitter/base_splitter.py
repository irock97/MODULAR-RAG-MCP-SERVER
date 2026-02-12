"""Abstract base class for Text Splitters.

This module defines the BaseSplitter interface that all text splitting
implementations must follow. This enables pluggable splitting strategies.

Design Principles:
    - Pluggable: All providers implement this interface
    - Type Safe: Full type hints for all methods
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class SplitResult:
    """Result from a text splitting operation.

    Attributes:
        chunks: List of text chunks
        metadata: Additional information about the split (e.g., chunk_count)
    """

    chunks: list[str]
    metadata: dict[str, Any] | None = None

    def __repr__(self) -> str:
        return f"SplitResult(chunks={len(self.chunks)}, metadata={self.metadata})"


class BaseSplitter(ABC):
    """Abstract base class for text splitters.

    All splitting implementations (Recursive, Semantic, Fixed, etc.) must
    inherit from this class and implement the split_text() method.

    Example:
        >>> class RecursiveSplitter(BaseSplitter):
        ...     def split_text(self, text: str) -> SplitResult:
        ...         # Implementation here
        ...         pass
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this provider.

        Returns:
            Provider identifier (e.g., 'recursive', 'semantic', 'fixed')
        """
        ...

    @abstractmethod
    def split_text(self, text: str, **kwargs: Any) -> SplitResult:
        """Split a single text into chunks.

        Args:
            text: The text to split
            **kwargs: Additional provider-specific arguments
                - chunk_size: Maximum size of each chunk
                - chunk_overlap: Overlap between consecutive chunks
                - trace: Tracing context for observability

        Returns:
            SplitResult containing list of chunks

        Raises:
            SplitterError: If splitting fails
        """
        ...

    @abstractmethod
    def split_documents(
        self,
        documents: list[str],
        **kwargs: Any
    ) -> list[SplitResult]:
        """Split multiple documents.

        Args:
            documents: List of documents to split
            **kwargs: Additional arguments

        Returns:
            List of SplitResult for each document

        Raises:
            SplitterError: If splitting fails
        """
        ...


class SplitterError(Exception):
    """Base exception for splitter-related errors."""

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.details = details or {}


class UnknownSplitterProviderError(SplitterError):
    """Raised when an unknown splitter provider is specified."""

    pass


class SplitterConfigurationError(SplitterError):
    """Raised when splitter configuration is invalid."""

    pass
