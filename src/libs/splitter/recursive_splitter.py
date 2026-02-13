"""Recursive Character Text Splitter implementation using LangChain.

This module provides a recursive character-based text splitter that wraps
LangChain's RecursiveCharacterTextSplitter.

Design Principles:
    - LangChain-based: Wraps LangChain's proven implementation
    - Markdown-aware: Preserve heading and code block boundaries
    - Configurable: chunk_size and chunk_overlap control splitting behavior
"""

from typing import Any

from libs.splitter.base_splitter import (
    BaseSplitter,
    SplitResult,
    SplitterConfigurationError,
)
from observability.logger import TraceContext, get_logger

logger = get_logger(__name__)


class RecursiveSplitter(BaseSplitter):
    """Recursive Character Text Splitter using LangChain.

    Wraps LangChain's RecursiveCharacterTextSplitter to provide
    reliable text splitting with configurable separators.

    Attributes:
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Overlap between consecutive chunks
        separators: List of separators to try (in order of priority)
        keep_separator: Whether to keep separators in chunks
    """

    # Default chunk size
    DEFAULT_CHUNK_SIZE = 1000
    # Default overlap between chunks
    DEFAULT_CHUNK_OVERLAP = 200

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        keep_separator: bool = True,
        separators: list[str] | None = None,
    ) -> None:
        """Initialize the RecursiveSplitter.

        Args:
            chunk_size: Maximum chunk size in characters. Defaults to 1000.
            chunk_overlap: Overlap between chunks. Defaults to 200.
            keep_separator: Whether to keep separators in output chunks.
            separators: Custom list of separators (in priority order).

        Raises:
            SplitterConfigurationError: If configuration is invalid.
        """
        # Try to import LangChain's splitter
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            self._RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter
        except ImportError:
            raise SplitterConfigurationError(
                "langchain-text-splitters is not installed. "
                "Install it with: pip install langchain-text-splitters",
                provider="recursive"
            )

        # Validate parameters
        actual_chunk_size = chunk_size if chunk_size is not None else self.DEFAULT_CHUNK_SIZE
        actual_chunk_overlap = chunk_overlap if chunk_overlap is not None else self.DEFAULT_CHUNK_OVERLAP

        if actual_chunk_size <= 0:
            raise SplitterConfigurationError(
                f"chunk_size must be positive, got {actual_chunk_size}",
                provider="recursive"
            )

        if actual_chunk_overlap < 0:
            raise SplitterConfigurationError(
                f"chunk_overlap must be non-negative, got {actual_chunk_overlap}",
                provider="recursive"
            )

        if actual_chunk_overlap >= actual_chunk_size:
            raise SplitterConfigurationError(
                f"chunk_overlap ({actual_chunk_overlap}) must be less than "
                f"chunk_size ({actual_chunk_size})",
                provider="recursive"
            )

        # Default separators - ordered by semantic importance
        # Markdown-aware: preserve structure
        self._separators = separators or self._default_separators()

        # Store config for later use
        self._chunk_size = actual_chunk_size
        self._chunk_overlap = actual_chunk_overlap
        self._keep_separator = keep_separator

        # Create LangChain splitter instance
        self._splitter = self._RecursiveCharacterTextSplitter(
            chunk_size=actual_chunk_size,
            chunk_overlap=actual_chunk_overlap,
            keep_separator=keep_separator,
            separators=self._separators,
        )

    @property
    def provider_name(self) -> str:
        """Return the name of this provider.

        Returns:
            Provider identifier: 'recursive'
        """
        return "recursive"

    def _default_separators(self) -> list[str]:
        """Get default separators ordered by priority.

        Returns:
            List of separator strings in priority order.
        """
        return [
            "\n\n",  # Paragraphs
            "\n",  # Newlines
            " ",  # Spaces
            "",  # Characters (fallback)
        ]

    def split_text(
        self,
        text: str,
        trace: TraceContext | None = None,
        **kwargs: Any
    ) -> SplitResult:
        """Split a single text into chunks.

        Args:
            text: The text to split.
            trace: Tracing context for observability.
            **kwargs: Additional arguments.

        Returns:
            SplitResult containing list of chunks.

        Raises:
            SplitterError: If splitting fails.
        """
        if not text:
            return SplitResult(chunks=[])

        logger.info(
            f"Recursive splitter: text_length={len(text)}, "
            f"chunk_size={self._chunk_size}, "
            f"overlap={self._chunk_overlap}"
        )

        if trace:
            trace.record_stage(
                "text_splitting",
                {
                    "provider": self.provider_name,
                    "text_length": len(text),
                    "chunk_size": self._chunk_size,
                    "chunk_overlap": self._chunk_overlap,
                }
            )

        try:
            # Use LangChain's splitter
            chunks = self._splitter.split_text(text)

            logger.info(
                f"Recursive splitter: produced {len(chunks)} chunks"
            )

            if trace:
                trace.record_stage(
                    "split_result",
                    {
                        "chunk_count": len(chunks),
                    }
                )

            return SplitResult(
                chunks=chunks,
                metadata={
                    "chunk_count": len(chunks),
                    "chunk_size": self._chunk_size,
                    "chunk_overlap": self._chunk_overlap,
                }
            )

        except Exception as e:
            raise SplitterConfigurationError(
                f"Failed to split text: {e}",
                provider=self.provider_name
            )

    def split_documents(
        self,
        documents: list[str],
        trace: TraceContext | None = None,
        **kwargs: Any
    ) -> list[SplitResult]:
        """Split multiple documents.

        Args:
            documents: List of documents to split.
            trace: Tracing context for observability.
            **kwargs: Additional arguments.

        Returns:
            List of SplitResult for each document.
        """
        if not documents:
            return []

        results = []
        for i, doc in enumerate(documents):
            if trace:
                trace.record_stage(
                    "document_splitting",
                    {
                        "document_index": i,
                        "document_length": len(doc),
                    }
                )

            result = self.split_text(doc, trace, **kwargs)
            results.append(result)

        return results

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"RecursiveSplitter("
            f"provider={self.provider_name}, "
            f"chunk_size={self._chunk_size}, "
            f"chunk_overlap={self._chunk_overlap})"
        )
