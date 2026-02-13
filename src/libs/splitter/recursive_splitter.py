"""Recursive Character Text Splitter implementation.

This module provides a recursive character-based text splitter that respects
Markdown structure (headings, code blocks) to prevent breaking semantic content.

Design Principles:
    - Recursive splitting: Try multiple separators in order
    - Markdown-aware: Preserve heading and code block boundaries
    - Configurable: chunk_size and chunk_overlap control splitting behavior
"""

import re
from typing import Any

from libs.splitter.base_splitter import (
    BaseSplitter,
    SplitResult,
    SplitterError,
    SplitterConfigurationError,
)
from observability.logger import TraceContext, get_logger

logger = get_logger(__name__)


class RecursiveSplitter(BaseSplitter):
    """Recursive Character Text Splitter.

    Splits text by recursively trying different separators, starting with
    those that preserve the most semantic structure (Markdown headings,
    code blocks, paragraphs).

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
        # Validate chunk_size first (before default assignment)
        if chunk_size is not None and chunk_size <= 0:
            raise SplitterConfigurationError(
                f"chunk_size must be positive, got {chunk_size}",
                provider="recursive"
            )

        # Validate chunk_overlap before default assignment
        if chunk_overlap is not None and chunk_overlap < 0:
            raise SplitterConfigurationError(
                f"chunk_overlap must be non-negative, got {chunk_overlap}",
                provider="recursive"
            )

        self._chunk_size = chunk_size if chunk_size is not None else self.DEFAULT_CHUNK_SIZE
        self._chunk_overlap = chunk_overlap if chunk_overlap is not None else self.DEFAULT_CHUNK_OVERLAP
        self._keep_separator = keep_separator

        if self._chunk_overlap >= self._chunk_size:
            raise SplitterConfigurationError(
                f"chunk_overlap ({self._chunk_overlap}) must be less than "
                f"chunk_size ({self._chunk_size})",
                provider="recursive"
            )

        # Default separators - ordered by semantic importance
        # Markdown-aware: preserve structure
        self._separators = separators or self._default_separators()

    @property
    def provider_name(self) -> str:
        """Return the name of this provider.

        Returns:
            Provider identifier: 'recursive'
        """
        return "recursive"

    def _default_separators(self) -> list[str]:
        """Get default separators ordered by priority.

        Separators are ordered to preserve the most semantic structure:
        1. Double newlines (paragraphs)
        2. Code blocks (```)
        3. Headings (#, ##, ###, etc.)
        4. Single newlines
        5. Sentences (periods, question marks, exclamation marks)
        6. Words (spaces)
        7. Characters

        Returns:
            List of separator strings in priority order.
        """
        return [
            "\n\n",  # Paragraphs
            "\n```\n",  # Code blocks (opening)
            "\n```",  # Code blocks (closing variant)
            "```\n",  # Code blocks (alternative)
            "```",  # Code blocks (no newline)
            "\n## ",  # H2 headings
            "\n### ",  # H3 headings
            "\n#### ",  # H4 headings
            "\n##### ",  # H5 headings
            "\n###### ",  # H6 headings
            "\n# ",  # H1 headings (after newline)
            "# ",  # H1 headings (start of line)
            "\n",  # Single newlines
            ". ",  # Sentence endings
            "? ",  # Question marks
            "! ",  # Exclamation marks
            " ",  # Spaces (words)
            "",  # Characters (fallback)
        ]

    def _merge_splits(self, splits: list[str], separator: str) -> list[str]:
        """Merge splits with separator.

        Args:
            splits: List of text fragments.
            separator: Separator to join with.

        Returns:
            Merged list of strings.
        """
        if not splits:
            return []

        if self._keep_separator and separator:
            # Add separator to each split except the last
            return [split + separator for split in splits[:-1]] + [splits[-1]]

        return splits

    def _split_text_recursive(
        self,
        text: str,
        separators: list[str],
        chunk_size: int,
        overlap: int
    ) -> list[str]:
        """Recursively split text using separators.

        Args:
            text: Text to split.
            separators: List of separators to try.
            chunk_size: Maximum chunk size.
            overlap: Overlap between chunks.

        Returns:
            List of text chunks.
        """
        if not text:
            return []

        # If text is small enough, return as single chunk
        if len(text) <= chunk_size:
            return [text]

        # Try each separator in order
        separator = separators[0] if separators else ""
        splits = text.split(separator)

        # If we got good splits (not too many), try to merge them into chunks
        if len(splits) > 1:
            # Check if we can use these splits directly
            merged_splits = self._merge_splits(splits, separator)

            # Group splits into chunks respecting chunk_size
            chunks: list[str] = []
            current_chunk = ""

            for split in merged_splits:
                # If split itself is too big, split recursively
                if len(split) > chunk_size:
                    # Save current chunk if not empty
                    if current_chunk:
                        chunks.append(current_chunk.rstrip())
                        current_chunk = ""
                    # Recursively split the large piece
                    chunks.extend(
                        self._split_text_recursive(
                            split, separators[1:], chunk_size, overlap
                        )
                    )
                # Check if adding this split would exceed chunk_size
                elif len(current_chunk) + len(split) <= chunk_size:
                    current_chunk += split
                else:
                    # Start new chunk
                    chunks.append(current_chunk.rstrip())
                    # Add overlap from end of previous chunk
                    if overlap > 0 and chunks:
                        overlap_text = chunks[-1][-overlap:]
                        current_chunk = overlap_text + split
                    else:
                        current_chunk = split

            # Don't forget the last chunk
            if current_chunk:
                chunks.append(current_chunk.rstrip())

            return chunks

        # If only one split, try the next separator
        if len(splits) == 1 and len(separators) > 1:
            return self._split_text_recursive(
                text, separators[1:], chunk_size, overlap
            )

        # If still no good split, split by character
        if len(text) <= chunk_size:
            return [text]

        # Fallback: split by character with overlap
        chunks: list[str] = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            chunks.append(chunk)

        return chunks

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
                - chunk_size: Override chunk size
                - chunk_overlap: Override chunk overlap

        Returns:
            SplitResult containing list of chunks.

        Raises:
            SplitterError: If splitting fails.
        """
        if not text:
            return SplitResult(chunks=[])

        # Allow runtime override of chunk parameters
        chunk_size = kwargs.get("chunk_size", self._chunk_size)
        chunk_overlap = kwargs.get("chunk_overlap", self._chunk_overlap)

        logger.info(
            f"Recursive splitter: text_length={len(text)}, "
            f"chunk_size={chunk_size}, overlap={chunk_overlap}"
        )

        if trace:
            trace.record_stage(
                "text_splitting",
                {
                    "provider": self.provider_name,
                    "text_length": len(text),
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                }
            )

        try:
            # Split the text
            chunks = self._split_text_recursive(
                text,
                self._separators,
                chunk_size,
                chunk_overlap
            )

            logger.info(
                f"Recursive splitter: produced {len(chunks)} chunks"
            )

            if trace:
                trace.record_stage(
                    "split_result",
                    {
                        "chunk_count": len(chunks),
                        "chunks": chunks[:5],  # First 5 chunks for debugging
                    }
                )

            return SplitResult(
                chunks=chunks,
                metadata={
                    "chunk_count": len(chunks),
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                }
            )

        except Exception as e:
            raise SplitterError(
                f"Failed to split text: {e}",
                provider=self.provider_name,
                details={"text_length": len(text)}
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
