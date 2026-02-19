"""Sparse Encoder - Generate sparse vectors for chunks (TF weights for BM25).

This module provides the SparseEncoder class that generates sparse vector
representations for document chunks using term frequency (TF) weighting.
The output is designed to be used by BM25Indexer (C11) for building
inverted indexes.

Design Principles:
    - TF Only: Outputs term frequencies (TF) for BM25Indexer to use
    - Fail-Safe: Handles empty input gracefully
    - Traceable: Supports optional trace context for observability
    - Metadata Forward: Preserves all original chunk metadata

Example:
    >>> from ingestion.embedding import SparseEncoder
    >>>
    >>> encoder = SparseEncoder()
    >>>
    >>> chunks = [chunk1, chunk2, chunk3]
    >>> records = encoder.encode(chunks)
"""

import re
from collections import Counter

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import Chunk, ChunkRecord
from observability.logger import get_logger

logger = get_logger(__name__)

# Default stop words for English text
DEFAULT_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "must", "shall",
    "can", "need", "dare", "ought", "used", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "up", "about", "into", "over",
    "after", "beneath", "under", "above", "i", "you", "he", "she",
    "it", "we", "they", "what", "which", "who", "whom", "this",
    "that", "these", "those", "am", "if", "then", "because", "while",
})


class SparseEncoder:
    """Sparse encoder for document chunks.

    This class generates sparse vector representations for chunks using
    term frequency (TF) weighting. The output is designed for BM25Indexer
    to consume and build inverted indexes.

    Note: IDF computation is handled by BM25Indexer (C11).

    Attributes:
        stop_words: Set of words to filter out
        min_term_length: Minimum character length for terms

    Example:
        >>> from ingestion.embedding import SparseEncoder
        >>>
        >>> encoder = SparseEncoder(
        ...     stop_words=DEFAULT_STOP_WORDS,
        ...     min_term_length=2
        ... )
        >>>
        >>> chunks = [chunk1, chunk2, chunk3]
        >>> records = encoder.encode(chunks)
        >>> records[0].sparse_vector
        {"algorithm": 1.0, "complexity": 0.5, ...}
    """

    def __init__(
        self,
        stop_words: frozenset[str] | None = None,
        min_term_length: int = 2,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the SparseEncoder.

        Args:
            stop_words: Set of stop words to filter out.
            min_term_length: Minimum character length for valid terms.
            settings: Optional settings for configuration.
        """
        self._stop_words = stop_words or DEFAULT_STOP_WORDS
        self._min_term_length = min_term_length
        self._settings = settings

        logger.info(
            f"SparseEncoder initialized: stop_words={len(self._stop_words)}, "
            f"min_term_length={min_term_length}"
        )

    @property
    def stop_words(self) -> frozenset[str]:
        """Get the stop words set.

        Returns:
            Set of words that are filtered out.
        """
        return self._stop_words

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into terms.

        Args:
            text: Text to tokenize.

        Returns:
            List of lowercase tokens.
        """
        # Convert to lowercase and extract alphanumeric tokens
        text = text.lower()
        tokens = re.findall(r'\b[a-z0-9]+\b', text)
        # Filter by length and stop words
        tokens = [
            t for t in tokens
            if len(t) >= self._min_term_length
            and t not in self._stop_words
        ]
        return tokens

    def _compute_tf(self, text: str) -> dict[str, float]:
        """Compute term frequencies (TF) for a document.

        TF = term_count / max_term_count (normalized by max frequency)

        Args:
            text: Document text.

        Returns:
            Dictionary mapping terms to TF weights (0.0-1.0).
        """
        tokens = self._tokenize(text)
        if not tokens:
            return {}

        # Count term frequencies
        term_counts = Counter(tokens)
        max_freq = max(term_counts.values()) if term_counts else 1

        # Normalize by max frequency (TF weighting)
        tf_weights: dict[str, float] = {
            term: count / max_freq
            for term, count in term_counts.items()
        }

        return tf_weights

    def encode(
        self,
        chunks: list[Chunk],
        trace: TraceContext | None = None,
    ) -> list[ChunkRecord]:
        """Encode chunks into ChunkRecords with sparse vectors.

        This method takes a list of chunks, extracts their text content,
        computes TF (term frequency) weights, and returns ChunkRecords
        containing the sparse vectors.

        Args:
            chunks: List of chunks to encode.
            trace: Optional trace context for observability.

        Returns:
            List of ChunkRecords with sparse_vector populated.

        Example:
            >>> chunks = [Chunk(id="1", text="Hello world")]
            >>> records = encoder.encode(chunks)
            >>> records[0].sparse_vector
            {"hello": 1.0, "world": 1.0}
        """
        if not chunks:
            logger.info("No chunks to encode")
            return []

        logger.info(
            f"Encoding {len(chunks)} chunks with SparseEncoder "
            f"(min_term_length={self._min_term_length})"
        )

        if trace:
            trace.record_stage(
                "sparse_encoding_start",
                {
                    "chunk_count": len(chunks),
                    "min_term_length": self._min_term_length,
                    "stop_words_count": len(self._stop_words),
                },
            )

        # Extract texts from chunks
        texts = [chunk.text for chunk in chunks]

        # Compute TF vectors for each chunk
        all_tokens: set[str] = set()
        sparse_vectors: list[dict[str, float]] = []

        for text in texts:
            tf_weights = self._compute_tf(text)
            sparse_vectors.append(tf_weights)
            all_tokens.update(tf_weights.keys())

        # Build ChunkRecords with sparse vectors
        records: list[ChunkRecord] = []
        for chunk, sparse_vector in zip(chunks, sparse_vectors):
            # Sort by weight for consistent output
            sorted_vector = dict(
                sorted(sparse_vector.items(), key=lambda x: (-x[1], x[0]))
            )

            record = ChunkRecord(
                id=chunk.id,
                text=chunk.text,
                metadata=chunk.metadata.copy(),
                dense_vector=None,
                sparse_vector=sorted_vector,
            )
            records.append(record)

        # Count total terms
        total_terms = sum(len(v) for v in sparse_vectors)
        logger.info(
            f"Encoded {len(records)} chunks, "
            f"total_terms={total_terms}, unique_terms={len(all_tokens)}"
        )

        if trace:
            trace.record_stage(
                "sparse_encoding_complete",
                {
                    "record_count": len(records),
                    "total_terms": total_terms,
                    "unique_terms": len(all_tokens),
                },
            )

        return records

    def encode_single(
        self,
        chunk: Chunk,
        trace: TraceContext | None = None,
    ) -> ChunkRecord:
        """Encode a single chunk into a ChunkRecord.

        This is a convenience method for encoding a single chunk.

        Args:
            chunk: Single chunk to encode.
            trace: Optional trace context for observability.

        Returns:
            ChunkRecord with sparse_vector populated.
        """
        if not chunk.text:
            logger.warning(f"Empty chunk text for chunk {chunk.id}")
            return ChunkRecord(
                id=chunk.id,
                text=chunk.text,
                metadata=chunk.metadata.copy(),
                dense_vector=None,
                sparse_vector={},
            )

        # Compute TF vector
        tf_weights = self._compute_tf(chunk.text)

        # Sort by weight
        sorted_vector = dict(
            sorted(tf_weights.items(), key=lambda x: (-x[1], x[0]))
        )

        return ChunkRecord(
            id=chunk.id,
            text=chunk.text,
            metadata=chunk.metadata.copy(),
            dense_vector=None,
            sparse_vector=sorted_vector,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"SparseEncoder("
            f"min_term_length={self._min_term_length}, "
            f"stop_words={len(self._stop_words)})"
        )
