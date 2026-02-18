"""Dense Encoder - Generate dense embeddings for chunks.

This module provides the DenseEncoder class that generates dense vector
representations for document chunks using an embedding model.

Design Principles:
    - Batch Processing: Efficiently processes multiple chunks in a single API call
    - Fail-Safe: Handles empty input gracefully
    - Traceable: Supports optional trace context for observability
    - Metadata Forward: Preserves all original chunk metadata

Example:
    >>> from libs.embedding.providers import OpenAIEmbedding
    >>> from ingestion.embedding import DenseEncoder
    >>>
    >>> embedding = OpenAIEmbedding(api_key="...")
    >>> encoder = DenseEncoder(embedding)
    >>>
    >>> chunks = [chunk1, chunk2, chunk3]
    >>> records = encoder.encode(chunks)
"""

from typing import Any

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import Chunk, ChunkRecord
from libs.embedding.base_embedding import BaseEmbedding, EmbeddingResult
from observability.logger import get_logger

logger = get_logger(__name__)


class DenseEncoder:
    """Dense embedding encoder for document chunks.

    This class generates dense vector representations for chunks using
    an embedding model. It efficiently batches multiple chunks for
    optimal API usage.

    Attributes:
        embedding: The embedding model used for encoding
        batch_size: Maximum chunks per embedding call

    Example:
        >>> from libs.embedding.providers import OpenAIEmbedding
        >>> from ingestion.embedding import DenseEncoder
        >>>
        >>> embedding = OpenAIEmbedding(model="text-embedding-3-small")
        >>> encoder = DenseEncoder(embedding, batch_size=100)
        >>>
        >>> chunks = [chunk1, chunk2, chunk3]
        >>> records = encoder.encode(chunks)
    """

    def __init__(
        self,
        embedding: BaseEmbedding,
        batch_size: int = 100,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the DenseEncoder.

        Args:
            embedding: Embedding model for generating vectors.
            batch_size: Maximum number of chunks per embedding call.
            settings: Optional settings for configuration.
        """
        self._embedding = embedding
        self._batch_size = batch_size
        self._settings = settings

        logger.info(
            f"DenseEncoder initialized: provider={embedding.provider_name}, "
            f"batch_size={batch_size}, dimensions={embedding.dimensions}"
        )

    @property
    def embedding(self) -> BaseEmbedding:
        """Get the embedding model.

        Returns:
            The embedding model used for encoding.
        """
        return self._embedding

    @property
    def provider_name(self) -> str:
        """Get the embedding provider name.

        Returns:
            Provider identifier (e.g., 'openai', 'qwen').
        """
        return self._embedding.provider_name

    @property
    def dimensions(self) -> int:
        """Get the embedding dimensions.

        Returns:
            Vector dimensions of the embedding model.
        """
        return self._embedding.dimensions

    def encode(
        self,
        chunks: list[Chunk],
        trace: TraceContext | None = None,
    ) -> list[ChunkRecord]:
        """Encode chunks into ChunkRecords with dense vectors.

        This method takes a list of chunks, extracts their text content,
        generates embeddings using the configured embedding model, and
        returns ChunkRecords containing the dense vectors.

        Args:
            chunks: List of chunks to encode.
            trace: Optional trace context for observability.

        Returns:
            List of ChunkRecords with dense_vector populated.

        Example:
            >>> chunks = [Chunk(id="1", text="Hello world")]
            >>> records = encoder.encode(chunks)
            >>> records[0].dense_vector
            [0.1, 0.2, 0.3, ...]
        """
        if not chunks:
            logger.info("No chunks to encode")
            return []

        logger.info(
            f"Encoding {len(chunks)} chunks with {self._embedding.provider_name} "
            f"(batch_size={self._batch_size})"
        )

        if trace:
            trace.record_stage(
                "dense_encoding_start",
                {
                    "chunk_count": len(chunks),
                    "provider": self.provider_name,
                    "dimensions": self.dimensions,
                },
            )

        # Batch processing: split chunks into batches
        all_records: list[ChunkRecord] = []
        total_tokens: dict[str, int] = {}

        for i in range(0, len(chunks), self._batch_size):
            batch_chunks = chunks[i : i + self._batch_size]
            batch_texts = [chunk.text for chunk in batch_chunks]

            # Generate embeddings for this batch
            result = self._embedding.embed(batch_texts, trace=trace)

            # Build ChunkRecords for this batch
            for chunk, vector in zip(batch_chunks, result.vectors):
                record = ChunkRecord(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=chunk.metadata.copy(),
                    dense_vector=vector,
                    sparse_vector=None,
                )
                all_records.append(record)

            # Accumulate token usage
            if result.usage:
                for key, value in result.usage.items():
                    total_tokens[key] = total_tokens.get(key, 0) + value

            batch_num = i // self._batch_size + 1
            logger.debug(f"Batch {batch_num}: encoded {len(batch_chunks)} chunks")

        logger.info(
            f"Encoded {len(all_records)} chunks in {(len(chunks) + self._batch_size - 1) // self._batch_size} batches, "
            f"vectors dimension={self.dimensions}"
        )

        if trace:
            trace.record_stage(
                "dense_encoding_complete",
                {
                    "record_count": len(all_records),
                    "dimensions": self.dimensions,
                    "tokens": total_tokens if total_tokens else None,
                },
            )

        return all_records

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
            ChunkRecord with dense_vector populated.
        """
        if not chunk.text:
            logger.warning(f"Empty chunk text for chunk {chunk.id}")
            return ChunkRecord(
                id=chunk.id,
                text=chunk.text,
                metadata=chunk.metadata.copy(),
                dense_vector=[],
                sparse_vector=None,
            )

        result = self._embedding.embed_single(chunk.text, trace=trace)

        return ChunkRecord(
            id=chunk.id,
            text=chunk.text,
            metadata=chunk.metadata.copy(),
            dense_vector=result,
            sparse_vector=None,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"DenseEncoder(provider={self.provider_name}, "
            f"dimensions={self.dimensions}, batch_size={self._batch_size})"
        )
