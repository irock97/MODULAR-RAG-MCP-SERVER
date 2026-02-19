"""Batch Processor - Batch orchestration for encoding operations.

This module provides the BatchProcessor class that splits chunks into batches,
drives dense/sparse encoding, and records batch timing for observability.

Design Principles:
    - Batch Management: Split chunks into configurable batch sizes
    - Encoder Agnostic: Works with any encoder implementing the encoding protocol
    - Trace Support: Records batch timing for observability
    - Order Preserved: Maintains chunk order through batch processing
    - Progress Reportable: Supports progress callbacks for long-running tasks

Example:
    >>> from ingestion.embedding import BatchProcessor
    >>>
    >>> processor = BatchProcessor(
    ...     batch_size=32,
    ...     dense_encoder=dense_encoder,
    ...     sparse_encoder=sparse_encoder
    ... )
    >>>
    >>> chunks = [chunk1, chunk2, chunk3, ...]
    >>> records = processor.process(chunks)
"""

import time
from typing import Callable

from core.trace.trace_context import TraceContext
from core.types import Chunk, ChunkRecord
from ingestion.embedding import DenseEncoder, SparseEncoder
from observability.logger import get_logger

logger = get_logger(__name__)


class BatchProcessor:
    """Batch processor for orchestrating encoding operations.

    This class manages the batched processing of chunks through encoding
    operations (dense and/or sparse), providing timing information for
    observability and progress reporting.

    Attributes:
        batch_size: Number of chunks to process per batch
        dense_encoder: Encoder for dense embeddings (optional)
        sparse_encoder: Encoder for sparse vectors (optional)
        progress_callback: Optional callback for progress updates

    Example:
        >>> processor = BatchProcessor(
        ...     batch_size=32,
        ...     dense_encoder=dense_encoder,
        ...     sparse_encoder=sparse_encoder
        ... )
        >>> records = processor.process(chunks)
    """

    def __init__(
        self,
        batch_size: int = 32,
        dense_encoder: DenseEncoder | None = None,
        sparse_encoder: SparseEncoder | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        """Initialize the BatchProcessor.

        Args:
            batch_size: Number of chunks to process per batch.
            dense_encoder: Optional encoder for dense embeddings.
            sparse_encoder: Optional encoder for sparse vectors.
            progress_callback: Optional callback(current_batch, total_batches).
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self._batch_size = batch_size
        self._dense_encoder = dense_encoder
        self._sparse_encoder = sparse_encoder
        self._progress_callback = progress_callback

        logger.info(
            f"BatchProcessor initialized: batch_size={batch_size}, "
            f"dense_encoder={'enabled' if dense_encoder else 'disabled'}, "
            f"sparse_encoder={'enabled' if sparse_encoder else 'disabled'}, "
            f"progress_callback={'enabled' if progress_callback else 'disabled'}"
        )

    @property
    def batch_size(self) -> int:
        """Get the batch size.

        Returns:
            Number of chunks processed per batch.
        """
        return self._batch_size

    def _split_into_batches(self, chunks: list[Chunk]) -> list[list[Chunk]]:
        """Split chunks into batches.

        Args:
            chunks: List of chunks to split.

        Returns:
            List of batches, each containing up to batch_size chunks.
        """
        return [
            chunks[i : i + self._batch_size]
            for i in range(0, len(chunks), self._batch_size)
        ]

    def process(
        self,
        chunks: list[Chunk],
        trace: TraceContext | None = None,
    ) -> list[ChunkRecord]:
        """Process chunks through encoding operations in batches.

        This method splits chunks into batches, processes each batch through
        the configured encoders (dense and/or sparse), and returns the combined
        ChunkRecords maintaining original order.

        Args:
            chunks: List of chunks to process.
            trace: Optional trace context for observability.

        Returns:
            List of ChunkRecords with encoded vectors, in original order.

        Example:
            >>> records = processor.process(chunks)
        """
        if not chunks:
            logger.info("No chunks to process")
            return []

        logger.info(
            f"Processing {len(chunks)} chunks with batch_size={self._batch_size}"
        )

        if trace:
            trace.record_stage(
                "batch_processing_start",
                {
                    "chunk_count": len(chunks),
                    "batch_size": self._batch_size,
                    "has_dense_encoder": self._dense_encoder is not None,
                    "has_sparse_encoder": self._sparse_encoder is not None,
                },
            )

        # Split into batches
        batches = self._split_into_batches(chunks)
        total_batches = len(batches)

        logger.info(f"Split into {total_batches} batches")

        all_records: list[ChunkRecord] = []

        for batch_idx, batch_chunks in enumerate(batches):
            batch_start_time = time.perf_counter()

            logger.info(
                f"Processing batch {batch_idx + 1}/{total_batches} "
                f"({len(batch_chunks)} chunks)"
            )

            # Process dense encoding
            dense_records: list[ChunkRecord] | None = None
            if self._dense_encoder is not None:
                dense_records = self._dense_encoder.encode(batch_chunks, trace=trace)

            # Process sparse encoding
            sparse_records: list[ChunkRecord] | None = None
            if self._sparse_encoder is not None:
                sparse_records = self._sparse_encoder.encode(batch_chunks, trace=trace)

            # Calculate batch timing BEFORE merge
            batch_duration = time.perf_counter() - batch_start_time

            # Merge records and add encoding duration to metadata
            batch_records = self._merge_records(
                batch_chunks, dense_records, sparse_records, batch_duration
            )
            all_records.extend(batch_records)
            logger.info(
                f"Batch {batch_idx + 1}/{total_batches} completed in "
                f"{batch_duration:.3f}s"
            )

            # Report progress
            if self._progress_callback:
                self._progress_callback(batch_idx + 1, total_batches)

            if trace:
                trace.record_stage(
                    "batch_completed",
                    {
                        "batch_index": batch_idx + 1,
                        "total_batches": total_batches,
                        "chunk_count": len(batch_chunks),
                        "duration_seconds": batch_duration,
                    },
                )

        # Calculate total encoding duration from record metadata
        total_duration = sum(
            r.metadata.get("_encoding_duration", 0) for r in all_records
        )

        logger.info(
            f"Batch processing complete: {len(all_records)} records, "
            f"total_chunks={len(chunks)}, batches={total_batches}, "
            f"total_duration={total_duration:.3f}s"
        )

        if trace:
            trace.record_stage(
                "batch_processing_complete",
                {
                    "record_count": len(all_records),
                    "total_chunks": len(chunks),
                    "total_batches": total_batches,
                    "total_duration_seconds": total_duration,
                },
            )

        return all_records

    def _merge_records(
        self,
        chunks: list[Chunk],
        dense_records: list[ChunkRecord] | None,
        sparse_records: list[ChunkRecord] | None,
        encoding_duration: float = 0.0,
    ) -> list[ChunkRecord]:
        """Merge dense and sparse encoding results into final records.

        Args:
            chunks: Original chunks being processed.
            dense_records: Records with dense vectors (may be None).
            sparse_records: Records with sparse vectors (may be None).
            encoding_duration: Batch encoding duration in seconds.

        Returns:
            Merged ChunkRecords with both dense and sparse vectors.
        """
        # Build lookup dictionaries by chunk id
        dense_map = {r.id: r for r in dense_records} if dense_records else {}
        sparse_map = {r.id: r for r in sparse_records} if sparse_records else {}

        merged_records: list[ChunkRecord] = []

        for chunk in chunks:
            dense_record = dense_map.get(chunk.id)
            sparse_record = sparse_map.get(chunk.id)

            # Merge: use dense_vector from dense_record, sparse_vector from sparse_record
            # Add encoding duration to metadata for trace purposes
            metadata = chunk.metadata.copy()
            metadata["_encoding_duration"] = encoding_duration

            merged = ChunkRecord(
                id=chunk.id,
                text=chunk.text,
                metadata=metadata,
                dense_vector=dense_record.dense_vector if dense_record else None,
                sparse_vector=sparse_record.sparse_vector if sparse_record else None,
            )
            merged_records.append(merged)

        return merged_records

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"BatchProcessor("
            f"batch_size={self._batch_size}, "
            f"dense_encoder={'enabled' if self._dense_encoder else 'disabled'}, "
            f"sparse_encoder={'enabled' if self._sparse_encoder else 'disabled'}, "
            f"progress_callback={'enabled' if self._progress_callback else 'disabled'})"
        )
