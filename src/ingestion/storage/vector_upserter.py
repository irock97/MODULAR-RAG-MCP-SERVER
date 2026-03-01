"""VectorUpserter - Idempotent vector storage for dense embeddings.

This module provides the VectorUpserter class that handles idempotent
upsertion of dense vectors into a vector store. It generates stable chunk
IDs based on source content to ensure the same content always produces
the same ID.

Design Principles:
    - Idempotent: Same content produces same chunk_id
    - Deterministic: ID generation is based on source content
    - Batch Support: Supports batch upsert with order preservation

Example:
    >>> from ingestion.storage import VectorUpserter
    >>> from libs.vector_store import ChromaVectorStore
    >>>
    >>> store = ChromaVectorStore(collection_name="docs")
    >>> upsert = VectorUpserter(vector_store=store)
    >>>
    >>> records = [ChunkRecord(id="chunk1", text="Hello world", dense_vector=[0.1, 0.2])]
    >>> result = upsert.upsert(records)
"""

import hashlib
import json
from typing import Any

from core.trace.trace_context import TraceContext
from core.types import ChunkRecord
from libs.vector_store.base_vector_store import BaseVectorStore, VectorRecord
from observability.logger import get_logger

logger = get_logger(__name__)


class VectorUpserter:
    """Idempotent vector upsert handler.

    This class handles upserting dense vectors into a vector store
    with idempotency guarantees. It generates stable chunk IDs based
    on source content (source_path + chunk_index + content_hash).

    Attributes:
        vector_store: The vector store to upsert into

    Example:
        >>> from ingestion.storage import VectorUpserter
        >>> upsert = VectorUpserter(vector_store=my_store)
        >>> records = [ChunkRecord(id="1", text="content", dense_vector=[0.1, 0.2])]
        >>> upsert.upsert(records)
    """

    # Number of characters to use from content hash
    CONTENT_HASH_PREFIX_LEN = 8

    def __init__(
        self,
        vector_store: BaseVectorStore,
    ) -> None:
        """Initialize the VectorUpserter.

        Args:
            vector_store: The vector store to upsert into.
        """
        self._vector_store = vector_store

        logger.info(
            f"VectorUpserter initialized with provider={vector_store.provider_name}"
        )

    def _generate_chunk_id(
        self,
        source_path: str,
        chunk_index: int,
        text: str,
    ) -> str:
        """Generate deterministic chunk ID.

        ID format: hash(source_path + chunk_index + content_hash[:8])

        Args:
            source_path: Path to the source document
            chunk_index: Index of the chunk within the document
            text: Text content of the chunk

        Returns:
            Deterministic chunk ID string
        """
        # Compute content hash from text
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        content_prefix = content_hash[: self.CONTENT_HASH_PREFIX_LEN]

        # Combine source_path + chunk_index + content_prefix
        id_source = f"{source_path}:{chunk_index}:{content_prefix}"

        # Generate final hash
        chunk_id = hashlib.sha256(id_source.encode("utf-8")).hexdigest()[:16]

        logger.debug(
            f"Generated chunk_id={chunk_id} from "
            f"source={source_path}, index={chunk_index}"
        )

        return chunk_id

    def _record_to_vector_record(
        self,
        record: ChunkRecord,
    ) -> VectorRecord:
        """Convert ChunkRecord to VectorRecord.

        Args:
            record: ChunkRecord with dense_vector

        Returns:
            VectorRecord for upserting

        Raises:
            ValueError: If dense_vector is missing
        """
        if record.dense_vector is None:
            raise ValueError(
                f"ChunkRecord {record.id} has no dense_vector for upsert"
            )

        # Extract source_path and chunk_index from metadata
        source_path = record.metadata.get("source_path", "")
        chunk_index = record.metadata.get("chunk_index", 0)

        # Generate deterministic chunk_id if not provided
        if hasattr(record, "id") and record.id:
            # Use provided id if it looks like a hash (16 hex chars)
            chunk_id = record.id
        else:
            chunk_id = self._generate_chunk_id(
                source_path=source_path,
                chunk_index=chunk_index,
                text=record.text,
            )

        # Convert metadata for ChromaDB compatibility
        # ChromaDB only supports str, int, float, bool in metadata
        # Convert complex types (dicts, lists, etc.) to JSON strings
        metadata = self._convert_metadata_for_chroma(record.metadata)

        # Include text content in metadata for retrieval
        if record.text:
            metadata["text"] = record.text

        return VectorRecord(
            id=chunk_id,
            vector=record.dense_vector,
            metadata=metadata,
        )

    def _convert_metadata_for_chroma(
        self,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Convert metadata to be ChromaDB compatible.

        ChromaDB only supports str, int, float, bool values in metadata.
        This method converts complex types to JSON strings.

        Args:
            metadata: Original metadata dict

        Returns:
            ChromaDB-compatible metadata dict
        """
        converted: dict[str, Any] = {}

        for key, value in metadata.items():
            if value is None:
                # Skip None values
                continue
            elif isinstance(value, (str, int, float, bool)):
                # Keep simple types as-is
                converted[key] = value
            elif isinstance(value, (list, dict)):
                # Convert complex types to JSON string
                converted[key] = json.dumps(value, ensure_ascii=False)
            else:
                # For any other type, convert to string
                converted[key] = str(value)

        return converted

    def upsert(
        self,
        records: list[ChunkRecord],
        trace: TraceContext | None = None,
    ) -> list[str]:
        """Upsert chunk records into the vector store.

        This method generates stable chunk IDs for each record and
        upserts them into the vector store. The operation is idempotent:
        the same content will always produce the same chunk_id.

        Args:
            records: List of ChunkRecords with dense_vector populated
            trace: Optional trace context for observability

        Returns:
            List of successfully upserted chunk IDs

        Example:
            >>> records = [
            ...     ChunkRecord(
            ...         id="chunk1",
            ...         text="Hello world",
            ...         dense_vector=[0.1, 0.2],
            ...         metadata={"source_path": "/docs/a.pdf", "chunk_index": 0}
            ...     )
            ... ]
            >>> upsert.upsert(records)
            ['a1b2c3d4e5f6g7h8']
        """
        if not records:
            logger.info("No records to upsert")
            return []

        logger.info(f"Upserting {len(records)} records")

        if trace:
            trace.record_stage(
                "vector_upsert_start",
                {"record_count": len(records)},
            )

        # Filter records with dense_vector
        valid_records = [r for r in records if r.dense_vector is not None]

        if len(valid_records) < len(records):
            missing_count = len(records) - len(valid_records)
            logger.warning(
                f"Skipping {missing_count} records without dense_vector"
            )

        if not valid_records:
            logger.warning("No valid records with dense_vector to upsert")
            return []

        # Convert to VectorRecords
        vector_records: list[VectorRecord] = []
        for record in valid_records:
            try:
                vector_record = self._record_to_vector_record(record)
                vector_records.append(vector_record)
            except ValueError as e:
                logger.warning(f"Skipping record {record.id}: {e}")

        # Upsert to vector store
        try:
            result_ids = self._vector_store.upsert(vector_records, trace=trace)

            logger.info(f"Successfully upserted {len(result_ids)} records")

            if trace:
                trace.record_stage(
                    "vector_upsert_complete",
                    {"upserted_count": len(result_ids)},
                )

            return result_ids

        except Exception as e:
            logger.error(f"Failed to upsert records: {e}")
            raise

    def upsert_single(
        self,
        record: ChunkRecord,
        trace: TraceContext | None = None,
    ) -> str:
        """Upsert a single chunk record.

        This is a convenience method for upserting a single record.

        Args:
            record: ChunkRecord with dense_vector populated
            trace: Optional trace context for observability

        Returns:
            The upserted chunk ID

        Raises:
            ValueError: If record has no dense_vector
        """
        results = self.upsert([record], trace=trace)

        if not results:
            raise ValueError(f"Failed to upsert record {record.id}")

        return results[0]

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"VectorUpserter(provider={self._vector_store.provider_name})"
        )
