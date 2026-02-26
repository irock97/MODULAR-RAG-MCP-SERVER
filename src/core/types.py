"""Core data types for ingestion and retrieval pipeline.

This module defines the shared data structures used across
the entire RAG pipeline: from document ingestion to retrieval.

Design Principles:
    - Serializable: All types can be converted to dict/JSON
    - Immutable: Core types use frozen dataclasses
    - Extensible: metadata allows incremental field addition
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    """Raw document entity.

    Represents a document before chunking. This is the input
    to the ingestion pipeline after being loaded by a Loader.

    Attributes:
        id: Unique identifier for the document
        text: Raw text content of the document
        metadata: Document-level metadata (source_path required at minimum)
    """
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            text=data["text"],
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class Chunk:
    """Chunked document entity.

    Represents a chunk derived from a Document after the splitting
    process. Chunks preserve source reference for traceability.

    Attributes:
        id: Unique identifier for the chunk
        text: Text content of this chunk
        metadata: Chunk-level metadata (inherits from Document + chunk info)
        start_offset: Character offset where chunk starts in source document
        end_offset: Character offset where chunk ends in source document
        source_ref: Reference to source document ID
    """
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    start_offset: int | None = None
    end_offset: int | None = None
    source_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "source_ref": self.source_ref,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            text=data["text"],
            metadata=data.get("metadata", {}),
            start_offset=data.get("start_offset"),
            end_offset=data.get("end_offset"),
            source_ref=data.get("source_ref"),
        )


@dataclass(frozen=True)
class ChunkRecord:
    """Record for vector storage.

    Represents a Chunk prepared for embedding and storage.
    Contains optional vector representations for dense/sparse retrieval.

    Attributes:
        id: Unique identifier (typically matches Chunk.id)
        text: Text content for embedding computation
        metadata: Storage-level metadata
        dense_vector: Optional pre-computed dense embedding
        sparse_vector: Optional sparse vector (e.g., for BM25)
    """
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    dense_vector: list[float] | None = None
    sparse_vector: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "dense_vector": self.dense_vector,
            "sparse_vector": self.sparse_vector,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChunkRecord":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            text=data["text"],
            metadata=data.get("metadata", {}),
            dense_vector=data.get("dense_vector"),
            sparse_vector=data.get("sparse_vector"),
        )


# Type aliases for convenience
IngestionEntity = Document | Chunk | ChunkRecord
"""Union type for any entity in the ingestion pipeline."""


@dataclass(frozen=True)
class ProcessedQuery:
    """Processed query result for retrieval.

    Represents a parsed query with extracted keywords, filters,
    and normalized text ready for retrieval.

    Attributes:
        keywords: List of extracted keywords
        filters: Parsed filter dictionary
        raw_query: Original raw query string
        normalized_query: Normalized query string
    """

    keywords: list[str]
    filters: dict[str, Any]
    raw_query: str
    normalized_query: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "keywords": self.keywords,
            "filters": self.filters,
            "raw_query": self.raw_query,
            "normalized_query": self.normalized_query,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProcessedQuery":
        """Create from dictionary."""
        return cls(
            keywords=data["keywords"],
            filters=data["filters"],
            raw_query=data["raw_query"],
            normalized_query=data["normalized_query"],
        )


@dataclass(frozen=True)
class RetrievalResult:
    """Result from retrieval operations (dense or sparse).

    Represents a single retrieved chunk with its relevance score
    and metadata. This is the primary return type for retrieval operations.

    Attributes:
        chunk_id: Unique identifier for the retrieved chunk
        score: Relevance score (higher is better)
        text: Text content of the retrieved chunk
        metadata: Additional metadata for the chunk
    """

    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "score": self.score,
            "text": self.text,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetrievalResult":
        """Create from dictionary."""
        return cls(
            chunk_id=data["chunk_id"],
            score=data["score"],
            text=data["text"],
            metadata=data.get("metadata", {}),
        )
