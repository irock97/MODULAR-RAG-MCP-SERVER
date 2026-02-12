"""Abstract base class for Vector Store providers.

This module defines the BaseVectorStore interface that all vector database
implementations must follow. This enables pluggable vector store providers.

Design Principles:
    - Pluggable: All providers implement this interface
    - Type Safe: Full type hints for all methods
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VectorRecord:
    """A single record for upserting to vector store.

    Attributes:
        id: Unique identifier for the record
        vector: The embedding vector
        metadata: Additional metadata for the record
    """

    id: str
    vector: list[float]
    metadata: dict[str, Any] | None = None


@dataclass
class QueryResult:
    """Result from a vector store query.

    Attributes:
        ids: List of matched record IDs
        scores: Similarity scores for each match
        metadata: Metadata for each matched record (aligned with ids)
    """

    ids: list[str]
    scores: list[float]
    metadata: list[dict[str, Any] | None] = field(default_factory=list)


class BaseVectorStore(ABC):
    """Abstract base class for vector store providers.

    All vector store implementations (Chroma, Qdrant, Pinecone, etc.) must
    inherit from this class and implement the core methods.

    Example:
        >>> class ChromaVectorStore(BaseVectorStore):
        ...     def upsert(self, records):
        ...         # Implementation here
        ...         pass
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this provider.

        Returns:
            Provider identifier (e.g., 'chroma', 'qdrant', 'pinecone')
        """
        ...

    @abstractmethod
    def upsert(
        self,
        records: list[VectorRecord],
        **kwargs: Any
    ) -> list[str]:
        """Upsert (insert or update) vectors into the store.

        Args:
            records: List of VectorRecord to upsert
            **kwargs: Additional provider-specific arguments
                - trace: Tracing context for observability
                - batch_size: Batch size for bulk operations

        Returns:
            List of successfully upserted record IDs

        Raises:
            VectorStoreError: If upsert fails
        """
        ...

    @abstractmethod
    def query(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        **kwargs: Any
    ) -> QueryResult:
        """Query the vector store for similar vectors.

        Args:
            query_vector: The query embedding vector
            top_k: Number of results to return
            filters: Metadata filters to apply
            **kwargs: Additional arguments

        Returns:
            QueryResult with matched IDs and scores

        Raises:
            VectorStoreError: If query fails
        """
        ...

    @abstractmethod
    def delete(
        self,
        ids: list[str],
        **kwargs: Any
    ) -> bool:
        """Delete records from the store.

        Args:
            ids: List of record IDs to delete
            **kwargs: Additional arguments

        Returns:
            True if deletion was successful

        Raises:
            VectorStoreError: If delete fails
        """
        ...

    @abstractmethod
    def count(self, **kwargs: Any) -> int:
        """Get the total number of records in the store.

        Args:
            **kwargs: Additional arguments

        Returns:
            Total number of records

        Raises:
            VectorStoreError: If count fails
        """
        ...

    @abstractmethod
    def clear(self, **kwargs: Any) -> bool:
        """Clear all records from the store.

        Args:
            **kwargs: Additional arguments

        Returns:
            True if clear was successful

        Raises:
            VectorStoreError: If clear fails
        """
        ...


class VectorStoreError(Exception):
    """Base exception for vector store-related errors."""

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


class UnknownVectorStoreProviderError(VectorStoreError):
    """Raised when an unknown vector store provider is specified."""

    pass


class VectorStoreConfigurationError(VectorStoreError):
    """Raised when vector store configuration is invalid."""

    pass
