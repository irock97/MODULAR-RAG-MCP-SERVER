"""ChromaDB Vector Store implementation.

This module provides ChromaDB-based vector storage that follows the BaseVectorStore interface.
It supports local persistence and metadata filtering.

Design Principles:
    - ChromaDB-backed: Uses ChromaDB for efficient vector storage and retrieval
    - Local persistence: Supports persistence to local directory
    - Filterable: Supports metadata filters in queries
"""

import uuid
from typing import Any

from libs.vector_store.base_vector_store import (
    BaseVectorStore,
    VectorRecord,
    QueryResult,
    VectorStoreConfigurationError,
)
from observability.logger import TraceContext, get_logger

logger = get_logger(__name__)


class ChromaStore(BaseVectorStore):
    """ChromaDB-backed Vector Store.

    Provides vector storage and retrieval using ChromaDB with
    local persistence support.

    Attributes:
        persist_directory: Directory for local persistence
        collection_name: Name of the ChromaDB collection
    """

    # Default collection name
    DEFAULT_COLLECTION_NAME = "default"
    # Default persistence directory
    DEFAULT_PERSIST_DIR = "data/db/chroma"

    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        """Initialize the ChromaStore.

        Args:
            persist_directory: Directory for local persistence.
            collection_name: Name of the ChromaDB collection.

        Raises:
            VectorStoreConfigurationError: If chromadb is not installed.
        """
        # Try to import chromadb
        try:
            import chromadb
            self._chromadb = chromadb
        except ImportError:
            raise VectorStoreConfigurationError(
                "chromadb is not installed. "
                "Install it with: pip install chromadb",
                provider="chroma"
            )

        self._persist_directory = persist_directory or self.DEFAULT_PERSIST_DIR
        self._collection_name = collection_name or self.DEFAULT_COLLECTION_NAME

        # Create ChromaDB client with persistence
        self._client = self._chromadb.PersistentClient(
            path=self._persist_directory
        )

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name
        )

    @property
    def provider_name(self) -> str:
        """Return the name of this provider.

        Returns:
            Provider identifier: 'chroma'
        """
        return "chroma"

    def upsert(
        self,
        records: list[VectorRecord],
        trace: TraceContext | None = None,
        **kwargs: Any
    ) -> list[str]:
        """Upsert vectors into the store.

        Args:
            records: List of VectorRecord to upsert
            trace: Tracing context for observability
            **kwargs: Additional arguments

        Returns:
            List of successfully upserted record IDs

        Raises:
            VectorStoreConfigurationError: If upsert fails
        """
        if not records:
            return []

        logger.info(
            f"ChromaStore upsert: record_count={len(records)}, "
            f"collection={self._collection_name}"
        )

        if trace:
            trace.record_stage(
                "vector_upsert",
                {
                    "provider": self.provider_name,
                    "record_count": len(records),
                    "collection": self._collection_name,
                }
            )

        try:
            ids = []
            embeddings = []
            metadatas = []
            documents = []

            for record in records:
                # Generate ID if not provided
                record_id = record.id or str(uuid.uuid4())
                ids.append(record_id)
                embeddings.append(record.vector)
                # ChromaDB requires non-empty metadata, use placeholder if empty
                metadatas.append(record.metadata or {"source": ""})
                # Use metadata source or empty document placeholder
                documents.append(record.metadata.get("source", "") if record.metadata else "")

            # Upsert to ChromaDB
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
            )

            logger.info(
                f"ChromaStore upsert complete: {len(ids)} records"
            )

            return ids

        except Exception as e:
            raise VectorStoreConfigurationError(
                f"Failed to upsert records: {e}",
                provider=self.provider_name,
                details={"record_count": len(records)}
            )

    def query(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        trace: TraceContext | None = None,
        **kwargs: Any
    ) -> QueryResult:
        """Query the vector store for similar vectors.

        Args:
            query_vector: The query embedding vector
            top_k: Number of results to return
            filters: Metadata filters to apply
            trace: Tracing context for observability
            **kwargs: Additional arguments

        Returns:
            QueryResult with matched IDs and scores

        Raises:
            VectorStoreConfigurationError: If query fails
        """
        if not query_vector:
            return QueryResult(ids=[], scores=[], metadata=[])

        logger.info(
            f"ChromaStore query: top_k={top_k}, "
            f"has_filters={filters is not None}"
        )

        if trace:
            trace.record_stage(
                "vector_query",
                {
                    "provider": self.provider_name,
                    "top_k": top_k,
                    "has_filters": filters is not None,
                    "vector_dimensions": len(query_vector),
                }
            )

        try:
            # Build where clause for filters
            where_clause = None
            if filters:
                where_clause = {}
                for key, value in filters.items():
                    where_clause[key] = value

            # Query ChromaDB
            results = self._collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=where_clause,
            )

            # Extract results
            ids = results.get("ids", [[]])[0] if results.get("ids") else []
            distances = results.get("distances", [[]])[0] if results.get("distances") else []
            metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []

            # Convert distances to similarity scores (ChromaDB uses distances)
            # Lower distance = higher similarity
            scores = [1.0 / (1.0 + d) for d in distances] if distances else []

            logger.info(
                f"ChromaStore query complete: {len(ids)} results"
            )

            return QueryResult(
                ids=ids,
                scores=scores,
                metadata=metadatas,
            )

        except Exception as e:
            raise VectorStoreConfigurationError(
                f"Failed to query vector store: {e}",
                provider=self.provider_name,
                details={"top_k": top_k, "vector_dimensions": len(query_vector)}
            )

    def delete(
        self,
        ids: list[str],
        trace: TraceContext | None = None,
        **kwargs: Any
    ) -> bool:
        """Delete records from the store.

        Args:
            ids: List of record IDs to delete
            trace: Tracing context for observability
            **kwargs: Additional arguments

        Returns:
            True if deletion was successful

        Raises:
            VectorStoreConfigurationError: If delete fails
        """
        if not ids:
            return True

        logger.info(
            f"ChromaStore delete: id_count={len(ids)}"
        )

        if trace:
            trace.record_stage(
                "vector_delete",
                {
                    "provider": self.provider_name,
                    "id_count": len(ids),
                }
            )

        try:
            self._collection.delete(ids=ids)
            logger.info(f"ChromaStore delete complete: {len(ids)} records")
            return True

        except Exception as e:
            raise VectorStoreConfigurationError(
                f"Failed to delete records: {e}",
                provider=self.provider_name,
                details={"id_count": len(ids)}
            )

    def count(
        self,
        trace: TraceContext | None = None,
        **kwargs: Any
    ) -> int:
        """Get the total number of records in the store.

        Args:
            trace: Tracing context for observability
            **kwargs: Additional arguments

        Returns:
            Total number of records
        """
        try:
            count = self._collection.count()
            logger.info(f"ChromaStore count: {count}")
            return count
        except Exception as e:
            raise VectorStoreConfigurationError(
                f"Failed to count records: {e}",
                provider=self.provider_name
            )

    def clear(
        self,
        trace: TraceContext | None = None,
        **kwargs: Any
    ) -> bool:
        """Clear all records from the store.

        Args:
            trace: Tracing context for observability
            **kwargs: Additional arguments

        Returns:
            True if clear was successful
        """
        logger.info(f"ChromaStore clear: collection={self._collection_name}")

        if trace:
            trace.record_stage(
                "vector_clear",
                {
                    "provider": self.provider_name,
                    "collection": self._collection_name,
                }
            )

        try:
            # Delete and recreate the collection
            self._client.delete_collection(name=self._collection_name)
            self._collection = self._client.create_collection(
                name=self._collection_name
            )
            logger.info("ChromaStore clear complete")
            return True

        except Exception as e:
            raise VectorStoreConfigurationError(
                f"Failed to clear collection: {e}",
                provider=self.provider_name
            )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"ChromaStore("
            f"provider={self.provider_name}, "
            f"collection={self._collection_name}, "
            f"persist_dir={self._persist_directory})"
        )
