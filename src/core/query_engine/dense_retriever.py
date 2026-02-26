"""Dense Retriever - Semantic vector retrieval.

This module provides the DenseRetriever class that performs semantic
retrieval using vector embeddings and a vector store.

Design Principles:
    - Semantic Retrieval: Uses dense embeddings for semantic similarity
    - Dependency Injection: Supports mock embedding and vector store for testing
    - Trace Support: Records retrieval stages in trace context

Example:
    >>> from core.query_engine import DenseRetriever
    >>> from libs.embedding import OpenAIEmbedding
    >>> from libs.vector_store import ChromaStore
    >>>
    >>> embedding = OpenAIEmbedding(...)
    >>> vector_store = ChromaStore(collection_name="docs")
    >>> retriever = DenseRetriever(embedding=embedding, vector_store=vector_store)
    >>>
    >>> results = retriever.retrieve("What is machine learning?", top_k=5)
"""

from typing import Any

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import RetrievalResult
from libs.embedding.base_embedding import BaseEmbedding
from libs.vector_store.base_vector_store import BaseVectorStore
from observability.logger import get_logger

logger = get_logger(__name__)


class DenseRetriever:
    """Dense retriever using vector embeddings and vector store.

    This class performs semantic retrieval by:
    1. Converting the query to a dense embedding
    2. Querying the vector store for similar vectors
    3. Returning structured retrieval results

    Attributes:
        embedding_client: The embedding client for query vectorization
        vector_store: The vector store for retrieval
    """

    def __init__(
        self,
        settings: Settings | None = None,
        embedding_client: BaseEmbedding | None = None,
        vector_store: BaseVectorStore | None = None,
        default_top_k: int = 10,
    ) -> None:
        """Initialize the DenseRetriever.

        Args:
            settings: Settings object for configuration
            embedding_client: Optional embedding client. If None, created from settings.
            vector_store: Optional vector store. If None, created from settings.
            default_top_k: Default number of results to return
        """
        self.embedding_client = embedding_client
        self.vector_store = vector_store

        # Extract default_top_k from settings if available
        self.default_top_k = default_top_k
        if settings is not None:
            retrieval_config = getattr(settings, "retrieval", None)
            if retrieval_config is not None:
                self.default_top_k = getattr(
                    retrieval_config, "dense_top_k", default_top_k
                )

        # Create dependencies from settings if not provided
        if embedding_client is None and settings is not None:
            from libs.embedding.embedding_factory import EmbeddingFactory

            self.embedding_client = EmbeddingFactory.create(settings)
            logger.info(
                f"Created embedding from settings: {self.embedding_client.provider_name}"
            )

        if vector_store is None and settings is not None:
            from libs.vector_store.vector_store_factory import VectorStoreFactory

            self.vector_store = VectorStoreFactory.create(settings)
            logger.info(
                f"Created vector store from settings: {self.vector_store.provider_name}"
            )

        logger.info(f"DenseRetriever initialized with default_top_k={self.default_top_k}")

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        trace: TraceContext | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve relevant chunks using dense embeddings.

        Args:
            query: The query string
            top_k: Number of results to retrieve. If None, uses default_top_k.
            filters: Optional metadata filters
            trace: Optional trace context for observability

        Returns:
            List of RetrievalResult sorted by score (descending)

        Raises:
            ValueError: If query is empty
        """
        # Validate inputs
        self._validate_query(query)
        self._validate_dependencies()

        # Use default top_k if not specified
        effective_top_k = top_k if top_k is not None else self.default_top_k

        logger.info(f"DenseRetriever retrieve: query='{query}', top_k={effective_top_k}")

        # Step 1: Generate query embedding
        query_vectors = self.embedding_client.embed([query], trace=trace)
        query_vector = query_vectors[0]
        logger.debug(f"Generated query embedding: {len(query_vector)} dims")

        if trace:
            trace.record_stage(
                "dense_embedding",
                {
                    "query": query,
                    "vector_dims": len(query_vector),
                },
            )

        # Step 2: Query vector store (returns list[dict])
        raw_results = self.vector_store.query(
            query_vector=query_vector,
            top_k=effective_top_k,
            filters=filters,
            trace=trace,
        )

        logger.debug(f"Vector store returned {len(raw_results)} results")

        # Step 3: Transform to RetrievalResult objects
        results = self._transform_results(raw_results)

        if trace:
            trace.record_stage(
                "dense_retrieval",
                {
                    "result_count": len(results),
                    "top_score": results[0].score if results else 0,
                },
            )

        logger.info(
            f"DenseRetriever complete: {len(results)} results, "
            f"top_score={results[0].score if results else 0}"
        )

        return results

    def _validate_query(self, query: str) -> None:
        """Validate the query string.

        Args:
            query: Query string to validate.

        Raises:
            ValueError: If query is empty or not a string.
        """
        if not isinstance(query, str):
            raise ValueError(
                f"Query must be a string, got {type(query).__name__}"
            )
        if not query.strip():
            raise ValueError("Query cannot be empty or whitespace-only")

    def _validate_dependencies(self) -> None:
        """Validate that required dependencies are configured.

        Raises:
            RuntimeError: If embedding_client or vector_store is None.
        """
        if self.embedding_client is None:
            raise RuntimeError(
                "DenseRetriever requires an embedding_client. "
                "Provide one during initialization or via setter."
            )
        if self.vector_store is None:
            raise RuntimeError(
                "DenseRetriever requires a vector_store. "
                "Provide one during initialization or via setter."
            )

    def _transform_results(
        self,
        raw_results: list[dict[str, Any]],
    ) -> list[RetrievalResult]:
        """Transform raw vector store results to RetrievalResult objects.

        Args:
            raw_results: Raw results from vector store query.
                         Each result should have: id, score, text, metadata.

        Returns:
            List of RetrievalResult objects.
        """
        results = []
        for raw in raw_results:
            try:
                result = RetrievalResult(
                    chunk_id=str(raw.get("id", "")),
                    score=float(raw.get("score", 0.0)),
                    text=str(raw.get("text", "")),
                    metadata=raw.get("metadata", {}),
                )
                results.append(result)
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Failed to transform result {raw.get('id', 'unknown')}: {e}. "
                    "Skipping this result."
                )
                continue

        return results
