"""Sparse Retriever - BM25 keyword-based retrieval.

This module provides the SparseRetriever class that performs keyword-based
retrieval using BM25 algorithm and a vector store for text retrieval.

Design Principles:
    - BM25 Retrieval: Uses BM25 algorithm for keyword-based ranking
    - Dependency Injection: Supports mock BM25Indexer and vector store for testing
    - Trace Support: Records retrieval stages in trace context
    - Hybrid Data: Combines BM25 scores with vector store text

Example:
    >>> from core.query_engine import SparseRetriever
    >>> from ingestion.storage import BM25Indexer
    >>> from libs.vector_store import ChromaStore
    >>>
    >>> indexer = BM25Indexer(index_dir="data/db/bm25/docs")
    >>> vector_store = ChromaStore(collection_name="docs")
    >>> retriever = SparseRetriever(bm25_indexer=indexer, vector_store=vector_store)
    >>>
    >>> results = retriever.retrieve(query="machine learning", top_k=5)
"""

from typing import Any, List

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import RetrievalResult
from libs.vector_store.base_vector_store import BaseVectorStore
from observability.logger import get_logger

logger = get_logger(__name__)


class SparseRetriever:
    """Sparse retriever using BM25 and vector store.

    This class performs keyword-based retrieval by:
    1. Using BM25Indexer to compute BM25 scores for query terms
    2. Using vector store to fetch text content for matched chunks
    3. Returning structured retrieval results with BM25 scores

    Attributes:
        bm25_indexer: The BM25 indexer for keyword search
        vector_store: The vector store for text retrieval
    """

    # Default index directory for BM25
    DEFAULT_BM25_INDEX_DIR = "data/db/bm25"

    def __init__(
        self,
        settings: Settings | None = None,
        bm25_indexer: Any | None = None,
        vector_store: BaseVectorStore | None = None,
        default_top_k: int = 10,
    ) -> None:
        """Initialize the SparseRetriever.

        Args:
            settings: Settings object for configuration (includes collection, index_dir).
            bm25_indexer: The BM25 indexer for keyword search. If None, will be
                created from settings.
            vector_store: The vector store for fetching text content. If None, will be
                created from settings.
            default_top_k: Default number of results to return
        """
        self._settings = settings
        self._bm25_indexer = bm25_indexer
        self._vector_store = vector_store
        self.default_top_k = default_top_k

        # Extract default_top_k from settings if available
        if settings is not None:
            retrieval_config = getattr(settings, "retrieval", None)
            if retrieval_config is not None:
                self.default_top_k = getattr(
                    retrieval_config, "sparse_top_k", default_top_k
                )

        # Create dependencies from settings if not provided
        if vector_store is None and settings is not None:
            from libs.vector_store.vector_store_factory import VectorStoreFactory

            self._vector_store = VectorStoreFactory.create(settings)
            logger.info(
                f"Created vector store from settings: {self._vector_store.provider_name}"
            )

        logger.info(f"SparseRetriever initialized with default_top_k={self.default_top_k}")

    @property
    def bm25_indexer(self) -> Any:
        """Get the BM25 indexer."""
        return self._bm25_indexer

    @bm25_indexer.setter
    def bm25_indexer(self, indexer: Any) -> None:
        """Set the BM25 indexer."""
        self._bm25_indexer = indexer

    @property
    def vector_store(self) -> BaseVectorStore | None:
        """Get the vector store."""
        return self._vector_store

    @vector_store.setter
    def vector_store(self, store: BaseVectorStore) -> None:
        """Set the vector store."""
        self._vector_store = store

    def retrieve(
        self,
        keywords: list[str],
        collection: str = "default",
        top_k: int | None = None,
        trace: TraceContext | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve relevant chunks using BM25 keyword search.

        Args:
            keywords: List of query keywords (e.g., ["machine", "learning"]).
            collection: Collection name for BM25 index.
            top_k: Number of results to retrieve. If None, uses default_top_k.
            trace: Optional trace context for observability.

        Returns:
            List of RetrievalResult sorted by score (descending)

        Raises:
            ValueError: If keywords is empty
            RuntimeError: If dependencies are not configured
        """
        # Validate inputs
        if not keywords:
            raise ValueError("keywords cannot be empty")
        self._validate_keywords(keywords)
        self._validate_dependencies(collection=collection)

        # Use default top_k if not specified
        effective_top_k = top_k if top_k is not None else self.default_top_k

        logger.info(
            f"SparseRetriever retrieve: keywords={keywords}, collection={collection}, "
            f"top_k={effective_top_k}"
        )

        if trace:
            trace.record_stage(
                "sparse_retrieval_start",
                {"keywords": keywords, "collection": collection, "top_k": effective_top_k},
            )

        # Load BM25 index if not already loaded for this collection
        self._ensure_bm25_indexer(collection=collection)

        # Step 1: BM25 search to get chunk_ids and scores
        bm25_results = self._bm25_indexer.search(
            query_terms=keywords,
            top_k=effective_top_k,
            trace=trace,
        )

        if not bm25_results:
            logger.info("BM25 search returned no results")
            if trace:
                trace.record_stage(
                    "sparse_retrieval_complete",
                    {"result_count": 0},
                )
            return []

        # Extract chunk_ids
        chunk_ids = [chunk_id for chunk_id, _ in bm25_results]
        bm25_scores = {chunk_id: score for chunk_id, score in bm25_results}

        logger.debug(f"BM25 search returned {len(chunk_ids)} results")

        # Step 2: Fetch text from vector store
        try:
            records = self._vector_store.get_by_ids(chunk_ids, trace=trace)
        except Exception as e:
            logger.error(f"Failed to get text from vector store: {e}")
            # Return results with empty text if vector store fetch fails
            records = []

        # Step 3: Combine BM25 scores with text
        results = self._build_results(records, bm25_scores)

        if trace:
            trace.record_stage(
                "sparse_retrieval_complete",
                {
                    "result_count": len(results),
                    "top_score": results[0].score if results else 0,
                },
            )

        logger.info(
            f"SparseRetriever complete: {len(results)} results, "
            f"top_score={results[0].score if results else 0}"
        )

        return results

    def _validate_keywords(self, keywords: List[str]) -> None:
        """Validate the keywords list.

        Args:
            keywords: Keywords list to validate.

        Raises:
            ValueError: If keywords is empty or not a list.
        """
        if not isinstance(keywords, list):
            raise ValueError(
                f"Keywords must be a list, got {type(keywords).__name__}"
            )
        if not keywords:
            raise ValueError("Keywords list cannot be empty")
        # Filter out empty strings but allow the call to proceed
        # (empty strings will simply not match anything)


    def _validate_dependencies(self, collection: str = "default") -> None:
        """Validate that required dependencies are configured.

        Args:
            collection: Collection name for BM25 index

        Raises:
            RuntimeError: If vector_store is None.
        """
        if self._vector_store is None:
            raise RuntimeError(
                "SparseRetriever requires a vector_store. "
                "Provide one during initialization or via settings."
            )

    def _ensure_bm25_indexer(self, collection: str = "default") -> None:
        """Ensure BM25 indexer is loaded for the given collection.

        Args:
            collection: Collection name for BM25 index
        """
        # If bm25_indexer was provided during init (not created from settings), use it as-is
        # Check if it's a real BM25Indexer by checking class name
        if self._bm25_indexer is not None:
            class_name = self._bm25_indexer.__class__.__name__
            if class_name == "BM25Indexer":
                # User provided a real BM25Indexer, check if index is already loaded
                # Load if index is empty (not yet loaded from disk)
                if self._bm25_indexer.num_docs == 0:
                    logger.info(f"BM25 index not loaded, attempting to load for collection '{collection}'")
                    loaded = self._bm25_indexer.load(collection=collection)
                    if not loaded:
                        logger.warning(f"BM25 index not found for collection '{collection}', index is empty")
                    return
                # Index already loaded, check collection match
                if self._bm25_indexer.collection != collection:
                    # Different collection requested, need to reload
                    logger.info(f"Switching BM25 index from {self._bm25_indexer.collection} to {collection}")
                    loaded = self._bm25_indexer.load(collection=collection)
                    if not loaded:
                        logger.warning(f"BM25 index not found for collection '{collection}', index is empty")
                return
            # If it's a mock or other type, don't overwrite - use as-is
            return

        # Create BM25 indexer from settings
        if self._settings is not None:
            # Get index_dir from settings
            index_dir = getattr(self._settings, "bm25_index_dir", self.DEFAULT_BM25_INDEX_DIR)
        else:
            index_dir = self.DEFAULT_BM25_INDEX_DIR

        # Create new BM25 indexer and load the collection
        from ingestion.storage import BM25Indexer

        self._bm25_indexer = BM25Indexer(index_dir=index_dir)
        loaded = self._bm25_indexer.load(collection=collection)

        if not loaded:
            logger.warning(f"BM25 index not found for collection '{collection}', index is empty")

    def _build_results(
        self,
        records: list[dict[str, Any]],
        bm25_scores: dict[str, float],
    ) -> list[RetrievalResult]:
        """Build RetrievalResult list from records and BM25 scores.

        Args:
            records: List of {id, text, metadata} dicts
            bm25_scores: Dict mapping chunk_id to BM25 score

        Returns:
            List of RetrievalResult sorted by score descending
        """
        # Convert list to dict for easier lookup
        records_by_id = {r["id"]: r for r in records}

        results = []

        for chunk_id, bm25_score in bm25_scores.items():
            record = records_by_id.get(chunk_id, {})

            try:
                result = RetrievalResult(
                    chunk_id=chunk_id,
                    score=bm25_score,
                    text=record.get("text", ""),
                    metadata=record.get("metadata", {}),
                )
                results.append(result)
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Failed to build result for {chunk_id}: {e}. "
                    "Skipping this result."
                )
                continue

        # Sort by score descending
        results.sort(key=lambda x: (-x.score, x.chunk_id))

        return results


    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"SparseRetriever("
            f"bm25_indexer={self._bm25_indexer}, "
            f"vector_store={self._vector_store}, "
            f"default_top_k={self.default_top_k})"
        )


# =============================================================================
# Factory Function
# =============================================================================


def create_sparse_retriever(
    settings: Settings | None = None,
    bm25_indexer: Any | None = None,
    vector_store: BaseVectorStore | None = None,
) -> SparseRetriever:
    """Create a SparseRetriever with optional configuration.

    Args:
        settings: Settings object for configuration
        bm25_indexer: Optional BM25 indexer. If None, created from default path.
        vector_store: Optional vector store. If None, created from settings.

    Returns:
        Configured SparseRetriever instance
    """
    # Create vector store if not provided
    if vector_store is None and settings is not None:
        from libs.vector_store.vector_store_factory import VectorStoreFactory

        collection_name = getattr(settings.retrieval, "collection", "default")
        vector_store = VectorStoreFactory.create(settings, collection_name=collection_name)

    return SparseRetriever(
        settings=settings,
        bm25_indexer=bm25_indexer,
        vector_store=vector_store,
    )
