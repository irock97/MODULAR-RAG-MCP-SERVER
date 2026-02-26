"""Hybrid Search - Orchestrates dense and sparse retrieval with fusion.

This module provides the HybridSearch class that combines QueryProcessor,
DenseRetriever, SparseRetriever, and RRFFusion to perform hybrid search.

Design Principles:
    - Parallel Execution: Dense and sparse retrieval run concurrently
    - Configurable: RRF k parameter, weights, top_k
    - Fallback Strategy: If one retriever fails, use the other's results
    - Trace Support: Records all stages in trace context

Example:
    >>> from core.query_engine import HybridSearch
    >>> from core.query_engine import DenseRetriever, SparseRetriever, RRFFusion
    >>> from core.query_engine import QueryProcessor
    >>>
    >>> processor = QueryProcessor()
    >>> dense_retriever = DenseRetriever(...)
    >>> sparse_retriever = SparseRetriever(...)
    >>> fusion = RRFFusion()
    >>>
    >>> hybrid = HybridSearch(
    ...     query_processor=processor,
    ...     dense_retriever=dense_retriever,
    ...     sparse_retriever=sparse_retriever,
    ...     fusion=fusion,
    ... )
    >>>
    >>> # Basic search
    >>> results = hybrid.search("machine learning", top_k=10)
    >>>
    >>> # Search with detailed results
    >>> detailed = hybrid.search("machine learning", top_k=10, return_details=True)
    >>> print(detailed.dense_results)
    >>> print(detailed.sparse_results)
"""

from __future__ import annotations

import concurrent.futures
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from core.query_engine.dense_retriever import DenseRetriever
from core.query_engine.fusion import RRFFusion
from core.query_engine.query_processor import QueryProcessor
from core.query_engine.sparse_retriever import SparseRetriever
from core.trace.trace_context import TraceContext
from core.types import ProcessedQuery, RetrievalResult
from observability.logger import get_logger

if TYPE_CHECKING:
    from core.settings import Settings

logger = get_logger(__name__)

# Default timeout for parallel retrieval (seconds)
DEFAULT_PARALLEL_TIMEOUT = 30.0


@dataclass
class HybridSearchResult:
    """Result from hybrid search operation with detailed information.

    Attributes:
        results: Final fused/interleaved results
        dense_results: Results from dense retriever (if available)
        sparse_results: Results from sparse retriever (if available)
        dense_error: Error message if dense retrieval failed
        sparse_error: Error message if sparse retrieval failed
        fusion_error: Error message if fusion failed
        used_fallback: Whether fallback strategy was used (interleave instead of fusion)
        processed_query: Processed query with keywords and filters
        execution_time_ms: Time taken for the search in milliseconds
        metadata: Additional debug information
    """

    results: list[RetrievalResult] = field(default_factory=list)
    dense_results: Optional[list[RetrievalResult]] = None
    sparse_results: Optional[list[RetrievalResult]] = None
    dense_error: Optional[str] = None
    sparse_error: Optional[str] = None
    fusion_error: Optional[str] = None
    used_fallback: bool = False
    processed_query: Optional[ProcessedQuery] = None
    execution_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_results(self) -> int:
        """Get total number of final results."""
        return len(self.results)

    @property
    def has_errors(self) -> bool:
        """Check if any retrieval had errors."""
        return any([
            self.dense_error is not None,
            self.sparse_error is not None,
            self.fusion_error is not None,
        ])


@dataclass
class HybridSearchConfig:
    """Configuration for HybridSearch.

    Attributes:
        dense_top_k: Number of results from dense retrieval
        sparse_top_k: Number of results from sparse retrieval
        fusion_top_k: Final number of results after fusion
        enable_dense: Whether to use dense retrieval
        enable_sparse: Whether to use sparse retrieval
        parallel_retrieval: Whether to run retrievals in parallel
        metadata_filter_post: Apply metadata filters after fusion (fallback)
        parallel_timeout: Timeout for parallel retrieval in seconds
    """

    dense_top_k: int = 20
    sparse_top_k: int = 20
    fusion_top_k: int = 10
    enable_dense: bool = True
    enable_sparse: bool = True
    parallel_retrieval: bool = True
    metadata_filter_post: bool = True
    parallel_timeout: float = DEFAULT_PARALLEL_TIMEOUT


class HybridSearch:
    """Hybrid Search orchestrator combining dense and sparse retrieval.

    This class orchestrates the complete retrieval pipeline:
    1. Query Processing: Extract keywords and filters from query
    2. Parallel Retrieval: Run dense and sparse retrievers concurrently
    3. Result Fusion: Combine results using RRF algorithm

    The search method returns either a list of RetrievalResult (default) or
    a HybridSearchResult (when return_details=True) with detailed information.

    Example:
        >>> config = HybridSearchConfig(dense_top_k=20, sparse_top_k=20, fusion_top_k=10)
        >>> hybrid = HybridSearch(config=config, ...)
        >>> results = hybrid.search("query", top_k=10)
        >>>
        >>> # Get detailed results
        >>> detailed = hybrid.search("query", top_k=10, return_details=True)
        >>> print(detailed.dense_results, detailed.sparse_results)
    """

    def __init__(
        self,
        query_processor: Optional[QueryProcessor] = None,
        dense_retriever: Optional[DenseRetriever] = None,
        sparse_retriever: Optional[SparseRetriever] = None,
        fusion: Optional[RRFFusion] = None,
        config: Optional[HybridSearchConfig] = None,
    ) -> None:
        """Initialize the HybridSearch.

        Args:
            query_processor: Query processor for keyword extraction.
                If None, a default QueryProcessor will be created.
            dense_retriever: Dense retriever for semantic search.
                If None, will be required at search time.
            sparse_retriever: Sparse retriever for BM25 search.
                If None, will be required at search time.
            fusion: RRF fusion instance.
                If None, will use default RRFFusion.
            config: HybridSearch configuration.
                If None, uses default config.
        """
        self._query_processor = query_processor or QueryProcessor()
        self._dense_retriever = dense_retriever
        self._sparse_retriever = sparse_retriever
        self._config = config or HybridSearchConfig()

        # Create fusion (RRFFusion with default k=60)
        self._fusion = fusion

        logger.info(
            f"HybridSearch initialized: parallel={self._config.parallel_retrieval}, "
            f"enable_dense={self._config.enable_dense}, "
            f"enable_sparse={self._config.enable_sparse}"
        )

    @property
    def query_processor(self) -> QueryProcessor:
        """Get the query processor."""
        return self._query_processor

    @property
    def dense_retriever(self) -> Optional[DenseRetriever]:
        """Get the dense retriever."""
        return self._dense_retriever

    @property
    def sparse_retriever(self) -> Optional[SparseRetriever]:
        """Get the sparse retriever."""
        return self._sparse_retriever

    @property
    def fusion(self) -> Optional[RRFFusion]:
        """Get the fusion instance."""
        return self._fusion

    @property
    def config(self) -> HybridSearchConfig:
        """Get the configuration."""
        return self._config

    @staticmethod
    def _extract_config(settings: Optional[Settings]) -> HybridSearchConfig:
        """Extract configuration from settings object.

        Args:
            settings: Settings object with retrieval configuration

        Returns:
            HybridSearchConfig with values from settings or defaults
        """
        config = HybridSearchConfig()

        if settings is None:
            return config

        # Try to get retrieval config from settings
        retrieval_config = getattr(settings, "retrieval", None)

        if retrieval_config is not None:
            config.dense_top_k = getattr(retrieval_config, "dense_top_k", config.dense_top_k)
            config.sparse_top_k = getattr(retrieval_config, "sparse_top_k", config.sparse_top_k)
            config.fusion_top_k = getattr(retrieval_config, "fusion_top_k", config.fusion_top_k)
            config.enable_dense = getattr(retrieval_config, "enable_dense", config.enable_dense)
            config.enable_sparse = getattr(retrieval_config, "enable_sparse", config.enable_sparse)
            config.parallel_retrieval = getattr(retrieval_config, "parallel_retrieval", config.parallel_retrieval)
            config.metadata_filter_post = getattr(retrieval_config, "metadata_filter_post", config.metadata_filter_post)

        return config

    def _merge_filters(
        self,
        query_filters: dict[str, Any],
        explicit_filters: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """Merge filters from query and explicit filters.

        Args:
            query_filters: Filters extracted from query string
            explicit_filters: Filters passed directly to search

        Returns:
            Merged filters dictionary
        """
        merged = dict(query_filters)
        if explicit_filters:
            merged.update(explicit_filters)
        return merged

    def _apply_metadata_filters(
        self,
        results: list[RetrievalResult],
        filters: dict[str, Any],
    ) -> list[RetrievalResult]:
        """Apply metadata filters with support for complex filter types.

        Filter types:
        - collection: Check multiple metadata keys (collection, source, doc_type)
        - tags: Support list intersection matching
        - source_path: Support partial matching
        - other keys: Exact matching

        Args:
            results: Results to filter
            filters: Filters to apply

        Returns:
            Filtered results
        """
        if not filters:
            return results

        filtered = []
        for result in results:
            if result.metadata is None:
                continue

            if self._matches_filters(result.metadata, filters):
                filtered.append(result)

        return filtered

    def _matches_filters(
        self,
        metadata: dict[str, Any],
        filters: dict[str, Any],
    ) -> bool:
        """Check if metadata matches the filters.

        Args:
            metadata: Result metadata
            filters: Filters to match against

        Returns:
            True if all filters match, False otherwise
        """
        for key, value in filters.items():
            if key == "collection":
                # Check multiple possible keys for collection
                if not self._match_collection(metadata, value):
                    return False
            elif key == "tags":
                # Support list intersection matching
                if not self._match_tags(metadata, value):
                    return False
            elif key == "source_path":
                # Support partial matching
                if not self._match_source_path(metadata, value):
                    return False
            else:
                # Exact matching for other keys
                if metadata.get(key) != value:
                    return False

        return True

    def _match_collection(
        self,
        metadata: dict[str, Any],
        value: Any,
    ) -> bool:
        """Match collection filter from various metadata keys.

        Args:
            metadata: Result metadata
            value: Expected collection value

        Returns:
            True if matches
        """
        # Check multiple possible keys
        for key in ("collection", "source", "doc_type", "category"):
            if metadata.get(key) == value:
                return True
        return False

    def _match_tags(
        self,
        metadata: dict[str, Any],
        value: Any,
    ) -> bool:
        """Match tags with list intersection.

        Args:
            metadata: Result metadata
            value: Expected tags (single or list)

        Returns:
            True if any tag matches
        """
        metadata_tags = metadata.get("tags", [])
        if not isinstance(metadata_tags, list):
            metadata_tags = [metadata_tags]

        if isinstance(value, list):
            # Intersection: any value in metadata_tags matches any in value
            return bool(set(metadata_tags) & set(value))
        else:
            # Single value: check if in metadata_tags
            return value in metadata_tags

    def _match_source_path(
        self,
        metadata: dict[str, Any],
        value: str,
    ) -> bool:
        """Match source_path with partial matching.

        Args:
            metadata: Result metadata
            value: Expected path substring

        Returns:
            True if value is substring of source_path
        """
        source_path = metadata.get("source_path", "")
        if not source_path:
            source_path = metadata.get("source", "")

        return value.lower() in str(source_path).lower()

    def _interleave_results(
        self,
        list1: list[RetrievalResult],
        list2: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Interleave results from two lists (round-robin) with deduplication.

        This is a fallback when fusion is not available.

        Args:
            list1: First list of results (e.g., dense)
            list2: Second list of results (e.g., sparse)
            top_k: Maximum number of results to return

        Returns:
            Interleaved results with deduplication
        """
        if not list1:
            return list2[:top_k]
        if not list2:
            return list1[:top_k]

        seen: set[str] = set()
        result: list[RetrievalResult] = []
        idx1, idx2 = 0, 0

        while len(result) < top_k and (idx1 < len(list1) or idx2 < len(list2)):
            # Take from list1 if available
            if idx1 < len(list1):
                item = list1[idx1]
                if item.chunk_id not in seen:
                    result.append(item)
                    seen.add(item.chunk_id)
                idx1 += 1

            # Take from list2 if available
            if len(result) >= top_k:
                break
            if idx2 < len(list2):
                item = list2[idx2]
                if item.chunk_id not in seen:
                    result.append(item)
                    seen.add(item.chunk_id)
                idx2 += 1

        return result

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None,
        collection: str = "default",
        trace: Optional[TraceContext] = None,
        return_details: bool = False,
    ) -> Any:
        """Execute hybrid search.

        This method:
        1. Processes the query to extract keywords and filters
        2. Runs dense and sparse retrieval (parallel if enabled)
        3. Fuses results using RRF (or interleaves as fallback)
        4. Optionally applies post-fusion metadata filters

        Args:
            query: Raw query string
            top_k: Number of final results to return (overrides config.fusion_top_k)
            filters: Additional filters to apply (merged with query filters)
            collection: Collection name for retrieval
            trace: Optional trace context
            return_details: If True, return HybridSearchResult with details;
                           if False, return list[RetrievalResult]

        Returns:
            If return_details=False: List of RetrievalResult sorted by fused score
            If return_details=True: HybridSearchResult with detailed information

        Raises:
            ValueError: If query is empty
        """
        start_time = time.time()

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        # Use fusion_top_k from config if not specified
        effective_top_k = top_k if top_k is not None else self._config.fusion_top_k

        if trace:
            trace.record_stage(
                "hybrid_search_start",
                {
                    "query": query,
                    "top_k": effective_top_k,
                    "collection": collection,
                    "return_details": return_details,
                },
            )

        # Step 1: Process query
        processed = self._query_processor.process(query)

        # Merge filters
        merged_filters = self._merge_filters(processed.filters, filters)

        logger.info(
            f"Hybrid search: query='{query}', keywords={processed.keywords}, "
            f"filters={merged_filters}, top_k={effective_top_k}"
        )

        if trace:
            trace.record_stage(
                "query_processed",
                {
                    "keywords": processed.keywords,
                    "filters": merged_filters,
                },
            )

        # Step 2: Retrieve from enabled sources
        dense_results, sparse_results, dense_error, sparse_error = self._run_retrievals(
            keywords=processed.keywords,
            filters=merged_filters,
            collection=collection,
            trace=trace,
        )

        # Step 3: Fuse results (or use fallback)
        fused, fusion_error, used_fallback = self._run_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            top_k=effective_top_k,
            trace=trace,
        )

        # Step 4: Post-fusion metadata filtering
        if self._config.metadata_filter_post and merged_filters:
            fused = self._apply_metadata_filters(fused, merged_filters)

        execution_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Hybrid search complete: {len(fused)} results, "
            f"dense={len(dense_results)}, sparse={len(sparse_results)}, "
            f"fallback={used_fallback}, time={execution_time_ms:.2f}ms"
        )

        if trace:
            trace.record_stage(
                "hybrid_search_complete",
                {
                    "result_count": len(fused),
                    "execution_time_ms": execution_time_ms,
                    "used_fallback": used_fallback,
                },
            )

        # Return based on return_details flag
        if return_details:
            return HybridSearchResult(
                results=fused,
                dense_results=dense_results if dense_results else None,
                sparse_results=sparse_results if sparse_results else None,
                dense_error=dense_error,
                sparse_error=sparse_error,
                fusion_error=fusion_error,
                used_fallback=used_fallback,
                processed_query=processed,
                execution_time_ms=execution_time_ms,
                metadata={
                    "top_k": effective_top_k,
                    "collection": collection,
                    "config": {
                        "enable_dense": self._config.enable_dense,
                        "enable_sparse": self._config.enable_sparse,
                        "parallel_retrieval": self._config.parallel_retrieval,
                    },
                },
            )

        return fused

    def _run_retrievals(
        self,
        keywords: list[str],
        filters: dict[str, Any],
        collection: str,
        trace: Optional[TraceContext] = None,
    ) -> tuple[
        list[RetrievalResult],
        list[RetrievalResult],
        Optional[str],
        Optional[str],
    ]:
        """Run dense and sparse retrievals.

        Args:
            keywords: Keywords for retrieval
            filters: Filters to apply
            collection: Collection name
            trace: Trace context

        Returns:
            Tuple of (dense_results, sparse_results, dense_error, sparse_error)
        """
        dense_top_k = self._config.dense_top_k
        sparse_top_k = self._config.sparse_top_k

        if self._config.parallel_retrieval:
            return self._run_parallel_retrievals(
                keywords=keywords,
                filters=filters,
                collection=collection,
                dense_top_k=dense_top_k,
                sparse_top_k=sparse_top_k,
                trace=trace,
            )
        else:
            return self._run_sequential_retrievals(
                keywords=keywords,
                filters=filters,
                collection=collection,
                dense_top_k=dense_top_k,
                sparse_top_k=sparse_top_k,
                trace=trace,
            )

    def _run_parallel_retrievals(
        self,
        keywords: list[str],
        filters: dict[str, Any],
        collection: str,
        dense_top_k: int,
        sparse_top_k: int,
        trace: Optional[TraceContext] = None,
    ) -> tuple[
        list[RetrievalResult],
        list[RetrievalResult],
        Optional[str],
        Optional[str],
    ]:
        """Run retrievals in parallel using ThreadPoolExecutor.

        Args:
            keywords: Keywords for retrieval
            filters: Filters to apply
            collection: Collection name
            dense_top_k: Top-k for dense
            sparse_top_k: Top-k for sparse
            trace: Trace context

        Returns:
            Tuple of (dense_results, sparse_results, dense_error, sparse_error)
        """
        timeout = self._config.parallel_timeout

        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit dense retrieval
            dense_future = executor.submit(
                self._run_dense_retrieval,
                keywords=keywords,
                filters=filters,
                top_k=dense_top_k,
                trace=trace,
            )

            # Submit sparse retrieval
            sparse_future = executor.submit(
                self._run_sparse_retrieval,
                keywords=keywords,
                collection=collection,
                top_k=sparse_top_k,
                trace=trace,
            )

            # Wait for results with timeout
            try:
                dense_results, dense_error = dense_future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                logger.error(f"Dense retrieval timed out after {timeout}s")
                dense_results = []
                dense_error = f"Timeout after {timeout}s"
            except Exception as e:
                logger.error(f"Dense retrieval failed: {e}")
                dense_results = []
                dense_error = str(e)

            try:
                sparse_results, sparse_error = sparse_future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                logger.error(f"Sparse retrieval timed out after {timeout}s")
                sparse_results = []
                sparse_error = f"Timeout after {timeout}s"
            except Exception as e:
                logger.error(f"Sparse retrieval failed: {e}")
                sparse_results = []
                sparse_error = str(e)

        return dense_results, sparse_results, dense_error, sparse_error

    def _run_sequential_retrievals(
        self,
        keywords: list[str],
        filters: dict[str, Any],
        collection: str,
        dense_top_k: int,
        sparse_top_k: int,
        trace: Optional[TraceContext] = None,
    ) -> tuple[
        list[RetrievalResult],
        list[RetrievalResult],
        Optional[str],
        Optional[str],
    ]:
        """Run retrievals sequentially.

        Args:
            keywords: Keywords for retrieval
            filters: Filters to apply
            collection: Collection name
            dense_top_k: Top-k for dense
            sparse_top_k: Top-k for sparse
            trace: Trace context

        Returns:
            Tuple of (dense_results, sparse_results, dense_error, sparse_error)
        """
        dense_results = []
        dense_error = None
        if self._config.enable_dense:
            dense_results, dense_error = self._run_dense_retrieval(
                keywords=keywords,
                filters=filters,
                top_k=dense_top_k,
                trace=trace,
            )

        sparse_results = []
        sparse_error = None
        if self._config.enable_sparse:
            sparse_results, sparse_error = self._run_sparse_retrieval(
                keywords=keywords,
                collection=collection,
                top_k=sparse_top_k,
                trace=trace,
            )

        return dense_results, sparse_results, dense_error, sparse_error

    def _run_dense_retrieval(
        self,
        keywords: list[str],
        filters: dict[str, Any],
        top_k: int,
        trace: Optional[TraceContext] = None,
    ) -> tuple[list[RetrievalResult], Optional[str]]:
        """Run dense retrieval with error handling.

        Args:
            keywords: Keywords for retrieval
            filters: Filters to apply
            top_k: Top-k
            trace: Trace context

        Returns:
            Tuple of (results, error_message)
        """
        if not self._config.enable_dense or self._dense_retriever is None:
            return [], None

        try:
            query_text = " ".join(keywords) if keywords else ""
            results = self._dense_retriever.retrieve(
                query=query_text,
                filters=filters,
                top_k=top_k,
                trace=trace,
            )
            logger.debug(f"Dense retrieval: {len(results)} results")
            return results, None
        except Exception as e:
            logger.error(f"Dense retrieval failed: {e}")
            return [], str(e)

    def _run_sparse_retrieval(
        self,
        keywords: list[str],
        collection: str,
        top_k: int,
        trace: Optional[TraceContext] = None,
    ) -> tuple[list[RetrievalResult], Optional[str]]:
        """Run sparse retrieval with error handling.

        Args:
            keywords: Keywords for retrieval
            collection: Collection name
            top_k: Top-k
            trace: Trace context

        Returns:
            Tuple of (results, error_message)
        """
        if self._sparse_retriever is None:
            return [], None

        try:
            results = self._sparse_retriever.retrieve(
                keywords=keywords,
                collection=collection,
                top_k=top_k,
                trace=trace,
            )
            logger.debug(f"Sparse retrieval: {len(results)} results")
            return results, None
        except Exception as e:
            logger.error(f"Sparse retrieval failed: {e}")
            return [], str(e)

    def _run_fusion(
        self,
        dense_results: list[RetrievalResult],
        sparse_results: list[RetrievalResult],
        top_k: int,
        trace: Optional[TraceContext] = None,
    ) -> tuple[list[RetrievalResult], Optional[str], bool]:
        """Run fusion or fallback.

        Args:
            dense_results: Results from dense retriever
            sparse_results: Results from sparse retriever
            top_k: Number of results to return
            trace: Trace context

        Returns:
            Tuple of (fused_results, error_message, used_fallback)
        """
        # Check if fusion is available
        if self._fusion is not None:
            try:
                fused = self._fusion.fuse(
                    [dense_results, sparse_results],
                    top_k=top_k,
                    trace=trace,
                )
                return fused, None, False
            except Exception as e:
                logger.error(f"Fusion failed: {e}, using fallback")

        # Fallback to interleaving
        logger.info("Using interleave fallback for fusion")
        fused = self._interleave_results(
            dense_results,
            sparse_results,
            top_k,
        )
        return fused, None, True

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"HybridSearch("
            f"dense={self._dense_retriever is not None}, "
            f"sparse={self._sparse_retriever is not None}, "
            f"enable_dense={self._config.enable_dense}, "
            f"enable_sparse={self._config.enable_sparse}, "
            f"parallel={self._config.parallel_retrieval}, "
            f"fusion={self._fusion is not None})"
        )


def create_hybrid_search(
    settings: Optional[Settings] = None,
    query_processor: Optional[QueryProcessor] = None,
    dense_retriever: Optional[DenseRetriever] = None,
    sparse_retriever: Optional[SparseRetriever] = None,
    fusion: Optional[RRFFusion] = None,
    config: Optional[HybridSearchConfig] = None,
) -> HybridSearch:
    """Factory function to create a HybridSearch instance.

    Args:
        settings: Settings object for configuration (used if config is None)
        query_processor: Query processor for keyword extraction
        dense_retriever: Dense retriever for semantic search
        sparse_retriever: Sparse retriever for BM25 search
        fusion: RRF fusion instance (if None, creates RRFFusion with k=60)
        config: HybridSearch configuration (if None, extracts from settings)

    Returns:
        Configured HybridSearch instance

    Example:
        >>> # Create with settings
        >>> from core.settings import Settings
        >>> settings = Settings()
        >>> hybrid = create_hybrid_search(settings)
        >>>
        >>> # Create with custom fusion
        >>> fusion = RRFFusion(k=50)
        >>> hybrid = create_hybrid_search(fusion=fusion)
    """
    # Extract config from settings if not provided
    if config is None and settings is not None:
        config = HybridSearch._extract_config(settings)

    # Create default fusion if not provided
    if fusion is None:
        fusion = RRFFusion()

    return HybridSearch(
        query_processor=query_processor,
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        fusion=fusion,
        config=config,
    )
