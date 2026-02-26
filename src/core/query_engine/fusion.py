"""Fusion - RRF (Reciprocal Rank Fusion) for combining retrieval results.

This module provides the RRFFusion class that combines ranked results
from multiple retrievers (dense and sparse) using the RRF algorithm.

Design Principles:
    - Deterministic: Same input produces same output
    - Configurable: k parameter for RRF constant
    - Pluggable: Works with any ranked input

Example:
    >>> from core.query_engine.fusion import RRFFusion
    >>> from core.types import RetrievalResult
    >>>
    >>> dense_results = [
    >>>     RetrievalResult(chunk_id="doc1", score=0.9, text="..."),
    >>>     RetrievalResult(chunk_id="doc2", score=0.8, text="..."),
    >>> ]
    >>> sparse_results = [
    >>>     RetrievalResult(chunk_id="doc2", score=0.95, text="..."),
    >>>     RetrievalResult(chunk_id="doc3", score=0.7, text="..."),
    >>> ]
    >>>
    >>> fusion = RRFFusion(k=60)
    >>> fused = fusion.fuse([dense_results, sparse_results])
"""

from typing import Any

from core.trace.trace_context import TraceContext
from core.types import RetrievalResult
from observability.logger import get_logger

logger = get_logger(__name__)


class RRFFusion:
    """Reciprocal Rank Fusion for combining retrieval results.

    This class implements the RRF algorithm to combine ranked results
    from multiple retrievers into a single ranked list.

    RRF Formula:
        score(d) = Σ (1 / (k + rank(d)))

    Where:
        - k is a constant (default 60) that controls ranking smoothness
        - rank(d) is the 1-indexed position of document d in each result list

    Attributes:
        k: RRF constant parameter (default 60)
    """

    # Default RRF k parameter
    DEFAULT_K = 60

    def __init__(self, k: float = DEFAULT_K) -> None:
        """Initialize the RRFFusion.

        Args:
            k: RRF constant parameter. Higher values give more weight to
               lower-ranked results. Default is 60.
        """
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        self._k = k
        logger.info(f"RRFFusion initialized with k={k}")

    @property
    def k(self) -> float:
        """Get the RRF k parameter."""
        return self._k

    def fuse(
        self,
        result_lists: list[list[RetrievalResult]],
        top_k: int | None = None,
        trace: TraceContext | None = None,
    ) -> list[RetrievalResult]:
        """Fuse multiple ranked result lists using RRF.

        This method combines results from multiple retrievers (e.g., dense
        and sparse) into a single ranked list using the Reciprocal Rank
        Fusion algorithm.

        Args:
            result_lists: List of ranked result lists to fuse.
                Each list should be sorted by relevance (best first).
            top_k: Number of top results to return. If None, returns all.
            trace: Optional trace context for observability.

        Returns:
            List of fused RetrievalResult sorted by RRF score (descending).
            Results include the combined text and metadata from input.

        Example:
            >>> dense = [RetrievalResult("a", 0.9, "text"), ...]
            >>> sparse = [RetrievalResult("b", 0.8, "text"), ...]
            >>> fused = fusion.fuse([dense, sparse], top_k=10)
        """
        if trace:
            trace.record_stage(
                "fusion_start",
                {"num_result_lists": len(result_lists), "k": self._k},
            )

        # Filter out empty lists
        non_empty_lists = [lst for lst in result_lists if lst]
        if not non_empty_lists:
            logger.info("All result lists are empty, returning empty results")
            return []

        logger.info(
            f"Fusing {len(non_empty_lists)} result lists with k={self._k}"
        )

        # Calculate RRF scores
        rrf_scores: dict[str, float] = {}
        chunk_data: dict[str, dict[str, Any]] = {}

        for result_list in non_empty_lists:
            for rank, result in enumerate(result_list, start=1):
                # RRF formula: 1 / (k + rank)
                rrf_score = 1.0 / (self._k + rank)

                # Accumulate scores
                rrf_scores[result.chunk_id] = (
                    rrf_scores.get(result.chunk_id, 0.0) + rrf_score
                )

                # Store text and metadata from first occurrence
                if result.chunk_id not in chunk_data:
                    chunk_data[result.chunk_id] = {
                        "text": result.text,
                        "metadata": result.metadata,
                    }

        # Sort by RRF score descending, then by chunk_id for determinism
        sorted_chunk_ids = sorted(
            rrf_scores.keys(),
            key=lambda chunk_id: (-rrf_scores[chunk_id], chunk_id)
        )

        # Apply top_k limit
        if top_k is not None:
            sorted_chunk_ids = sorted_chunk_ids[:top_k]

        # Build result list
        fused_results = []
        for chunk_id in sorted_chunk_ids:
            data = chunk_data[chunk_id]
            fused_results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    score=rrf_scores[chunk_id],
                    text=data["text"],
                    metadata=data["metadata"],
                )
            )

        logger.info(f"Fusion complete: {len(fused_results)} results")

        if trace:
            trace.record_stage(
                "fusion_complete",
                {"result_count": len(fused_results)},
            )

        return fused_results

    def __repr__(self) -> str:
        """Return string representation."""
        return f"RRFFusion(k={self._k})"
