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

from typing import Any, List, Optional

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
        ranking_lists: list[list[RetrievalResult]],
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
                {"num_result_lists": len(ranking_lists), "k": self._k},
            )

        # Filter out empty lists
        non_empty_lists = [lst for lst in ranking_lists if lst]
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


    def fuse_with_weights(
        self,
        ranking_lists: List[List[RetrievalResult]],
        weights: Optional[List[float]] = None,
        top_k: Optional[int] = None,
        trace: Optional[Any] = None,
    ) -> List[RetrievalResult]:
        """Fuse multiple ranked result lists using weighted RRF.

        This method combines results from multiple retrievers using weighted
        Reciprocal Rank Fusion, allowing different weights for each retriever.

        Weighted RRF Formula:
            score(d) = Σ (weight_i / (k + rank_i(d)))

        Where:
            - k is the RRF constant (default 60)
            - weight_i is the weight for the i-th ranking list
            - rank_i(d) is the 1-indexed position of document d in list i

        Args:
            ranking_lists: List of ranked result lists to fuse.
                Each list should be sorted by relevance (best first).
            weights: List of weights for each ranking list. If None, equal
                weights (1.0) are used for all lists. Weights should be
                non-negative. Higher weights give more importance to that
                retriever's rankings.
            top_k: Number of top results to return. If None, returns all.
            trace: Optional trace context for observability.

        Returns:
            List of fused RetrievalResult sorted by weighted RRF score (descending).
            Results include the combined text and metadata from input.

        Example:
            >>> dense = [RetrievalResult("a", 0.9, "text"), ...]
            >>> sparse = [RetrievalResult("b", 0.8, "text"), ...]
            >>> # Give 1.5x weight to dense, 1.0x to sparse
            >>> fused = fusion.fuse_with_weights([dense, sparse], weights=[1.5, 1.0])
        """
        if trace:
            trace.record_stage(
                "weighted_fusion_start",
                {
                    "num_ranking_lists": len(ranking_lists),
                    "weights": weights,
                    "k": self._k,
                },
            )

        # Filter out empty lists and get their indices
        non_empty_data = [
            (idx, lst) for idx, lst in enumerate(ranking_lists) if lst
        ]

        if not non_empty_data:
            logger.info("All ranking lists are empty, returning empty results")
            return []

        # Prepare weights
        num_lists = len(non_empty_data)
        if weights is None:
            # Default: equal weights of 1.0
            effective_weights = [1.0] * num_lists
        else:
            if len(weights) != num_lists:
                raise ValueError(
                    f"Number of weights ({len(weights)}) must match "
                    f"number of non-empty ranking lists ({num_lists})"
                )
            # Validate weights are non-negative
            for w in weights:
                if w < 0:
                    raise ValueError(f"Weights must be non-negative, got {w}")
            effective_weights = list(weights)

        logger.info(
            f"Weighted fusion: {num_lists} lists, weights={effective_weights}, k={self._k}"
        )

        # Calculate weighted RRF scores
        weighted_scores: dict[str, float] = {}
        chunk_data: dict[str, dict[str, Any]] = {}

        for (orig_idx, result_list), weight in zip(non_empty_data, effective_weights):
            for rank, result in enumerate(result_list, start=1):
                # Weighted RRF formula: weight / (k + rank)
                weighted_rrf_score = weight / (self._k + rank)

                # Accumulate scores
                weighted_scores[result.chunk_id] = (
                    weighted_scores.get(result.chunk_id, 0.0) + weighted_rrf_score
                )

                # Store text and metadata from first occurrence
                if result.chunk_id not in chunk_data:
                    chunk_data[result.chunk_id] = {
                        "text": result.text,
                        "metadata": result.metadata,
                    }

        # Sort by weighted score descending, then by chunk_id for determinism
        sorted_chunk_ids = sorted(
            weighted_scores.keys(),
            key=lambda chunk_id: (-weighted_scores[chunk_id], chunk_id)
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
                    score=weighted_scores[chunk_id],
                    text=data["text"],
                    metadata=data["metadata"],
                )
            )

        logger.info(f"Weighted fusion complete: {len(fused_results)} results")

        if trace:
            trace.record_stage(
                "weighted_fusion_complete",
                {"result_count": len(fused_results)},
            )

        return fused_results

    def __repr__(self) -> str:
        """Return string representation."""
        return f"RRFFusion(k={self._k})"
