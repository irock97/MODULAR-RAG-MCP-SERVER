"""CoreReranker - Core layer reranker with fallback.

This module provides the CoreReranker class that wraps libs.reranker backends
and provides fallback to fusion ranking when reranking fails.

Design Principles:
    - Pluggable: Uses BaseReranker from libs.reranker
    - Fallback: Graceful degradation to fusion ranking on failure
    - Observable: Trace context for monitoring reranking stages
    - Config-Driven: Configuration via settings or explicit config

Features:
    - Integrates with Cross-Encoder, LLM, or None rerankers
    - Automatic fallback to fusion ranking on error
    - Detailed metadata tracking (original_score, rerank_score)
    - Enhanced result format with reranker type detection
"""

from dataclasses import dataclass, field
from typing import Any

from core.trace.trace_context import TraceContext
from core.types import RetrievalResult
from libs.reranker.base_reranker import BaseReranker, Candidate
from observability.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class CoreRerankerConfig:
    """Configuration for CoreReranker.

    Attributes:
        enabled: Whether to enable reranking (default: True)
        top_k: Number of results to return after reranking (default: 5)
        fallback_to_fusion: Whether to fall back to fusion on error (default: True)
    """

    enabled: bool = True
    top_k: int = 5
    fallback_to_fusion: bool = True


# =============================================================================
# Result Types
# =============================================================================


@dataclass
class CoreRerankerResult:
    """Result from CoreReranker operation.

    This wraps retrieval results with additional metadata about the reranking process.

    Attributes:
        results: List of RetrievalResult in ranked order
        used_fallback: Whether fallback to fusion was triggered
        fallback_reason: Reason for fallback (if triggered)
        reranker_type: Type of reranker used (llm, cross_encoder, none)
        original_order: Original order of results before reranking (for debugging)
    """

    results: list[RetrievalResult]
    used_fallback: bool = False
    fallback_reason: str | None = None
    reranker_type: str = "none"
    original_order: list[RetrievalResult] = field(default_factory=list)


# =============================================================================
# CoreReranker
# =============================================================================


class CoreReranker:
    """Core layer reranker with fallback to fusion ranking.

    This class orchestrates reranking by:
    1. Using a configured reranker (Cross-Encoder, LLM, or None)
    2. Converting RetrievalResult to dict format for reranking
    3. Falling back to fusion ranking if reranking fails
    4. Returning results with enhanced metadata

    Example:
        >>> from core.query_engine import CoreReranker
        >>> from libs.reranker import CrossEncoderReranker
        >>>
        >>> # Create reranker
        >>> reranker = CoreReranker(
        ...     reranker_backend=CrossEncoderReranker(),
        ...     config=CoreRerankerConfig(enabled=True, top_k=5)
        ... )
        >>>
        >>> # Rerank results
        >>> result = reranker.rerank(
        ...     query="What is machine learning?",
        ...     candidates=retrieval_results,
        ... )
        >>> if result.used_fallback:
        ...     print("Used fusion fallback")
    """

    def __init__(
        self,
        settings: Any | None = None,
        reranker_backend: BaseReranker | None = None,
        config: CoreRerankerConfig | None = None,
    ) -> None:
        """Initialize the CoreReranker.

        Args:
            settings: Settings object for configuration (optional).
            reranker_backend: The reranking backend from libs.reranker.
                If None, reranking is effectively disabled.
            config: Configuration for reranking behavior.
                If None, extracted from settings or defaults.
        """
        self._reranker = reranker_backend

        # Extract config from settings if not provided
        if config is None:
            self._config = self._extract_config(settings)
        else:
            self._config = config

        # Detect and store reranker type
        self._reranker_type = self._get_reranker_type()

        logger.info(
            f"CoreReranker initialized: enabled={self._config.enabled}, "
            f"top_k={self._config.top_k}, "
            f"reranker_type={self._reranker_type}, "
            f"provider={reranker_backend.provider_name if reranker_backend else 'none'}"
        )

    @property
    def config(self) -> CoreRerankerConfig:
        """Get the reranker configuration."""
        return self._config

    @property
    def reranker_backend(self) -> BaseReranker | None:
        """Get the reranker backend."""
        return self._reranker

    @property
    def reranker_type(self) -> str:
        """Get the reranker type (llm, cross_encoder, none)."""
        return self._reranker_type

    def _get_reranker_type(self) -> str:
        """Detect and return the reranker type based on class name or provider name.

        Returns:
            Reranker type: 'llm', 'cross_encoder', or 'none'
        """
        if self._reranker is None:
            return "none"

        # First try class name
        class_name = self._reranker.__class__.__name__.lower()

        if "llm" in class_name:
            return "llm"
        elif "cross" in class_name or "encoder" in class_name:
            return "cross_encoder"

        # Fallback to provider_name if class name doesn't match
        provider_name = getattr(self._reranker, 'provider_name', None)
        if provider_name:
            provider_lower = provider_name.lower()
            if "llm" in provider_lower:
                return "llm"
            elif "cross" in provider_lower or "encoder" in provider_lower:
                return "cross_encoder"

        return "none"

    def _extract_config(self, settings: Any | None) -> CoreRerankerConfig:
        """Extract configuration from settings object.

        Args:
            settings: Settings object with rerank configuration

        Returns:
            CoreRerankerConfig with values from settings or defaults
        """

        try:
            rerank_settings = settings.rerank
            return CoreRerankerConfig(
                enabled=bool(rerank_settings.enabled) if rerank_settings else False,
                top_k=int(rerank_settings.top_k) if rerank_settings and hasattr(rerank_settings, 'top_k') else 5,
                fallback_to_fusion=True,
            )
        except AttributeError:
            logger.warning("Missing rerank configuration, using defaults (disabled)")
            return CoreRerankerConfig(enabled=False)

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int | None = None,
        trace: TraceContext | None = None,
        **kwargs: Any,
    ) -> CoreRerankerResult:
        """Rerank candidates using the configured reranker with fallback.

        This method:
        1. Checks if reranking is enabled
        2. Converts RetrievalResult to dict format
        3. Attempts reranking
        4. Falls back to fusion ranking on any error
        5. Returns results with enhanced metadata

        Args:
            query: The original query string
            candidates: List of RetrievalResult from hybrid search
            top_k: Override for number of results to return
            trace: Optional trace context for observability
            **kwargs: Additional parameters to pass to the reranker backend

        Returns:
            CoreRerankerResult containing ranked results and metadata
        """
        # Use config top_k if not specified
        effective_top_k = top_k if top_k is not None else self._config.top_k

        # Store original order for debugging
        original_order = candidates.copy()

        # Handle disabled reranking
        if not self._config.enabled or self._reranker is None:
            logger.info("Reranking disabled, returning fusion results")
            return CoreRerankerResult(
                results=self._apply_top_k(candidates, effective_top_k),
                used_fallback=False,
                reranker_type="none",
                original_order=original_order,
            )

        # Handle empty candidates
        if not candidates:
            logger.info("No candidates to rerank")
            return CoreRerankerResult(
                results=[],
                used_fallback=False,
                reranker_type=self._reranker_type,
                original_order=original_order,
            )

        # Handle single result - no need to rerank
        if len(candidates) == 1:
            logger.info("Single candidate, skipping rerank")
            return CoreRerankerResult(
                results=candidates[:],
                used_fallback=False,
                reranker_type=self._reranker_type,
                original_order=original_order,
            )

        if trace:
            trace.record_stage(
                "rerank_start",
                {
                    "query": query,
                    "candidate_count": len(candidates),
                    "top_k": effective_top_k,
                    "reranker_type": self._reranker_type,
                },
            )

        # Convert RetrievalResult to dict format
        rerank_input = self._results_to_candidates(candidates)

        try:
            # Call reranker backend with kwargs passthrough
            rerank_result = self._reranker.rerank(
                query=query,
                candidates=rerank_input,
                top_k=effective_top_k,
                trace=trace,
                **kwargs,
            )

            # Convert back to RetrievalResult with enhanced metadata
            results = self._candidates_to_results(rerank_result, candidates)

            # Apply top_k limit
            results = self._apply_top_k(results, effective_top_k)

            logger.info(
                f"Reranking successful: {len(results)} results, "
                f"reranker_type={self._reranker_type}"
            )

            if trace:
                trace.record_stage(
                    "rerank_complete",
                    {"result_count": len(results), "used_fallback": False},
                )

            return CoreRerankerResult(
                results=results,
                used_fallback=False,
                reranker_type=self._reranker_type,
                original_order=original_order,
            )

        except Exception as e:
            # Fallback to fusion ranking on error
            logger.warning(f"Reranking failed: {e}, falling back to fusion ranking")

            if trace:
                trace.record_stage(
                    "rerank_fallback",
                    {"error": str(e), "candidate_count": len(candidates)},
                )

            # Return original candidates as fallback
            fallback_results = self._apply_top_k(candidates, effective_top_k)

            return CoreRerankerResult(
                results=fallback_results,
                used_fallback=True,
                fallback_reason=str(e),
                reranker_type=self._reranker_type,
                original_order=original_order,
            )

    def _results_to_candidates(
        self,
        results: list[RetrievalResult],
    ) -> list[Candidate]:
        """Convert RetrievalResult list to Candidate objects for reranking.

        This method converts RetrievalResult objects to Candidate objects
        that can be used by the reranker backend.

        Args:
            results: List of RetrievalResult from retrieval

        Returns:
            List of Candidate objects with id, content, score, metadata
        """
        return [
            Candidate(
                id=result.chunk_id,
                content=result.text,
                score=result.score,
                metadata=result.metadata.copy() if result.metadata else None,
            )
            for result in results
        ]

    def _candidates_to_results(
        self,
        rerank_result: Any,
        original_results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Convert rerank result back to RetrievalResult format with enhanced metadata.

        Args:
            rerank_result: Result from reranker backend
            original_results: Original retrieval results

        Returns:
            List of RetrievalResult in reranked order with enhanced metadata
        """
        # Create lookup from original results
        original_by_id = {r.chunk_id: r for r in original_results}

        # Get reranked IDs and scores
        reranked_ids = rerank_result.ids if hasattr(rerank_result, 'ids') else []
        reranked_scores = rerank_result.scores if hasattr(rerank_result, 'scores') else []

        results = []
        for idx, chunk_id in enumerate(reranked_ids):
            original = original_by_id.get(chunk_id)

            # Get scores
            rerank_score = reranked_scores[idx] if idx < len(reranked_scores) else 0.0
            original_score = original.score if original else 0.0

            # Build enhanced metadata
            enhanced_metadata = {}
            if original and original.metadata:
                enhanced_metadata = original.metadata.copy()

            # Add reranking-specific metadata
            enhanced_metadata["_rerank"] = {
                "original_score": original_score,
                "rerank_score": rerank_score,
                "reranked": True,
                "reranker_type": self._reranker_type,
            }

            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    score=rerank_score,
                    text=original.text if original else "",
                    metadata=enhanced_metadata,
                )
            )

        return results

    def _apply_top_k(
        self,
        results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Apply top_k limit to results.

        Args:
            results: List of results
            top_k: Maximum number to return

        Returns:
            Top-k results
        """
        return results[:top_k]


# =============================================================================
# Factory Function
# =============================================================================


def create_core_reranker(
    settings: Any | None = None,
    reranker_backend: BaseReranker | None = None,
) -> CoreReranker:
    """Create a CoreReranker with optional configuration.

    Args:
        settings: Settings object for configuration
        reranker_backend: Pre-configured reranker backend

    Returns:
        Configured CoreReranker instance

    Example:
        >>> from core.query_engine import create_core_reranker
        >>> from libs.reranker import CrossEncoderReranker
        >>>
        >>> reranker = create_core_reranker(
        ...     reranker_backend=CrossEncoderReranker(),
        ... )
    """
    # If settings provided but no backend, try to create from settings
    if reranker_backend is None and settings is not None:
        try:
            from libs.reranker.reranker_factory import RerankerFactory

            reranker_backend = RerankerFactory.create(settings)
            logger.info(f"Created reranker from settings: {reranker_backend.provider_name}")
        except Exception as e:
            logger.warning(f"Could not create reranker from settings: {e}")

    return CoreReranker(
        settings=settings,
        reranker_backend=reranker_backend,
        config=None,
    )


# =============================================================================
# Backward Compatibility
# =============================================================================

# Keep old names for backward compatibility
Reranker = CoreReranker
RerankerConfig = CoreRerankerConfig
RerankerResult = CoreRerankerResult
create_reranker = create_core_reranker
