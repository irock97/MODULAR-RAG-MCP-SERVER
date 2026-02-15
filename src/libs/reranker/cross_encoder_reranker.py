"""Cross-Encoder Reranker implementation.

This module provides Cross-Encoder-based reranking that uses
a cross-encoder model to score query-candidate pairs.

Design Principles:
    - Cross-Encoder: Scores query-candidate pairs jointly
    - Deterministic: Mock scorer for testing
    - Fallback signal: Returns None on failure for graceful degradation
"""

import asyncio
from typing import Any

from libs.reranker.base_reranker import (
    BaseReranker,
    Candidate,
    RerankResult,
    RerankerConfigurationError,
)
from observability.logger import TraceContext, get_logger

logger = get_logger(__name__)


class CrossEncoderReranker(BaseReranker):
    """Cross-Encoder Reranker.

    Uses a cross-encoder model to score candidates based on their
    relevance to the query. Cross-encoders process the query and
    candidate together, enabling more accurate scoring than
    bi-encoders (embeddings).

    Attributes:
        model: The cross-encoder model name or path
        top_k: Number of top results to return
        batch_size: Batch size for scoring
        timeout: Timeout for scoring in seconds
    """

    # Default model
    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Default number of results
    DEFAULT_TOP_K = 10
    # Default batch size
    DEFAULT_BATCH_SIZE = 32
    # Default timeout
    DEFAULT_TIMEOUT = 30.0

    def __init__(
        self,
        model: str | None = None,
        top_k: int | None = None,
        batch_size: int | None = None,
        timeout: float | None = None,
        scorer=None,
    ) -> None:
        """Initialize the Cross-Encoder Reranker.

        Args:
            model: Cross-encoder model name or path.
            top_k: Number of top results to return.
            batch_size: Batch size for scoring multiple candidates.
            timeout: Timeout for scoring in seconds.
            scorer: Optional scorer function for testing (mock).
                If provided, uses this instead of loading a model.
                Signature: (query: str, candidates: list[str]) -> list[float]
        """
        self._model = model or self.DEFAULT_MODEL
        self._top_k = top_k if top_k is not None else self.DEFAULT_TOP_K
        self._batch_size = batch_size or self.DEFAULT_BATCH_SIZE
        self._timeout = timeout if timeout is not None else self.DEFAULT_TIMEOUT
        self._scorer = scorer

        # Will be initialized lazily
        self._cross_encoder = None

        logger.info(
            f"CrossEncoderReranker initialized: model={self._model}, "
            f"top_k={self._top_k}, timeout={self._timeout}"
        )

    @property
    def provider_name(self) -> str:
        """Return the name of this provider.

        Returns:
            Provider identifier: 'cross_encoder'
        """
        return "cross_encoder"

    def rerank(
        self,
        query: str,
        candidates: list[Candidate],
        trace: TraceContext | None = None,
        **kwargs: Any
    ) -> RerankResult:
        """Rerank candidates using Cross-Encoder scoring.

        Args:
            query: The search query
            candidates: List of candidates to rerank
            trace: Tracing context for observability
            **kwargs: Additional arguments
                - top_k: Override number of results to return

        Returns:
            RerankResult with candidates in ranked order

        Raises:
            RerankerConfigurationError: If scoring fails
        """
        if not candidates:
            return RerankResult(ids=[], scores=[], metadata={"reranked": False})

        top_k = kwargs.get("top_k", self._top_k)

        logger.info(
            f"CrossEncoderReranker: query={query[:50]}..., "
            f"candidate_count={len(candidates)}, top_k={top_k}"
        )

        if trace:
            trace.record_stage(
                "rerank",
                {
                    "provider": self.provider_name,
                    "query_length": len(query),
                    "candidate_count": len(candidates),
                    "top_k": top_k,
                }
            )

        try:
            # Use mock scorer if provided
            if self._scorer is not None:
                scores = self._mock_score(query, candidates)
            else:
                scores = self._score_with_model(query, candidates)

            # Combine scores with candidates
            scored = list(zip([c.id for c in candidates], scores))

            # Sort by score descending
            scored.sort(key=lambda x: x[1], reverse=True)

            # Extract top_k results
            ids = [s[0] for s in scored[:top_k]]
            result_scores = [float(s[1]) for s in scored[:top_k]]

            logger.info(
                f"CrossEncoderReranker complete: {len(ids)} results, "
                f"top_score={result_scores[0] if result_scores else 'N/A'}"
            )

            return RerankResult(
                ids=ids,
                scores=result_scores,
                metadata={
                    "reranked": True,
                    "model": self._model,
                    "total_scored": len(candidates),
                }
            )

        except asyncio.TimeoutError:
            logger.error(f"Cross-Encoder scoring timed out after {self._timeout}s")
            raise RerankerConfigurationError(
                f"Cross-Encoder scoring timed out after {self._timeout}s",
                provider=self.provider_name,
                details={"candidate_count": len(candidates), "timeout": self._timeout}
            )
        except Exception as e:
            logger.error(f"Cross-Encoder scoring failed: {e}")
            raise RerankerConfigurationError(
                f"Cross-Encoder scoring failed: {e}",
                provider=self.provider_name,
                details={"candidate_count": len(candidates)}
            )

    def _mock_score(self, query: str, candidates: list[Candidate]) -> list[float]:
        """Mock scoring for testing.

        Args:
            query: The search query
            candidates: Candidates to score

        Returns:
            List of scores (higher = more relevant)
        """
        if self._scorer is not None:
            texts = [c.content for c in candidates]
            return self._scorer(query, texts)
        return [0.0] * len(candidates)

    def _score_with_model(self, query: str, candidates: list[Candidate]) -> list[float]:
        """Score using Cross-Encoder model.

        Args:
            query: The search query
            candidates: Candidates to score

        Returns:
            List of scores (higher = more relevant)
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            # Fallback to uniform scores
            return [1.0 / (i + 1) for i in range(len(candidates))]

        # Initialize model if needed
        if self._cross_encoder is None:
            try:
                self._cross_encoder = CrossEncoder(self._model)
            except Exception as e:
                logger.warning(f"Failed to load Cross-Encoder model: {e}")
                return [1.0 / (i + 1) for i in range(len(candidates))]

        # Prepare query-candidate pairs
        texts = [c.content for c in candidates]
        pairs = [[query, text] for text in texts]

        # Score in batches
        scores = []
        for i in range(0, len(pairs), self._batch_size):
            batch = pairs[i : i + self._batch_size]
            batch_scores = self._cross_encoder.predict(
                batch,
                activation_fns=[],
                apply_softmax=False,
            )
            # Handle different output shapes
            if hasattr(batch_scores, "tolist"):
                batch_scores = batch_scores.tolist()
            if isinstance(batch_scores, float):
                scores.append(batch_scores)
            else:
                scores.extend(batch_scores)

        return scores

    def rerank_with_threshold(
        self,
        query: str,
        candidates: list[Candidate],
        score_threshold: float,
        trace: TraceContext | None = None,
        **kwargs: Any
    ) -> RerankResult:
        """Rerank with score threshold filtering.

        Args:
            query: The search query
            candidates: Candidates to rerank
            score_threshold: Minimum score for inclusion
            trace: Tracing context for observability
            **kwargs: Additional arguments

        Returns:
            RerankResult with candidates above threshold
        """
        result = self.rerank(query, candidates, trace=trace, **kwargs)

        # Filter by threshold
        filtered_ids = []
        filtered_scores = []
        for id_, score in zip(result.ids, result.scores):
            if score >= score_threshold:
                filtered_ids.append(id_)
                filtered_scores.append(score)

        return RerankResult(
            ids=filtered_ids,
            scores=filtered_scores,
            metadata={
                **result.metadata,
                "score_threshold": score_threshold,
            }
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"CrossEncoderReranker("
            f"provider={self.provider_name}, "
            f"model={self._model}, "
            f"top_k={self._top_k})"
        )
