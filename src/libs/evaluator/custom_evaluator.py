"""Custom evaluator implementation with standard retrieval metrics.

This module provides the CustomEvaluator class that computes
standard retrieval quality metrics (hit_rate, mrr, precision, recall, etc.).

Design Principles:
    - Deterministic: Returns stable metrics for same inputs
    - Lightweight: No external dependencies required
    - Observable: trace parameter for tracing integration
"""

from typing import Any

from libs.evaluator.base_evaluator import (
    BaseEvaluator,
    EvaluationResult,
    EvaluatorError,
)
from observability.logger import TraceContext, get_logger

logger = get_logger(__name__)


class CustomEvaluator(BaseEvaluator):
    """Custom evaluator for retrieval quality metrics.

    This evaluator computes standard retrieval metrics:
    - hit_rate: Whether any relevant doc was retrieved
    - mrr (Mean Reciprocal Rank): 1/rank of first relevant doc
    - precision@k: Fraction of retrieved docs that are relevant
    - recall@k: Fraction of relevant docs that are retrieved

    Attributes:
        metrics: List of metric names to compute
    """

    def __init__(
        self,
        metrics: list[str] | None = None,
        k_values: list[int] | None = None,
        **kwargs: Any
    ) -> None:
        """Initialize the Custom Evaluator.

        Args:
            metrics: List of metric names to compute (default: ["hit_rate", "mrr"])
            k_values: List of k values for precision@k and recall@k
            **kwargs: Extra arguments (ignored for compatibility)
        """
        self._metrics = metrics or ["hit_rate", "mrr"]
        self._k_values = k_values or [1, 5, 10]
        self.call_count = 0

    @property
    def provider_name(self) -> str:
        return "custom"

    def _compute_hit_rate(
        self,
        retrieved_ids: list[str],
        golden_ids: list[str]
    ) -> float:
        """Compute hit rate (whether any relevant doc was retrieved).

        Args:
            retrieved_ids: List of retrieved document IDs
            golden_ids: List of relevant/golden document IDs

        Returns:
            1.0 if any golden ID in retrieved IDs, 0.0 otherwise
        """
        retrieved_set = set(retrieved_ids)
        golden_set = set(golden_ids)
        hits = len(retrieved_set & golden_set)
        return 1.0 if hits > 0 else 0.0

    def _compute_mrr(
        self,
        retrieved_ids: list[str],
        golden_ids: list[str]
    ) -> float:
        """Compute Mean Reciprocal Rank.

        Args:
            retrieved_ids: List of retrieved document IDs (in order)
            golden_ids: List of relevant/golden document IDs

        Returns:
            1/rank of first relevant doc, 0.0 if none found
        """
        golden_set = set(golden_ids)
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in golden_set:
                return 1.0 / rank
        return 0.0

    def _compute_precision_at_k(
        self,
        retrieved_ids: list[str],
        golden_ids: list[str],
        k: int
    ) -> float:
        """Compute precision at k.

        Args:
            retrieved_ids: List of retrieved document IDs
            golden_ids: List of relevant/golden document IDs
            k: Consider only top-k retrieved documents

        Returns:
            Fraction of top-k docs that are relevant
        """
        retrieved_set = set(retrieved_ids[:k])
        golden_set = set(golden_ids)
        hits = len(retrieved_set & golden_set)
        return hits / k if k > 0 else 0.0

    def _compute_recall_at_k(
        self,
        retrieved_ids: list[str],
        golden_ids: list[str],
        k: int
    ) -> float:
        """Compute recall at k.

        Args:
            retrieved_ids: List of retrieved document IDs
            golden_ids: List of relevant/golden document IDs
            k: Consider only top-k retrieved documents

        Returns:
            Fraction of relevant docs found in top-k
        """
        retrieved_set = set(retrieved_ids[:k])
        golden_set = set(golden_ids)
        hits = len(retrieved_set & golden_set)
        return hits / len(golden_set) if len(golden_set) > 0 else 0.0

    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        trace: TraceContext | None = None,
        **kwargs: Any
    ) -> EvaluationResult:
        """Evaluate retrieval quality for a single query.

        Args:
            query: The original query string
            retrieved_ids: List of IDs returned by the retriever
            golden_ids: List of IDs that are relevant to the query
            trace: Tracing context for observability
            **kwargs: Additional arguments

        Returns:
            EvaluationResult containing requested metrics

        Raises:
            EvaluatorError: If evaluation fails
        """
        self.call_count += 1

        try:
            metrics: dict[str, float] = {}

            # Standard metrics
            if "hit_rate" in self._metrics:
                metrics["hit_rate"] = self._compute_hit_rate(retrieved_ids, golden_ids)

            if "mrr" in self._metrics:
                metrics["mrr"] = self._compute_mrr(retrieved_ids, golden_ids)

            # Precision and Recall at various k values
            for k in self._k_values:
                if "precision" in self._metrics:
                    metrics[f"precision@{k}"] = self._compute_precision_at_k(
                        retrieved_ids, golden_ids, k
                    )
                if "recall" in self._metrics:
                    metrics[f"recall@{k}"] = self._compute_recall_at_k(
                        retrieved_ids, golden_ids, k
                    )

            details = {
                "query_length": len(query),
                "retrieved_count": len(retrieved_ids),
                "golden_count": len(golden_ids),
            }

            logger.debug(
                f"Evaluated query: hit_rate={metrics.get('hit_rate', 'N/A')}, "
                f"mrr={metrics.get('mrr', 'N/A')}"
            )

            return EvaluationResult(metrics=metrics, details=details)

        except Exception as e:
            raise EvaluatorError(
                f"Evaluation failed: {e}",
                provider=self.provider_name,
                details={"query": query}
            ) from e

    def evaluate_batch(
        self,
        queries: list[str],
        retrieved_ids_list: list[list[str]],
        golden_ids_list: list[list[str]],
        trace: TraceContext | None = None,
        **kwargs: Any
    ) -> list[EvaluationResult]:
        """Evaluate retrieval quality for multiple queries.

        Args:
            queries: List of query strings
            retrieved_ids_list: List of retrieved ID lists
            golden_ids_list: List of golden ID lists
            trace: Tracing context for observability
            **kwargs: Additional arguments

        Returns:
            List of EvaluationResult for each query
        """
        results = []
        for query, retrieved_ids, golden_ids in zip(
            queries, retrieved_ids_list, golden_ids_list
        ):
            result = self.evaluate(query, retrieved_ids, golden_ids, trace=trace)
            results.append(result)

        # Aggregate metrics for batch
        all_metrics: dict[str, list[float]] = {}
        for result in results:
            for metric_name, value in result.metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)

        # Compute averages
        aggregated: dict[str, float] = {}
        for metric_name, values in all_metrics.items():
            aggregated[f"avg_{metric_name}"] = sum(values) / len(values)

        logger.info(
            f"Batch evaluation complete: {len(queries)} queries, "
            f"avg_hit_rate={aggregated.get('avg_hit_rate', 'N/A'):.4f}"
        )

        # Add aggregated metrics to last result
        if results:
            results[-1].details = results[-1].details or {}
            results[-1].details["aggregated_metrics"] = aggregated

        return results
