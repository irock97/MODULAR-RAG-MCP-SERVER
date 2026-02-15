"""LLM-based Reranker implementation.

This module provides LLM-based reranking that uses an LLM to score
and reorder candidates based on query relevance.

Design Principles:
    - LLM-powered: Uses LLM for intelligent scoring
    - Prompt-driven: Reads reranking prompt from config/prompts/rerank.txt
    - Fallback signal: Returns None on failure for graceful degradation
"""

import re
from pathlib import Path
from typing import Any

from libs.llm.base_llm import BaseLLM
from libs.reranker.base_reranker import (
    BaseReranker,
    Candidate,
    RerankResult,
    RerankerConfigurationError,
)
from observability.logger import TraceContext, get_logger

logger = get_logger(__name__)


class LLMReranker(BaseReranker):
    """LLM-based Reranker.

    Uses an LLM to score and reorder candidates based on their
    relevance to the query.

    Attributes:
        llm: The LLM to use for scoring
        prompt_template: The prompt template for reranking
        top_k: Number of top results to return
    """

    # Default prompt path
    DEFAULT_PROMPT_PATH = "config/prompts/rerank.txt"
    # Default number of results to return
    DEFAULT_TOP_K = 10
    # Default score threshold
    DEFAULT_SCORE_THRESHOLD = 0.0

    def __init__(
        self,
        llm: BaseLLM | None = None,
        model: str | None = None,
        prompt_path: str | None = None,
        prompt_template: str | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> None:
        """Initialize the LLM Reranker.

        Args:
            llm: The LLM to use for scoring. If None, must be provided later.
            model: Model name (for factory compatibility, not used directly).
            prompt_path: Path to the rerank prompt template file.
            prompt_template: Raw prompt template string (overrides prompt_path if provided).
            top_k: Number of top results to return.
            score_threshold: Minimum score threshold for inclusion.

        Raises:
            RerankerConfigurationError: If no LLM is provided and none is configured.
        """
        self._llm = llm

        # Load prompt template
        if prompt_template:
            self._prompt_template = prompt_template
        elif prompt_path:
            self._prompt_template = self._load_prompt(prompt_path)
        else:
            self._prompt_template = self._load_prompt(self.DEFAULT_PROMPT_PATH)

        self._top_k = top_k if top_k is not None else self.DEFAULT_TOP_K
        self._score_threshold = (
            score_threshold if score_threshold is not None else self.DEFAULT_SCORE_THRESHOLD
        )

        logger.info(
            f"LLM Reranker initialized: top_k={self._top_k}, "
            f"score_threshold={self._score_threshold}"
        )

    def _load_prompt(self, path: str) -> str:
        """Load prompt template from file.

        Args:
            path: Path to the prompt file.

        Returns:
            The prompt template string.

        Raises:
            RerankerConfigurationError: If the prompt file cannot be loaded.
        """
        try:
            prompt_path = Path(path)
            if prompt_path.exists():
                return prompt_path.read_text()
            else:
                # Try relative to project root
                project_root = Path(__file__).parent.parent.parent
                full_path = project_root / path
                if full_path.exists():
                    return full_path.read_text()
                logger.warning(f"Prompt file not found: {path}")
                return self._get_default_prompt()
        except Exception as e:
            logger.warning(f"Failed to load prompt from {path}: {e}")
            return self._get_default_prompt()

    def _get_default_prompt(self) -> str:
        """Return the default reranking prompt.

        Returns:
            Default prompt template.
        """
        return """You are an AI assistant specialized in evaluating the relevance of text passages to a given query.

Given a query and a list of candidate passages, score each passage on its relevance to the query.

Scoring criteria:
- 3 (Highly Relevant): The passage directly answers or addresses the query
- 2 (Partially Relevant): The passage contains related information but doesn't fully address the query
- 1 (Marginally Relevant): The passage has some tangential connection to the query
- 0 (Not Relevant): The passage has no meaningful connection to the query

Output format:
Return a JSON array of objects with:
- passage_id: The identifier of the passage
- score: The relevance score (0-3)"""

    @property
    def provider_name(self) -> str:
        """Return the name of this provider.

        Returns:
            Provider identifier: 'llm'
        """
        return "llm"

    def rerank(
        self,
        query: str,
        candidates: list[Candidate],
        trace: TraceContext | None = None,
        **kwargs: Any
    ) -> RerankResult:
        """Rerank candidates using LLM scoring.

        Args:
            query: The search query
            candidates: List of candidates to rerank
            trace: Tracing context for observability
            **kwargs: Additional arguments
                - top_k: Override number of results to return

        Returns:
            RerankResult with candidates in ranked order

        Raises:
            RerankerConfigurationError: If LLM is not configured
            RerankerError: If reranking fails
        """
        if not candidates:
            return RerankResult(ids=[], scores=[], metadata={"reranked": False})

        if self._llm is None:
            raise RerankerConfigurationError(
                "LLM is not configured for LLM Reranker",
                provider=self.provider_name
            )

        top_k = kwargs.get("top_k", self._top_k)

        logger.info(
            f"LLM Reranker: query={query[:50]}..., "
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
            # Build the prompt
            prompt = self._build_prompt(query, candidates)

            # Call LLM
            response = self._llm.complete(prompt=prompt, max_tokens=4096)

            # Parse the response
            parsed = self._parse_response(response.content, candidates)

            # Filter and sort by score
            filtered = [r for r in parsed if r["score"] >= self._score_threshold]
            sorted_results = sorted(filtered, key=lambda x: x["score"], reverse=True)

            # Extract ids and scores
            ids = [r["id"] for r in sorted_results[:top_k]]
            scores = [r["score"] for r in sorted_results[:top_k]]

            logger.info(
                f"LLM Reranker complete: {len(ids)} results, "
                f"top_score={scores[0] if scores else 'N/A'}"
            )

            return RerankResult(
                ids=ids,
                scores=scores,
                metadata={
                    "reranked": True,
                    "model": self._llm.provider_name if self._llm else "unknown",
                    "total_scored": len(candidates),
                    "score_threshold": self._score_threshold,
                }
            )

        except Exception as e:
            logger.error(f"LLM Reranker failed: {e}")
            # Return fallback signal - None indicates failure
            raise RerankerConfigurationError(
                f"Failed to rerank candidates: {e}",
                provider=self.provider_name,
                details={"candidate_count": len(candidates)}
            )

    def _build_prompt(self, query: str, candidates: list[Candidate]) -> str:
        """Build the reranking prompt.

        Args:
            query: The search query
            candidates: List of candidates to score

        Returns:
            The formatted prompt string.
        """
        # Build candidate list
        candidate_text = ""
        for i, c in enumerate(candidates):
            candidate_text += f"\n[{i}] ID: {c.id}\nContent: {c.content[:500]}"

        return f"""{self._prompt_template}

Query: {query}

Candidates:
{candidate_text}

Please score each candidate and return your scores in the following JSON format:
```json
[
  {{"passage_id": "id1", "score": 3, "reasoning": "brief explanation"}},
  {{"passage_id": "id2", "score": 2, "reasoning": "brief explanation"}}
]
```"""

    def _parse_response(
        self,
        response: str,
        candidates: list[Candidate]
    ) -> list[dict[str, Any]]:
        """Parse the LLM response into structured scores.

        Args:
            response: The LLM response text
            candidates: List of original candidates for ID mapping

        Returns:
            List of dicts with 'id', 'score', and 'reasoning'

        Raises:
            RerankerConfigurationError: If response cannot be parsed
        """
        # Extract JSON from response
        json_match = re.search(r"```json\s*([\s\S]*?)\s*```", response)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON array without code fences
            json_match = re.search(r"\[[\s\S]*?\]", response)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise RerankerConfigurationError(
                    f"Could not parse JSON from LLM response",
                    provider=self.provider_name,
                    details={"response_preview": response[:200]}
                )

        # Parse JSON
        import json
        try:
            scores = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise RerankerConfigurationError(
                f"Failed to parse JSON from LLM response: {e}",
                provider=self.provider_name,
                details={"json_str": json_str[:500]}
            )

        # Validate and normalize results
        results = []
        candidate_ids = {c.id for c in candidates}

        for item in scores:
            if not isinstance(item, dict):
                continue
            passage_id = item.get("passage_id") or item.get("id")
            if passage_id and passage_id in candidate_ids:
                score = item.get("score", 0)
                # Normalize score to 0-1 range if needed
                if score > 1:
                    score = min(score / 3.0, 1.0)  # Normalize 0-3 to 0-1
                results.append({
                    "id": passage_id,
                    "score": score,
                    "reasoning": item.get("reasoning", "")
                })

        if not results:
            # Fallback: return original order with zero scores
            logger.warning("No valid scores parsed, returning original order")
            for c in candidates:
                results.append({
                    "id": c.id,
                    "score": 0.0,
                    "reasoning": "Fallback: no score from LLM"
                })

        return results

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"LLMReranker("
            f"provider={self.provider_name}, "
            f"top_k={self._top_k}, "
            f"score_threshold={self._score_threshold})"
        )
