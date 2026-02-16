"""ChunkRefiner - Rule-based and LLM-based text refinement.

This module provides the ChunkRefiner class that cleans and improves
chunk text quality through rule-based patterns and optional LLM enhancement.

Design Principles:
    - Fail-Safe: LLM failures gracefully fall back to rule-based results
    - Configurable: LLM usage controlled via settings.yaml
    - Traceable: All refinements logged to trace context
    - Metadata Preserved: Original and refined metadata tracked

Refinement Pipeline:
    1. Rule-based cleaning (always runs)
    2. Optional LLM enhancement (if enabled and available)
    3. Fallback to rule-only if LLM fails
"""

import re
from pathlib import Path
from typing import Any

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.base_transform import BaseTransform
from libs.llm.base_llm import ChatMessage
from libs.llm.llm_factory import LLMFactory
from observability.logger import get_logger

logger = get_logger(__name__)


class ChunkRefinementError(Exception):
    """Error during chunk refinement."""

    pass


class ChunkRefiner(BaseTransform):
    """Refines chunk text using rules and optional LLM enhancement.

    This class applies two-stage refinement:
    1. **Rule-based cleaning**: Removes common noise patterns
    2. **LLM enhancement**: Optionally improves quality via LLM

    The LLM enhancement is controlled by `settings.ingestion.chunk_refiner.use_llm`.
    If LLM is enabled but fails, the refiner falls back to rule-based results.
    """

    # Common noise patterns for rule-based cleaning
    # Noise patterns with their replacement values
    NOISE_PATTERNS = {
        # Excessive whitespace (multiple spaces/newlines)
        "excessive_whitespace": (re.compile(r"[ \t]{2,}"), " "),
        # Multiple consecutive newlines
        "consecutive_newlines": (re.compile(r"\n{3,}"), "\n\n"),
        # Page headers/footers (common patterns)
        "page_markers": (re.compile(
            r"(?i)(page\s+\d+\s+of\s+\d+|"
            r"---\s*$|"
            r"^\s*={3,}\s*$|"
            r"(?:\bCONFIDENTIAL\b|\bINTERNAL\b|\bDRAFT\b).*$)"
        ), ""),
        # HTML/Markdown comments
        "html_comments": (re.compile(r"<!--.*?-->"), ""),
        # Leading/trailing whitespace on lines
        "line_whitespace": (re.compile(r"^[ \t]+|[ \t]+$", re.MULTILINE), ""),
        # Page break markers
        "page_breaks": (re.compile(r"(?i)(\f|page\s*break|={20,})"), ""),
    }

    def __init__(
        self,
        settings: Settings,
        llm: Any | None = None,
        prompt_path: str | Path | None = None,
    ) -> None:
        """Initialize the ChunkRefiner.

        Args:
            settings: Settings object with configuration.
            llm: Optional LLM instance. If None, created from settings.
            prompt_path: Optional path to prompt template file.
        """
        super().__init__(settings)

        self._llm = llm
        self._prompt_path = Path(prompt_path) if prompt_path else None
        self._use_llm = getattr(
            settings.ingestion.chunk_refiner, "use_llm", False
        )
        self._prompt_template: str | None = None

        # Load prompt template if LLM is enabled
        if self._use_llm:
            self._load_prompt()

    @property
    def use_llm(self) -> bool:
        """Check if LLM enhancement is enabled."""
        return self._use_llm

    def _load_prompt(self) -> None:
        """Load the prompt template from file or use default."""
        if self._prompt_path and self._prompt_path.exists():
            self._prompt_template = self._prompt_path.read_text()
            logger.info(f"Loaded prompt from: {self._prompt_path}")
        else:
            # Default prompt path
            default_prompt = Path(__file__).parent / "prompts" / "chunk_refinement.txt"
            if default_prompt.exists():
                self._prompt_template = default_prompt.read_text()
                logger.info(f"Loaded default prompt from: {default_prompt}")
            else:
                logger.warning("No prompt template found, LLM mode may not work properly")
                self._prompt_template = None

    def _get_llm(self) -> Any | None:
        """Get or create the LLM instance."""
        if self._llm is None and self._use_llm:
            try:
                self._llm = LLMFactory.create(self._settings)
                logger.info(f"Created LLM for chunk refinement: {self._llm.provider_name}")
            except Exception as e:
                logger.warning(f"Failed to create LLM for refinement: {e}")
                self._llm = None
        return self._llm

    def transform(
        self,
        chunks: list[Chunk],
        trace: TraceContext | None = None,
    ) -> list[Chunk]:
        """Transform chunks by refining their text.

        This method applies rule-based cleaning to all chunks, and
        optionally applies LLM enhancement if enabled.

        Args:
            chunks: List of chunks to refine.
            trace: Optional trace context for observability.

        Returns:
            List of refined chunks. Metadata includes refinement info.

        Raises:
            ChunkRefinementError: If all refinements fail unexpectedly.
        """
        if not chunks:
            return []

        logger.info(f"Refining {len(chunks)} chunks (LLM: {self._use_llm})")

        refined_chunks: list[Chunk] = []
        errors: list[dict[str, Any]] = []

        for index, chunk in enumerate(chunks):
            try:
                # Step 1: Rule-based cleaning
                refined_text = self._rule_based_refine(chunk.text)
                refined_by = "rule"

                # Step 2: Optional LLM enhancement
                if self._use_llm:
                    llm_result = self._llm_refine(refined_text, trace)
                    if llm_result is not None:
                        refined_text = llm_result
                        refined_by = "llm"

                # Create refined chunk with metadata
                refined_chunk = Chunk(
                    id=chunk.id,
                    text=refined_text,
                    metadata=dict(chunk.metadata),
                    source_ref=chunk.source_ref,
                )

                # Add refinement metadata
                refined_chunk.metadata["refinement"] = {
                    "refined_by": refined_by,
                    "rule_based_applied": True,
                }
                refined_chunk.metadata["chunk_index"] = chunk.metadata.get(
                    "chunk_index", index
                )

                refined_chunks.append(refined_chunk)

            except Exception as e:
                logger.warning(f"Failed to refine chunk {chunk.id}: {e}")
                # Fallback: return original chunk with error metadata
                error_chunk = Chunk(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=dict(chunk.metadata),
                    source_ref=chunk.source_ref,
                )
                error_chunk.metadata["refinement"] = {
                    "refined_by": "rule",
                    "error": str(e),
                    "fallback": True,
                }
                errors.append({"chunk_id": chunk.id, "error": str(e)})
                refined_chunks.append(error_chunk)

        # Record stage in trace
        if trace:
            trace.record_stage(
                "chunk_refinement",
                {
                    "input_count": len(chunks),
                    "output_count": len(refined_chunks),
                    "llm_used": self._use_llm,
                    "errors": errors,
                },
            )

        logger.info(
            f"Refined {len(refined_chunks)} chunks, {len(errors)} errors"
        )
        return refined_chunks

    def _rule_based_refine(self, text: str) -> str:
        """Apply rule-based text cleaning.

        This method removes common noise patterns from extracted text:
        - Excessive whitespace
        - Page markers (headers, footers, separators)
        - HTML/Markdown comments
        - Leading/trailing whitespace

        Args:
            text: Raw text from document extraction.

        Returns:
            Cleaned text with noise patterns removed.
        """
        if not text:
            return ""

        result = text

        # Apply each noise pattern with its replacement
        for pattern_name, (pattern, replacement) in self.NOISE_PATTERNS.items():
            result = pattern.sub(replacement, result)

        # Clean up any resulting double newlines
        result = re.sub(r"\n\n+", "\n\n", result)

        # Strip leading/trailing whitespace
        result = result.strip()

        return result

    def _llm_refine(
        self,
        text: str,
        trace: TraceContext | None = None,
    ) -> str | None:
        """Apply LLM-based text enhancement.

        This method sends the text to an LLM for intelligent refinement
        if an LLM is configured and available.

        Args:
            text: Rule-cleaned text to further refine.
            trace: Optional trace context for observability.

        Returns:
            LLM-refined text, or None if LLM fails or is disabled.

        Raises:
            ChunkRefinementError: If LLM call fails unexpectedly.
        """
        if not self._use_llm or not self._prompt_template:
            return None

        llm = self._get_llm()
        if llm is None:
            logger.warning("LLM not available, skipping LLM refinement")
            return None

        try:
            # Format prompt with the text using {text} placeholder
            if "{text}" in self._prompt_template:
                system_prompt = self._prompt_template.replace("{text}", text)
            else:
                # Fallback: append text to prompt
                system_prompt = self._prompt_template + f"\n\n{text}"

            # Use chat() with system and user messages
            messages = [
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content="Please refine the above text chunk."),
            ]

            # Call LLM with trace context
            response = llm.chat(messages=messages, trace=trace)

            # Parse response
            refined = response.content.strip()

            logger.debug(f"LLM refined text (len: {len(text)} -> {len(refined)})")

            # Record to trace
            if trace:
                trace.record_stage(
                    "chunk_llm_refinement",
                    {
                        "input_length": len(text),
                        "output_length": len(refined),
                        "llm_provider": llm.provider_name,
                    },
                )

            return refined

        except Exception as e:
            logger.warning(f"LLM refinement failed: {e}")
            # Return None to trigger fallback
            return None

    def get_refinement_stats(self, chunks: list[Chunk]) -> dict[str, Any]:
        """Get statistics about chunk refinements.

        Args:
            chunks: List of refined chunks.

        Returns:
            Dictionary with refinement statistics.
        """
        stats = {
            "total": len(chunks),
            "by_method": {"rule": 0, "llm": 0},
            "errors": 0,
        }

        for chunk in chunks:
            refinement = chunk.metadata.get("refinement", {})
            refined_by = refinement.get("refined_by", "rule")
            stats["by_method"][refined_by] = stats["by_method"].get(refined_by, 0) + 1

            if refinement.get("fallback"):
                stats["errors"] += 1

        return stats
