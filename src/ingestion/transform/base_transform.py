"""Base Transform - Abstract interface for chunk transformations.

This module defines the BaseTransform abstract base class that all
transformations must implement. Transformations process chunks after
chunking to refine content, enrich metadata, or perform other modifications.

Design Principles:
    - Same Interface Pattern: All transforms share the same transform() method
    - Fail-Safe: Transforms should not crash the pipeline; use fallback strategies
    - Traceable: All transforms support optional trace context
    - Metadata Forward: Original and refined metadata are preserved
"""

from abc import ABC, abstractmethod
from typing import Any

from core.settings import Settings
from core.trace.trace_context import TraceContext


class BaseTransform(ABC):
    """Abstract base class for all chunk transformations.

    Transformations operate on chunks after they have been created by
    the DocumentChunker. Examples include:
    - ChunkRefiner: Rule-based and LLM-based text refinement
    - MetadataEnricher: Adding extracted metadata to chunks
    - ImageCaptioner: Generating captions for images in chunks

    Example:
        >>> from core.settings import load_settings
        >>> from ingestion.transform import ChunkRefiner
        >>>
        >>> settings = load_settings()
        >>> refiner = ChunkRefiner(settings)
        >>>
        >>> chunks = [chunk1, chunk2, chunk3]
        >>> refined = refiner.transform(chunks)
    """

    def __init__(
        self,
        settings: Settings,
    ) -> None:
        """Initialize the transform with settings.

        Args:
            settings: Settings object containing configuration.
        """
        self._settings = settings

    @property
    def settings(self) -> Settings:
        """Get the settings object."""
        return self._settings

    @abstractmethod
    def transform(
        self,
        chunks: list[Any],
        trace: TraceContext | None = None,
    ) -> list[Any]:
        """Transform a list of chunks.

        This is the main entry point for all transformations. It processes
        each chunk and returns a list of transformed chunks.

        Args:
            chunks: List of chunks to transform.
            trace: Optional trace context for observability.

        Returns:
            List of transformed chunks. May be same length as input,
            or fewer/more depending on the transformation.

        Raises:
            TransformError: If transformation fails unexpectedly.
        """
        pass

    def _create_error_result(
        self,
        original_chunk: Any,
        reason: str,
    ) -> dict[str, Any]:
        """Create an error result for a failed transformation.

        This is a helper method for implementing fail-safe transforms.

        Args:
            original_chunk: The original chunk that failed.
            reason: Description of why the transformation failed.

        Returns:
            A dictionary with error information.
        """
        return {
            "original": original_chunk,
            "error": reason,
            "fallback": True,
        }
