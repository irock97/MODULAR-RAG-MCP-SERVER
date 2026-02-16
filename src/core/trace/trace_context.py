"""Trace Context - Minimal implementation for ingestion pipeline.

This module provides minimal trace context for tracking pipeline stages.
A full implementation will be added in Phase F (Observability).
"""

import uuid
from typing import Any


class TraceContext:
    """Minimal trace context for tracking ingestion pipeline stages.

    This is a placeholder implementation. A full implementation with
    structured logging, spans, and metrics will be added in Phase F.
    """

    def __init__(self) -> None:
        """Initialize trace context with a unique trace ID."""
        self._trace_id: str = str(uuid.uuid4())
        self._stages: dict[str, Any] = {}

    @property
    def trace_id(self) -> str:
        """Get the unique trace ID for this pipeline run."""
        return self._trace_id

    def record_stage(self, stage_name: str, data: dict[str, Any]) -> None:
        """Record data for a pipeline stage.

        Args:
            stage_name: Name of the pipeline stage (e.g., "chunking", "refining").
            data: Dictionary of stage data to record.
        """
        self._stages[stage_name] = data

    def get_stage(self, stage_name: str) -> dict[str, Any] | None:
        """Get recorded data for a stage.

        Args:
            stage_name: Name of the pipeline stage.

        Returns:
            Dictionary of stage data, or None if not recorded.
        """
        return self._stages.get(stage_name)

    def get_all_stages(self) -> dict[str, dict[str, Any]]:
        """Get all recorded stage data.

        Returns:
            Dictionary of all stage data.
        """
        return dict(self._stages)

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the trace context.

        Args:
            key: Metadata key.
            value: Metadata value.
        """
        if "metadata" not in self._stages:
            self._stages["metadata"] = {}
        self._stages["metadata"][key] = value

    def __repr__(self) -> str:
        """String representation of trace context."""
        return f"TraceContext(trace_id={self._trace_id}, stages={list(self._stages.keys())})"
