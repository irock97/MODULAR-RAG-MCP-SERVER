"""Observability and logging utilities for Modular RAG MCP Server.

This module provides logging infrastructure for the application.
Currently provides basic stderr logging. Will be expanded in Phase F.

Design Principles:
    - Observable: All components should emit structured logs
    - Fail-Safe: Logging failures should not crash the application
"""

import logging
import sys
from typing import Any


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with standard configuration.

    This function creates a logger that outputs to stderr with a standard
    format. In Phase F, this will be expanded to support structured JSON
    logging and trace file output.

    Args:
        name: Logger name (typically __name__ from the calling module)

    Returns:
        Configured Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing document", extra={"doc_id": "123"})
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers if already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Set default level if not already set
    if not logger.level:
        logger.setLevel(logging.INFO)

    return logger


def configure_logger(
    level: str = "INFO",
    format: str | None = None,
    log_file: str | None = None
) -> None:
    """Configure the root logger for the application.

    This function allows runtime reconfiguration of logging behavior.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        format: Custom log format string (optional)
        log_file: Path to log file (optional, for file logging)

    Example:
        >>> configure_logger(level="DEBUG", log_file="./app.log")
    """
    # Convert string level to int
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    log_level = level_map.get(level.upper(), logging.INFO)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_formatter = logging.Formatter(
        fmt=format or "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(console_formatter)
        root_logger.addHandler(file_handler)


class TraceContext:
    """Context manager for request tracing.

    This is a placeholder for the full TraceContext implementation
    that will be added in Phase F (Observability).

    Attributes:
        trace_id: Unique identifier for this trace
        spans: List of spans within this trace
    """

    def __init__(self, trace_id: str | None = None) -> None:
        self.trace_id = trace_id or self._generate_id()
        self.spans: list[dict[str, Any]] = []

    @staticmethod
    def _generate_id() -> str:
        """Generate a unique trace ID."""
        import uuid
        return uuid.uuid4().hex[:16]

    def __enter__(self) -> "TraceContext":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass
