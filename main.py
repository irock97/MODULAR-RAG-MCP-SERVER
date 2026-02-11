#!/usr/bin/env python3
"""Modular RAG MCP Server - Main Entry Point

This module provides the main entry point for the MCP Server.
See DEV_SPEC.md for architecture details.
"""

import sys
from pathlib import Path

from core.settings import Settings, SettingsError, load_settings
from observability.logger import get_logger

logger = get_logger(__name__)

# Default settings path
SETTINGS_PATH = Path(__file__).parent / "config" / "settings.yaml"


def main(settings_path: Path | None = None) -> int:
    """Main entry point.

    Args:
        settings_path: Optional path to settings.yaml. Defaults to config/settings.yaml.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    path = settings_path or SETTINGS_PATH

    logger.info("Modular RAG MCP Server - Starting...")

    try:
        settings = load_settings(path)
        logger.info(f"Configuration loaded successfully from {path}")
        logger.info(f"LLM provider: {settings.llm.provider}, model: {settings.llm.model}")
        logger.info(f"Embedding provider: {settings.embedding.provider}")
        logger.info(f"Vector store provider: {settings.vector_store.provider}")

    except SettingsError as e:
        logger.error(f"Configuration error: {e}")
        return 1

    logger.info("Server initialization complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
