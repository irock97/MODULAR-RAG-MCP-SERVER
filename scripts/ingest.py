#!/usr/bin/env python
"""Ingestion script for the Modular RAG MCP Server.

This script provides a command-line interface for running the ingestion pipeline
on documents.

Usage:
    python scripts/ingest.py --path /path/to/documents --collection my_docs
    python scripts/ingest.py --path /path/to/documents --collection my_docs --force

Options:
    --path PATH           Path to file or directory to ingest (required)
    --collection NAME    Collection name for organizing documents (default: default)
    --force              Force reprocessing even if file unchanged
    --settings PATH     Path to settings.yaml (default: config/settings.yaml)
    -h, --help          Show this help message
"""

import argparse
import sys
from pathlib import Path

from core.settings import load_settings
from ingestion.pipeline import IngestionPipeline
from observability.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into the RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to file or directory to ingest"
    )

    parser.add_argument(
        "--collection",
        type=str,
        default="default",
        help="Collection name (default: default)"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if file unchanged"
    )

    parser.add_argument(
        "--settings",
        type=str,
        default="config/settings.yaml",
        help="Path to settings.yaml (default: config/settings.yaml)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Validate path
    path = Path(args.path)
    if not path.exists():
        logger.error(f"Path does not exist: {path}")
        sys.exit(1)

    # Load settings
    try:
        settings = load_settings(args.settings)
        logger.info(f"Loaded settings from: {args.settings}")
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Ingestion Script")
    logger.info("=" * 60)
    logger.info(f"Path: {args.path}")
    logger.info(f"Collection: {args.collection}")
    logger.info(f"Force: {args.force}")
    logger.info("=" * 60)

    # Run pipeline
    pipeline = IngestionPipeline(
        settings=settings,
        collection=args.collection,
        force=args.force
    )

    try:
        result = pipeline.run(path)

        # Print summary
        logger.info("=" * 60)
        logger.info("Ingestion Complete!")
        logger.info("=" * 60)
        logger.info(f"Success: {result.success}")
        logger.info(f"File: {result.file_path}")
        logger.info(f"Doc ID: {result.doc_id}")
        logger.info(f"Chunks: {result.chunk_count}")
        logger.info(f"Images: {result.image_count}")
        logger.info(f"Vectors: {result.vector_ids_count if hasattr(result, 'vector_ids_count') else len(result.vector_ids)}")

        if result.error:
            logger.error(f"Error: {result.error}")

        if result.stages:
            logger.info("Stages:")
            for stage_name, stage_data in result.stages.items():
                logger.info(f"  {stage_name}: {stage_data}")

        # Exit with error code if failed
        if not result.success:
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
