"""Ingestion Pipeline - Document Chunking Module.

This module provides document chunking functionality for the ingestion pipeline.

Classes:
    DocumentChunker: Adapter that converts Document to List[Chunk]
"""

from ingestion.chunking.document_chunker import (
    DocumentChunker,
    DocumentChunkingError,
)

__all__ = [
    "DocumentChunker",
    "DocumentChunkingError",
]
