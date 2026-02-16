"""Document Chunker - Adapter between libs.splitter and Ingestion Pipeline.

This module provides the DocumentChunker class that transforms Document objects
into Chunk objects with business logic增值.

Design Principles:
    - Adapter Pattern: Wraps libs.splitter for pipeline use
    - Deterministic: Chunk IDs are stable across runs
    - Metadata Inheritance: Chunks inherit document metadata
    - Traceable: Chunks reference their parent document

Core Responsibilities:
    1. Generate stable Chunk IDs: `{doc_id}_{index:04d}_{hash_8chars}`
    2. Inherit Document.metadata to each Chunk
    3. Add chunk_index for ordering
    4. Set source_ref to parent Document.id
    5. Convert List[str] from splitter to List[Chunk]
"""

import hashlib
from typing import Any

from core.settings import Settings
from core.types import Chunk, Document
from libs.splitter.base_splitter import BaseSplitter
from libs.splitter.splitter_factory import SplitterFactory
from observability.logger import get_logger

logger = get_logger(__name__)


class DocumentChunkingError(Exception):
    """Error during document chunking."""

    pass


class DocumentChunker:
    """Adapter that converts Document objects to Chunk objects.

    This class adds business logic on top of libs.splitter:
    - Generates deterministic chunk IDs
    - Inherits document metadata
    - Tracks chunk positions
    - Enables source tracing

    Example:
        >>> from core.settings import load_settings
        >>> from libs.splitter.providers import RecursiveSplitter
        >>> from ingestion.chunking import DocumentChunker
        >>>
        >>> SplitterFactory.register("recursive", RecursiveSplitter)
        >>> settings = load_settings()
        >>> chunker = DocumentChunker(settings)
        >>>
        >>> doc = Document(id="doc1", text="Long text...", metadata={})
        >>> chunks = chunker.split_document(doc)
    """

    def __init__(
        self,
        settings: Settings,
        splitter: BaseSplitter | None = None,
    ) -> None:
        """Initialize the DocumentChunker.

        Args:
            settings: Settings object containing ingestion configuration.
            splitter: Optional splitter instance. If None, creates one from settings.
        """
        self._settings = settings

        if splitter is None:
            self._splitter = SplitterFactory.create(settings)
            logger.info(
                f"Created splitter from settings: {self._splitter.provider_name}"
            )
        else:
            self._splitter = splitter
            logger.info(f"Using provided splitter: {self._splitter.provider_name}")

    @property
    def splitter(self) -> BaseSplitter:
        """Get the underlying splitter instance."""
        return self._splitter

    def split_document(self, document: Document) -> list[Chunk]:
        """Split a Document into Chunks with metadata and IDs.

        This is the main entry point for document chunking. It:
        1. Calls libs.splitter to get text chunks
        2. Generates stable chunk IDs
        3. Inherits document metadata
        4. Adds chunk_index for ordering
        5. Sets source_ref for tracing

        Args:
            document: The Document to split.

        Returns:
            List of Chunk objects with inherited metadata and stable IDs.

        Raises:
            DocumentChunkingError: If splitting fails.
        """
        logger.info(f"Chunking document: {document.id}")

        try:
            # Step 1: Use libs.splitter to get text chunks
            result = self._splitter.split_text(document.text)
            text_chunks = result.chunks
            logger.info(f"Splitter produced {len(text_chunks)} chunks")
        except Exception as e:
            raise DocumentChunkingError(
                f"Failed to split document {document.id}: {e}"
            ) from e

        # Step 2-5: Transform to Chunk objects with business logic
        chunks: list[Chunk] = []
        for index, text_chunk in enumerate(text_chunks):
            # Generate stable chunk ID
            chunk_id = self._generate_chunk_id(document.id, index, text_chunk)

            # Inherit and extend metadata
            chunk_metadata = self._inherit_metadata(document, index)

            # Create the Chunk object
            chunk = Chunk(
                id=chunk_id,
                text=text_chunk,
                metadata=chunk_metadata,
                source_ref=document.id,
            )
            chunks.append(chunk)

        logger.info(f"Created {len(chunks)} chunks for document {document.id}")
        return chunks

    def _generate_chunk_id(
        self,
        doc_id: str,
        index: int,
        text_chunk: str,
    ) -> str:
        """Generate a deterministic chunk ID.

        Format: `{doc_id}_{index:04d}_{hash_8chars}`

        This ensures:
        - Uniqueness: Each chunk has a unique ID within the document
        - Stability: Same content produces same ID across runs
        - Traceability: ID contains document reference and position

        Args:
            doc_id: The parent document's ID.
            index: The chunk's position (0-based).
            text_chunk: The text content of the chunk.

        Returns:
            A deterministic chunk ID string.
        """
        # Create hash from content for stability
        content_hash = hashlib.md5(text_chunk.encode("utf-8")).hexdigest()[:8]

        # Format: doc_id_0001_abc12345
        chunk_id = f"{doc_id}_{index:04d}_{content_hash}"

        return chunk_id

    def _inherit_metadata(
        self,
        document: Document,
        chunk_index: int,
    ) -> dict[str, Any]:
        """Inherit document metadata and add chunk-specific fields.

        All fields from document.metadata are copied to chunk.metadata,
        plus additional fields for chunking context.

        Args:
            document: The source Document.
            chunk_index: The chunk's position in the document.

        Returns:
            A metadata dictionary with inherited and added fields.
        """
        # Copy existing metadata
        inherited: dict[str, Any] = dict(document.metadata)

        # Add chunking-specific fields
        inherited["chunk_index"] = chunk_index

        return inherited


