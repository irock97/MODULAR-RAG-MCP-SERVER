"""Document Chunker - Adapter between libs.splitter and Ingestion Pipeline.

This module provides the DocumentChunker class that transforms Document objects
into Chunk objects with business logic增值.

Design Principles:
    - Adapter Pattern: Wraps libs.splitter for pipeline use
    - Deterministic: Chunk IDs are stable across runs
    - Metadata Inheritance: Chunks inherit document metadata
    - Traceable: Chunks reference their parent document
    - Image Extraction: Uses regex to extract [IMAGE: xxx] placeholders from chunk text

Core Responsibilities:
    1. Generate stable Chunk IDs: `{doc_id}_{index:04d}_{hash_8chars}`
    2. Inherit Document.metadata to each Chunk
    3. Add chunk_index for ordering
    4. Set source_ref to parent Document.id
    5. Extract image_refs from chunk text using regex
    6. Convert List[str] from splitter to List[Chunk]
"""

import hashlib
import re
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
    - Extracts image references from chunk text

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

    # Regex pattern to match [IMAGE: xxx] placeholders
    IMAGE_PLACEHOLDER_PATTERN = re.compile(r'\[IMAGE:\s*([^\]]+)\]')

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
        3. Inherits document metadata and extracts image refs from text
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
        # Get page boundaries from document metadata for page number tracking
        page_boundaries = document.metadata.get("page_boundaries", [])
        chunks: list[Chunk] = []
        cumulative_offset = 0

        for index, text_chunk in enumerate(text_chunks):
            # Generate stable chunk ID
            chunk_id = self._generate_chunk_id(document.id, index, text_chunk)

            # Compute page number based on text offset
            chunk_start_offset = cumulative_offset
            page_num = self._compute_page_number(chunk_start_offset, page_boundaries)

            # Inherit and extend metadata, extracting images from text
            chunk_metadata = self._inherit_metadata(
                document, index, text_chunk, page_num
            )

            # Create the Chunk object (no start_offset/end_offset needed)
            chunk = Chunk(
                id=chunk_id,
                text=text_chunk,
                metadata=chunk_metadata,
                source_ref=document.id,
            )
            chunks.append(chunk)

            # Update cumulative offset for next chunk
            cumulative_offset += len(text_chunk) + 1  # +1 for separator

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

    def _compute_page_number(
        self, char_offset: int, page_boundaries: list[int]
    ) -> int | None:
        """Compute page number from character offset.

        Args:
            char_offset: Character offset in the document text.
            page_boundaries: List of character offsets where each page starts.

        Returns:
            Page number (1-based) or None if page_boundaries not available.
        """
        if not page_boundaries:
            return None

        # Find which page this offset belongs to
        for page_num, boundary in enumerate(page_boundaries, start=1):
            if char_offset < boundary:
                return page_num

        # If offset is beyond last boundary, return last page
        return len(page_boundaries)

    def _inherit_metadata(
        self,
        document: Document,
        chunk_index: int,
        chunk_text: str,
        page_num: int | None = None,
    ) -> dict[str, Any]:
        """Inherit document metadata and add chunk-specific fields.

        Extracts image references from chunk text using regex, then looks up
        full image metadata from document.metadata["images"].

        Args:
            document: The source Document.
            chunk_index: The chunk's position in the document.
            chunk_text: The text content of the chunk (used to extract image refs).
            page_num: Page number for this chunk (if available).

        Returns:
            A metadata dictionary with inherited and enriched fields.
        """
        # Copy existing metadata
        inherited: dict[str, Any] = dict(document.metadata)

        # Add chunking-specific fields
        inherited["chunk_index"] = chunk_index

        # Add page number if available
        if page_num is not None:
            inherited["page"] = page_num

        # Remove document-level images - we'll add chunk-specific images below
        inherited.pop("images", None)

        # Extract image_refs from chunk text using regex
        # This automatically handles overlap - if text contains [IMAGE: xxx], it's captured
        image_refs = []
        if chunk_text:
            matches = self.IMAGE_PLACEHOLDER_PATTERN.findall(chunk_text)
            image_refs = [m.strip() for m in matches]

        inherited["image_refs"] = image_refs

        # Build chunk-specific images list with full metadata
        # This is needed by ImageCaptioner to access image paths
        chunk_images = []
        if image_refs:
            doc_images = document.metadata.get("images", [])
            if doc_images:
                # Create lookup by image id
                image_lookup = {img.get("id"): img for img in doc_images}
                for img_id in image_refs:
                    if img_id in image_lookup:
                        chunk_images.append(image_lookup[img_id])

        if chunk_images:
            inherited["images"] = chunk_images

        return inherited
