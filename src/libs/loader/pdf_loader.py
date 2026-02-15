"""PDF Loader using MarkItDown for text extraction and Markdown conversion.

This loader:
1. Extracts text from PDF and converts to Markdown
2. Extracts images and saves to data/images/{doc_hash}/
3. Inserts image placeholders in the format [IMAGE: {image_id}]
4. Records image metadata in Document.metadata.images

Configuration:
    extract_images: Enable/disable image extraction (default: True)
    image_storage_dir: Base directory for image storage (default: data/images)

Graceful Degradation:
    If image extraction fails, logs warning and continues with text-only parsing.
"""

import hashlib
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from core.types import Document
from libs.loader.base_loader import (
    BaseLoader,
    LoadError,
    UnsupportedFormatError,
)

logger = logging.getLogger(__name__)


class PdfLoader(BaseLoader):
    """PDF document loader using MarkItDown.

    Extracts text and images from PDF files, converting to Markdown format.
    Supports image extraction with graceful degradation.

    Attributes:
        supported_extensions: List of supported file extensions.
        extract_images: Whether to extract images from PDF.
        image_storage_dir: Base directory for storing extracted images.
    """

    supported_extensions = [".pdf"]

    def __init__(
        self,
        extract_images: bool = True,
        image_storage_dir: str = "data/images",
    ) -> None:
        """Initialize the PDF loader.

        Args:
            extract_images: Enable/disable image extraction.
            image_storage_dir: Base directory for image storage.
        """
        self._extract_images = extract_images
        self._image_storage_dir = Path(image_storage_dir)

    @property
    def provider_name(self) -> str:
        """Return the name of this loader provider.

        Returns:
            Provider identifier: 'pdf'
        """
        return "pdf"

    def get_supported_extensions(self) -> list[str]:
        """Get the list of supported file extensions.

        Returns:
            List of supported extensions.
        """
        return self.supported_extensions

    def load(self, path: str | Path) -> Document:
        """Load a PDF document and convert to markdown.

        Args:
            path: Path to the PDF file.

        Returns:
            Document object with markdown content and metadata.

        Raises:
            LoadError: If the file cannot be read or parsed.
            UnsupportedFormatError: If the file is not a PDF.
        """
        path = Path(path)

        # Validate file exists
        if not path.exists():
            raise LoadError(f"File not found: {path}")

        # Validate extension
        if path.suffix.lower() not in self.supported_extensions:
            raise UnsupportedFormatError(
                f"Unsupported format: {path.suffix}. Expected PDF."
            )

        try:
            from markitdown import MarkItDown
        except ImportError:
            raise LoadError(
                "markitdown is not installed. "
                "Install with: pip install markitdown"
            )

        # Generate document ID from file hash first (needed for image directory)
        file_hash = self._compute_file_hash(path)
        doc_id = f"pdf:{file_hash}"

        try:
            md = MarkItDown()
            result = md.convert(str(path))

            # Extract markdown content
            text_content = result.text_content

            # Build metadata
            metadata: dict[str, Any] = {
                "source_path": str(path.absolute()),
                "doc_type": "pdf",
                "file_name": path.name,
                "file_size": path.stat().st_size,
                "modified_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            }

            # Extract images if enabled
            images_metadata: list[dict[str, Any]] = []
            if self._extract_images:
                try:
                    text_content, images_metadata = self._extract_and_process_images(
                        pdf_path=path,
                        text_content=text_content,
                        doc_hash=file_hash,
                    )
                except Exception as e:
                    logger.warning(
                        f"Image extraction failed for {path}: {e}. "
                        "Continuing with text-only parsing."
                    )

            if images_metadata:
                metadata["images"] = images_metadata

            return Document(
                id=doc_id,
                text=text_content,
                metadata=metadata,
            )

        except LoadError:
            raise
        except Exception as e:
            raise LoadError(f"Failed to parse PDF: {e}")

    def _extract_and_process_images(
        self,
        pdf_path: Path,
        text_content: str,
        doc_hash: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Extract images from PDF and insert placeholders.

        Uses PyMuPDF (fitz) to extract images, save them to disk, and insert
        placeholders in the text content.

        Args:
            pdf_path: Path to PDF file.
            text_content: Extracted text content.
            doc_hash: Document hash for image directory.

        Returns:
            Tuple of (modified_text, images_metadata_list)
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.warning(f"PyMuPDF not available, skipping image extraction for {pdf_path}")
            return text_content, []

        images_metadata: list[dict[str, Any]] = []
        modified_text = text_content

        # Create image directory
        image_dir = self._image_storage_dir / doc_hash
        image_dir.mkdir(parents=True, exist_ok=True)

        # Open PDF with PyMuPDF
        doc = fitz.open(str(pdf_path))

        image_count = 0
        for page_num, page in enumerate(doc):
            # Get image list from page
            image_list = page.get_images(full=True)

            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                if base_image is None:
                    continue

                image_ext = base_image.get("ext", "png")
                image_filename = f"image_{image_count}.{image_ext}"
                image_path = image_dir / image_filename

                # Save image
                with open(image_path, "wb") as f:
                    f.write(base_image["image"])

                # Create placeholder and calculate text_offset
                placeholder = f"\n[IMAGE: image_{image_count}]\n"
                text_offset = len(modified_text)
                text_length = len(placeholder)

                # Create image metadata following C1 spec
                img_metadata = {
                    "id": f"image_{image_count}",
                    "path": str(image_path),
                    "page": page_num + 1,
                    "text_offset": text_offset,
                    "text_length": text_length,
                    "position": {
                        "width": base_image.get("width"),
                        "height": base_image.get("height"),
                    },
                }
                images_metadata.append(img_metadata)

                # Insert placeholder in text
                modified_text += placeholder

                image_count += 1

        doc.close()

        if images_metadata:
            logger.info(
                f"Extracted {len(images_metadata)} images from {pdf_path.name} "
                f"to {image_dir}"
            )

        return modified_text, images_metadata

    def _compute_file_hash(self, path: Path) -> str:
        """Compute SHA256 hash of the file for ID generation.

        Args:
            path: Path to the file.

        Returns:
            Hexadecimal hash string (first 16 chars).
        """
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:16]
