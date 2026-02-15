"""Base Loader interface for document extraction.

This module defines the abstract base class for all document loaders.
Each loader is responsible for extracting text and metadata from a specific
file format and producing a standardized Document.

Design Principles:
    - Abstract Interface: BaseLoader defines the contract
    - Factory Compatible: Can be used with LoaderFactory pattern
    - Metadata Enrichment: Each loader should extract relevant metadata
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from core.types import Document


class LoaderError(Exception):
    """Base exception for loader operations."""

    pass


class LoadError(LoaderError):
    """Failed to load document."""

    pass


class UnsupportedFormatError(LoaderError):
    """File format not supported by this loader."""

    pass


class BaseLoader(ABC):
    """Abstract base class for document loaders.

    All document loaders should inherit from this class and implement
    the `load` method. Loaders are responsible for:
    1. Reading file content from the specified path
    2. Extracting relevant metadata (source_path, page numbers, etc.)
    3. Producing a standardized Document object

    Attributes:
        supported_extensions: List of file extensions this loader supports.
    """

    # Override in subclasses
    supported_extensions: list[str] = []

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this loader provider.

        Returns:
            Provider identifier (e.g., 'pdf', 'markdown', 'text')
        """
        pass

    @abstractmethod
    def load(self, path: str | Path) -> Document:
        """Load a document from the specified path.

        Args:
            path: Path to the file to load.

        Returns:
            Document object containing text content and metadata.

        Raises:
            LoadError: If the file cannot be read or parsed.
            UnsupportedFormatError: If the file format is not supported.
        """
        pass

    def can_load(self, path: str | Path) -> bool:
        """Check if this loader can handle the given file.

        Args:
            path: Path to check.

        Returns:
            True if the file extension is supported.
        """
        path = Path(path)
        ext = path.suffix.lower()
        return ext in self.supported_extensions

    @abstractmethod
    def get_supported_extensions(self) -> list[str]:
        """Get the list of supported file extensions.

        Returns:
            List of file extensions (with dot, e.g., ['.pdf', '.txt'])
        """
        pass
