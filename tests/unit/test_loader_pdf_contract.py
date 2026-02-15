"""Unit tests for PDF Loader contract.

This module tests the BaseLoader abstract class and PdfLoader implementation.

Design Principles:
    - Contract Testing: Verify interface compliance
    - Mock-based: No actual PDF parsing in unit tests
    - Coverage: Initialization, load, metadata extraction, error handling
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from libs.loader.base_loader import (
    BaseLoader,
    LoaderError,
    LoadError,
    UnsupportedFormatError,
)
from libs.loader.pdf_loader import PdfLoader
from core.types import Document


def create_mock_markitdown(text_content: str, page_count: int | None = None):
    """Helper to create a mock MarkItDown module and class.

    Returns a tuple of (mock_result, mock_module) that can be used with mock_markitdown_module.
    """
    mock_result = MagicMock()
    mock_result.text_content = text_content
    mock_result.page_count = page_count
    mock_result.images = []

    mock_md_instance = MagicMock()
    mock_md_instance.convert.return_value = mock_result

    # MarkItDown class that returns our mock instance
    mock_md_class = MagicMock(return_value=mock_md_instance)

    # Create a mock module with MarkItDown attribute
    mock_module = MagicMock()
    mock_module.MarkItDown = mock_md_class

    return mock_result, mock_module


class MockMarkItDown:
    """Context manager to mock the markitdown module."""

    def __init__(self, mock_module):
        self.mock_module = mock_module

    def __enter__(self):
        # Store original module if it exists
        self._original = sys.modules.get("markitdown", None)
        # Replace with mock
        sys.modules["markitdown"] = self.mock_module
        return self.mock_module

    def __exit__(self, *args):
        # Restore original module
        if self._original is not None:
            sys.modules["markitdown"] = self._original
        else:
            sys.modules.pop("markitdown", None)


class TestBaseLoaderInterface:
    """Tests for BaseLoader abstract interface."""

    def test_base_loader_is_abstract(self):
        """BaseLoader cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseLoader()

    def test_base_loader_has_required_methods(self):
        """BaseLoader should have required abstract methods."""
        assert hasattr(BaseLoader, 'load')
        assert hasattr(BaseLoader, 'can_load')
        assert hasattr(BaseLoader, 'get_supported_extensions')
        assert hasattr(BaseLoader, 'provider_name')

    def test_concrete_loader_inherits_correctly(self):
        """PdfLoader should properly inherit from BaseLoader."""
        loader = PdfLoader()
        assert isinstance(loader, BaseLoader)


class TestPdfLoader:
    """Tests for PdfLoader implementation."""

    def test_initialization(self):
        """Test PdfLoader initialization."""
        loader = PdfLoader()

        assert loader.provider_name == "pdf"
        assert ".pdf" in loader.supported_extensions

    def test_initialization_with_default_params(self):
        """Test PdfLoader initialization with default parameters."""
        loader = PdfLoader()

        assert loader.provider_name == "pdf"
        assert loader._extract_images is True
        assert str(loader._image_storage_dir) == "data/images"

    def test_initialization_with_custom_params(self):
        """Test PdfLoader initialization with custom parameters."""
        loader = PdfLoader(extract_images=False, image_storage_dir="/custom/images")

        assert loader._extract_images is False
        assert str(loader._image_storage_dir) == "/custom/images"

    def test_get_supported_extensions(self):
        """Test supported extensions return."""
        loader = PdfLoader()
        exts = loader.get_supported_extensions()

        assert isinstance(exts, list)
        assert ".pdf" in exts

    def test_can_load_pdf(self):
        """Test can_load identifies PDF files."""
        loader = PdfLoader()

        assert loader.can_load("test.pdf") is True
        assert loader.can_load("document.PDF") is True
        assert loader.can_load("/path/to/file.pdf") is True

    def test_cannot_load_non_pdf(self):
        """Test can_load rejects non-PDF files."""
        loader = PdfLoader()

        assert loader.can_load("test.txt") is False
        assert loader.can_load("document.md") is False
        assert loader.can_load("image.png") is False

    def test_load_nonexistent_file_raises_error(self):
        """Test loading non-existent file raises LoadError."""
        loader = PdfLoader()

        with pytest.raises(LoadError) as exc_info:
            loader.load("/path/that/does/not/exist.pdf")

        assert "not found" in str(exc_info.value).lower()

    def test_load_unsupported_format_raises_error(self):
        """Test loading unsupported format raises UnsupportedFormatError."""
        loader = PdfLoader()

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test content")
            path = f.name

        try:
            with pytest.raises(UnsupportedFormatError):
                loader.load(path)
        finally:
            Path(path).unlink()


class TestPdfLoaderMetadata:
    """Tests for PDF loader metadata extraction."""

    def test_metadata_contains_source_path(self):
        """Verify metadata contains source_path as required."""
        _, mock_module = create_mock_markitdown("# Test Document", page_count=1)

        with MockMarkItDown(mock_module):
            loader = PdfLoader()
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(b"fake pdf content")
                path = f.name

            try:
                doc = loader.load(path)

                assert "source_path" in doc.metadata
                assert doc.metadata["source_path"] == str(Path(path).absolute())
            finally:
                Path(path).unlink()

    def test_metadata_contains_doc_type(self):
        """Verify metadata contains doc_type."""
        _, mock_module = create_mock_markitdown("# Test Document")

        with MockMarkItDown(mock_module):
            loader = PdfLoader()
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(b"fake pdf")
                path = f.name

            try:
                doc = loader.load(path)

                assert "doc_type" in doc.metadata
                assert doc.metadata["doc_type"] == "pdf"
            finally:
                Path(path).unlink()

    def test_metadata_contains_file_name(self):
        """Verify metadata contains file_name."""
        _, mock_module = create_mock_markitdown("# Test Document")

        with MockMarkItDown(mock_module):
            loader = PdfLoader()
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(b"fake pdf")
                path = f.name

            try:
                doc = loader.load(path)

                assert "file_name" in doc.metadata
                assert doc.metadata["file_name"] == Path(path).name
            finally:
                Path(path).unlink()


class TestPdfLoaderDocument:
    """Tests for PDF loader Document output."""

    def test_document_has_id(self):
        """Verify loaded document has an ID."""
        _, mock_module = create_mock_markitdown("# Test Document", page_count=1)

        with MockMarkItDown(mock_module):
            loader = PdfLoader()
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(b"fake pdf content")
                path = f.name

            try:
                doc = loader.load(path)

                assert doc.id is not None
                assert len(doc.id) > 0
                assert doc.id.startswith("pdf:")
            finally:
                Path(path).unlink()

    def test_document_has_text(self):
        """Verify loaded document has text content."""
        expected_text = "# Test Document\nSome markdown content"
        _, mock_module = create_mock_markitdown(expected_text)

        with MockMarkItDown(mock_module):
            loader = PdfLoader()
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(b"fake pdf")
                path = f.name

            try:
                doc = loader.load(path)

                assert doc.text == expected_text
            finally:
                Path(path).unlink()

    def test_document_has_metadata(self):
        """Verify loaded document has metadata dict."""
        _, mock_module = create_mock_markitdown("# Test Document")

        with MockMarkItDown(mock_module):
            loader = PdfLoader()
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(b"fake pdf")
                path = f.name

            try:
                doc = loader.load(path)

                assert isinstance(doc.metadata, dict)
            finally:
                Path(path).unlink()


class TestPdfLoaderErrorHandling:
    """Tests for PDF loader error handling."""

    def test_load_error_on_parse_failure(self):
        """Test LoadError raised when parsing fails."""
        mock_md_instance = MagicMock()
        mock_md_instance.convert.side_effect = Exception("Parse error")
        mock_md_class = MagicMock(return_value=mock_md_instance)

        mock_module = MagicMock()
        mock_module.MarkItDown = mock_md_class

        with MockMarkItDown(mock_module):
            loader = PdfLoader()
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(b"fake pdf")
                path = f.name

            try:
                with pytest.raises(LoadError) as exc_info:
                    loader.load(path)

                assert "Failed to parse PDF" in str(exc_info.value)
            finally:
                Path(path).unlink()


class TestPdfLoaderFactoryCompatible:
    """Tests for factory compatibility."""

    def test_provider_name_for_factory(self):
        """Verify provider_name is suitable for factory registration."""
        loader = PdfLoader()

        # Provider name should be a valid identifier string
        assert isinstance(loader.provider_name, str)
        assert len(loader.provider_name) > 0
        assert " " not in loader.provider_name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
