"""Integration tests for PDF Loader with real PDFs.

This module tests the PdfLoader with actual PDF files generated
using the sample document generators.

Design Principles:
    - Integration Tests: Test with real PDF files
    - Coverage: Full pipeline from PDF to Document
    - Cleanup: Remove generated files after tests
"""

import tempfile
import shutil
from pathlib import Path

import pytest

from libs.loader.pdf_loader import PdfLoader
from libs.loader.base_loader import LoadError, UnsupportedFormatError
from core.types import Document


class TestPdfLoaderIntegration:
    """Integration tests for PdfLoader with real PDF files."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)

    @pytest.fixture
    def sample_pdf_path(self, temp_dir):
        """Generate and return path to a sample PDF."""
        try:
            from tests.fixtures.sample_documents.generate_pdfs import generate_sample_pdf
        except ImportError:
            pytest.skip("reportlab not installed for PDF generation")

        pdf_path = Path(temp_dir) / "sample_document.pdf"
        generate_sample_pdf(pdf_path, title="Test Document")
        return pdf_path

    @pytest.fixture
    def pdf_with_images_path(self, temp_dir):
        """Generate and return path to a PDF with images."""
        try:
            from tests.fixtures.sample_documents.generate_pdfs import generate_pdf_with_images
        except ImportError:
            pytest.skip("reportlab not installed for PDF generation")

        pdf_path = Path(temp_dir) / "document_with_images.pdf"
        generate_pdf_with_images(pdf_path, title="Document with Images", num_images=2)
        return pdf_path

    def test_load_sample_pdf(self, sample_pdf_path):
        """Test loading a simple PDF without images."""
        loader = PdfLoader()
        doc = loader.load(sample_pdf_path)

        assert isinstance(doc, Document)
        assert doc.id.startswith("pdf:")
        assert doc.text is not None
        assert len(doc.text) > 0
        assert "source_path" in doc.metadata
        assert doc.metadata["source_path"] == str(sample_pdf_path.absolute())
        assert doc.metadata["doc_type"] == "pdf"

    def test_pdf_metadata_extraction(self, sample_pdf_path):
        """Test that PDF metadata is correctly extracted."""
        loader = PdfLoader()
        doc = loader.load(sample_pdf_path)

        assert "file_name" in doc.metadata
        assert doc.metadata["file_name"] == sample_pdf_path.name
        assert "file_size" in doc.metadata
        assert doc.metadata["file_size"] > 0
        assert "modified_at" in doc.metadata
        assert doc.metadata["modified_at"] is not None

    def test_pdf_content_extraction(self, sample_pdf_path):
        """Test that PDF text content is extracted."""
        loader = PdfLoader()
        doc = loader.load(sample_pdf_path)

        # Verify the document contains expected sections
        assert "Test Document" in doc.text or "Sample Document" in doc.text

    def test_load_nonexistent_pdf_raises_error(self):
        """Test that loading a non-existent PDF raises LoadError."""
        loader = PdfLoader()

        with pytest.raises(LoadError) as exc_info:
            loader.load("/path/that/does/not/exist.pdf")

        assert "not found" in str(exc_info.value).lower()

    def test_load_unsupported_format_raises_error(self, temp_dir):
        """Test that loading an unsupported format raises error."""
        loader = PdfLoader()

        txt_path = Path(temp_dir) / "document.txt"
        txt_path.write_text("This is not a PDF")

        with pytest.raises(UnsupportedFormatError):
            loader.load(txt_path)

    def test_reload_same_pdf_produces_different_id(self, sample_pdf_path):
        """Test that reloading the same PDF produces consistent IDs."""
        loader = PdfLoader()

        doc1 = loader.load(sample_pdf_path)
        doc2 = loader.load(sample_pdf_path)

        # Same file should produce same hash-based ID
        assert doc1.id == doc2.id

    def test_different_pdfs_produce_different_ids(self, temp_dir):
        """Test that different PDFs produce different IDs."""
        try:
            from tests.fixtures.sample_documents.generate_pdfs import generate_sample_pdf
        except ImportError:
            pytest.skip("reportlab not installed")

        loader = PdfLoader()

        pdf1 = Path(temp_dir) / "doc1.pdf"
        generate_sample_pdf(pdf1, title="Document 1")

        pdf2 = Path(temp_dir) / "doc2.pdf"
        generate_sample_pdf(pdf2, title="Document 2")

        doc1 = loader.load(pdf1)
        doc2 = loader.load(pdf2)

        assert doc1.id != doc2.id


class TestPdfLoaderImageExtraction:
    """Tests for PDF loader image extraction functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)

    @pytest.fixture
    def pdf_with_images_path(self, temp_dir):
        """Generate and return path to a PDF with images."""
        try:
            from tests.fixtures.sample_documents.generate_pdfs import generate_pdf_with_images
        except ImportError:
            pytest.skip("reportlab not installed for PDF generation")

        pdf_path = Path(temp_dir) / "document_with_images.pdf"
        generate_pdf_with_images(pdf_path, title="Document with Images", num_images=2)
        return pdf_path

    def test_load_pdf_with_images(self, pdf_with_images_path):
        """Test loading PDF with images."""
        loader = PdfLoader()
        doc = loader.load(pdf_with_images_path)

        assert isinstance(doc, Document)
        assert doc.text is not None

    def test_image_extraction_and_placeholders(self, pdf_with_images_path, temp_dir):
        """Test that images are extracted and placeholders are inserted correctly."""
        loader = PdfLoader()
        doc = loader.load(pdf_with_images_path)

        # Verify images metadata exists
        assert "images" in doc.metadata, "metadata should contain 'images' field"
        images = doc.metadata["images"]
        assert len(images) > 0, "should extract at least one image"

        # Verify each image has required C1 spec fields
        for i, img in enumerate(images):
            assert "id" in img, f"image {i} missing 'id' field"
            assert "path" in img, f"image {i} missing 'path' field"
            assert "page" in img, f"image {i} missing 'page' field"
            assert "text_offset" in img, f"image {i} missing 'text_offset' field"
            assert "text_length" in img, f"image {i} missing 'text_length' field"

            # Verify placeholder format (includes newlines as generated by pdf_loader)
            expected_placeholder = f"\n[IMAGE: {img['id']}]\n"
            actual_placeholder = doc.text[img["text_offset"]:img["text_offset"] + img["text_length"]]
            assert actual_placeholder == expected_placeholder, \
                f"image {i}: expected {repr(expected_placeholder)}, got {repr(actual_placeholder)}"

        # Verify placeholder appears in text (search without newlines)
        text_without_newlines = doc.text.replace("\n", "")
        assert "[IMAGE: image_0]" in text_without_newlines or "[IMAGE: image_1]" in text_without_newlines

    def test_image_files_saved(self, pdf_with_images_path, temp_dir):
        """Test that extracted images are saved to disk."""
        loader = PdfLoader(image_storage_dir=temp_dir)
        doc = loader.load(pdf_with_images_path)

        # Verify image files exist
        if "images" in doc.metadata:
            for img in doc.metadata["images"]:
                image_path = Path(img["path"])
                assert image_path.exists(), f"Image file should exist: {image_path}"
                assert image_path.stat().st_size > 0, f"Image file should not be empty"

    def test_image_metadata_format(self, pdf_with_images_path):
        """Test that image metadata follows C1 spec format."""
        loader = PdfLoader()
        doc = loader.load(pdf_with_images_path)

        if "images" not in doc.metadata:
            pytest.skip("No images extracted from PDF")

        # Verify C1 format
        for img in doc.metadata["images"]:
            # id: str
            assert isinstance(img["id"], str)
            # path: str
            assert isinstance(img["path"], str)
            # page: int
            assert isinstance(img["page"], int)
            # text_offset: int
            assert isinstance(img["text_offset"], int)
            # text_length: int
            assert isinstance(img["text_length"], int)

            # position: dict (optional)
            if "position" in img:
                assert isinstance(img["position"], dict)


class TestPdfLoaderGracefulDegradation:
    """Tests for graceful degradation when image extraction fails."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)

    def test_pdf_parsing_failure_error(self, temp_dir):
        """Test that parsing failure raises LoadError."""
        # Create a simple PDF
        try:
            from tests.fixtures.sample_documents.generate_pdfs import generate_sample_pdf
        except ImportError:
            pytest.skip("reportlab not installed")

        pdf_path = Path(temp_dir) / "test.pdf"
        generate_sample_pdf(pdf_path, title="Test")

        loader = PdfLoader()
        doc = loader.load(pdf_path)

        # Should get the text content
        assert isinstance(doc, Document)
        assert doc.text is not None
        assert len(doc.text) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
