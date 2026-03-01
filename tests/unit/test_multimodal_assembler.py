"""Unit tests for multimodal_assembler.

These tests verify the MultimodalAssembler functionality.
"""

import base64
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.response.multimodal_assembler import (
    ImageContent,
    ImageReference,
    MultimodalAssembler,
    create_multimodal_assembler,
)
from core.types import RetrievalResult


class TestImageReference:
    """Tests for ImageReference dataclass."""

    def test_image_reference_creation(self):
        """Test creating an ImageReference instance."""
        ref = ImageReference(
            image_id="img_001",
            file_path="/path/to/image.png",
            page=1,
            text_offset=100,
            text_length=15,
            caption="A chart",
        )
        assert ref.image_id == "img_001"
        assert ref.file_path == "/path/to/image.png"
        assert ref.page == 1
        assert ref.caption == "A chart"

    def test_image_reference_to_dict(self):
        """Test ImageReference to_dict method."""
        ref = ImageReference(
            image_id="img_001",
            file_path="/path/to/image.png",
            caption="A chart",
        )
        result = ref.to_dict()
        assert result["image_id"] == "img_001"
        assert result["file_path"] == "/path/to/image.png"
        assert result["caption"] == "A chart"


class TestImageContent:
    """Tests for ImageContent dataclass."""

    def test_image_content_creation(self):
        """Test creating an ImageContent instance."""
        img = ImageContent(
            image_id="img_001",
            data="SGVsbG8gV29ybGQ=",
            mime_type="image/png",
            caption="A test image",
        )
        assert img.image_id == "img_001"
        assert img.data == "SGVsbG8gV29ybGQ="
        assert img.mime_type == "image/png"
        assert img.caption == "A test image"

    def test_image_content_to_mcp_content(self):
        """Test conversion to MCP ImageContent."""
        from mcp import types

        img = ImageContent(
            image_id="img_001",
            data="SGVsbG8gV29ybGQ=",
            mime_type="image/png",
        )
        mcp_content = img.to_mcp_content()
        assert isinstance(mcp_content, types.ImageContent)
        assert mcp_content.type == "image"
        assert mcp_content.data == "SGVsbG8gV29ybGQ="
        assert mcp_content.mimeType == "image/png"


class TestMultimodalAssembler:
    """Tests for MultimodalAssembler."""

    def test_assembler_creation(self):
        """Test creating a MultimodalAssembler instance."""
        assembler = MultimodalAssembler()
        assert assembler is not None
        assert assembler.max_images_per_result == 5
        assert assembler.include_captions is True

    def test_assembler_with_custom_settings(self):
        """Test creating with custom settings."""
        assembler = MultimodalAssembler(
            max_images_per_result=10,
            include_captions=False,
        )
        assert assembler.max_images_per_result == 10
        assert assembler.include_captions is False

    def test_extract_image_refs_from_metadata(self):
        """Test extracting image references from metadata."""
        assembler = MultimodalAssembler()
        result = RetrievalResult(
            chunk_id="chunk1",
            score=0.9,
            text="Some text",
            metadata={
                "images": [
                    {"id": "img1", "path": "/images/chart.png", "page": 1},
                    {"id": "img2", "path": "/images/graph.png"},
                ],
                "image_captions": {"img1": "A chart showing growth"},
            },
        )

        refs = assembler.extract_image_refs(result)
        assert len(refs) == 2
        assert refs[0].image_id == "img1"
        assert refs[0].file_path == "/images/chart.png"
        assert refs[0].page == 1
        assert refs[0].caption == "A chart showing growth"
        assert refs[1].image_id == "img2"

    def test_extract_image_refs_from_text_placeholder(self):
        """Test extracting from text placeholder."""
        assembler = MultimodalAssembler()
        result = RetrievalResult(
            chunk_id="chunk1",
            score=0.9,
            text="Here is a chart: [IMAGE: img1] and another [IMAGE: img2]",
            metadata={},
        )

        refs = assembler.extract_image_refs(result)
        assert len(refs) == 2
        assert refs[0].image_id == "img1"
        assert refs[1].image_id == "img2"

    def test_has_images(self):
        """Test has_images method."""
        assembler = MultimodalAssembler()

        # With images
        result_with_images = RetrievalResult(
            chunk_id="chunk1",
            score=0.9,
            text="text",
            metadata={"images": [{"id": "img1"}]},
        )
        assert assembler.has_images(result_with_images) is True

        # Without images
        result_without_images = RetrievalResult(
            chunk_id="chunk1",
            score=0.9,
            text="text",
            metadata={},
        )
        assert assembler.has_images(result_without_images) is False

    def test_count_images(self):
        """Test count_images method."""
        assembler = MultimodalAssembler()
        results = [
            RetrievalResult(
                chunk_id="chunk1",
                score=0.9,
                text="text",
                metadata={"images": [{"id": "img1"}]},
            ),
            RetrievalResult(
                chunk_id="chunk2",
                score=0.9,
                text="text",
                metadata={"images": [{"id": "img2"}, {"id": "img3"}]},
            ),
        ]

        total = assembler.count_images(results)
        assert total == 3

    def test_resolve_image_path_explicit(self):
        """Test resolving path with explicit file_path."""
        assembler = MultimodalAssembler()
        ref = ImageReference(
            image_id="img1",
            file_path="/tmp/test.png",
        )

        with patch("pathlib.Path.exists", return_value=True):
            path = assembler.resolve_image_path(ref)
            assert path is not None
            assert path.endswith("test.png")

    def test_load_image(self):
        """Test loading an image file."""
        assembler = MultimodalAssembler()

        # Mock Path operations
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.read_bytes", return_value=b"fake image data"):
                with patch("pathlib.Path.suffix", ".png"):
                    content = assembler.load_image("/tmp/test.png")
                    assert content is not None
                    assert content.image_id == "test"
                    # Verify it's base64 encoded
                    decoded = base64.b64decode(content.data)
                    assert decoded == b"fake image data"

    def test_assemble_for_result_with_images(self):
        """Test assembling content for a result with images."""
        assembler = MultimodalAssembler()
        result = RetrievalResult(
            chunk_id="chunk1",
            score=0.9,
            text="Here is a chart",
            metadata={
                "images": [{"id": "img1", "path": "/tmp/test.png"}],
                "image_captions": {"img1": "A chart"},
            },
        )

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.read_bytes", return_value=b"fake"):
                with patch("pathlib.Path.suffix", ".png"):
                    blocks = assembler.assemble_for_result(result)
                    # Should have image block + caption text block
                    assert len(blocks) == 2

    def test_assemble_deduplicates(self):
        """Test that assemble deduplicates images."""
        assembler = MultimodalAssembler()
        results = [
            RetrievalResult(
                chunk_id="chunk1",
                score=0.9,
                text="text",
                metadata={"images": [{"id": "img1", "path": "/tmp/test.png"}]},
            ),
            RetrievalResult(
                chunk_id="chunk2",
                score=0.9,
                text="text",
                metadata={"images": [{"id": "img1", "path": "/tmp/test.png"}]},
            ),
        ]

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.read_bytes", return_value=b"fake data that is long enough"):
                with patch("pathlib.Path.suffix", ".png"):
                    blocks = assembler.assemble(results)
                    # Should only have one image (deduplicated)
                    image_count = sum(1 for b in blocks if hasattr(b, 'type') and b.type == 'image')
                    assert image_count == 1


class TestCreateMultimodalAssembler:
    """Tests for create_multimodal_assembler factory function."""

    def test_create_with_defaults(self):
        """Test creating with default settings."""
        assembler = create_multimodal_assembler()
        assert isinstance(assembler, MultimodalAssembler)
        assert assembler.max_images_per_result == 5
        assert assembler.include_captions is True

    def test_create_with_custom_settings(self):
        """Test creating with custom settings."""
        assembler = create_multimodal_assembler(
            max_images_per_result=10,
            include_captions=False,
        )
        assert isinstance(assembler, MultimodalAssembler)
        assert assembler.max_images_per_result == 10
        assert assembler.include_captions is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
