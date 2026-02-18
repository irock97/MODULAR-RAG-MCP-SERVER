"""Unit tests for ImageCaptioner.

This module tests the ImageCaptioner class with mock Vision LLM responses.
It covers caption generation, fallback scenarios, and metadata handling.

Design Principles:
    - Mock-based: Uses unittest.mock for Vision LLM client
    - Contract Testing: Verify interface compliance
    - Coverage: Caption generation, fallback, error handling, metadata
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.image_captioner import ImageCaptioner
from libs.llm.base_vision_llm import VisionResponse


class MockVisionLLM:
    """A mock Vision LLM for testing."""

    def __init__(self, response_content: str = "") -> None:
        self._response_content = response_content
        self.call_count = 0
        self.last_messages = None
        self.last_image = None

    @property
    def provider_name(self) -> str:
        return "mock-vision"

    @property
    def supported_formats(self) -> list[str]:
        return ["image/png", "image/jpeg"]

    def chat_with_image(
        self,
        text: str,
        image: str | bytes,
        trace: Any = None,
        **kwargs: Any
    ) -> VisionResponse:
        self.call_count += 1
        self.last_messages = text
        self.last_image = image

        return VisionResponse(
            content=self._response_content,
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )


class TestImageCaptioner:
    """Tests for ImageCaptioner class."""

    def test_initialization_disabled_by_default(self):
        """Test initialization when captioning is disabled."""
        settings = Settings()
        settings.ingestion = MagicMock()
        settings.ingestion.image_captioner = MagicMock()
        settings.ingestion.image_captioner.enabled = False

        captioner = ImageCaptioner(settings)

        assert captioner.enabled is False

    def test_initialization_enabled(self):
        """Test initialization when captioning is enabled."""
        settings = Settings()
        settings.ingestion = MagicMock()
        settings.ingestion.image_captioner = MagicMock()
        settings.ingestion.image_captioner.enabled = True

        mock_vision = MockVisionLLM("A test caption")
        captioner = ImageCaptioner(settings, vision_llm=mock_vision)

        assert captioner.enabled is True

    def test_initialization_no_config(self):
        """Test initialization when image_captioner config doesn't exist."""
        settings = Settings()
        # No ingestion.image_captioner attribute

        captioner = ImageCaptioner(settings)

        assert captioner.enabled is False


class TestImageCaptionerTransform:
    """Tests for transform method."""

    def test_transform_empty_list(self):
        """Test transforming empty chunk list."""
        settings = Settings()
        settings.ingestion = MagicMock()
        settings.ingestion.image_captioner = MagicMock()
        settings.ingestion.image_captioner.enabled = True

        captioner = ImageCaptioner(settings)

        result = captioner.transform([])

        assert result == []

    def test_transform_chunk_without_images(self):
        """Test processing chunk without images."""
        settings = Settings()
        settings.ingestion = MagicMock()
        settings.ingestion.image_captioner = MagicMock()
        settings.ingestion.image_captioner.enabled = True

        mock_vision = MockVisionLLM("Caption")
        captioner = ImageCaptioner(settings, vision_llm=mock_vision)

        chunk = Chunk(
            id="test_0001_abc12345",
            text="Some text without images",
            metadata={},
            source_ref="doc_001",
        )

        result = captioner.transform([chunk])

        assert len(result) == 1
        assert result[0].id == chunk.id
        assert "image_captioning" in result[0].metadata
        assert result[0].metadata["image_captioning"]["enabled"] is True
        assert result[0].metadata["image_captioning"]["captions_count"] == 0

    def test_transform_generates_captions(self, tmp_path):
        """Test caption generation for chunk with images."""
        settings = Settings()
        settings.ingestion = MagicMock()
        settings.ingestion.image_captioner = MagicMock()
        settings.ingestion.image_captioner.enabled = True

        # Create a real temp image file for testing
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        image_file = image_dir / "test_image.png"
        image_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        mock_vision = MockVisionLLM("This is a chart showing sales growth over time.")
        captioner = ImageCaptioner(settings, vision_llm=mock_vision)

        chunk = Chunk(
            id="test_0001_abc12345",
            text="See the chart below:",
            metadata={
                "images": [
                    {
                        "id": "doc_hash_001_page_0_1",
                        "path": str(image_file),
                    }
                ]
            },
            source_ref="doc_001",
        )

        result = captioner.transform([chunk])

        assert len(result) == 1
        enriched = result[0]
        assert enriched.metadata["image_captions"]["doc_hash_001_page_0_1"] == "This is a chart showing sales growth over time."
        assert enriched.metadata["image_captioning"]["captions_count"] == 1
        assert enriched.metadata["image_captioning"]["vision_provider"] == "mock-vision"

    def test_transform_multiple_images(self, tmp_path):
        """Test caption generation for chunk with multiple images."""
        settings = Settings()
        settings.ingestion = MagicMock()
        settings.ingestion.image_captioner = MagicMock()
        settings.ingestion.image_captioner.enabled = True

        # Create real temp image files
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        image_file1 = image_dir / "img_1.png"
        image_file1.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        image_file2 = image_dir / "img_2.png"
        image_file2.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        mock_vision = MockVisionLLM("Diagram showing system architecture flow.")
        captioner = ImageCaptioner(settings, vision_llm=mock_vision)

        chunk = Chunk(
            id="test_0001_abc12345",
            text="The system consists of several components:",
            metadata={
                "images": [
                    {
                        "id": "img_1",
                        "path": str(image_file1),
                    },
                    {
                        "id": "img_2",
                        "path": str(image_file2),
                    },
                ]
            },
            source_ref="doc_001",
        )

        result = captioner.transform([chunk])

        assert len(result) == 1
        captions = result[0].metadata.get("image_captions", {})
        assert len(captions) == 2
        assert "img_1" in captions
        assert "img_2" in captions

    def test_transform_disabled_marks_unprocessed(self):
        """Test that disabled captioning marks chunks with images."""
        settings = Settings()
        settings.ingestion = MagicMock()
        settings.ingestion.image_captioner = MagicMock()
        settings.ingestion.image_captioner.enabled = False

        captioner = ImageCaptioner(settings)

        chunk = Chunk(
            id="test_0001_abc12345",
            text="See the chart:",
            metadata={
                "images": [
                    {"id": "img_1", "path": "/test/images/img_1.png"}
                ]
            },
            source_ref="doc_001",
        )

        result = captioner.transform([chunk])

        assert len(result) == 1
        assert result[0].metadata.get("has_unprocessed_images") is True
        # Check failed_count in image_captioning metadata
        assert result[0].metadata["image_captioning"]["failed_count"] == 1


class TestImageCaptionerFallback:
    """Tests for fallback scenarios."""

    def test_fallback_on_vision_llm_unavailable(self):
        """Test fallback when Vision LLM is not available."""
        settings = Settings()
        settings.ingestion = MagicMock()
        settings.ingestion.image_captioner = MagicMock()
        settings.ingestion.image_captioner.enabled = True

        # Vision LLM that returns None (unavailable)
        captioner = ImageCaptioner(settings, vision_llm=None)

        chunk = Chunk(
            id="test_0001_abc12345",
            text="Image description:",
            metadata={
                "images": [{"id": "img_1", "path": "/test/images/img_1.png"}]
            },
            source_ref="doc_001",
        )

        result = captioner.transform([chunk])

        assert len(result) == 1
        assert result[0].metadata.get("has_unprocessed_images") is True

    def test_fallback_on_missing_image_path(self):
        """Test fallback when image path is missing."""
        settings = Settings()
        settings.ingestion = MagicMock()
        settings.ingestion.image_captioner = MagicMock()
        settings.ingestion.image_captioner.enabled = True

        mock_vision = MockVisionLLM("Caption")
        captioner = ImageCaptioner(settings, vision_llm=mock_vision)

        chunk = Chunk(
            id="test_0001_abc12345",
            text="Image here:",
            metadata={
                "images": [{"id": "img_1"}]  # No path
            },
            source_ref="doc_001",
        )

        result = captioner.transform([chunk])

        assert len(result) == 1
        captions = result[0].metadata.get("image_captions", {})
        assert "img_1" not in captions

    def test_fallback_on_missing_image_file(self):
        """Test fallback when image file doesn't exist."""
        settings = Settings()
        settings.ingestion = MagicMock()
        settings.ingestion.image_captioner = MagicMock()
        settings.ingestion.image_captioner.enabled = True

        mock_vision = MockVisionLLM("Caption")
        captioner = ImageCaptioner(settings, vision_llm=mock_vision)

        chunk = Chunk(
            id="test_0001_abc12345",
            text="Image here:",
            metadata={
                "images": [{"id": "img_1", "path": "/nonexistent/img_1.png"}]
            },
            source_ref="doc_001",
        )

        result = captioner.transform([chunk])

        assert len(result) == 1
        captions = result[0].metadata.get("image_captions", {})
        assert "img_1" not in captions

    def test_fallback_on_vision_llm_error(self, tmp_path):
        """Test fallback when Vision LLM raises an error."""
        settings = Settings()
        settings.ingestion = MagicMock()
        settings.ingestion.image_captioner = MagicMock()
        settings.ingestion.image_captioner.enabled = True

        # Create a real temp image file
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        image_file = image_dir / "img_1.png"
        image_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        # Vision LLM that raises an exception
        mock_vision = MagicMock()
        mock_vision.provider_name = "mock"
        mock_vision.chat_with_image.side_effect = Exception("API error")

        captioner = ImageCaptioner(settings, vision_llm=mock_vision)

        chunk = Chunk(
            id="test_0001_abc12345",
            text="Image:",
            metadata={
                "images": [{"id": "img_1", "path": str(image_file)}]
            },
            source_ref="doc_001",
        )

        result = captioner.transform([chunk])

        assert len(result) == 1
        # Should still mark the image as unprocessed
        assert result[0].metadata.get("has_unprocessed_images") is True


class TestImageCaptionerMetadata:
    """Tests for metadata handling."""

    def test_preserves_original_metadata(self, tmp_path):
        """Test that original metadata is preserved."""
        settings = Settings()
        settings.ingestion = MagicMock()
        settings.ingestion.image_captioner = MagicMock()
        settings.ingestion.image_captioner.enabled = True

        # Create a temp image file
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        image_file = image_dir / "img_1.png"
        image_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        mock_vision = MockVisionLLM("Caption")
        captioner = ImageCaptioner(settings, vision_llm=mock_vision)

        original_metadata = {
            "source_path": "/test/doc.pdf",
            "page": 5,
            "custom_field": "value",
            "images": [{"id": "img_1", "path": str(image_file)}],
        }
        chunk = Chunk(
            id="test_0001_abc12345",
            text="Test content",
            metadata=original_metadata,
            source_ref="doc_001",
        )

        result = captioner.transform([chunk])

        assert result[0].metadata["source_path"] == "/test/doc.pdf"
        assert result[0].metadata["page"] == 5
        assert result[0].metadata["custom_field"] == "value"

    def test_appends_captions_to_existing(self, tmp_path):
        """Test that new captions are appended to existing captions."""
        settings = Settings()
        settings.ingestion = MagicMock()
        settings.ingestion.image_captioner = MagicMock()
        settings.ingestion.image_captioner.enabled = True

        # Create a temp image file
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        image_file = image_dir / "img_new.png"
        image_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        mock_vision = MockVisionLLM("New caption")
        captioner = ImageCaptioner(settings, vision_llm=mock_vision)

        chunk = Chunk(
            id="test_0001_abc12345",
            text="Content",
            metadata={
                "images": [{"id": "img_new", "path": str(image_file)}],
                "image_captions": {"img_existing": "Existing caption"},
            },
            source_ref="doc_001",
        )

        result = captioner.transform([chunk])

        captions = result[0].metadata["image_captions"]
        assert "img_existing" in captions
        assert "img_new" in captions
        assert captions["img_new"] == "New caption"

    def test_preserves_chunk_text(self, tmp_path):
        """Test that chunk text is not modified."""
        settings = Settings()
        settings.ingestion = MagicMock()
        settings.ingestion.image_captioner = MagicMock()
        settings.ingestion.image_captioner.enabled = True

        # Create a temp image file
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        image_file = image_dir / "img_1.png"
        image_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        mock_vision = MockVisionLLM("Caption")
        captioner = ImageCaptioner(settings, vision_llm=mock_vision)

        original_text = "This is the original chunk text with [IMAGE: img_1] placeholder."
        chunk = Chunk(
            id="test_0001_abc12345",
            text=original_text,
            metadata={"images": [{"id": "img_1", "path": str(image_file)}]},
            source_ref="doc_001",
        )

        result = captioner.transform([chunk])

        assert result[0].text == original_text


class TestImageCaptionerStats:
    """Tests for captioning statistics."""

    def test_get_captioning_stats(self, tmp_path):
        """Test getting captioning statistics."""
        settings = Settings()
        settings.ingestion = MagicMock()
        settings.ingestion.image_captioner = MagicMock()
        settings.ingestion.image_captioner.enabled = True

        # Create temp image files
        image_dir = tmp_path / "images"
        image_dir.mkdir()

        mock_vision = MockVisionLLM("Caption")
        captioner = ImageCaptioner(settings, vision_llm=mock_vision)

        chunks = []
        for i in range(3):
            image_file = image_dir / f"img_{i}.png"
            image_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
            chunks.append(
                Chunk(
                    id=f"test_{i:04d}_abc12345",
                    text=f"Content {i}",
                    metadata={"images": [{"id": f"img_{i}", "path": str(image_file)}]},
                    source_ref="doc_001",
                )
            )

        result = captioner.transform(chunks)
        stats = captioner.get_captioning_stats(result)

        assert stats["total"] == 3
        assert stats["with_images"] == 3
        assert stats["with_captions"] == 3
        assert stats["total_images"] == 3
        assert stats["total_captions"] == 3
        assert stats["unprocessed"] == 0


class TestImageCaptionerEdgeCases:
    """Tests for edge cases."""

    def test_transform_with_empty_text(self):
        """Test handling chunk with empty text."""
        settings = Settings()
        settings.ingestion = MagicMock()
        settings.ingestion.image_captioner = MagicMock()
        settings.ingestion.image_captioner.enabled = True

        mock_vision = MockVisionLLM("Caption")
        captioner = ImageCaptioner(settings, vision_llm=mock_vision)

        chunk = Chunk(
            id="test_0001_abc12345",
            text="",
            metadata={},
            source_ref="doc_001",
        )

        result = captioner.transform([chunk])

        assert len(result) == 1
        assert "image_captioning" in result[0].metadata

    def test_transform_with_empty_images_list(self):
        """Test handling chunk with empty images list."""
        settings = Settings()
        settings.ingestion = MagicMock()
        settings.ingestion.image_captioner = MagicMock()
        settings.ingestion.image_captioner.enabled = True

        mock_vision = MockVisionLLM("Caption")
        captioner = ImageCaptioner(settings, vision_llm=mock_vision)

        chunk = Chunk(
            id="test_0001_abc12345",
            text="Content without images",
            metadata={"images": []},
            source_ref="doc_001",
        )

        result = captioner.transform([chunk])

        assert len(result) == 1
        assert result[0].metadata["image_captioning"]["captions_count"] == 0

    def test_captions_preserved_in_output(self, tmp_path):
        """Test that captions are correctly stored and retrievable."""
        settings = Settings()
        settings.ingestion = MagicMock()
        settings.ingestion.image_captioner = MagicMock()
        settings.ingestion.image_captioner.enabled = True

        # Create a temp image file
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        image_file = image_dir / "chart_001.png"
        image_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        expected_caption = "A bar chart comparing Q1-Q4 revenue across regions."
        mock_vision = MockVisionLLM(expected_caption)
        captioner = ImageCaptioner(settings, vision_llm=mock_vision)

        chunk = Chunk(
            id="test_0001_abc12345",
            text="Revenue breakdown:",
            metadata={
                "images": [
                    {"id": "chart_001", "path": str(image_file)}
                ]
            },
            source_ref="doc_001",
        )

        result = captioner.transform([chunk])

        captions = result[0].metadata.get("image_captions", {})
        assert "chart_001" in captions
        assert captions["chart_001"] == expected_caption


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
