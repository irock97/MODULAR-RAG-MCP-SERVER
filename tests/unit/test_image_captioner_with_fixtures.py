"""Unit tests for ImageCaptioner with fixtures.

This module tests the ImageCaptioner class using actual sample documents
and images from the tests/fixtures directory.

Design Principles:
    - Fixture-based: Uses real sample files for testing
    - Uses Mock Vision LLM to avoid real API calls
    - Complete Flow: Tests the full pipeline from image to caption
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from core.settings import Settings
from core.types import Chunk
from ingestion.transform.image_captioner import ImageCaptioner
from libs.llm.base_vision_llm import VisionResponse


# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
SAMPLE_IMAGES_DIR = FIXTURES_DIR / "sample_documents"
CAT_IMAGE_PATH = SAMPLE_IMAGES_DIR / "cat.png"


class MockVisionLLM:
    """A mock Vision LLM for testing."""

    def __init__(self, response_content: str = "A descriptive caption") -> None:
        self._response_content = response_content
        self.call_count = 0

    @property
    def provider_name(self) -> str:
        return "mock-vision"

    @property
    def supported_formats(self) -> list[str]:
        return ["image/png", "image/jpeg", "image/gif", "image/webp"]

    def chat_with_image(
        self,
        text: str,
        image: Any,
        trace: Any = None,
        **kwargs: Any
    ) -> VisionResponse:
        self.call_count += 1
        return VisionResponse(
            content=self._response_content,
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )


class TestImageCaptionerWithFixtures:
    """Tests for ImageCaptioner using fixtures."""

    @pytest.fixture
    def sample_images_dir(self):
        """Path to sample images directory."""
        return SAMPLE_IMAGES_DIR

    @pytest.fixture
    def cat_image_path(self):
        """Path to cat.png test image."""
        assert CAT_IMAGE_PATH.exists(), f"Test image not found: {CAT_IMAGE_PATH}"
        return CAT_IMAGE_PATH

    @pytest.fixture
    def mock_vision_llm(self):
        """Create a mock Vision LLM."""
        return MockVisionLLM(response_content="A cute cat sitting on a windowsill")

    @pytest.fixture
    def image_captioner_settings(self):
        """Create Settings with image captioner enabled."""
        settings = Settings()
        settings.ingestion = MagicMock()
        settings.ingestion.image_captioner = MagicMock()
        settings.ingestion.image_captioner.enabled = True
        return settings

    def test_caption_cat_image(
        self, cat_image_path, mock_vision_llm, image_captioner_settings
    ):
        """Test captioning cat.png with mock Vision LLM.

        This test verifies the complete flow:
        1. Load a real image (cat.png)
        2. Create a mock Vision LLM
        3. Process through ImageCaptioner
        4. Verify caption was generated
        """
        captioner = ImageCaptioner(
            image_captioner_settings, vision_llm=mock_vision_llm
        )

        # Create chunk with image metadata matching PDF loader output format
        chunk = Chunk(
            id="test_cat_001_abc12345",
            text="See the cat image below:",
            metadata={
                "images": [
                    {
                        "id": "image_cat",
                        "path": str(cat_image_path),
                        "page": 1,
                        "text_offset": 19,
                        "text_length": 10,
                        "position": {"width": 800, "height": 600},
                    }
                ]
            },
            source_ref="cat_document.pdf",
        )

        result = captioner.transform([chunk])

        assert len(result) == 1
        enriched = result[0]

        # Verify caption was generated
        assert "image_captions" in enriched.metadata
        assert "image_cat" in enriched.metadata["image_captions"]
        caption = enriched.metadata["image_captions"]["image_cat"]
        assert isinstance(caption, str)
        assert len(caption) > 0

        # Verify metadata
        assert "image_captioning" in enriched.metadata
        assert enriched.metadata["image_captioning"]["captions_count"] == 1
        assert enriched.metadata["image_captioning"]["vision_provider"] == "mock-vision"
        assert enriched.metadata["image_captioning"]["failed_count"] == 0

        # Verify original metadata is preserved
        assert enriched.source_ref == "cat_document.pdf"
        assert enriched.metadata["images"][0]["id"] == "image_cat"

        # Verify Vision LLM was called
        assert mock_vision_llm.call_count == 1

    def test_captioner_with_disabled_vision(
        self, cat_image_path, image_captioner_settings
    ):
        """Test that disabled captioner marks images as unprocessed."""
        image_captioner_settings.ingestion.image_captioner.enabled = False

        captioner = ImageCaptioner(image_captioner_settings)

        chunk = Chunk(
            id="test_disabled_001_xyz",
            text="Cat image:",
            metadata={
                "images": [
                    {"id": "image_cat", "path": str(cat_image_path)}
                ]
            },
            source_ref="doc.pdf",
        )

        result = captioner.transform([chunk])

        assert len(result) == 1
        enriched = result[0]

        # Verify images are marked as unprocessed
        assert enriched.metadata.get("has_unprocessed_images") is True
        assert enriched.metadata["image_captioning"]["enabled"] is False
        assert enriched.metadata["image_captioning"]["failed_count"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
