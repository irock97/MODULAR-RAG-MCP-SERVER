"""Unit tests for ImageCaptioner with real fixtures and real Vision LLM.

This module tests the ImageCaptioner class using actual sample documents
and images from the tests/fixtures directory with real QwenVisionLLM.

Design Principles:
    - Fixture-based: Uses real sample files for testing
    - Real LLM: Uses actual QwenVisionLLM for caption generation
    - Complete Flow: Tests the full pipeline from image to caption
"""

from pathlib import Path


import pytest

from core.settings import Settings, load_settings
from core.types import Chunk
from ingestion.transform.image_captioner import ImageCaptioner
from libs.llm.llm_factory import LLMFactory
from libs.llm.qwen_vision_llm import QwenVisionLLM


# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
SAMPLE_IMAGES_DIR = FIXTURES_DIR / "sample_documents"
CAT_IMAGE_PATH = SAMPLE_IMAGES_DIR / "cat.png"


class TestImageCaptionerWithRealVisionLLM:
    """Tests for ImageCaptioner using real QwenVisionLLM and real images."""

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
    def settings(self):
        """Load settings from config file."""
        return load_settings()

    @pytest.fixture
    def vision_llm(self, settings):
        """Create a real QwenVisionLLM instance using loaded settings."""
        vision_llm = LLMFactory.create_vision_llm(settings)
        assert vision_llm is not None
        assert isinstance(vision_llm, QwenVisionLLM)
        return vision_llm

    @pytest.fixture
    def image_captioner_settings(self, settings):
        """Configure Settings with image captioner enabled."""
        settings.ingestion.image_captioner.enabled = True
        return settings

    def test_caption_cat_image_with_real_vision_llm(
        self, cat_image_path, vision_llm, image_captioner_settings
    ):
        """Test captioning cat.png with real QwenVisionLLM.

        This test verifies the complete flow:
        1. Load a real image (cat.png)
        2. Create a real QwenVisionLLM instance
        3. Process through ImageCaptioner
        4. Verify caption was generated using real Vision API
        """
        captioner = ImageCaptioner(
            image_captioner_settings, vision_llm=vision_llm
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
        assert enriched.metadata["image_captioning"]["vision_provider"] == "qwen-vision"
        assert enriched.metadata["image_captioning"]["failed_count"] == 0

        # Verify original metadata is preserved
        # Note: source_ref is a Chunk attribute, not in metadata
        assert enriched.source_ref == "cat_document.pdf"
        assert enriched.metadata["images"][0]["id"] == "image_cat"

        print(f"Generated caption: {caption}")

    def test_captioner_with_disabled_vision(
        self, cat_image_path, settings
    ):
        """Test that disabled captioner marks images as unprocessed."""
        settings.ingestion.image_captioner.enabled = False

        captioner = ImageCaptioner(settings)

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
