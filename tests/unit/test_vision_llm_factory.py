"""Unit tests for Vision LLM factory and BaseVisionLLM.

This module tests the BaseVisionLLM abstract class and the
LLMFactory.create_vision_llm method.

Design Principles:
    - Mock-based: Uses FakeVisionLLM for testing
    - Contract Testing: Verify interface compliance
    - Coverage: Registration, creation, error handling
"""

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from core.settings import Settings
from core.trace.trace_context import TraceContext
from libs.llm.azure_vision_llm import AzureVisionLLM
from libs.llm.base_vision_llm import (
    BaseVisionLLM,
    ImageInput,
    ImagePreprocessor,
    UnsupportedImageFormatError,
    VisionResponse,
)
from libs.llm.llm_factory import LLMFactory
from libs.llm.qwen_vision_llm import QwenVisionLLM


class FakeVisionLLM(BaseVisionLLM):
    """A fake Vision LLM for testing."""

    def __init__(
        self,
        response_content: str = "Vision response",
        **kwargs: Any
    ) -> None:
        self._response_content = response_content
        self.call_count = 0
        self.last_text = None
        self.last_image = None

    @property
    def provider_name(self) -> str:
        return "fake-vision"

    def chat_with_image(
        self,
        text: str,
        image: ImageInput | str | bytes,
        trace: TraceContext | None = None,
        **kwargs: Any
    ) -> VisionResponse:
        self.call_count += 1
        self.last_text = text
        self.last_image = image

        return VisionResponse(content=self._response_content)


class FakeVisionLLMError(BaseVisionLLM):
    """A fake Vision LLM that raises errors."""

    def __init__(self, **kwargs: Any) -> None:
        pass

    @property
    def provider_name(self) -> str:
        return "fake-vision-error"

    def chat_with_image(
        self,
        text: str,
        image: ImageInput | str | bytes,
        trace: TraceContext | None = None,
        **kwargs: Any
    ) -> VisionResponse:
        raise Exception("Vision LLM API error")


class TestImageInput:
    """Tests for ImageInput class."""

    def test_from_path(self):
        """Test creating ImageInput from path."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"fake image data")
            path = f.name

        img = ImageInput(path=path)
        assert img.path == path
        assert img.base64 is None
        assert img.mime_type == "image/png"

    def test_from_base64(self):
        """Test creating ImageInput from base64."""
        base64_data = "SGVsbG8gV29ybGQ="
        img = ImageInput(base64=base64_data, mime_type="image/jpeg")
        assert img.base64 == base64_data
        assert img.path is None
        assert img.mime_type == "image/jpeg"

    def test_get_bytes_from_path(self):
        """Test getting bytes from path."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"test image bytes")
            path = f.name

        img = ImageInput(path=path)
        data = img.get_bytes()
        assert data == b"test image bytes"

    def test_get_bytes_from_base64(self):
        """Test getting bytes from base64."""
        import base64
        original = b"hello world"
        base64_data = base64.b64encode(original).decode()

        img = ImageInput(base64=base64_data)
        data = img.get_bytes()
        assert data == original

    def test_validation_no_data(self):
        """Test that at least one of path or base64 is required."""
        with pytest.raises(ValueError):
            ImageInput()

    def test_path_takes_precedence(self):
        """Test that path takes precedence when both are set."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"path data")
            path = f.name

        img = ImageInput(path=path, base64="base64_data")
        # Both are stored, but get_bytes uses path first
        assert img.path == path
        assert img.base64 == "base64_data"


class TestBaseVisionLLM:
    """Tests for BaseVisionLLM abstract class."""

    def test_provider_name_property_required(self):
        """Test that provider_name is an abstract property."""
        # Should not be able to instantiate directly
        with pytest.raises(TypeError):
            BaseVisionLLM()

    def test_chat_with_image_required(self):
        """Test that chat_with_image is an abstract method."""

        class IncompleteVisionLLM(BaseVisionLLM):
            @property
            def provider_name(self) -> str:
                return "incomplete"

        # Should not be able to instantiate (chat_with_image not implemented)
        with pytest.raises(TypeError):
            IncompleteVisionLLM()

    def test_supported_formats_default(self):
        """Test default supported formats."""
        class TestVisionLLM(FakeVisionLLM):
            pass

        llm = TestVisionLLM()
        assert "image/png" in llm.supported_formats
        assert "image/jpeg" in llm.supported_formats

    def test_max_image_size_default(self):
        """Test default max image size."""
        llm = FakeVisionLLM()
        assert llm.max_image_size == (2048, 2048)

    def test_max_token_limit_default(self):
        """Test default max token limit."""
        llm = FakeVisionLLM()
        assert llm.max_token_limit == 10

    def test_preprocess_image_with_path(self):
        """Test preprocessing image from path."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"original data")
            path = f.name

        llm = FakeVisionLLM()
        result = llm.preprocess_image(path)
        assert result == b"original data"

    def test_preprocess_image_with_base64(self):
        """Test preprocessing image from base64."""
        import base64
        original = b"base64 data"
        b64 = base64.b64encode(original).decode()

        llm = FakeVisionLLM()
        result = llm.preprocess_image(b64)
        assert result == original

    def test_preprocess_image_with_image_input(self):
        """Test preprocessing with ImageInput object."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"input data")
            path = f.name

        img_input = ImageInput(path=path, mime_type="image/png")
        llm = FakeVisionLLM()
        result = llm.preprocess_image(img_input)
        assert result == b"input data"

    def test_preprocess_image_with_custom_preprocessor(self):
        """Test preprocessing with custom preprocessor."""
        import base64
        original = b"original"
        b64 = base64.b64encode(original).decode()

        class UppercasePreprocessor(ImagePreprocessor):
            @property
            def name(self) -> str:
                return "uppercase"

            def process(self, image_data: bytes) -> bytes:
                return image_data.upper()

        llm = FakeVisionLLM()
        result = llm.preprocess_image(b64, preprocessor=UppercasePreprocessor())
        assert result == b"ORIGINAL"

    def test_chat_with_images_limits(self):
        """Test chat_with_images enforces max limit."""
        llm = FakeVisionLLM()

        # Create many images
        images = [ImageInput(base64="aGVsbG8=") for _ in range(20)]

        with pytest.raises(ValueError, match="Too many images"):
            llm.chat_with_images("test", images)


class TestLLMFactoryVision:
    """Tests for LLMFactory Vision LLM methods."""

    def setup_method(self):
        """Clear vision providers before each test."""
        LLMFactory.clear_vision()

    def teardown_method(self):
        """Clear vision providers after each test."""
        LLMFactory.clear_vision()

    def test_register_vision_provider(self):
        """Test registering a Vision LLM provider."""
        LLMFactory.register_vision("test-vision", FakeVisionLLM)

        assert LLMFactory.has_vision_provider("test-vision")
        assert "test-vision" in LLMFactory.get_vision_provider_names()

    def test_register_vision_case_insensitive(self):
        """Test that vision provider registration is case insensitive."""
        LLMFactory.register_vision("TEST-VISION", FakeVisionLLM)

        assert LLMFactory.has_vision_provider("test-vision")
        assert LLMFactory.has_vision_provider("TEST-VISION")

    def test_unregister_vision_provider(self):
        """Test unregistering a Vision LLM provider."""
        LLMFactory.register_vision("test-vision", FakeVisionLLM)
        assert LLMFactory.has_vision_provider("test-vision")

        result = LLMFactory.unregister_vision("test-vision")
        assert result is True
        assert not LLMFactory.has_vision_provider("test-vision")

    def test_unregister_nonexistent_provider(self):
        """Test unregistering a provider that doesn't exist."""
        result = LLMFactory.unregister_vision("nonexistent")
        assert result is False

    def test_get_vision_provider_names(self):
        """Test getting list of registered vision providers."""
        LLMFactory.register_vision("provider1", FakeVisionLLM)
        LLMFactory.register_vision("provider2", FakeVisionLLM)

        names = LLMFactory.get_vision_provider_names()
        assert "provider1" in names
        assert "provider2" in names

    def test_clear_vision_providers(self):
        """Test clearing all vision providers."""
        LLMFactory.register_vision("test1", FakeVisionLLM)
        LLMFactory.register_vision("test2", FakeVisionLLM)

        LLMFactory.clear_vision()

        assert LLMFactory.get_vision_provider_names() == []

    def test_create_vision_llm_success(self):
        """Test creating a Vision LLM instance."""
        settings = Settings()
        settings.llm.provider = "test-vision"
        settings.llm.model = "test-model"
        settings.llm.api_key = "test-key"

        LLMFactory.register_vision("test-vision", FakeVisionLLM)

        llm = LLMFactory.create_vision_llm(settings)

        assert isinstance(llm, FakeVisionLLM)
        assert llm.provider_name == "fake-vision"

    def test_create_vision_llm_unregistered_provider(self):
        """Test error when provider not registered."""
        settings = Settings()
        settings.llm.provider = "unregistered-vision"
        settings.llm.model = "test-model"

        with pytest.raises(Exception):
            LLMFactory.create_vision_llm(settings)

    def test_create_vision_llm_no_provider_configured(self):
        """Test error when no provider configured."""
        settings = Settings()
        settings.llm.provider = None
        settings.llm.model = "test-model"

        with pytest.raises(Exception):
            LLMFactory.create_vision_llm(settings)

    def test_create_vision_llm_with_overrides(self):
        """Test creating Vision LLM with parameter overrides."""
        settings = Settings()
        settings.llm.provider = "test-vision"
        settings.llm.model = "original-model"
        settings.llm.api_key = "original-key"

        LLMFactory.register_vision("test-vision", FakeVisionLLM)

        # Note: FakeVisionLLM doesn't use these parameters
        llm = LLMFactory.create_vision_llm(
            settings,
            model="override-model",
            api_key="override-key"
        )

        assert isinstance(llm, FakeVisionLLM)

    def test_create_vision_llm_chat(self):
        """Test calling chat_with_image on created Vision LLM."""
        settings = Settings()
        settings.llm.provider = "test-vision"
        settings.llm.model = "test-model"

        LLMFactory.register_vision("test-vision", FakeVisionLLM)

        llm = LLMFactory.create_vision_llm(settings)
        trace = TraceContext()

        result = llm.chat_with_image("What do you see?", "test.png", trace)

        assert result.content == "Vision response"
        assert llm.call_count == 1
        assert llm.last_text == "What do you see?"

    def test_create_vision_llm_error_handling(self):
        """Test error handling when Vision LLM fails."""
        settings = Settings()
        settings.llm.provider = "error-vision"
        settings.llm.model = "test-model"

        LLMFactory.register_vision("error-vision", FakeVisionLLMError)

        llm = LLMFactory.create_vision_llm(settings)

        with pytest.raises(Exception, match="Vision LLM API error"):
            llm.chat_with_image("test", "test.png")


class TestVisionPreprocessor:
    """Tests for ImagePreprocessor abstract class."""

    def test_preprocessor_must_implement_process(self):
        """Test that preprocessor must implement process method."""

        class IncompletePreprocessor(ImagePreprocessor):
            @property
            def name(self) -> str:
                return "incomplete"

        with pytest.raises(TypeError):
            IncompletePreprocessor()

    def test_preprocessor_must_implement_name(self):
        """Test that preprocessor must implement name property."""

        class IncompletePreprocessor(ImagePreprocessor):
            def process(self, image_data: bytes) -> bytes:
                return image_data

        with pytest.raises(TypeError):
            IncompletePreprocessor()


class TestVisionResponse:
    """Tests for VisionResponse dataclass."""

    def test_vision_response_creation(self):
        """Test creating a VisionResponse."""
        response = VisionResponse(content="Hello world")

        assert response.content == "Hello world"
        assert response.raw_response is None
        assert response.usage is None
        assert response.image_size is None

    def test_vision_response_with_metadata(self):
        """Test creating VisionResponse with all fields."""
        response = VisionResponse(
            content="Response text",
            raw_response={"raw": "data"},
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            image_size=(1024, 768)
        )

        assert response.content == "Response text"
        assert response.raw_response == {"raw": "data"}
        assert response.usage == {"prompt_tokens": 10, "completion_tokens": 5}
        assert response.image_size == (1024, 768)


class TestVisionLLMWithVisionConfig:
    """Tests for using vision_llm config in create_vision_llm."""

    def setup_method(self):
        """Re-register vision providers since previous tests may clear them."""
        # Re-register the default providers
        LLMFactory.register_vision("azure", AzureVisionLLM)
        LLMFactory.register_vision("qwen", QwenVisionLLM)

    def teardown_method(self):
        """Clear vision providers after each test."""
        LLMFactory.clear_vision()

    def test_create_vision_llm_with_vision_llm_config(self):
        """Test creating Vision LLM using vision_llm config section."""
        settings = Settings()
        # Set up vision_llm config
        settings.vision_llm = MagicMock()
        settings.vision_llm.provider = "qwen"
        settings.vision_llm.model = "qwen-vl-max"
        settings.vision_llm.api_key = "test-key"
        settings.vision_llm.base_url = None
        settings.vision_llm.temperature = None
        settings.vision_llm.max_tokens = None
        settings.vision_llm.timeout = 60
        settings.vision_llm.azure_endpoint = None
        settings.vision_llm.api_version = None
        settings.vision_llm.deployment_name = None

        llm = LLMFactory.create_vision_llm(settings)

        assert isinstance(llm, QwenVisionLLM)
        assert llm.provider_name == "qwen-vision"
        assert llm._model == "qwen-vl-max"

    def test_create_vision_llm_falls_back_to_llm_config(self):
        """Test falling back to llm config when vision_llm is not set."""
        settings = Settings()
        settings.llm.provider = "qwen"
        settings.llm.model = "qwen-vl-plus"
        settings.llm.api_key = "llm-key"
        settings.llm.base_url = None
        settings.llm.temperature = None
        settings.llm.max_tokens = None
        settings.llm.timeout = None
        settings.llm.azure_endpoint = None
        settings.llm.azure_api_version = None
        settings.llm.azure_deployment = None
        # No vision_llm set

        llm = LLMFactory.create_vision_llm(settings)

        assert isinstance(llm, QwenVisionLLM)
        assert llm._model == "qwen-vl-plus"

    def test_create_vision_llm_vision_config_takes_precedence(self):
        """Test that vision_llm config takes precedence over llm config."""
        settings = Settings()
        settings.llm.provider = "openai"
        settings.llm.model = "gpt-4o"
        settings.llm.api_key = "llm-key"
        # Set up vision_llm to override
        settings.vision_llm = MagicMock()
        settings.vision_llm.provider = "qwen"
        settings.vision_llm.model = "qwen-vl-max"
        settings.vision_llm.api_key = "vision-key"
        settings.vision_llm.base_url = None
        settings.vision_llm.temperature = None
        settings.vision_llm.max_tokens = None
        settings.vision_llm.timeout = None
        settings.vision_llm.azure_endpoint = None
        settings.vision_llm.api_version = None
        settings.vision_llm.deployment_name = None

        llm = LLMFactory.create_vision_llm(settings)

        assert isinstance(llm, QwenVisionLLM)
        assert llm._model == "qwen-vl-max"

    def test_create_vision_llm_error_no_provider(self):
        """Test error when no provider is configured."""
        settings = Settings()
        # No llm.provider and no vision_llm
        settings.llm = MagicMock()
        settings.llm.provider = None

        with pytest.raises(Exception, match="Vision LLM provider is not configured"):
            LLMFactory.create_vision_llm(settings)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
