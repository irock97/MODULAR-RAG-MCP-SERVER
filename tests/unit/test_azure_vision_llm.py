"""Unit tests for Azure Vision LLM.

This module tests the AzureVisionLLM implementation with mock HTTP responses.
It covers normal calls, image compression, timeout, and auth failure scenarios.

Design Principles:
    - Mock-based: Uses unittest.mock for HTTP client
    - Contract Testing: Verify interface compliance
    - Coverage: Registration, creation, error handling, image compression
"""

import tempfile
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import httpx

from core.settings import Settings
from libs.llm.azure_vision_llm import AzureVisionLLM
from libs.llm.base_vision_llm import ImageInput, VisionResponse
from libs.llm.llm_factory import LLMFactory


class TestAzureVisionLLM:
    """Tests for AzureVisionLLM class."""

    def test_initialization_success(self):
        """Test successful initialization with all required parameters."""
        llm = AzureVisionLLM(
            azure_endpoint="https://test.openai.azure.com/",
            deployment_name="gpt-4o",
            api_key="test-api-key"
        )

        assert llm.provider_name == "azure-vision"
        assert llm._azure_endpoint == "https://test.openai.azure.com/"
        assert llm._deployment_name == "gpt-4o"

    def test_initialization_missing_endpoint(self):
        """Test that missing endpoint raises configuration error."""
        with pytest.raises(Exception) as exc_info:
            AzureVisionLLM(
                deployment_name="gpt-4o",
                api_key="test-api-key"
            )

        assert "endpoint" in str(exc_info.value).lower()

    def test_initialization_missing_deployment(self):
        """Test that missing deployment name raises configuration error."""
        with pytest.raises(Exception) as exc_info:
            AzureVisionLLM(
                azure_endpoint="https://test.openai.azure.com/",
                api_key="test-api-key"
            )

        assert "deployment" in str(exc_info.value).lower()

    def test_initialization_missing_api_key(self):
        """Test that missing API key raises configuration error."""
        with pytest.raises(Exception) as exc_info:
            AzureVisionLLM(
                azure_endpoint="https://test.openai.azure.com/",
                deployment_name="gpt-4o"
            )

        assert "api key" in str(exc_info.value).lower()

    def test_custom_max_image_size(self):
        """Test custom max image size setting."""
        llm = AzureVisionLLM(
            azure_endpoint="https://test.openai.azure.com/",
            deployment_name="gpt-4o",
            api_key="test-api-key",
            max_image_size=1024
        )

        assert llm._max_image_size == 1024
        assert llm.max_image_size == (1024, 1024)

    def test_default_api_version(self):
        """Test that default API version is set correctly."""
        llm = AzureVisionLLM(
            azure_endpoint="https://test.openai.azure.com/",
            deployment_name="gpt-4o",
            api_key="test-api-key"
        )

        assert llm._api_version == "2024-02-15-preview"

    def test_custom_api_version(self):
        """Test custom API version setting."""
        llm = AzureVisionLLM(
            azure_endpoint="https://test.openai.azure.com/",
            deployment_name="gpt-4o",
            api_key="test-api-key",
            api_version="2024-05-01-preview"
        )

        assert llm._api_version == "2024-05-01-preview"

    def test_supported_formats(self):
        """Test supported image formats."""
        llm = AzureVisionLLM(
            azure_endpoint="https://test.openai.azure.com/",
            deployment_name="gpt-4o",
            api_key="test-api-key"
        )

        assert "image/png" in llm.supported_formats
        assert "image/jpeg" in llm.supported_formats


class TestAzureVisionLLMImageCompression:
    """Tests for image compression functionality."""

    def test_compress_image(self):
        """Test that image compression works."""
        llm = AzureVisionLLM(
            azure_endpoint="https://test.openai.azure.com/",
            deployment_name="gpt-4o",
            api_key="test-api-key",
            max_image_size=1024
        )

        # Create a test image
        from PIL import Image
        img = Image.new("RGB", (2000, 2000), color="red")
        img_bytes = BytesIO()
        img.save(img_bytes, format="PNG")
        original_size = len(img_bytes.getvalue())

        # Compress
        compressed = llm._compress_image(img_bytes.getvalue())

        # Should return valid PNG bytes
        assert isinstance(compressed, bytes)
        assert len(compressed) > 0

        # Should be able to open the compressed image
        compressed_img = Image.open(BytesIO(compressed))
        assert compressed_img.size[0] <= 1024
        assert compressed_img.size[1] <= 1024


class TestAzureVisionLLMChatWithImage:
    """Tests for chat_with_image method."""

    def test_chat_with_image_bytes(self):
        """Test chat_with_image with image bytes."""
        llm = AzureVisionLLM(
            azure_endpoint="https://test.openai.azure.com/",
            deployment_name="gpt-4o",
            api_key="test-api-key"
        )

        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "This is a test response"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        # Call with bytes and pass mock_client via kwargs
        response = llm.chat_with_image(
            "What do you see?",
            b"fake png image data",
            http_client=mock_client
        )

        assert isinstance(response, VisionResponse)
        assert response.content == "This is a test response"

    def test_chat_with_image_base64(self):
        """Test chat_with_image with base64-encoded image."""
        llm = AzureVisionLLM(
            azure_endpoint="https://test.openai.azure.com/",
            deployment_name="gpt-4o",
            api_key="test-api-key"
        )

        import base64
        b64_image = base64.b64encode(b"fake image").decode()

        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Base64 image response"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        response = llm.chat_with_image(
            "Describe this",
            b64_image,
            http_client=mock_client
        )

        assert response.content == "Base64 image response"

    def test_chat_with_image_image_input(self):
        """Test chat_with_image with ImageInput object."""
        llm = AzureVisionLLM(
            azure_endpoint="https://test.openai.azure.com/",
            deployment_name="gpt-4o",
            api_key="test-api-key"
        )

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"fake image")
            image_path = f.name

        image_input = ImageInput(path=image_path, mime_type="image/png")

        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ImageInput response"}}],
            "usage": {}
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        response = llm.chat_with_image(
            "What is this?",
            image_input,
            http_client=mock_client
        )

        assert response.content == "ImageInput response"


class TestAzureVisionLLMErrors:
    """Tests for error handling scenarios."""

    def test_timeout_error(self):
        """Test timeout error handling."""
        llm = AzureVisionLLM(
            azure_endpoint="https://test.openai.azure.com/",
            deployment_name="gpt-4o",
            api_key="test-api-key"
        )

        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.RequestError(
            "Connection timeout",
            request=MagicMock()
        )

        with pytest.raises(Exception) as exc_info:
            llm.chat_with_image(
                "test",
                b"fake image",
                http_client=mock_client
            )

        assert "connect" in str(exc_info.value).lower() or "timeout" in str(exc_info.value).lower()

    def test_auth_failure(self):
        """Test authentication failure error handling."""
        llm = AzureVisionLLM(
            azure_endpoint="https://test.openai.azure.com/",
            deployment_name="gpt-4o",
            api_key="invalid-key"
        )

        # Mock 401 response
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {
            "error": {
                "message": "Invalid authentication credentials",
                "code": "invalid_api_key"
            }
        }

        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.HTTPStatusError(
            "401 Unauthorized",
            request=MagicMock(),
            response=mock_response
        )

        with pytest.raises(Exception) as exc_info:
            llm.chat_with_image(
                "test",
                b"fake image",
                http_client=mock_client
            )

        # Should contain error info
        assert "api" in str(exc_info.value).lower() or "authentication" in str(exc_info.value).lower()

    def test_deployment_not_found(self):
        """Test deployment not found error handling."""
        llm = AzureVisionLLM(
            azure_endpoint="https://test.openai.azure.com/",
            deployment_name="nonexistent-deployment",
            api_key="test-key"
        )

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = {
            "error": {
                "message": "Deployment not found",
                "code": "deployment_not_found"
            }
        }

        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.HTTPStatusError(
            "404 Not Found",
            request=MagicMock(),
            response=mock_response
        )

        with pytest.raises(Exception) as exc_info:
            llm.chat_with_image(
                "test",
                b"fake image",
                http_client=mock_client
            )

        error_msg = str(exc_info.value).lower()
        assert "deployment" in error_msg or "404" in error_msg or "not found" in error_msg


class TestAzureVisionLLMFactory:
    """Tests for LLMFactory integration with Azure Vision."""

    def setup_method(self):
        """Clear vision providers before each test."""
        if "azure" in LLMFactory._vision_providers:
            del LLMFactory._vision_providers["azure"]

    def teardown_method(self):
        """Clear vision providers after each test."""
        if "azure" in LLMFactory._vision_providers:
            del LLMFactory._vision_providers["azure"]

    def test_create_azure_vision_llm(self):
        """Test creating Azure Vision LLM via factory."""
        settings = Settings()
        settings.llm.provider = "azure"
        settings.llm.model = "gpt-4o"
        settings.llm.api_key = "test-key"
        settings.llm.azure_endpoint = "https://test.openai.azure.com/"
        settings.llm.azure_deployment = "gpt-4o"
        settings.llm.azure_api_version = "2024-02-15-preview"

        # Register Azure Vision provider
        LLMFactory.register_vision("azure", AzureVisionLLM)

        llm = LLMFactory.create_vision_llm(settings)

        assert isinstance(llm, AzureVisionLLM)
        assert llm.provider_name == "azure-vision"
        assert llm._deployment_name == "gpt-4o"
        assert llm._azure_endpoint == "https://test.openai.azure.com/"

    def test_create_azure_vision_with_overrides(self):
        """Test creating Azure Vision LLM with parameter overrides."""
        settings = Settings()
        settings.llm.provider = "azure"
        settings.llm.model = "gpt-4o"
        settings.llm.api_key = "original-key"
        settings.llm.azure_endpoint = "https://original.openai.azure.com/"
        settings.llm.azure_deployment = "original-deployment"

        # Register Azure Vision provider
        LLMFactory.register_vision("azure", AzureVisionLLM)

        llm = LLMFactory.create_vision_llm(
            settings,
            api_key="override-key",
            deployment_name="override-deployment"
        )

        assert isinstance(llm, AzureVisionLLM)
        # Override should take precedence
        assert llm._api_key == "override-key"
        assert llm._deployment_name == "override-deployment"


class TestAzureVisionLLMRepr:
    """Tests for string representation."""

    def test_repr(self):
        """Test __repr__ method."""
        llm = AzureVisionLLM(
            azure_endpoint="https://test.openai.azure.com/",
            deployment_name="gpt-4o",
            api_key="test-api-key"
        )

        repr_str = repr(llm)
        assert "azure-vision" in repr_str
        assert "gpt-4o" in repr_str
        assert "test.openai.azure.com" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
