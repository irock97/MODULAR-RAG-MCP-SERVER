"""Unit tests for QwenVisionLLM.

This module tests the QwenVisionLLM class with mock HTTP responses.
It covers initialization, image encoding, request building, and response parsing.

Design Principles:
    - Mock-based: Uses unittest.mock for HTTP client
    - Contract Testing: Verify interface compliance
    - Coverage: Initialization, image handling, API calls, error handling
"""

import base64
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from libs.llm.qwen_vision_llm import QwenVisionLLM


class MockResponse:
    """Mock httpx response for testing."""

    def __init__(self, json_data: dict[str, Any], status_code: int = 200) -> None:
        self._json_data = json_data
        self.status_code = status_code

    def json(self) -> dict[str, Any]:
        return self._json_data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import httpx
            raise httpx.HTTPStatusError(
                "Error",
                request=MagicMock(),
                response=MagicMock(status_code=self.status_code),
            )


class TestQwenVisionLLM:
    """Tests for QwenVisionLLM class."""

    def test_initialization_with_api_key(self):
        """Test initialization with explicit API key."""
        llm = QwenVisionLLM(api_key="sk-test-key", model="qwen-vl-max")

        assert llm.provider_name == "qwen-vision"
        assert llm._model == "qwen-vl-max"
        assert llm._api_key == "sk-test-key"
        assert llm._base_url == QwenVisionLLM.DEFAULT_BASE_URL

    def test_initialization_default_model(self):
        """Test initialization uses default model."""
        llm = QwenVisionLLM(api_key="sk-test-key")

        assert llm._model == "qwen-vl-max"

    def test_initialization_with_env_var(self):
        """Test initialization reads API key from environment."""
        with patch.dict("os.environ", {"DASHSCOPE_API_KEY": "env-api-key"}):
            llm = QwenVisionLLM()

        assert llm._api_key == "env-api-key"

    def test_initialization_without_api_key_raises(self):
        """Test initialization raises error without API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(Exception):
                QwenVisionLLM()

    def test_supported_formats(self):
        """Test supported image formats."""
        llm = QwenVisionLLM(api_key="sk-test-key")

        assert "image/png" in llm.supported_formats
        assert "image/jpeg" in llm.supported_formats
        assert "image/gif" in llm.supported_formats
        assert "image/webp" in llm.supported_formats

    def test_max_image_size(self):
        """Test maximum image size."""
        llm = QwenVisionLLM(api_key="sk-test-key")

        assert llm.max_image_size == (4096, 4096)

    def test_repr(self):
        """Test string representation."""
        llm = QwenVisionLLM(api_key="sk-test-key", model="qwen-vl-plus")

        repr_str = repr(llm)
        assert "QwenVisionLLM" in repr_str
        assert "qwen-vision" in repr_str
        assert "qwen-vl-plus" in repr_str


class TestQwenVisionLLMEncodeImage:
    """Tests for image encoding."""

    def test_encode_image_from_path(self, tmp_path):
        """Test encoding image from file path."""
        llm = QwenVisionLLM(api_key="sk-test-key")

        # Create a test image
        image_file = tmp_path / "test.png"
        image_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        result = llm._encode_image(str(image_file))

        assert result.startswith("data:image/png;base64,")

    def test_encode_image_from_base64(self):
        """Test encoding image from base64 string."""
        llm = QwenVisionLLM(api_key="sk-test-key")

        # Create a simple base64 encoded PNG
        test_b64 = base64.b64encode(b"test image data").decode()

        result = llm._encode_image(test_b64)

        assert result.startswith("data:image/png;base64,")

    def test_encode_image_from_bytes(self):
        """Test encoding image from bytes."""
        llm = QwenVisionLLM(api_key="sk-test-key")

        test_bytes = b"test image data"

        result = llm._encode_image(test_bytes)

        assert result.startswith("data:image/png;base64,")

    def test_encode_image_unsupported_type(self):
        """Test encoding raises error for unsupported type."""
        llm = QwenVisionLLM(api_key="sk-test-key")

        with pytest.raises(Exception):
            llm._encode_image(12345)  # type: ignore


class TestQwenVisionLLMPayload:
    """Tests for request payload building."""

    def test_build_payload(self):
        """Test building request payload."""
        llm = QwenVisionLLM(
            api_key="sk-test-key",
            model="qwen-vl-max",
            max_tokens=1000,
        )

        payload = llm._build_request_payload(
            "Describe this image",
            "data:image/png;base64,abc123"
        )

        assert payload["model"] == "qwen-vl-max"
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == "user"
        assert payload["max_tokens"] == 1000

    def test_build_payload_with_temperature(self):
        """Test building payload with temperature."""
        llm = QwenVisionLLM(
            api_key="sk-test-key",
            model="qwen-vl-plus",
            temperature=0.5,
        )

        payload = llm._build_request_payload(
            "Describe this image",
            "data:image/png;base64,abc123"
        )

        assert payload["temperature"] == 0.5


class TestQwenVisionLLMResponse:
    """Tests for response parsing."""

    def test_parse_response_success(self):
        """Test successful response parsing."""
        llm = QwenVisionLLM(api_key="sk-test-key")

        response_data = {
            "choices": [
                {"message": {"content": "This is a chart showing growth."}}
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        }

        content = llm._parse_response(response_data)

        assert content == "This is a chart showing growth."

    def test_parse_response_empty_choices(self):
        """Test response parsing with empty choices."""
        llm = QwenVisionLLM(api_key="sk-test-key")

        with pytest.raises(Exception):
            llm._parse_response({"choices": []})

    def test_parse_response_no_content(self):
        """Test response parsing with no content in message."""
        llm = QwenVisionLLM(api_key="sk-test-key")

        with pytest.raises(Exception):
            llm._parse_response({"choices": [{"message": {}}]})


class TestQwenVisionLLMChat:
    """Tests for chat_with_image method."""

    def test_chat_with_image_success(self, tmp_path):
        """Test successful image chat."""
        llm = QwenVisionLLM(api_key="sk-test-key")

        # Create a test image
        image_file = tmp_path / "test.png"
        image_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        mock_response = {
            "choices": [
                {"message": {"content": "A test image caption"}}
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        }

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = MockResponse(mock_response)

            response = llm.chat_with_image(
                "Describe this image",
                str(image_file)
            )

            assert response.content == "A test image caption"
            assert response.usage["prompt_tokens"] == 100
            assert response.usage["completion_tokens"] == 50

    def test_chat_with_image_with_base64(self):
        """Test chat with base64 encoded image."""
        llm = QwenVisionLLM(api_key="sk-test-key")

        mock_response = {
            "choices": [
                {"message": {"content": "Image description"}}
            ],
            "usage": {}
        }

        test_b64 = base64.b64encode(b"test").decode()

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = MockResponse(mock_response)

            response = llm.chat_with_image(
                "What do you see?",
                test_b64
            )

            assert response.content == "Image description"

    def test_chat_with_image_http_error(self, tmp_path):
        """Test chat handles HTTP errors gracefully."""
        llm = QwenVisionLLM(api_key="sk-test-key")

        # Create a test image
        image_file = tmp_path / "test.png"
        image_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = MockResponse(
                {"message": "Invalid API key"},
                status_code=401
            )

            with pytest.raises(Exception):
                llm.chat_with_image(
                    "Describe this",
                    str(image_file)
                )


class TestQwenVisionLLMErrorHandling:
    """Tests for error handling."""

    def test_handle_http_error(self):
        """Test HTTP error handling."""
        llm = QwenVisionLLM(api_key="sk-test-key")

        import httpx

        error = httpx.HTTPStatusError(
            "Error",
            request=MagicMock(),
            response=MagicMock(
                status_code=401,
                json=lambda: {"type": "invalid_api_key", "message": "Bad key"}
            ),
        )

        with pytest.raises(Exception):
            llm._handle_http_error(error)

    def test_unsupported_image_type(self):
        """Test error handling for unsupported image type."""
        llm = QwenVisionLLM(api_key="sk-test-key")

        with pytest.raises(Exception):
            llm._encode_image(12345)  # type: ignore


class TestQwenVisionLLMIntegration:
    """Integration tests with mocked API calls."""

    def test_full_chat_flow(self, tmp_path):
        """Test complete chat flow from request to response."""
        llm = QwenVisionLLM(
            api_key="sk-test-key",
            model="qwen-vl-plus",
            max_tokens=2000,
        )

        # Create test image
        image_file = tmp_path / "chart.png"
        image_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 200)

        # Mock API response
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": "A bar chart showing quarterly revenue growth"
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 250,
                "completion_tokens": 75,
                "total_tokens": 325
            }
        }

        with patch("httpx.Client") as mock_client:
            mock_client.return_value.post.return_value = MockResponse(mock_response)
            mock_client.return_value.__enter__ = MagicMock(
                return_value=mock_client.return_value
            )
            mock_client.return_value.__exit__ = MagicMock(return_value=False)

            response = llm.chat_with_image(
                "Analyze this chart and describe the key insights.",
                str(image_file)
            )

            assert "bar chart" in response.content.lower() or \
                   "quarterly" in response.content.lower() or \
                   "revenue" in response.content.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
