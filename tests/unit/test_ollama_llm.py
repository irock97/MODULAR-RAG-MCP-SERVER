"""Smoke tests for Ollama LLM provider.

This module contains unit tests for Ollama LLM provider.
Tests focus on configuration, message building, and basic functionality.

Design Principles:
    - Deterministic: All tests use fixed responses
    - Coverage: Tests provider creation, configuration, and message building
"""

import pytest
from unittest.mock import MagicMock
from typing import Any

import httpx

from libs.llm.base_llm import (
    ChatMessage,
    LLMConfigurationError,
    LLMError,
)
from libs.llm.ollama_llm import OllamaLLM, DEFAULT_OLLAMA_BASE_URL
from libs.llm.llm_factory import LLMFactory


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_messages() -> list[ChatMessage]:
    """Create sample chat messages for testing."""
    return [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="Hello, world!"),
    ]


# ============================================================================
# Ollama LLM Tests
# ============================================================================

class TestOllamaLLM:
    """Tests for Ollama LLM implementation."""

    def test_provider_name(self):
        """Test that provider name is correct."""
        llm = OllamaLLM(model="llama3.2")
        assert llm.provider_name == "ollama"

    def test_initialization_with_defaults(self):
        """Test initialization with default values."""
        llm = OllamaLLM(model="llama3.2")
        assert llm._model == "llama3.2"
        assert llm._base_url == DEFAULT_OLLAMA_BASE_URL
        assert llm._temperature == 0.0
        assert llm._keep_alive == "5m"

    def test_initialization_with_custom_values(self):
        """Test initialization with custom values."""
        llm = OllamaLLM(
            base_url="http://localhost:11435",
            model="llama3.2",
            temperature=0.8,
            max_tokens=2048,
            keep_alive="10m"
        )
        assert llm._base_url == "http://localhost:11435"
        assert llm._model == "llama3.2"
        assert llm._temperature == 0.8
        assert llm._max_tokens == 2048
        assert llm._keep_alive == "10m"

    def test_initialization_without_model_raises_error(self):
        """Test that initialization without model raises error."""
        with pytest.raises(LLMConfigurationError) as exc_info:
            OllamaLLM()
        assert "model" in str(exc_info.value).lower()

    def test_chat_empty_messages_raises_error(self):
        """Test that empty messages list raises error."""
        llm = OllamaLLM(model="llama3.2")
        with pytest.raises(LLMError) as exc_info:
            llm.chat([])
        assert "no messages" in str(exc_info.value).lower()

    def test_repr(self):
        """Test string representation."""
        llm = OllamaLLM(model="llama3.2", temperature=0.7)
        repr_str = repr(llm)
        assert "OllamaLLM" in repr_str
        assert "ollama" in repr_str
        assert "llama3.2" in repr_str


# ============================================================================
# Ollama Factory Tests
# ============================================================================

class TestOllamaFactory:
    """Tests for Ollama provider registration with factory."""

    def setup_method(self):
        """Clear factory registry before each test."""
        LLMFactory.clear()
        LLMFactory.register("ollama", OllamaLLM)

    def teardown_method(self):
        """Clear factory registry after each test."""
        LLMFactory.clear()

    def test_ollama_provider_registered(self):
        """Test that Ollama provider is registered."""
        assert LLMFactory.has_provider("ollama")
        assert "ollama" in LLMFactory.get_provider_names()

    def test_create_ollama_instance(self):
        """Test creating Ollama instance through factory."""
        mock_settings = MagicMock()
        mock_settings.llm.provider = "ollama"
        mock_settings.llm.api_key = None
        mock_settings.llm.model = "llama3.2"
        mock_settings.llm.base_url = "http://localhost:11434"
        mock_settings.llm.temperature = 0.0
        mock_settings.llm.max_tokens = None

        llm = LLMFactory.create(mock_settings)

        assert isinstance(llm, OllamaLLM)
        assert llm.provider_name == "ollama"
        assert llm._model == "llama3.2"

    def test_create_ollama_with_custom_base_url(self):
        """Test creating Ollama instance with custom base URL."""
        mock_settings = MagicMock()
        mock_settings.llm.provider = "ollama"
        mock_settings.llm.api_key = None
        mock_settings.llm.model = "llama3.2"
        mock_settings.llm.base_url = "http://custom:11434"
        mock_settings.llm.temperature = 0.0
        mock_settings.llm.max_tokens = None

        llm = LLMFactory.create(mock_settings)

        assert isinstance(llm, OllamaLLM)
        assert llm._base_url == "http://custom:11434"


# ============================================================================
# Message Building Tests
# ============================================================================

class TestOllamaMessageBuilding:
    """Tests for Ollama message building functionality."""

    def test_single_user_message(self):
        """Test building a single user message."""
        messages = [ChatMessage(role="user", content="Hello")]
        llm = OllamaLLM(model="test-model")
        payload = llm._build_request_payload(messages)

        assert payload["messages"] == [{"role": "user", "content": "Hello"}]
        assert payload["model"] == "test-model"

    def test_multi_turn_conversation(self):
        """Test building multi-turn conversation."""
        messages = [
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assistant", content="Hello!"),
            ChatMessage(role="user", content="How are you?"),
        ]
        llm = OllamaLLM(model="test-model")
        payload = llm._build_request_payload(messages)

        assert len(payload["messages"]) == 4
        assert payload["messages"][0]["role"] == "system"
        assert payload["options"]["temperature"] == 0.0  # default

    def test_custom_options(self):
        """Test building payload with custom options."""
        messages = [ChatMessage(role="user", content="Hello")]
        llm = OllamaLLM(model="test-model", temperature=0.5, max_tokens=100)
        payload = llm._build_request_payload(messages)

        assert payload["options"]["temperature"] == 0.5
        assert payload["options"]["num_predict"] == 100

    def test_keep_alive_in_payload(self):
        """Test that keep_alive is included in payload."""
        messages = [ChatMessage(role="user", content="Hello")]
        llm = OllamaLLM(model="test-model", keep_alive="10m")
        payload = llm._build_request_payload(messages)

        assert payload["keep_alive"] == "10m"


# ============================================================================
# Response Parsing Tests
# ============================================================================

class TestOllamaResponseParsing:
    """Tests for Ollama response parsing."""

    def test_parse_response_with_usage(self):
        """Test parsing response with usage info."""
        response_data = {
            "id": "test-id",
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "Hello!"},
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }

        llm = OllamaLLM(model="test-model")
        response = llm._parse_response(response_data)

        assert response.content == "Hello!"
        assert response.usage is not None
        assert response.usage["total_tokens"] == 15

    def test_parse_response_without_usage(self):
        """Test parsing response without usage info."""
        response_data = {
            "id": "test-id",
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "Hello!"}
        }

        llm = OllamaLLM(model="test-model")
        response = llm._parse_response(response_data)

        assert response.content == "Hello!"
        assert response.usage is None


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestOllamaErrorHandling:
    """Tests for Ollama error handling."""

    def test_handle_http_error(self):
        """Test HTTP error handling."""
        llm = OllamaLLM(model="test-model")

        error_response = {
            "error": "model not found",
            "error_type": "model_error"
        }

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = error_response

        error = httpx.HTTPStatusError(
            "Error",
            request=MagicMock(),
            response=mock_response
        )

        with pytest.raises(LLMError) as exc_info:
            llm._handle_http_error(error)

        assert exc_info.value.provider == "ollama"
        assert "model not found" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
