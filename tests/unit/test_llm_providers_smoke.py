"""Smoke tests for LLM providers (OpenAI, Azure, DeepSeek).

This module contains mock HTTP tests for LLM providers.
Tests use httpx_mock to simulate API responses without making real HTTP calls.

Design Principles:
    - Mock HTTP: Uses httpx_mock to intercept requests
    - Deterministic: All tests use fixed mock responses
    - Coverage: Tests provider creation, routing, and error handling
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import Any

import httpx
import respx

from libs.llm.base_llm import (
    BaseLLM,
    ChatMessage,
    LLMResponse,
    LLMConfigurationError,
    LLMError,
)
from libs.llm.openai_llm import OpenAILLM
from libs.llm.azure_llm import AzureOpenAILLM
from libs.llm.deepseek_llm import DeepSeekLLM
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


@pytest.fixture
def openai_api_response() -> dict[str, Any]:
    """Create a mock OpenAI API response."""
    return {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 13,
            "completion_tokens": 10,
            "total_tokens": 23
        }
    }


@pytest.fixture
def azure_api_response() -> dict[str, Any]:
    """Create a mock Azure OpenAI API response."""
    return {
        "id": "azure-chatcmpl-xyz789",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello from Azure!"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 8,
            "total_tokens": 23
        }
    }


@pytest.fixture
def deepseek_api_response() -> dict[str, Any]:
    """Create a mock DeepSeek API response."""
    return {
        "id": "ds-chatcmpl-123456",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "deepseek-chat",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello from DeepSeek!"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 7,
            "total_tokens": 19
        }
    }


@pytest.fixture
def error_api_response() -> dict[str, Any]:
    """Create a mock API error response."""
    return {
        "error": {
            "message": "Invalid API key",
            "type": "authentication_error",
            "code": "invalid_api_key"
        }
    }


# ============================================================================
# OpenAI LLM Tests
# ============================================================================

class TestOpenAILLM:
    """Tests for OpenAI LLM implementation."""

    def test_provider_name(self):
        """Test that provider name is correct."""
        llm = OpenAILLM(api_key="test-key", model="gpt-4o-mini")
        assert llm.provider_name == "openai"

    def test_initialization_with_api_key(self):
        """Test initialization with explicit API key."""
        llm = OpenAILLM(api_key="sk-test", model="gpt-4o-mini")
        assert llm._api_key == "sk-test"
        assert llm._model == "gpt-4o-mini"
        assert llm._base_url == "https://api.openai.com/v1"

    def test_initialization_with_custom_base_url(self):
        """Test initialization with custom base URL."""
        llm = OpenAILLM(
            api_key="sk-test",
            model="gpt-4o-mini",
            base_url="https://custom.openai.proxy/v1"
        )
        assert llm._base_url == "https://custom.openai.proxy/v1"

    def test_initialization_without_model_raises_error(self):
        """Test that initialization without model raises error."""
        with pytest.raises(LLMConfigurationError) as exc_info:
            OpenAILLM(api_key="sk-test")
        assert "model" in str(exc_info.value).lower()

    @respx.mock
    def test_chat_success(self, mock_messages, openai_api_response):
        """Test successful chat completion."""
        # Mock the API endpoint
        mock_route = respx.post(
            "https://api.openai.com/v1/chat/completions"
        ).mock(
            return_value=httpx.Response(200, json=openai_api_response)
        )

        llm = OpenAILLM(api_key="sk-test", model="gpt-4o-mini")
        response = llm.chat(mock_messages)

        assert isinstance(response, LLMResponse)
        assert response.content == "Hello! How can I help you today?"
        assert response.usage is not None
        assert response.usage["total_tokens"] == 23
        assert mock_route.called

    @respx.mock
    def test_chat_with_custom_temperature(self, mock_messages, openai_api_response):
        """Test chat with custom temperature."""
        mock_route = respx.post(
            "https://api.openai.com/v1/chat/completions"
        ).mock(
            return_value=httpx.Response(200, json=openai_api_response)
        )

        llm = OpenAILLM(api_key="sk-test", model="gpt-4o-mini", temperature=0.8)
        response = llm.chat(mock_messages)

        assert response is not None
        assert mock_route.called
        # Verify request payload contains temperature
        request_body = mock_route.calls[0].request.content
        assert b'"temperature":0.8' in request_body

    def test_chat_empty_messages_raises_error(self):
        """Test that empty messages list raises error."""
        llm = OpenAILLM(api_key="sk-test", model="gpt-4o-mini")
        with pytest.raises(LLMError) as exc_info:
            llm.chat([])
        assert "no messages" in str(exc_info.value).lower()

    @respx.mock
    def test_chat_authentication_error(self, mock_messages, error_api_response):
        """Test handling of authentication errors."""
        respx.post(
            "https://api.openai.com/v1/chat/completions"
        ).mock(
            return_value=httpx.Response(401, json=error_api_response)
        )

        llm = OpenAILLM(api_key="sk-invalid", model="gpt-4o-mini")
        with pytest.raises(LLMError) as exc_info:
            llm.chat(mock_messages)

        assert exc_info.value.provider == "openai"
        assert exc_info.value.code == 401

    def test_repr(self):
        """Test string representation."""
        llm = OpenAILLM(api_key="sk-test", model="gpt-4o-mini", temperature=0.7)
        repr_str = repr(llm)
        assert "OpenAILLM" in repr_str
        assert "openai" in repr_str
        assert "gpt-4o-mini" in repr_str


# ============================================================================
# Azure LLM Tests
# ============================================================================

class TestAzureOpenAILLM:
    """Tests for Azure OpenAI LLM implementation."""

    def test_provider_name(self):
        """Test that provider name is correct."""
        llm = AzureOpenAILLM(
            api_key="test-key",
            base_url="https://resource.openai.azure.com/openai/deployments/deployment",
            deployment="gpt-4o-mini"
        )
        assert llm.provider_name == "azure"

    def test_initialization(self):
        """Test initialization with required parameters."""
        llm = AzureOpenAILLM(
            api_key="azure-key",
            base_url="https://my-resource.openai.azure.com/openai/deployments/my-deployment",
            deployment="gpt-4o-mini"
        )
        assert llm._api_key == "azure-key"
        assert llm._deployment == "gpt-4o-mini"
        assert "my-deployment" in llm._base_url

    def test_initialization_without_base_url_raises_error(self):
        """Test that initialization without base URL raises error."""
        with pytest.raises(LLMConfigurationError) as exc_info:
            AzureOpenAILLM(api_key="test-key", deployment="test")
        assert "base_url" in str(exc_info.value).lower()

    @respx.mock
    def test_chat_success(self, mock_messages, azure_api_response):
        """Test successful chat completion."""
        mock_route = respx.post(
            url__startswith="https://my-resource.openai.azure.com/"
        ).mock(
            return_value=httpx.Response(200, json=azure_api_response)
        )

        llm = AzureOpenAILLM(
            api_key="azure-key",
            base_url="https://my-resource.openai.azure.com/openai/deployations/my-deployment",
            deployment="gpt-4o-mini"
        )
        response = llm.chat(mock_messages)

        assert isinstance(response, LLMResponse)
        assert response.content == "Hello from Azure!"
        assert mock_route.called

    def test_deployment_name_property(self):
        """Test deployment name property."""
        llm = AzureOpenAILLM(
            api_key="test-key",
            base_url="https://test.openai.azure.com/openai/deployments/my-model",
            deployment="my-model"
        )
        assert llm.deployment_name == "my-model"


# ============================================================================
# DeepSeek LLM Tests
# ============================================================================

class TestDeepSeekLLM:
    """Tests for DeepSeek LLM implementation."""

    def test_provider_name(self):
        """Test that provider name is correct."""
        llm = DeepSeekLLM(api_key="test-key", model="deepseek-chat")
        assert llm.provider_name == "deepseek"

    def test_initialization_with_defaults(self):
        """Test initialization with default values."""
        llm = DeepSeekLLM(api_key="ds-test")
        assert llm._api_key == "ds-test"
        assert llm._model == "deepseek-chat"
        assert llm._base_url == "https://api.deepseek.com"

    @respx.mock
    def test_chat_success(self, mock_messages, deepseek_api_response):
        """Test successful chat completion."""
        mock_route = respx.post(
            "https://api.deepseek.com/chat/completions"
        ).mock(
            return_value=httpx.Response(200, json=deepseek_api_response)
        )

        llm = DeepSeekLLM(api_key="ds-test", model="deepseek-chat")
        response = llm.chat(mock_messages)

        assert isinstance(response, LLMResponse)
        assert response.content == "Hello from DeepSeek!"
        assert mock_route.called

    @respx.mock
    def test_chat_with_custom_base_url(self, mock_messages, deepseek_api_response):
        """Test chat with custom base URL."""
        mock_route = respx.post(
            "https://custom.deepseek.endpoint/chat/completions"
        ).mock(
            return_value=httpx.Response(200, json=deepseek_api_response)
        )

        llm = DeepSeekLLM(
            api_key="ds-test",
            model="deepseek-chat",
            base_url="https://custom.deepseek.endpoint"
        )
        response = llm.chat(mock_messages)

        assert response is not None
        assert mock_route.called


# ============================================================================
# LLM Factory Provider Registration Tests
# ============================================================================

class TestLLMFactoryProviders:
    """Tests for LLM provider registration with factory."""

    def setup_method(self):
        """Clear factory registry before each test."""
        LLMFactory.clear()

    def teardown_method(self):
        """Clear factory registry after each test."""
        LLMFactory.clear()

    def test_register_openai_provider(self):
        """Test registering OpenAI provider."""
        LLMFactory.register("openai", OpenAILLM)
        assert LLMFactory.has_provider("openai")
        assert "openai" in LLMFactory.get_provider_names()

    def test_register_azure_provider(self):
        """Test registering Azure provider."""
        LLMFactory.register("azure", AzureOpenAILLM)
        assert LLMFactory.has_provider("azure")
        assert "azure" in LLMFactory.get_provider_names()

    def test_register_deepseek_provider(self):
        """Test registering DeepSeek provider."""
        LLMFactory.register("deepseek", DeepSeekLLM)
        assert LLMFactory.has_provider("deepseek")
        assert "deepseek" in LLMFactory.get_provider_names()

    @respx.mock
    def test_create_openai_instance(self, mock_messages, openai_api_response):
        """Test creating OpenAI instance through factory."""
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=openai_api_response)
        )

        LLMFactory.register("openai", OpenAILLM)

        # Create settings mock
        mock_settings = MagicMock()
        mock_settings.llm.provider = "openai"
        mock_settings.llm.api_key = "sk-test"
        mock_settings.llm.model = "gpt-4o-mini"
        mock_settings.llm.base_url = None
        mock_settings.llm.temperature = 0.0
        mock_settings.llm.max_tokens = None

        llm = LLMFactory.create(mock_settings)

        assert isinstance(llm, OpenAILLM)
        assert llm.provider_name == "openai"

    @respx.mock
    def test_create_deepseek_instance(self, mock_messages, deepseek_api_response):
        """Test creating DeepSeek instance through factory."""
        respx.post("https://api.deepseek.com/chat/completions").mock(
            return_value=httpx.Response(200, json=deepseek_api_response)
        )

        LLMFactory.register("deepseek", DeepSeekLLM)

        mock_settings = MagicMock()
        mock_settings.llm.provider = "deepseek"
        mock_settings.llm.api_key = "ds-test"
        mock_settings.llm.model = "deepseek-chat"
        mock_settings.llm.base_url = None
        mock_settings.llm.temperature = 0.0
        mock_settings.llm.max_tokens = None

        llm = LLMFactory.create(mock_settings)

        assert isinstance(llm, DeepSeekLLM)
        assert llm.provider_name == "deepseek"

    def test_create_unknown_provider_raises_error(self):
        """Test that unknown provider raises appropriate error."""
        mock_settings = MagicMock()
        mock_settings.llm.provider = "unknown-provider"

        with pytest.raises(LLMError) as exc_info:
            LLMFactory.create(mock_settings)

        assert "unknown" in str(exc_info.value).lower()
        assert "unknown-provider" in str(exc_info.value)


# ============================================================================
# Message Building Tests
# ============================================================================

class TestMessageBuilding:
    """Tests for message building functionality."""

    def test_single_user_message(self):
        """Test building a single user message."""
        messages = [ChatMessage(role="user", content="Hello")]
        llm = OpenAILLM(api_key="test", model="test-model")
        payload = llm._build_request_payload(messages)

        assert payload["messages"] == [{"role": "user", "content": "Hello"}]

    def test_multi_turn_conversation(self):
        """Test building multi-turn conversation."""
        messages = [
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assistant", content="Hello!"),
            ChatMessage(role="user", content="How are you?"),
        ]
        llm = OpenAILLM(api_key="test", model="test-model")
        payload = llm._build_request_payload(messages)

        assert len(payload["messages"]) == 4
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][0]["content"] == "You are helpful."


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Tests for error handling."""

    @respx.mock
    def test_rate_limit_error(self, mock_messages):
        """Test handling of rate limit errors."""
        rate_limit_response = {
            "error": {
                "message": "Rate limit exceeded",
                "type": "rate_limit_error",
                "code": "rate_limit_exceeded"
            }
        }
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(429, json=rate_limit_response)
        )

        llm = OpenAILLM(api_key="sk-test", model="gpt-4o-mini")

        with pytest.raises(LLMError) as exc_info:
            llm.chat(mock_messages)

        assert exc_info.value.provider == "openai"
        assert exc_info.value.code == 429

    @respx.mock
    def test_connection_error(self, mock_messages):
        """Test handling of connection errors."""
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=httpx.RequestError("Connection failed", request=MagicMock())
        )

        llm = OpenAILLM(api_key="sk-test", model="gpt-4o-mini")

        with pytest.raises(LLMError) as exc_info:
            llm.chat(mock_messages)

        assert "connection" in str(exc_info.value).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
