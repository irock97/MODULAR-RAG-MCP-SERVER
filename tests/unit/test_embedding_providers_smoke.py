"""Smoke tests for Embedding providers (OpenAI, Azure).

This module contains unit tests for OpenAI and Azure Embedding providers.
Tests focus on configuration, message building, and basic functionality.

Design Principles:
    - Deterministic: All tests use fixed responses
    - Coverage: Tests provider creation, configuration, and message building
"""

import pytest
from unittest.mock import MagicMock
from typing import Any

import httpx

from libs.embedding.base_embedding import (
    EmbeddingResult,
    EmbeddingConfigurationError,
    EmbeddingError,
)
from libs.embedding.openai_embedding import OpenAIEmbedding
from libs.embedding.azure_embedding import AzureOpenAIEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_texts() -> list[str]:
    """Create sample texts for testing."""
    return [
        "Hello world",
        "This is a test",
        "OpenAI embeddings are useful"
    ]


@pytest.fixture
def openai_embedding_response() -> dict[str, Any]:
    """Create a mock OpenAI embeddings API response."""
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "index": 0,
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]
            },
            {
                "object": "embedding",
                "index": 1,
                "embedding": [0.6, 0.7, 0.8, 0.9, 1.0]
            },
            {
                "object": "embedding",
                "index": 2,
                "embedding": [0.11, 0.22, 0.33, 0.44, 0.55]
            }
        ],
        "model": "text-embedding-3-small",
        "usage": {
            "prompt_tokens": 15,
            "total_tokens": 15
        }
    }


@pytest.fixture
def openai_error_response() -> dict[str, Any]:
    """Create a mock OpenAI API error response."""
    return {
        "error": {
            "message": "Invalid API key",
            "type": "authentication_error",
            "code": "invalid_api_key"
        }
    }


# ============================================================================
# OpenAI Embedding Tests
# ============================================================================

class TestOpenAIEmbedding:
    """Tests for OpenAI Embedding implementation."""

    def test_provider_name(self):
        """Test that provider name is correct."""
        embedding = OpenAIEmbedding(api_key="test-key")
        assert embedding.provider_name == "openai"

    def test_initialization_with_defaults(self):
        """Test initialization with default values."""
        embedding = OpenAIEmbedding(api_key="sk-test")
        assert embedding._model == "text-embedding-3-small"
        assert embedding._base_url == "https://api.openai.com/v1"
        assert embedding._dimensions is None

    def test_initialization_with_custom_values(self):
        """Test initialization with custom values."""
        embedding = OpenAIEmbedding(
            api_key="sk-test",
            base_url="https://custom.openai.proxy/v1",
            model="text-embedding-3-large",
            dimensions=1024,
            timeout=60.0
        )
        assert embedding._api_key == "sk-test"
        assert embedding._base_url == "https://custom.openai.proxy/v1"
        assert embedding._model == "text-embedding-3-large"
        assert embedding._dimensions == 1024

    def test_initialization_without_api_key_raises_error(self):
        """Test that initialization without API key raises error."""
        with pytest.raises(EmbeddingConfigurationError) as exc_info:
            OpenAIEmbedding()
        assert "api key" in str(exc_info.value).lower()

    def test_dimensions_property_with_default(self):
        """Test dimensions property returns default when not set."""
        embedding = OpenAIEmbedding(api_key="sk-test")
        assert embedding.dimensions == 1536  # default for text-embedding-3-small

    def test_dimensions_property_with_custom(self):
        """Test dimensions property returns custom value."""
        embedding = OpenAIEmbedding(api_key="sk-test", dimensions=1024)
        assert embedding.dimensions == 1024

    def test_repr(self):
        """Test string representation."""
        embedding = OpenAIEmbedding(api_key="sk-test", model="text-embedding-3-large")
        repr_str = repr(embedding)
        assert "OpenAIEmbedding" in repr_str
        assert "openai" in repr_str
        assert "text-embedding-3-large" in repr_str


# ============================================================================
# Embedding Factory Tests
# ============================================================================

class TestEmbeddingFactory:
    """Tests for Embedding provider registration with factory."""

    def setup_method(self):
        """Clear factory registry before each test."""
        EmbeddingFactory.clear()
        EmbeddingFactory.register("openai", OpenAIEmbedding)

    def teardown_method(self):
        """Clear factory registry after each test."""
        EmbeddingFactory.clear()

    def test_openai_provider_registered(self):
        """Test that OpenAI provider is registered."""
        assert EmbeddingFactory.has_provider("openai")
        assert "openai" in EmbeddingFactory.get_provider_names()

    def test_create_openai_instance(self):
        """Test creating OpenAI instance through factory."""
        mock_settings = MagicMock()
        mock_settings.embedding.provider = "openai"
        mock_settings.embedding.api_key = "sk-test"
        mock_settings.embedding.model = "text-embedding-3-small"
        mock_settings.embedding.base_url = None
        mock_settings.embedding.dimensions = None

        embedding = EmbeddingFactory.create(mock_settings)

        assert isinstance(embedding, OpenAIEmbedding)
        assert embedding.provider_name == "openai"
        assert embedding._model == "text-embedding-3-small"

    def test_create_openai_with_custom_dimensions(self):
        """Test creating OpenAI instance with custom dimensions."""
        mock_settings = MagicMock()
        mock_settings.embedding.provider = "openai"
        mock_settings.embedding.api_key = "sk-test"
        mock_settings.embedding.model = "text-embedding-3-small"
        mock_settings.embedding.base_url = None
        mock_settings.embedding.dimensions = 1024

        embedding = EmbeddingFactory.create(mock_settings)

        assert isinstance(embedding, OpenAIEmbedding)
        assert embedding._dimensions == 1024


# ============================================================================
# Message Building Tests
# ============================================================================

class TestOpenAIEmbeddingMessageBuilding:
    """Tests for OpenAI Embedding message building functionality."""

    def test_single_text(self):
        """Test building payload with a single text."""
        embedding = OpenAIEmbedding(api_key="sk-test")
        payload = embedding._build_request_payload(["Hello"])

        assert payload["input"] == ["Hello"]
        assert payload["model"] == "text-embedding-3-small"

    def test_multiple_texts(self, sample_texts):
        """Test building payload with multiple texts."""
        embedding = OpenAIEmbedding(api_key="sk-test")
        payload = embedding._build_request_payload(sample_texts)

        assert payload["input"] == sample_texts
        assert len(payload["input"]) == 3

    def test_with_custom_dimensions(self):
        """Test building payload with custom dimensions."""
        embedding = OpenAIEmbedding(api_key="sk-test", dimensions=1024)
        payload = embedding._build_request_payload(["Hello"])

        assert payload["dimensions"] == 1024

    def test_with_encoding_format(self):
        """Test building payload with encoding format."""
        embedding = OpenAIEmbedding(api_key="sk-test")
        payload = embedding._build_request_payload(["Hello"], encoding_format="base64")

        assert payload["encoding_format"] == "base64"

    def test_with_user_parameter(self):
        """Test building payload with user parameter."""
        embedding = OpenAIEmbedding(api_key="sk-test")
        payload = embedding._build_request_payload(["Hello"], user="user-123")

        assert payload["user"] == "user-123"


# ============================================================================
# Response Parsing Tests
# ============================================================================

class TestOpenAIEmbeddingResponseParsing:
    """Tests for OpenAI Embedding response parsing."""

    def test_parse_response_with_usage(self, openai_embedding_response):
        """Test parsing response with usage info."""
        embedding = OpenAIEmbedding(api_key="sk-test")
        result = embedding._parse_response(openai_embedding_response)

        assert isinstance(result, EmbeddingResult)
        assert len(result.vectors) == 3
        assert len(result.vectors[0]) == 5
        assert result.usage is not None
        assert result.usage["total_tokens"] == 15

    def test_parse_empty_response(self):
        """Test parsing empty response raises error."""
        embedding = OpenAIEmbedding(api_key="sk-test")

        with pytest.raises(EmbeddingError) as exc_info:
            embedding._parse_response({"data": []})

        assert "empty" in str(exc_info.value).lower()


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestOpenAIEmbeddingErrorHandling:
    """Tests for OpenAI Embedding error handling."""

    def test_handle_http_error(self):
        """Test HTTP error handling."""
        embedding = OpenAIEmbedding(api_key="sk-test")

        error_response = {
            "error": {
                "message": "Invalid API key",
                "type": "authentication_error"
            }
        }

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = error_response

        error = httpx.HTTPStatusError(
            "Error",
            request=MagicMock(),
            response=mock_response
        )

        with pytest.raises(EmbeddingError) as exc_info:
            embedding._handle_http_error(error)

        assert exc_info.value.provider == "openai"
        assert "invalid api key" in str(exc_info.value).lower()


# ============================================================================
# Embedding Generation Tests
# ============================================================================

class TestOpenAIEmbeddingGeneration:
    """Tests for OpenAI Embedding generation functionality."""

    def test_embed_empty_list(self):
        """Test embedding empty list returns empty result."""
        embedding = OpenAIEmbedding(api_key="sk-test")
        result = embedding.embed([])

        assert isinstance(result, EmbeddingResult)
        assert result.vectors == []

    def test_embed_single_text(self):
        """Test embedding a single text returns one vector."""
        embedding = OpenAIEmbedding(api_key="sk-test")
        embedding._parse_response = MagicMock(return_value=EmbeddingResult(
            vectors=[[0.1, 0.2, 0.3]],
            usage={"total_tokens": 2}
        ))

        # Mock the HTTP client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response
        embedding._http_client = mock_client

        result = embedding.embed(["Hello"])

        assert len(result.vectors) == 1

    def test_embed_single_helper(self):
        """Test embed_single helper method."""
        embedding = OpenAIEmbedding(api_key="sk-test")
        embedding.embed = MagicMock(return_value=EmbeddingResult(
            vectors=[[0.1, 0.2, 0.3]],
            usage={"total_tokens": 2}
        ))

        result = embedding.embed_single("Hello")

        assert result == [0.1, 0.2, 0.3]

    def test_embed_single_empty_text(self):
        """Test embed_single with empty text returns empty list."""
        embedding = OpenAIEmbedding(api_key="sk-test")
        result = embedding.embed_single("")

        assert result == []


# ============================================================================
# Azure OpenAI Embedding Tests
# ============================================================================

class TestAzureOpenAIEmbedding:
    """Tests for Azure OpenAI Embedding implementation."""

    def test_provider_name(self):
        """Test that provider name is correct."""
        embedding = AzureOpenAIEmbedding(
            api_key="test-key",
            base_url="https://resource.openai.azure.com/openai/deployments/deployment",
            deployment="embedding-deployment"
        )
        assert embedding.provider_name == "azure"

    def test_initialization(self):
        """Test initialization with required parameters."""
        embedding = AzureOpenAIEmbedding(
            api_key="azure-key",
            base_url="https://my-resource.openai.azure.com/openai/deployments/my-deployment",
            deployment="text-embedding-3-small"
        )
        assert embedding._api_key == "azure-key"
        assert embedding._deployment == "text-embedding-3-small"
        assert "my-deployment" in embedding._base_url

    def test_initialization_without_base_url_raises_error(self):
        """Test that initialization without base URL raises error."""
        with pytest.raises(EmbeddingConfigurationError) as exc_info:
            AzureOpenAIEmbedding(api_key="test-key", deployment="test")
        assert "base_url" in str(exc_info.value).lower()

    def test_deployment_name_property(self):
        """Test deployment name property."""
        embedding = AzureOpenAIEmbedding(
            api_key="test-key",
            base_url="https://test.openai.azure.com/openai/deployments/my-model",
            deployment="my-model"
        )
        assert embedding.deployment_name == "my-model"

    def test_repr(self):
        """Test string representation."""
        embedding = AzureOpenAIEmbedding(
            api_key="test-key",
            base_url="https://test.openai.azure.com/openai/deployments/my-model",
            deployment="my-model"
        )
        repr_str = repr(embedding)
        assert "AzureOpenAIEmbedding" in repr_str
        assert "azure" in repr_str
        assert "my-model" in repr_str


# ============================================================================
# Azure Embedding Factory Tests
# ============================================================================

class TestAzureEmbeddingFactory:
    """Tests for Azure Embedding provider registration with factory."""

    def setup_method(self):
        """Clear factory registry before each test."""
        EmbeddingFactory.clear()
        EmbeddingFactory.register("azure", AzureOpenAIEmbedding)

    def teardown_method(self):
        """Clear factory registry after each test."""
        EmbeddingFactory.clear()

    def test_azure_provider_registered(self):
        """Test that Azure provider is registered."""
        assert EmbeddingFactory.has_provider("azure")
        assert "azure" in EmbeddingFactory.get_provider_names()

    def test_create_azure_instance(self):
        """Test creating Azure instance through factory."""
        mock_settings = MagicMock()
        mock_settings.embedding.provider = "azure"
        mock_settings.embedding.api_key = "azure-key"
        mock_settings.embedding.deployment = "text-embedding-3-small"
        mock_settings.embedding.base_url = "https://my-resource.openai.azure.com/openai/deployments/my-deployment"
        mock_settings.embedding.model = None
        mock_settings.embedding.dimensions = None
        mock_settings.embedding.api_version = "2024-02-15-preview"

        embedding = EmbeddingFactory.create(mock_settings)

        assert isinstance(embedding, AzureOpenAIEmbedding)
        assert embedding.provider_name == "azure"
        assert embedding._deployment == "text-embedding-3-small"


# ============================================================================
# Azure Message Building Tests
# ============================================================================

class TestAzureEmbeddingMessageBuilding:
    """Tests for Azure Embedding message building functionality."""

    def test_single_text(self):
        """Test building payload with a single text."""
        embedding = AzureOpenAIEmbedding(
            api_key="test-key",
            base_url="https://test.openai.azure.com/openai/deployments/my-model",
            deployment="my-model"
        )
        payload = embedding._build_request_payload(["Hello"])

        assert payload["input"] == ["Hello"]

    def test_multiple_texts(self, sample_texts):
        """Test building payload with multiple texts."""
        embedding = AzureOpenAIEmbedding(
            api_key="test-key",
            base_url="https://test.openai.azure.com/openai/deployments/my-model",
            deployment="my-model"
        )
        payload = embedding._build_request_payload(sample_texts)

        assert payload["input"] == sample_texts
        assert len(payload["input"]) == 3

    def test_with_custom_dimensions(self):
        """Test building payload with custom dimensions."""
        embedding = AzureOpenAIEmbedding(
            api_key="test-key",
            base_url="https://test.openai.azure.com/openai/deployments/my-model",
            deployment="my-model",
            dimensions=1024
        )
        payload = embedding._build_request_payload(["Hello"])

        assert payload["dimensions"] == 1024


# ============================================================================
# Azure Response Parsing Tests
# ============================================================================

class TestAzureEmbeddingResponseParsing:
    """Tests for Azure Embedding response parsing."""

    def test_parse_response_with_usage(self, openai_embedding_response):
        """Test parsing response with usage info."""
        embedding = AzureOpenAIEmbedding(
            api_key="test-key",
            base_url="https://test.openai.azure.com/openai/deployments/my-model",
            deployment="my-model"
        )
        result = embedding._parse_response(openai_embedding_response)

        assert isinstance(result, EmbeddingResult)
        assert len(result.vectors) == 3


# ============================================================================
# Azure Error Handling Tests
# ============================================================================

class TestAzureEmbeddingErrorHandling:
    """Tests for Azure Embedding error handling."""

    def test_handle_http_error(self):
        """Test HTTP error handling."""
        embedding = AzureOpenAIEmbedding(
            api_key="test-key",
            base_url="https://test.openai.azure.com/openai/deployments/my-model",
            deployment="my-model"
        )

        error_response = {
            "error": {
                "message": "Deployment not found",
                "type": "resource_not_found_error"
            }
        }

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = error_response

        error = httpx.HTTPStatusError(
            "Error",
            request=MagicMock(),
            response=mock_response
        )

        with pytest.raises(EmbeddingError) as exc_info:
            embedding._handle_http_error(error)

        assert exc_info.value.provider == "azure"


# ============================================================================
# Azure Embedding Generation Tests
# ============================================================================

class TestAzureEmbeddingGeneration:
    """Tests for Azure Embedding generation functionality."""

    def test_embed_empty_list(self):
        """Test embedding empty list returns empty result."""
        embedding = AzureOpenAIEmbedding(
            api_key="test-key",
            base_url="https://test.openai.azure.com/openai/deployments/my-model",
            deployment="my-model"
        )
        result = embedding.embed([])

        assert isinstance(result, EmbeddingResult)
        assert result.vectors == []

    def test_embed_single_empty_text(self):
        """Test embed_single with empty text returns empty list."""
        embedding = AzureOpenAIEmbedding(
            api_key="test-key",
            base_url="https://test.openai.azure.com/openai/deployments/my-model",
            deployment="my-model"
        )
        result = embedding.embed_single("")

        assert result == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
