"""Tests for Ollama Embedding provider.

This module contains unit tests for OllamaEmbedding provider.
Tests use MagicMock for HTTP mocking to ensure deterministic testing.
"""

import pytest
from unittest.mock import MagicMock, patch

from libs.embedding.ollama_embedding import OllamaEmbedding
from libs.embedding.base_embedding import (
    EmbeddingResult,
    EmbeddingConfigurationError,
    EmbeddingError,
)


class TestOllamaEmbedding:
    """Tests for OllamaEmbedding provider."""

    def test_initialization_default(self):
        """Test OllamaEmbedding initialization with defaults."""
        import os
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://localhost:11434"}):
            embedding = OllamaEmbedding()

            assert embedding.provider_name == "ollama"
            assert embedding._base_url == "http://localhost:11434"
            assert embedding._model == "nomic-embed-text"
            assert embedding.dimensions == 768

    def test_initialization_custom(self):
        """Test OllamaEmbedding initialization with custom parameters."""
        embedding = OllamaEmbedding(
            base_url="http://localhost:8080",
            model="mxbai-embed-large",
            timeout=120.0
        )

        assert embedding._base_url == "http://localhost:8080"
        assert embedding._model == "mxbai-embed-large"
        assert embedding._timeout == 120.0

    def test_initialization_missing_url(self):
        """Test initialization fails without base_url."""
        import os

        # Clear OLLAMA_BASE_URL from environment
        env_backup = os.environ.get("OLLAMA_BASE_URL")
        if "OLLAMA_BASE_URL" in os.environ:
            del os.environ["OLLAMA_BASE_URL"]

        try:
            with pytest.raises(EmbeddingConfigurationError):
                embedding = OllamaEmbedding()
        finally:
            if env_backup:
                os.environ["OLLAMA_BASE_URL"] = env_backup

    def test_initialization_from_env(self):
        """Test initialization reads from OLLAMA_BASE_URL env var."""
        import os
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://custom:9999"}):
            embedding = OllamaEmbedding()

            assert embedding._base_url == "http://custom:9999"

    def test_embed_empty_texts(self):
        """Test embed with empty input."""
        import os
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://localhost:11434"}):
            embedding = OllamaEmbedding()

            # Empty input should return early without making HTTP calls
            result = embedding.embed([])

            assert isinstance(result, EmbeddingResult)
            assert result.vectors == []

    def test_embed_single_text(self):
        """Test embed with single text using mock HTTP client."""
        import os
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://localhost:11434"}):
            embedding = OllamaEmbedding()

            # Create mock HTTP client and response
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]
            }
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response

            # Use mock client
            embedding._http_client = mock_client

            result = embedding.embed(["hello"])

            assert len(result.vectors) == 1
            assert result.vectors[0] == [0.1, 0.2, 0.3, 0.4, 0.5]
            mock_client.post.assert_called_once()

    def test_embed_batch_texts(self):
        """Test embed with multiple texts."""
        import os
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://localhost:11434"}):
            embedding = OllamaEmbedding()

            # Create mock HTTP client
            mock_client = MagicMock()

            # Mock responses for each call
            def mock_post(url, json):
                mock_response = MagicMock()
                if json["input"] == "hello":
                    mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
                elif json["input"] == "world":
                    mock_response.json.return_value = {"embedding": [0.4, 0.5, 0.6]}
                else:
                    mock_response.json.return_value = {"embedding": [0.7, 0.8, 0.9]}
                mock_response.raise_for_status = MagicMock()
                return mock_response

            mock_client.post.side_effect = mock_post
            embedding._http_client = mock_client

            result = embedding.embed(["hello", "world", "test"])

            assert len(result.vectors) == 3
            assert result.vectors[0] == [0.1, 0.2, 0.3]
            assert result.vectors[1] == [0.4, 0.5, 0.6]
            assert result.vectors[2] == [0.7, 0.8, 0.9]
            assert mock_client.post.call_count == 3

    def test_embed_single_method(self):
        """Test embed_single method."""
        import os
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://localhost:11434"}):
            embedding = OllamaEmbedding()

            # Mock the embed method
            with patch.object(embedding, 'embed', return_value=EmbeddingResult(vectors=[[0.1, 0.2, 0.3]])):
                vec = embedding.embed_single("hello")

            assert vec == [0.1, 0.2, 0.3]

    def test_embed_single_empty_text(self):
        """Test embed_single with empty text."""
        import os
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://localhost:11434"}):
            embedding = OllamaEmbedding()

            vec = embedding.embed_single("")

            assert vec == []

    def test_embed_connection_failure(self):
        """Test embed handles connection failure."""
        import os
        import httpx

        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://localhost:11434"}):
            embedding = OllamaEmbedding()

            # Create mock client that raises connection error
            mock_client = MagicMock()
            mock_client.post.side_effect = httpx.ConnectError("Connection refused")
            embedding._http_client = mock_client

            with pytest.raises(EmbeddingError) as exc_info:
                embedding.embed(["test"])

            assert "Failed to connect" in str(exc_info.value)
            assert exc_info.value.provider == "ollama"

    def test_embed_timeout(self):
        """Test embed handles timeout."""
        import os
        import httpx

        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://localhost:11434"}):
            embedding = OllamaEmbedding()

            # Create mock client that raises timeout
            mock_client = MagicMock()
            mock_client.post.side_effect = httpx.TimeoutException("Request timeout")
            embedding._http_client = mock_client

            with pytest.raises(EmbeddingError) as exc_info:
                embedding.embed(["test"])

            assert "Failed to connect" in str(exc_info.value)

    def test_embed_http_error(self):
        """Test embed handles HTTP error response."""
        import os
        import httpx

        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://localhost:11434"}):
            embedding = OllamaEmbedding()

            # Create mock client that returns error response
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.json.return_value = {"error": "Model not found"}
            error = httpx.HTTPStatusError(
                "Internal Server Error",
                request=MagicMock(),
                response=mock_response
            )
            mock_client.post.side_effect = error
            embedding._http_client = mock_client

            with pytest.raises(EmbeddingError) as exc_info:
                embedding.embed(["test"])

            assert "Ollama API error" in str(exc_info.value)
            assert exc_info.value.code == 500

    def test_embed_empty_response(self):
        """Test embed handles empty response."""
        import os

        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://localhost:11434"}):
            embedding = OllamaEmbedding()

            # Create mock client that returns empty response
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {"embedding": []}
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response
            embedding._http_client = mock_client

            with pytest.raises(EmbeddingError) as exc_info:
                embedding.embed(["test"])

            assert "Empty response" in str(exc_info.value)

    def test_repr(self):
        """Test string representation."""
        import os
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://localhost:11434"}):
            embedding = OllamaEmbedding(model="custom-model")

            repr_str = repr(embedding)

            assert "OllamaEmbedding" in repr_str
            assert "custom-model" in repr_str

    def test_truncation_enabled(self):
        """Test text truncation when enabled."""
        import os
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://localhost:11434"}):
            embedding = OllamaEmbedding(truncate_length=10)

            # Create mock HTTP client
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response
            embedding._http_client = mock_client

            # Long text should be truncated
            long_text = "This is a very long text that exceeds the truncation limit"
            embedding.embed([long_text])

            # Verify truncation was applied (check payload)
            call_args = mock_client.post.call_args
            payload = call_args.kwargs.get("json", call_args[1].get("json", {}))
            assert len(payload["input"]) <= 10

    def test_truncation_disabled(self):
        """Test no truncation when disabled (default)."""
        import os
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://localhost:11434"}):
            embedding = OllamaEmbedding()  # No truncation by default

            # Create mock HTTP client
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
            mock_response.raise_for_status = MagicMock()
            mock_client.post.return_value = mock_response
            embedding._http_client = mock_client

            # Long text should not be truncated
            long_text = "This is a very long text that exceeds the truncation limit"
            embedding.embed([long_text])

            # Verify full text was sent
            call_args = mock_client.post.call_args
            payload = call_args.kwargs.get("json", call_args[1].get("json", {}))
            assert payload["input"] == long_text


class TestOllamaEmbeddingFactory:
    """Tests for EmbeddingFactory with Ollama provider."""

    def test_factory_create_ollama(self):
        """Test EmbeddingFactory can create Ollama provider."""
        from libs.embedding.embedding_factory import EmbeddingFactory

        # Register Ollama provider
        from libs.embedding.ollama_embedding import OllamaEmbedding
        EmbeddingFactory.register("ollama", OllamaEmbedding)

        # Create mock settings
        mock_settings = MagicMock()
        mock_settings.embedding.provider = "ollama"
        mock_settings.embedding.base_url = "http://localhost:11434"
        mock_settings.embedding.model = "nomic-embed-text"
        mock_settings.embedding.dimensions = None
        mock_settings.embedding.api_key = None
        mock_settings.embedding.deployment = None
        mock_settings.embedding.api_version = None

        # Test factory creation
        embedding = EmbeddingFactory.create(mock_settings)

        assert isinstance(embedding, OllamaEmbedding)
        assert embedding._model == "nomic-embed-text"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
