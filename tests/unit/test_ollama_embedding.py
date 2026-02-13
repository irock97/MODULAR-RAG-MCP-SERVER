"""Tests for Ollama Embedding provider.

This module contains unit tests for OllamaEmbedding provider.
Tests use MagicMock for HTTP mocking to ensure deterministic testing.
"""

import pytest
from unittest.mock import MagicMock, patch

from libs.embedding.ollama_embedding import OllamaEmbedding
from libs.embedding.base_embedding import EmbeddingResult, EmbeddingConfigurationError


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
        from libs.embedding.base_embedding import EmbeddingConfigurationError

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

    def test_embed_empty_texts(self):
        """Test embed with empty input."""
        import os
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://localhost:11434"}):
            embedding = OllamaEmbedding()

            # Empty input should return early without making HTTP calls
            result = embedding.embed([])

            assert isinstance(result, EmbeddingResult)
            assert result.vectors == []

    def test_embed_single(self):
        """Test embed_single method."""
        import os
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://localhost:11434"}):
            embedding = OllamaEmbedding()

            # Mock the embed method to avoid actual HTTP calls
            with patch.object(embedding, 'embed', return_value=EmbeddingResult(vectors=[[0.1, 0.2, 0.3]])):
                vec = embedding.embed_single("hello")

            assert isinstance(vec, list)
            assert vec == [0.1, 0.2, 0.3]

    def test_repr(self):
        """Test string representation."""
        import os
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://localhost:11434"}):
            embedding = OllamaEmbedding(model="custom-model")

            repr_str = repr(embedding)

            assert "OllamaEmbedding" in repr_str
            assert "custom-model" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
