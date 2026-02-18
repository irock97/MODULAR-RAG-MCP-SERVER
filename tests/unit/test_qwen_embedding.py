"""Unit tests for DashScopeEmbedding.

This module tests the DashScopeEmbedding class with mocks
to verify the implementation works correctly.

Design Principles:
    - Mock-based: Uses mocks for API responses
    - Contract Testing: Verify interface compliance
"""

from unittest.mock import MagicMock, patch

import pytest

from libs.embedding.qwen_embedding import DashScopeEmbedding


class TestDashScopeEmbedding:
    """Tests for DashScopeEmbedding class."""

    def test_initialization_with_api_key(self):
        """Test initialization with explicit API key."""
        embedding = DashScopeEmbedding(
            api_key="test-api-key",
            model="text-embedding-v4",
            dimensions=1024
        )

        assert embedding.provider_name == "qwen"
        assert embedding.dimensions == 1024
        assert embedding._model == "text-embedding-v4"

    def test_initialization_with_env_var(self, monkeypatch):
        """Test initialization using environment variable."""
        monkeypatch.setenv("DASHSCOPE_API_KEY", "env-api-key")

        embedding = DashScopeEmbedding()

        assert embedding._api_key == "env-api-key"

    def test_initialization_without_api_key_raises(self):
        """Test that initialization fails without API key."""
        import os
        # Clear the environment variable
        original_key = os.environ.get("DASHSCOPE_API_KEY")
        try:
            if "DASHSCOPE_API_KEY" in os.environ:
                del os.environ["DASHSCOPE_API_KEY"]

            with pytest.raises(Exception):
                DashScopeEmbedding()
        finally:
            if original_key is not None:
                os.environ["DASHSCOPE_API_KEY"] = original_key

    def test_build_request_payload(self):
        """Test request payload construction."""
        embedding = DashScopeEmbedding(api_key="test-key")

        payload = embedding._build_request_payload(["Hello", "World"])

        assert payload["model"] == "text-embedding-v4"
        assert payload["input"] == ["Hello", "World"]

    def test_build_request_payload_with_dimensions(self):
        """Test request payload with dimensions."""
        embedding = DashScopeEmbedding(api_key="test-key", dimensions=512)

        payload = embedding._build_request_payload(["Hello"])

        assert payload["dimensions"] == 512

    def test_get_api_url(self):
        """Test API URL construction."""
        embedding = DashScopeEmbedding(
            api_key="test-key",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        url = embedding._get_api_url()

        assert url == "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"

    def test_parse_response(self):
        """Test response parsing."""
        embedding = DashScopeEmbedding(api_key="test-key")

        response_data = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]}
            ],
            "usage": {
                "prompt_tokens": 10,
                "total_tokens": 20
            }
        }

        result = embedding._parse_response(response_data)

        assert len(result.vectors) == 2
        assert result.vectors[0] == [0.1, 0.2, 0.3]
        assert result.usage["prompt_tokens"] == 10

    def test_embed_single_text(self):
        """Test embedding a single text."""
        embedding = DashScopeEmbedding(api_key="test-key")

        # Mock the embed method
        with patch.object(embedding, 'embed') as mock_embed:
            mock_embed.return_value = MagicMock(
                vectors=[[0.1, 0.2, 0.3]]
            )

            result = embedding.embed_single("Hello")

            assert result == [0.1, 0.2, 0.3]
            mock_embed.assert_called_once_with(["Hello"], None)

    def test_embed_empty_texts(self):
        """Test embedding empty text list."""
        embedding = DashScopeEmbedding(api_key="test-key")

        result = embedding.embed([])

        assert result.vectors == []

    def test_embed_single_empty_text(self):
        """Test embedding empty single text."""
        embedding = DashScopeEmbedding(api_key="test-key")

        result = embedding.embed_single("")

        assert result == []

    def test_repr(self):
        """Test string representation."""
        embedding = DashScopeEmbedding(
            api_key="test-key",
            model="text-embedding-v3",
            dimensions=512
        )

        repr_str = repr(embedding)

        assert "DashScopeEmbedding" in repr_str
        assert "qwen" in repr_str
        assert "text-embedding-v3" in repr_str
        assert "512" in repr_str

    def test_unsupported_model_warning(self):
        """Test warning for unsupported model."""
        import logging

        with patch('libs.embedding.qwen_embedding.logger') as mock_logger:
            embedding = DashScopeEmbedding(
                api_key="test-key",
                model="custom-model"
            )

            mock_logger.warning.assert_called_once()

    def test_default_dimensions(self):
        """Test default dimensions for text-embedding-v4."""
        embedding = DashScopeEmbedding(api_key="test-key")

        assert embedding.dimensions == 1024


class TestDashScopeEmbeddingWithHttpClient:
    """Tests for DashScopeEmbedding with custom HTTP client."""

    def test_embed_with_custom_client(self):
        """Test embedding with custom HTTP client."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2]}],
            "usage": {"prompt_tokens": 5, "total_tokens": 5}
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response

        embedding = DashScopeEmbedding(
            api_key="test-key",
            http_client=mock_client
        )

        result = embedding.embed(["Hello"])

        assert result.vectors == [[0.1, 0.2]]
        mock_client.post.assert_called_once()

    def test_embed_closes_client_when_not_provided(self):
        """Test that client is closed when not provided by user."""
        embedding = DashScopeEmbedding(api_key="test-key")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1]}],
            "usage": {}
        }
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            embedding.embed(["Hello"])

            mock_client.close.assert_called_once()


class TestDashScopeEmbeddingErrorHandling:
    """Tests for DashScopeEmbedding error handling."""

    def test_http_error_handling(self):
        """Test HTTP error handling."""
        import httpx

        embedding = DashScopeEmbedding(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.json.return_value = {
            "error": {"type": "rate_limited", "message": "Rate limit exceeded"}
        }
        error = httpx.HTTPStatusError(
            "Rate limit",
            request=MagicMock(),
            response=mock_response
        )

        with pytest.raises(Exception) as exc_info:
            embedding._handle_http_error(error)

        assert "429" in str(exc_info.value)

    def test_empty_response_error(self):
        """Test error handling for empty response."""
        embedding = DashScopeEmbedding(api_key="test-key")

        with pytest.raises(Exception) as exc_info:
            embedding._parse_response({"data": []})

        assert "empty" in str(exc_info.value).lower()


class TestDashScopeEmbeddingCompleteFlow:
    """Integration tests for complete embedding flow with sample documents."""

    def test_embed_sample_document_flow(self):
        """Test complete embedding flow using sample document.

        This test simulates a real-world scenario:
        1. Read a sample document (LeetCode.md)
        2. Split it into sections based on headings
        3. Generate embeddings for each section
        4. Verify embeddings are generated correctly
        """
        import os

        # Read the sample document
        fixture_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "fixtures",
            "sample_documents",
            "LeetCode.md"
        )
        with open(fixture_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Split document into sections by headings (##)
        sections = []
        current_section = ""

        for line in content.split("\n"):
            if line.startswith("## "):
                if current_section:
                    sections.append(current_section.strip())
                current_section = line
            else:
                current_section += "\n" + line

        if current_section:
            sections.append(current_section.strip())

        # Initialize embedding with test configuration
        embedding = DashScopeEmbedding(
            api_key="test-api-key",
            model="text-embedding-v4",
            dimensions=1024
        )

        # Mock HTTP client for predictable results
        mock_client = MagicMock()
        mock_response = MagicMock()

        # Generate predictable embeddings based on section index
        def generate_mock_embedding(section_text: str, index: int) -> list[float]:
            # Create deterministic mock vectors
            return [
                float(len(section_text) % 10) * 0.1,
                float(index) * 0.1,
                float(hash(section_text) % 100) / 100.0,
                0.5,
            ]

        vectors = [
            generate_mock_embedding(text, i) for i, text in enumerate(sections)
        ]

        mock_response.json.return_value = {
            "data": [{"embedding": v} for v in vectors],
            "usage": {
                "prompt_tokens": sum(len(s.split()) for s in sections),
                "total_tokens": sum(len(s.split()) for s in sections)
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response

        embedding = DashScopeEmbedding(
            api_key="test-api-key",
            model="text-embedding-v4",
            dimensions=1024,
            http_client=mock_client
        )

        # Generate embeddings for all sections
        result = embedding.embed(sections)

        # Verify results
        assert len(result.vectors) == len(sections)
        assert all(len(v) == 4 for v in result.vectors)  # 4 dimensions from mock
        assert result.usage is not None
        assert result.usage["prompt_tokens"] > 0

        # Verify sections are embedded with correct IDs mapping
        for i, (section, vector) in enumerate(zip(sections, result.vectors)):
            assert len(vector) == 4
            assert vector[0] == float(len(section) % 10) * 0.1

    def test_embed_single_from_sample_document(self):
        """Test embedding a single section from sample document.

        This tests the embed_single method with real content from
        the sample document.
        """
        import os

        # Read the sample document
        fixture_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "fixtures",
            "sample_documents",
            "LeetCode.md"
        )
        with open(fixture_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract first section
        first_section = content.split("## ")[1].split("## ")[0].strip()

        # Mock HTTP client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3, 0.4]}],
            "usage": {"prompt_tokens": 50, "total_tokens": 50}
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response

        embedding = DashScopeEmbedding(
            api_key="test-api-key",
            model="text-embedding-v4",
            dimensions=1024,
            http_client=mock_client
        )

        # Embed single section
        vector = embedding.embed_single(first_section)

        # Verify result
        assert len(vector) == 4
        assert vector == [0.1, 0.2, 0.3, 0.4]

        # Verify HTTP call was made with correct payload
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert "json" in call_args.kwargs
        assert call_args.kwargs["json"]["input"] == [first_section]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
