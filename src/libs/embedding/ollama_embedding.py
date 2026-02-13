"""Ollama Embedding implementation.

This module provides Ollama embeddings API implementation that follows the BaseEmbedding interface.
It connects to a local Ollama server to generate embeddings.

Design Principles:
    - Local-first: Runs models locally without external API calls
    - OpenAI-compatible: Uses OpenAI-compatible embeddings API format
    - Testable: Supports mock-based testing
"""

from typing import Any

from libs.embedding.base_embedding import (
    BaseEmbedding,
    EmbeddingResult,
    EmbeddingConfigurationError,
    EmbeddingError,
)
from observability.logger import TraceContext, get_logger

logger = get_logger(__name__)


class OllamaEmbedding(BaseEmbedding):
    """Ollama embeddings API implementation.

    Connects to a local Ollama server to generate embeddings using
    Ollama's embeddings API format.

    Attributes:
        base_url: Base URL of the Ollama server
        model: Model name to use
        timeout: Request timeout in seconds
        http_client: Optional HTTP client for custom configuration
        max_tokens: Maximum number of tokens per input (0 for no limit)
        truncate_length: Truncation length for long texts (None to truncate)
    """

    # Default Ollama embedding model
    DEFAULT_MODEL = "nomic-embed-text"
    # Default dimensions for nomic-embed-text
    DEFAULT_DIMENSIONS = 768
    # Default max input length for Ollama (8192 tokens approx 32KB)
    DEFAULT_MAX_TOKENS = 8192
    # Default truncation length (None = truncate to this character limit)
    DEFAULT_TRUNCATE_LENGTH = 30000

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        http_client: Any | None = None,
        max_tokens: int = 0,
        truncate_length: int | None = None,
    ) -> None:
        """Initialize the Ollama Embedding.

        Args:
            base_url: Ollama server URL. Required if OLLAMA_BASE_URL env var not set.
            model: Model name. Defaults to nomic-embed-text.
            timeout: Request timeout in seconds.
            http_client: Optional HTTP client.
            max_tokens: Maximum tokens per input (0 = no limit).
            truncate_length: Max characters per input (None = no truncation).

        Raises:
            EmbeddingConfigurationError: If base_url is not configured.
        """
        import os

        # Get base_url from parameter or environment
        env_base_url = os.getenv("OLLAMA_BASE_URL")

        # base_url parameter takes precedence over environment
        if base_url is not None:
            self._base_url = base_url
        elif env_base_url is not None:
            self._base_url = env_base_url
        else:
            self._base_url = None

        self._model = model or self.DEFAULT_MODEL
        self._timeout = timeout
        self._http_client = http_client
        self._max_tokens = max_tokens if max_tokens > 0 else None
        self._truncate_length = truncate_length

        if not self._base_url:
            raise EmbeddingConfigurationError(
                "Ollama base_url is required. Set 'base_url' in settings or "
                "OLLAMA_BASE_URL env var.",
                provider="ollama"
            )

    def _truncate_text(self, text: str) -> str:
        """Truncate text to max length.

        Args:
            text: Input text.

        Returns:
            Truncated text.
        """
        if self._truncate_length and len(text) > self._truncate_length:
            return text[:self._truncate_length]
        return text

    @property
    def provider_name(self) -> str:
        """Return the name of this provider.

        Returns:
            Provider identifier: 'ollama'
        """
        return "ollama"

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensions.

        Returns:
            Default embedding dimensions.
        """
        return self.DEFAULT_DIMENSIONS

    def _parse_response(self, response_data: dict[str, Any]) -> EmbeddingResult:
        """Parse Ollama API response into EmbeddingResult.

        Args:
            response_data: Raw API response dictionary.

        Returns:
            Parsed EmbeddingResult object.
        """
        # Ollama embeddings API returns embeddings in 'embeddings' array for batch input
        # or 'embedding' for single input
        embeddings = response_data.get("embeddings", response_data.get("embedding", []))
        if not embeddings:
            raise EmbeddingError(
                "Empty response from Ollama Embeddings API",
                provider=self.provider_name,
                details=response_data
            )

        # Ensure we have a list of embeddings
        if isinstance(embeddings[0], (int, float)):
            # Single embedding returned (old API format)
            embeddings = [embeddings]

        return EmbeddingResult(
            vectors=embeddings,
        )

    def _embed_single(self, text: str, client: Any) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Single text string to embed.
            client: HTTP client to use.

        Returns:
            Single embedding vector.
        """
        import httpx

        # Apply truncation if configured
        text_to_embed = self._truncate_text(text)

        url = f"{self._base_url}/api/embeddings"
        payload = {
            "model": self._model,
            "input": text_to_embed,
        }

        try:
            response = client.post(url, json=payload)
            response.raise_for_status()
            response_data = response.json()

            result = self._parse_response(response_data)
            if result.vectors:
                return result.vectors[0]
            return []
        except httpx.HTTPStatusError as e:
            try:
                error_data = e.response.json()
                error_message = error_data.get("error", str(e))
            except Exception:
                error_message = str(e)

            raise EmbeddingError(
                f"Ollama API error: {error_message}",
                provider=self.provider_name,
                code=e.response.status_code,
            )
        except httpx.RequestError as e:
            raise EmbeddingError(
                f"Failed to connect to Ollama API: {e}",
                provider=self.provider_name,
                details={"url": url, "error": str(e)}
            )

    def embed(
        self,
        texts: list[str],
        trace: TraceContext | None = None,
        **kwargs: Any
    ) -> EmbeddingResult:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.
            trace: Tracing context for observability.
            **kwargs: Additional provider-specific arguments.

        Returns:
            EmbeddingResult containing list of vectors.

        Raises:
            EmbeddingError: If embedding fails.
        """
        if not texts:
            return EmbeddingResult(vectors=[])

        logger.info(
            f"Ollama embedding request: model={self._model}, "
            f"text_count={len(texts)}"
        )

        if trace:
            trace.record_stage(
                "embedding_request",
                {
                    "provider": self.provider_name,
                    "model": self._model,
                    "text_count": len(texts)
                }
            )

        import httpx

        client = self._http_client or httpx.Client(timeout=self._timeout)

        try:
            # Handle single text directly
            if len(texts) == 1:
                embedding = self._embed_single(texts[0], client)
                result = EmbeddingResult(vectors=[embedding])
            else:
                # For batch, we need to call API for each text
                # Ollama embeddings API currently only supports single input
                embeddings = []
                for text in texts:
                    embedding = self._embed_single(text, client)
                    embeddings.append(embedding)
                result = EmbeddingResult(vectors=embeddings)

            logger.info(
                f"Ollama embedding response: vector_count={len(result.vectors)}, "
                f"dimensions={self.dimensions}"
            )

            if trace:
                trace.record_stage(
                    "embedding_response",
                    {
                        "vector_count": len(result.vectors),
                        "dimensions": self.dimensions,
                    }
                )

            return result

        finally:
            if self._http_client is None:
                client.close()

    def embed_single(
        self,
        text: str,
        trace: TraceContext | None = None,
        **kwargs: Any
    ) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Single text string to embed.
            trace: Tracing context for observability.
            **kwargs: Additional arguments.

        Returns:
            Single embedding vector.
        """
        if not text:
            return []

        result = self.embed([text], trace, **kwargs)

        if result.vectors:
            return result.vectors[0]

        return []

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"OllamaEmbedding("
            f"provider={self.provider_name}, "
            f"model={self._model}, "
            f"dimensions={self.dimensions})"
        )
