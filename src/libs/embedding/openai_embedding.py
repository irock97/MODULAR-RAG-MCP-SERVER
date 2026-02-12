"""OpenAI Embedding implementation.

This module provides OpenAI Embedding implementation that follows the BaseEmbedding interface.
It supports OpenAI's embeddings API and is compatible with any provider that uses
the OpenAI embeddings format.

Design Principles:
    - OpenAI-compatible: Follows OpenAI embeddings API conventions
    - Type Safe: Full type hints for all methods
    - Observable: trace parameter for tracing integration
    - Configurable: Supports all OpenAI API parameters
"""

import json
from typing import Any

import httpx

from libs.embedding.base_embedding import (
    BaseEmbedding,
    EmbeddingResult,
    EmbeddingConfigurationError,
    EmbeddingError,
)
from observability.logger import TraceContext, get_logger

logger = get_logger(__name__)


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI Embedding implementation.

    This class implements the BaseEmbedding interface for OpenAI's embeddings API.

    Attributes:
        api_key: OpenAI API key
        base_url: Base URL for the API endpoint
        model: Model name to use
        dimensions: Embedding dimensions (if supported by model)
        timeout: Request timeout in seconds
        http_client: Optional HTTP client for custom configuration

    Example:
        >>> embedding = OpenAIEmbedding(
        ...     api_key="sk-...",
        ...     model="text-embedding-3-small"
        ... )
        >>> result = embedding.embed(["Hello world"])
        >>> print(result.vectors[0][:5])
    """

    # Default model
    DEFAULT_MODEL = "text-embedding-3-small"
    # Default dimensions for the default model
    DEFAULT_DIMENSIONS = 1536

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
        timeout: float = 30.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        """Initialize the OpenAI Embedding.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            base_url: Base URL for the API. Defaults to OpenAI's official API.
            model: Model name. Defaults to text-embedding-3-small.
            dimensions: Embedding dimensions. Optional for compatible models.
            timeout: Request timeout in seconds.
            http_client: Optional pre-configured HTTP client.

        Raises:
            EmbeddingConfigurationError: If API key is not configured.
        """
        self._api_key = api_key
        self._base_url = base_url or "https://api.openai.com/v1"
        self._model = model or self.DEFAULT_MODEL
        self._dimensions = dimensions
        self._timeout = timeout
        self._http_client = http_client

        # Get API key from environment if not provided
        if not self._api_key:
            import os
            self._api_key = os.getenv("OPENAI_API_KEY")
            if not self._api_key:
                raise EmbeddingConfigurationError(
                    "OpenAI API key is not configured. Set 'api_key' in settings or "
                    "OPENAI_API_KEY env var.",
                    provider="openai"
                )

    @property
    def provider_name(self) -> str:
        """Return the name of this provider.

        Returns:
            Provider identifier: 'openai'
        """
        return "openai"

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensions.

        Returns:
            Embedding dimensions.
        """
        return self._dimensions or self.DEFAULT_DIMENSIONS

    def _build_request_payload(
        self,
        texts: list[str],
        **kwargs: Any
    ) -> dict[str, Any]:
        """Build the request payload for OpenAI embeddings API.

        Args:
            texts: List of texts to embed.
            **kwargs: Additional API parameters.

        Returns:
            Request payload dictionary.
        """
        payload: dict[str, Any] = {
            "input": texts,
            "model": self._model,
        }

        # Add dimensions if specified and model supports it
        if self._dimensions is not None:
            payload["dimensions"] = self._dimensions

        # Add encoding format if specified
        if "encoding_format" in kwargs:
            payload["encoding_format"] = kwargs["encoding_format"]

        # Add user parameter if specified
        if "user" in kwargs:
            payload["user"] = kwargs["user"]

        return payload

    def _parse_response(self, response_data: dict[str, Any]) -> EmbeddingResult:
        """Parse OpenAI API response into EmbeddingResult.

        Args:
            response_data: Raw API response dictionary.

        Returns:
            Parsed EmbeddingResult object.
        """
        data = response_data.get("data", [])
        if not data:
            raise EmbeddingError(
                "Empty response from OpenAI Embeddings API",
                provider=self.provider_name,
                details=response_data
            )

        # Extract embedding vectors
        vectors: list[list[float]] = []
        for item in data:
            embedding = item.get("embedding", [])
            vectors.append(embedding)

        # Extract usage
        usage = response_data.get("usage", {})
        usage_info: dict[str, int] | None = None

        if usage:
            usage_info = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }

        return EmbeddingResult(
            vectors=vectors,
            usage=usage_info
        )

    def _handle_http_error(
        self,
        error: httpx.HTTPStatusError,
        trace: TraceContext | None = None
    ) -> None:
        """Handle HTTP errors from OpenAI API.

        Args:
            error: HTTP status error.
            trace: Tracing context.

        Raises:
            EmbeddingError: Structured error with provider and details.
        """
        try:
            error_data = error.response.json()
            error_message = error_data.get("error", {}).get("message", str(error))
            error_type = error_data.get("error", {}).get("type", "api_error")

            raise EmbeddingError(
                f"OpenAI API error: {error_message}",
                provider=self.provider_name,
                code=error.response.status_code,
                details={
                    "error_type": error_type,
                    "status_code": error.response.status_code,
                    "response_body": error_data
                }
            )
        except json.JSONDecodeError:
            raise EmbeddingError(
                f"OpenAI API error: HTTP {error.response.status_code}",
                provider=self.provider_name,
                code=error.response.status_code,
                details={"status_code": error.response.status_code}
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
                - encoding_format: 'float' or 'base64'
                - user: User identifier for API tracking

        Returns:
            EmbeddingResult containing list of vectors.

        Raises:
            EmbeddingError: If embedding fails.
        """
        # Validate input
        if not texts:
            return EmbeddingResult(vectors=[])

        logger.info(
            f"OpenAI embedding request: model={self._model}, "
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

        # Build request
        payload = self._build_request_payload(texts, **kwargs)
        url = f"{self._base_url}/embeddings"

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": "ModularRAG-MCP-Server/1.0",
        }

        client = self._http_client or httpx.Client(timeout=self._timeout)

        try:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            response_data = response.json()

            result = self._parse_response(response_data)

            logger.info(
                f"OpenAI embedding response: vector_count={len(result.vectors)}, "
                f"dimensions={self.dimensions}"
            )

            if trace:
                trace.record_stage(
                    "embedding_response",
                    {
                        "vector_count": len(result.vectors),
                        "dimensions": self.dimensions,
                        "tokens": result.usage
                    }
                )

            return result

        except httpx.HTTPStatusError as e:
            self._handle_http_error(e, trace)
        except httpx.RequestError as e:
            raise EmbeddingError(
                f"Failed to connect to OpenAI API: {e}",
                provider=self.provider_name,
                details={"url": url, "error": str(e)}
            )
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

        Raises:
            EmbeddingError: If embedding fails.
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
            f"OpenAIEmbedding(provider={self.provider_name}, "
            f"model={self._model}, "
            f"dimensions={self.dimensions})"
        )
