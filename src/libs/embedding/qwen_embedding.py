"""DashScope Embedding implementation.

This module provides Alibaba DashScope Embedding implementation that follows
the BaseEmbedding interface. DashScope supports the text-embedding-v4 model
and is compatible with OpenAI's embedding API format.

Design Principles:
    - DashScope-compatible: Follows DashScope API conventions
    - Type Safe: Full type hints for all methods
    - Observable: trace parameter for tracing integration
    - Configurable: Supports DashScope-specific parameters

Example Configuration:
    embedding:
      provider: qwen
      model: text-embedding-v4
      api_key: your-dashscope-api-key
      base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
"""

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


class DashScopeEmbedding(BaseEmbedding):
    """DashScope Embedding implementation.

    This class implements the BaseEmbedding interface for Alibaba DashScope's
    embeddings API. It supports the text-embedding-v4 model family.

    Attributes:
        api_key: DashScope API key
        model: Model name (text-embedding-v4, text-embedding-v3)
        base_url: DashScope API base URL
        dimensions: Embedding dimensions
        timeout: Request timeout in seconds

    Example:
        >>> embedding = DashScopeEmbedding(
        ...     api_key="...",
        ...     model="text-embedding-v4"
        ... )
        >>> result = embedding.embed(["Hello world"])
    """

    # Default DashScope API base URL (compatible mode)
    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # Supported models
    SUPPORTED_MODELS = ["text-embedding-v4", "text-embedding-v3"]

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        dimensions: int | None = None,
        timeout: float = 60.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        """Initialize the DashScope Embedding.

        Args:
            api_key: DashScope API key. If None, reads from DASHSCOPE_API_KEY env var.
            model: Model name. Defaults to text-embedding-v4.
            base_url: DashScope API base URL. Defaults to compatible mode endpoint.
            dimensions: Embedding dimensions. Optional.
            timeout: Request timeout in seconds.
            http_client: Optional pre-configured HTTP client.

        Raises:
            EmbeddingConfigurationError: If required configuration is missing.
        """
        import os

        self._api_key = api_key
        self._model = model or "text-embedding-v4"
        self._base_url = base_url or self.DEFAULT_BASE_URL
        self._dimensions = dimensions
        self._timeout = timeout
        self._http_client = http_client

        # Validate model
        if self._model not in self.SUPPORTED_MODELS:
            logger.warning(
                f"Model {self._model} not in explicitly supported models: "
                f"{self.SUPPORTED_MODELS}. Proceeding anyway."
            )

        # Get API key from environment if not provided
        if not self._api_key:
            self._api_key = os.getenv("DASHSCOPE_API_KEY")
            if not self._api_key:
                raise EmbeddingConfigurationError(
                    "DashScope API key is not configured. Set 'api_key' or "
                    "DASHSCOPE_API_KEY env var.",
                    provider="qwen"
                )

    @property
    def provider_name(self) -> str:
        """Return the name of this provider.

        Returns:
            Provider identifier: 'qwen'
        """
        return "qwen"

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensions.

        Returns:
            Embedding dimensions.
        """
        return self._dimensions or 1024  # Default for text-embedding-v4

    def _build_request_payload(
        self,
        texts: list[str],
        **kwargs: Any
    ) -> dict[str, Any]:
        """Build the request payload for DashScope Embedding API.

        Args:
            texts: List of texts to embed.
            **kwargs: Additional API parameters.

        Returns:
            Request payload dictionary.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "input": texts,
        }

        # Add dimensions if specified and model supports it
        if self._dimensions is not None:
            payload["dimensions"] = self._dimensions

        return payload

    def _get_api_url(self) -> str:
        """Get the full API URL for embeddings endpoint.

        Returns:
            Complete URL for embeddings endpoint.
        """
        base = self._base_url.rstrip("/")
        return f"{base}/embeddings"

    def _parse_response(self, response_data: dict[str, Any]) -> EmbeddingResult:
        """Parse DashScope Embedding API response into EmbeddingResult.

        Args:
            response_data: Raw API response dictionary.

        Returns:
            Parsed EmbeddingResult object.
        """
        data = response_data.get("data", [])
        if not data:
            raise EmbeddingError(
                "Empty response from DashScope Embedding API",
                provider=self.provider_name,
                details=response_data
            )

        vectors: list[list[float]] = []
        for item in data:
            embedding = item.get("embedding", [])
            vectors.append(embedding)

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
        """Handle HTTP errors from DashScope API.

        Args:
            error: HTTP status error.
            trace: Tracing context.

        Raises:
            EmbeddingError: Structured error with provider and details.
        """
        try:
            error_data = error.response.json()
            error_message = error_data.get("message", str(error))
            error_type = error_data.get("type", "api_error")

            # Map common DashScope error types
            error_messages = {
                "invalid_api_key": "Invalid DashScope API key. Please check your credentials.",
                "rate_limited": "DashScope API rate limit exceeded. Please try again later.",
                "quota_exceeded": "DashScope quota exceeded. Please check your subscription.",
            }

            user_message = error_messages.get(error_type, error_message)

            raise EmbeddingError(
                f"DashScope API error: {user_message}",
                provider=self.provider_name,
                code=error.response.status_code,
                details={
                    "error_type": error_type,
                    "status_code": error.response.status_code,
                    "response_body": error_data
                }
            )

        except Exception:
            raise EmbeddingError(
                f"DashScope API error: HTTP {error.response.status_code}",
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

        Returns:
            EmbeddingResult containing list of vectors.

        Raises:
            EmbeddingError: If embedding fails.
        """
        if not texts:
            return EmbeddingResult(vectors=[])

        logger.info(
            f"DashScope embedding request: model={self._model}, "
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

        payload = self._build_request_payload(texts, **kwargs)
        url = self._get_api_url()

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
                f"DashScope embedding response: vector_count={len(result.vectors)}, "
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
                f"Failed to connect to DashScope API: {e}",
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
            f"DashScopeEmbedding("
            f"provider={self.provider_name}, "
            f"model={self._model}, "
            f"dimensions={self.dimensions})"
        )
