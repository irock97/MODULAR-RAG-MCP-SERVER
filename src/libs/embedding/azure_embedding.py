"""Azure OpenAI Embedding implementation.

This module provides Azure OpenAI Embedding implementation that follows the BaseEmbedding interface.
Azure OpenAI uses the OpenAI API format but with additional authentication
(Azure AD, API key) and deployment naming conventions.

Design Principles:
    - Azure-compatible: Follows Azure OpenAI API conventions
    - Type Safe: Full type hints for all methods
    - Observable: trace parameter for tracing integration
    - Configurable: Supports Azure-specific parameters (deployment, api-version)

Example Configuration:
    embedding:
      provider: azure
      api_key: your-azure-api-key
      base_url: https://resource-name.openai.azure.com/openai/deployments/deployment-name
      model: text-embedding-3-small
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

# Default Azure OpenAI API version
DEFAULT_AZURE_API_VERSION = "2024-02-15-preview"


class AzureOpenAIEmbedding(BaseEmbedding):
    """Azure OpenAI Embedding implementation.

    This class implements the BaseEmbedding interface for Azure OpenAI's embeddings API.

    Attributes:
        api_key: Azure API key
        base_url: Base URL for the Azure OpenAI endpoint
        deployment: Deployment name (model deployment)
        api_version: API version string
        dimensions: Embedding dimensions (if supported)
        timeout: Request timeout in seconds

    Example:
        >>> embedding = AzureOpenAIEmbedding(
        ...     api_key="...",
        ...     base_url="https://my-resource.openai.azure.com/openai/deployments/my-deployment",
        ...     model="text-embedding-3-small"
        ... )
        >>> result = embedding.embed(["Hello world"])
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        deployment: str | None = None,
        api_version: str = DEFAULT_AZURE_API_VERSION,
        model: str | None = None,
        dimensions: int | None = None,
        timeout: float = 60.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        """Initialize the Azure OpenAI Embedding.

        Args:
            api_key: Azure API key. If None, reads from AZURE_OPENAI_API_KEY env var.
            base_url: Base URL for the API endpoint.
            deployment: Deployment name (model deployment).
            api_version: API version string. Defaults to a recent stable version.
            model: Optional model name (for reference/tracking).
            dimensions: Embedding dimensions. Optional for compatible models.
            timeout: Request timeout in seconds.
            http_client: Optional pre-configured HTTP client.

        Raises:
            EmbeddingConfigurationError: If required configuration is missing.
        """
        self._api_key = api_key
        self._base_url = base_url
        self._deployment = deployment
        self._api_version = api_version
        self._model = model
        self._dimensions = dimensions
        self._timeout = timeout
        self._http_client = http_client

        # Validate required fields
        if not self._deployment and not self._base_url:
            raise EmbeddingConfigurationError(
                "Azure deployment name is required. Set 'deployment' in settings.",
                provider="azure"
            )

        # Validate base_url format
        if not self._base_url:
            raise EmbeddingConfigurationError(
                "Azure base_url is required. Set 'base_url' in settings.",
                provider="azure"
            )

        # Get API key from environment if not provided
        if not self._api_key:
            import os
            self._api_key = os.getenv("AZURE_OPENAI_API_KEY")
            if not self._api_key:
                raise EmbeddingConfigurationError(
                    "Azure API key is not configured. Set 'api_key' in settings or "
                    "AZURE_OPENAI_API_KEY env var.",
                    provider="azure"
                )

    @property
    def provider_name(self) -> str:
        """Return the name of this provider.

        Returns:
            Provider identifier: 'azure'
        """
        return "azure"

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensions.

        Returns:
            Embedding dimensions.
        """
        return self._dimensions or 1536  # Default for text-embedding-3-small

    @property
    def deployment_name(self) -> str:
        """Return the deployment name.

        Returns:
            Deployment name configured for this instance.
        """
        return self._deployment or ""

    def _build_request_payload(
        self,
        texts: list[str],
        **kwargs: Any
    ) -> dict[str, Any]:
        """Build the request payload for Azure OpenAI API.

        Args:
            texts: List of texts to embed.
            **kwargs: Additional API parameters.

        Returns:
            Request payload dictionary.
        """
        payload: dict[str, Any] = {
            "input": texts,
        }

        # Add dimensions if specified and model supports it
        if self._dimensions is not None:
            payload["dimensions"] = self._dimensions

        # Add encoding format if specified
        if "encoding_format" in kwargs:
            payload["encoding_format"] = kwargs["encoding_format"]

        return payload

    def _get_api_url(self) -> str:
        """Get the full API URL for embeddings endpoint.

        Returns:
            Complete URL for embeddings endpoint.
        """
        # Azure format: {base_url}/embeddings?api-version={api_version}
        base = self._base_url.rstrip("/")
        return f"{base}/embeddings?api-version={self._api_version}"

    def _parse_response(self, response_data: dict[str, Any]) -> EmbeddingResult:
        """Parse Azure OpenAI API response into EmbeddingResult.

        Args:
            response_data: Raw API response dictionary.

        Returns:
            Parsed EmbeddingResult object.
        """
        data = response_data.get("data", [])
        if not data:
            raise EmbeddingError(
                "Empty response from Azure OpenAI Embeddings API",
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
        """Handle HTTP errors from Azure OpenAI API.

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
            code = error_data.get("error", {}).get("code")

            raise EmbeddingError(
                f"Azure OpenAI API error: {error_message}",
                provider=self.provider_name,
                code=code or error.response.status_code,
                details={
                    "error_type": error_type,
                    "status_code": error.response.status_code,
                    "deployment": self._deployment,
                    "response_body": error_data
                }
            )
        except json.JSONDecodeError:
            raise EmbeddingError(
                f"Azure OpenAI API error: HTTP {error.response.status_code}",
                provider=self.provider_name,
                code=error.response.status_code,
                details={
                    "status_code": error.response.status_code,
                    "deployment": self._deployment
                }
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

        Returns:
            EmbeddingResult containing list of vectors.

        Raises:
            EmbeddingError: If embedding fails.
        """
        if not texts:
            return EmbeddingResult(vectors=[])

        logger.info(
            f"Azure OpenAI embedding request: deployment={self._deployment}, "
            f"text_count={len(texts)}"
        )

        if trace:
            trace.record_stage(
                "embedding_request",
                {
                    "provider": self.provider_name,
                    "deployment": self._deployment,
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
                f"Azure OpenAI embedding response: vector_count={len(result.vectors)}"
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
                f"Failed to connect to Azure OpenAI API: {e}",
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
            f"AzureOpenAIEmbedding(provider={self.provider_name}, "
            f"deployment={self._deployment}, "
            f"dimensions={self.dimensions})"
        )
