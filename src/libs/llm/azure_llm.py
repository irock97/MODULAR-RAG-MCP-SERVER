"""Azure OpenAI LLM implementation.

This module provides Azure OpenAI LLM implementation that follows the BaseLLM interface.
Azure OpenAI uses the OpenAI API format but with additional authentication
(Azure AD, API key) and deployment naming conventions.

Design Principles:
    - Azure-compatible: Follows Azure OpenAI API conventions
    - Type Safe: Full type hints for all methods
    - Observable: trace parameter for tracing integration
    - Configurable: Supports Azure-specific parameters (deployment, api-version)

Example Configuration:
    llm:
      provider: azure
      api_key: your-azure-api-key
      base_url: https://resource-name.openai.azure.com/openai/deployments/deployment-name
      model: gpt-4o-mini
      azure_api_version: 2024-02-15-preview
"""

import json
from typing import Any

import httpx

from libs.llm.base_llm import (
    BaseLLM,
    ChatMessage,
    LLMResponse,
    LLMConfigurationError,
    LLMError,
)
from observability.logger import TraceContext, get_logger

logger = get_logger(__name__)

# Default Azure OpenAI API version
DEFAULT_AZURE_API_VERSION = "2024-02-15-preview"


class AzureOpenAILLM(BaseLLM):
    """Azure OpenAI LLM implementation.

    This class implements the BaseLLM interface for Azure OpenAI's chat completions API.

    Attributes:
        api_key: Azure API key
        base_url: Base URL for the Azure OpenAI endpoint
        deployment: Deployment name (model deployment)
        api_version: API version string
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds

    Example:
        >>> llm = AzureOpenAILLM(
        ...     api_key="...",
        ...     base_url="https://my-resource.openai.azure.com/openai/deployments/my-deployment",
        ...     deployment="gpt-4o-mini"
        ... )
        >>> response = llm.chat([ChatMessage(role="user", content="Hello")])
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        deployment: str | None = None,
        api_version: str = DEFAULT_AZURE_API_VERSION,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: float = 60.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        """Initialize the Azure OpenAI LLM.

        Args:
            api_key: Azure API key. If None, reads from AZURE_OPENAI_API_KEY env var.
            base_url: Base URL for the API endpoint.
            deployment: Deployment name (model deployment).
            api_version: API version string. Defaults to a recent stable version.
            model: Optional model name (for reference/tracking).
            temperature: Sampling temperature (0.0-2.0).
            max_tokens: Maximum tokens to generate.
            timeout: Request timeout in seconds.
            http_client: Optional pre-configured HTTP client.

        Raises:
            LLMConfigurationError: If required configuration is missing.
        """
        self._api_key = api_key
        self._base_url = base_url
        self._deployment = deployment
        self._api_version = api_version
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._http_client = http_client

        # Validate required fields
        if not self._deployment and not self._base_url:
            raise LLMConfigurationError(
                "Azure deployment name is required. Set 'deployment' in settings.",
                provider="azure"
            )

        # Validate base_url format
        if not self._base_url:
            raise LLMConfigurationError(
                "Azure base_url is required. Set 'base_url' in settings.",
                provider="azure"
            )

        # Get API key from environment if not provided
        if not self._api_key:
            import os
            self._api_key = os.getenv("AZURE_OPENAI_API_KEY")
            if not self._api_key:
                raise LLMConfigurationError(
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
    def deployment_name(self) -> str:
        """Return the deployment name.

        Returns:
            Deployment name configured for this instance.
        """
        return self._deployment or ""

    def _build_request_payload(
        self,
        messages: list[ChatMessage],
        **kwargs: Any
    ) -> dict[str, Any]:
        """Build the request payload for Azure OpenAI API.

        Args:
            messages: List of chat messages.
            **kwargs: Additional API parameters.

        Returns:
            Request payload dictionary.
        """
        # Convert ChatMessage objects to API format
        messages_data = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        # Build payload (Azure uses same format as OpenAI)
        payload: dict[str, Any] = {
            "messages": messages_data,
            "temperature": kwargs.get("temperature", self._temperature),
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
        }

        # Add optional parameters
        if kwargs.get("stream"):
            payload["stream"] = True

        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]

        if "presence_penalty" in kwargs:
            payload["presence_penalty"] = kwargs["presence_penalty"]

        if "frequency_penalty" in kwargs:
            payload["frequency_penalty"] = kwargs["frequency_penalty"]

        return payload

    def _get_api_url(self) -> str:
        """Get the full API URL for chat completions.

        Returns:
            Complete URL for chat completions endpoint.
        """
        # Azure format: {base_url}/chat/completions?api-version={api_version}
        # The base_url should already include /openai/deployments/{deployment}
        base = self._base_url.rstrip("/")
        return f"{base}/chat/completions?api-version={self._api_version}"

    def _parse_response(self, response_data: dict[str, Any]) -> LLMResponse:
        """Parse Azure OpenAI API response into LLMResponse.

        Args:
            response_data: Raw API response dictionary.

        Returns:
            Parsed LLMResponse object.
        """
        # Same parsing as OpenAI
        choices = response_data.get("choices", [])
        if not choices:
            raise LLMError(
                "Empty response from Azure OpenAI API",
                provider=self.provider_name,
                details=response_data
            )

        message = choices[0].get("message", {})
        content = message.get("content", "")

        usage = response_data.get("usage", {})
        usage_info: dict[str, int] = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }

        return LLMResponse(
            content=content,
            raw_response=response_data,
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
            LLMError: Structured error with provider and details.
        """
        try:
            error_data = error.response.json()
            error_message = error_data.get("error", {}).get("message", str(error))
            error_type = error_data.get("error", {}).get("type", "api_error")
            code = error_data.get("error", {}).get("code")

            raise LLMError(
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
            raise LLMError(
                f"Azure OpenAI API error: HTTP {error.response.status_code}",
                provider=self.provider_name,
                code=error.response.status_code,
                details={
                    "status_code": error.response.status_code,
                    "deployment": self._deployment
                }
            )

    def chat(
        self,
        messages: list[ChatMessage],
        trace: TraceContext | None = None,
        **kwargs: Any
    ) -> LLMResponse:
        """Send a chat request to Azure OpenAI.

        Args:
            messages: List of chat messages in conversation order.
            trace: Tracing context for observability.
            **kwargs: Additional provider-specific arguments.

        Returns:
            LLMResponse containing the generated text.

        Raises:
            LLMError: If the request fails.
        """
        logger.info(
            f"Azure OpenAI chat request: deployment={self._deployment}, "
            f"message_count={len(messages)}"
        )

        if trace:
            trace.record_stage(
                "llm_request",
                {
                    "provider": self.provider_name,
                    "deployment": self._deployment,
                    "message_count": len(messages)
                }
            )

        if not messages:
            raise LLMError(
                "No messages provided to chat",
                provider=self.provider_name,
                details={"message_count": 0}
            )

        # Build request
        payload = self._build_request_payload(messages, **kwargs)
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

            llm_response = self._parse_response(response_data)

            logger.info(
                f"Azure OpenAI chat response: content_length={len(llm_response.content)}"
            )

            if trace:
                trace.record_stage(
                    "llm_response",
                    {
                        "content_length": len(llm_response.content),
                        "tokens": llm_response.usage
                    }
                )

            return llm_response

        except httpx.HTTPStatusError as e:
            self._handle_http_error(e, trace)
        except httpx.RequestError as e:
            raise LLMError(
                f"Failed to connect to Azure OpenAI API: {e}",
                provider=self.provider_name,
                details={"url": url, "error": str(e)}
            )
        finally:
            if self._http_client is None:
                client.close()

    def chat_stream(
        self,
        messages: list[ChatMessage],
        trace: TraceContext | None = None,
        **kwargs: Any
    ):
        """Stream a chat response from Azure OpenAI.

        Args:
            messages: List of chat messages.
            trace: Tracing context for observability.
            **kwargs: Additional arguments.

        Yields:
            str: Chunks of the response text.
        """
        logger.info(
            f"Azure OpenAI streaming chat request: deployment={self._deployment}"
        )

        payload = self._build_request_payload(messages, stream=True, **kwargs)
        url = self._get_api_url()

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": "ModularRAG-MCP-Server/1.0",
        }

        client = self._http_client or httpx.Client(timeout=self._timeout)

        try:
            with client.stream("POST", url, headers=headers, json=payload) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line:
                        continue

                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue

        except httpx.HTTPStatusError as e:
            self._handle_http_error(e, trace)
        except httpx.RequestError as e:
            raise LLMError(
                f"Failed to connect to Azure OpenAI API: {e}",
                provider=self.provider_name
            )
        finally:
            if self._http_client is None:
                client.close()

        if trace:
            trace.record_stage("llm_stream_complete", {"provider": self.provider_name})

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"AzureOpenAILLM(provider={self.provider_name}, "
            f"deployment={self._deployment}, "
            f"temperature={self._temperature})"
        )
