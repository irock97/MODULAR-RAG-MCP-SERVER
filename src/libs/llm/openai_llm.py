"""OpenAI-compatible LLM implementation.

This module provides OpenAI LLM implementation that follows the BaseLLM interface.
It supports OpenAI's chat completions API and is compatible with any provider
that uses the OpenAI API format (Azure OpenAI, DeepSeek, local servers, etc.).

Design Principles:
    - OpenAI-compatible: Follows OpenAI API conventions
    - Type Safe: Full type hints for all methods
    - Observable: trace parameter for tracing integration
    - Configurable: Supports all OpenAI API parameters
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


class OpenAILLM(BaseLLM):
    """OpenAI LLM implementation.

    This class implements the BaseLLM interface for OpenAI's chat completions API.

    Attributes:
        api_key: OpenAI API key
        base_url: Base URL for the API endpoint
        model: Model name to use
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds
        http_client: Optional HTTP client for custom configuration

    Example:
        >>> llm = OpenAILLM(
        ...     api_key="sk-...",
        ...     model="gpt-4o-mini",
        ...     temperature=0.7
        ... )
        >>> response = llm.chat([ChatMessage(role="user", content="Hello")])
        >>> print(response.content)
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: float = 30.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        """Initialize the OpenAI LLM.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            base_url: Base URL for the API. Defaults to OpenAI's official API.
            model: Model name. Required for API calls.
            temperature: Sampling temperature (0.0-2.0). Lower is more deterministic.
            max_tokens: Maximum tokens to generate.
            timeout: Request timeout in seconds.
            http_client: Optional pre-configured HTTP client.

        Raises:
            LLMConfigurationError: If model is not provided and cannot be determined.
        """
        self._api_key = api_key
        self._base_url = base_url or "https://api.openai.com/v1"
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._http_client = http_client

        # Validate required fields
        if not self._model:
            raise LLMConfigurationError(
                "OpenAI model is not configured. Set 'model' in settings or pass it directly.",
                provider="openai"
            )

        # Get API key from environment if not provided
        if not self._api_key:
            import os
            self._api_key = os.getenv("OPENAI_API_KEY")
            if not self._api_key:
                raise LLMConfigurationError(
                    "OpenAI API key is not configured. Set 'api_key' in settings or OPENAI_API_KEY env var.",
                    provider="openai"
                )

    @property
    def provider_name(self) -> str:
        """Return the name of this provider.

        Returns:
            Provider identifier: 'openai'
        """
        return "openai"

    def _build_request_payload(
        self,
        messages: list[ChatMessage],
        **kwargs: Any
    ) -> dict[str, Any]:
        """Build the request payload for OpenAI chat completions API.

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

        # Build payload
        payload: dict[str, Any] = {
            "model": self._model,
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

        if "logit_bias" in kwargs:
            payload["logit_bias"] = kwargs["logit_bias"]

        if "response_format" in kwargs:
            payload["response_format"] = kwargs["response_format"]

        return payload

    def _parse_response(self, response_data: dict[str, Any]) -> LLMResponse:
        """Parse OpenAI API response into LLMResponse.

        Args:
            response_data: Raw API response dictionary.

        Returns:
            Parsed LLMResponse object.
        """
        # Extract content
        choices = response_data.get("choices", [])
        if not choices:
            raise LLMError(
                "Empty response from OpenAI API",
                provider=self.provider_name,
                details=response_data
            )

        message = choices[0].get("message", {})
        content = message.get("content", "")

        # Extract usage
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

    def _make_request(
        self,
        payload: dict[str, Any],
        stream: bool = False,
        trace: TraceContext | None = None,
    ) -> LLMResponse:
        """Make HTTP request to OpenAI API.

        Args:
            payload: Request payload.
            stream: Whether to stream the response.
            trace: Tracing context.

        Returns:
            LLMResponse from the API.

        Raises:
            LLMError: On API errors with provider and error details.
        """
        url = f"{self._base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": "ModularRAG-MCP-Server/1.0",
        }

        # Use provided HTTP client or create a new one
        client = self._http_client or httpx.Client(timeout=self._timeout)

        try:
            if stream:
                # Handle streaming response
                return self._handle_streaming_response(
                    url, headers, payload, client, trace
                )

            # Make synchronous request
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()

            return self._parse_response(response.json())

        except httpx.HTTPStatusError as e:
            self._handle_http_error(e, trace)
        except httpx.RequestError as e:
            raise LLMError(
                f"Failed to connect to OpenAI API: {e}",
                provider=self.provider_name,
                details={"url": url, "error": str(e)}
            )
        finally:
            if self._http_client is None:
                client.close()

    def _handle_streaming_response(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        client: httpx.Client,
        trace: TraceContext | None = None,
    ) -> LLMResponse:
        """Handle streaming response from OpenAI API.

        Args:
            url: API URL.
            headers: Request headers.
            payload: Request payload.
            client: HTTP client.
            trace: Tracing context.

        Returns:
            LLMResponse with accumulated content.
        """
        # For streaming, we need to use a streaming request
        accumulated_content = ""
        raw_chunks: list[dict[str, Any]] = []

        try:
            with client.stream("POST", url, headers=headers, json=payload) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line:
                        continue

                    # Parse SSE format
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data)
                            raw_chunks.append(chunk)

                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                accumulated_content += content
                        except json.JSONDecodeError:
                            continue

            return LLMResponse(
                content=accumulated_content,
                raw_response={"chunks": raw_chunks},
                usage=None  # Streaming doesn't return usage
            )

        except httpx.HTTPStatusError as e:
            self._handle_http_error(e, trace)

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
            LLMError: Structured error with provider and details.
        """
        try:
            error_data = error.response.json()
            error_message = error_data.get("error", {}).get("message", str(error))
            error_type = error_data.get("error", {}).get("type", "api_error")

            raise LLMError(
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
            raise LLMError(
                f"OpenAI API error: HTTP {error.response.status_code}",
                provider=self.provider_name,
                code=error.response.status_code,
                details={"status_code": error.response.status_code}
            )

    def chat(
        self,
        messages: list[ChatMessage],
        trace: TraceContext | None = None,
        **kwargs: Any
    ) -> LLMResponse:
        """Send a chat request to OpenAI.

        Args:
            messages: List of chat messages in conversation order.
            trace: Tracing context for observability.
            **kwargs: Additional provider-specific arguments.
                - temperature: Sampling temperature (0.0-2.0)
                - max_tokens: Maximum tokens to generate
                - stream: Whether to stream the response
                - top_p: Top-p sampling parameter
                - presence_penalty: Presence penalty
                - frequency_penalty: Frequency penalty

        Returns:
            LLMResponse containing the generated text.

        Raises:
            LLMError: If the request fails.
        """
        # Log request start
        logger.info(
            f"OpenAI chat request: model={self._model}, "
            f"message_count={len(messages)}"
        )

        # Record trace if provided
        if trace:
            trace.record_stage(
                "llm_request",
                {
                    "provider": self.provider_name,
                    "model": self._model,
                    "message_count": len(messages)
                }
            )

        # Validate input
        if not messages:
            raise LLMError(
                "No messages provided to chat",
                provider=self.provider_name,
                details={"message_count": 0}
            )

        # Build and send request
        payload = self._build_request_payload(messages, **kwargs)
        response = self._make_request(
            payload,
            stream=kwargs.get("stream", False),
            trace=trace
        )

        # Log response
        logger.info(
            f"OpenAI chat response: content_length={len(response.content)}, "
            f"tokens={response.usage}"
        )

        # Record trace
        if trace:
            trace.record_stage(
                "llm_response",
                {
                    "content_length": len(response.content),
                    "tokens": response.usage
                }
            )

        return response

    def chat_stream(
        self,
        messages: list[ChatMessage],
        trace: TraceContext | None = None,
        **kwargs: Any
    ):
        """Stream a chat response from OpenAI.

        Args:
            messages: List of chat messages.
            trace: Tracing context for observability.
            **kwargs: Additional arguments.

        Yields:
            str: Chunks of the response text.

        Raises:
            LLMError: If the request fails.
        """
        # Log request start
        logger.info(
            f"OpenAI streaming chat request: model={self._model}, "
            f"message_count={len(messages)}"
        )

        # Build request
        payload = self._build_request_payload(messages, stream=True, **kwargs)

        # Get client
        client = self._http_client or httpx.Client(timeout=self._timeout)
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": "ModularRAG-MCP-Server/1.0",
        }

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
                f"Failed to connect to OpenAI API: {e}",
                provider=self.provider_name
            )
        finally:
            if self._http_client is None:
                client.close()

        # Record trace
        if trace:
            trace.record_stage("llm_stream_complete", {"provider": self.provider_name})

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"OpenAILLM(provider={self.provider_name}, "
            f"model={self._model}, "
            f"temperature={self._temperature})"
        )
