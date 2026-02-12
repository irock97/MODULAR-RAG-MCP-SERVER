"""Ollama LLM implementation.

This module provides Ollama LLM implementation that follows the BaseLLM interface.
Ollama is a local LLM runtime with an OpenAI-compatible API.

Design Principles:
    - Local-first: Designed for local Ollama runtime
    - OpenAI-compatible API: Uses similar API format to OpenAI
    - Type Safe: Full type hints for all methods
    - Observable: trace parameter for tracing integration

Example Configuration:
    llm:
      provider: ollama
      base_url: http://localhost:11434
      model: llama3.2
      temperature: 0.7
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

# Default Ollama API base URL
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

# Default Ollama API version endpoint
DEFAULT_OLLAMA_API_VERSION = "v1"


class OllamaLLM(BaseLLM):
    """Ollama LLM implementation.

    This class implements the BaseLLM interface for Ollama's chat completions API.
    Ollama provides an OpenAI-compatible API endpoint.

    Attributes:
        base_url: Base URL for the Ollama API endpoint
        model: Model name to use
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds
        keep_alive: Keepalive duration for the model (e.g., "5m", "0")

    Example:
        >>> llm = OllamaLLM(
        ...     base_url="http://localhost:11434",
        ...     model="llama3.2"
        ... )
        >>> response = llm.chat([ChatMessage(role="user", content="Hello")])
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: float = 120.0,
        keep_alive: str = "5m",
        http_client: httpx.Client | None = None,
    ) -> None:
        """Initialize the Ollama LLM.

        Args:
            base_url: Base URL for the Ollama API. Defaults to localhost:11434.
            model: Model name. Required.
            temperature: Sampling temperature (0.0-2.0).
            max_tokens: Maximum tokens to generate.
            timeout: Request timeout in seconds (higher for local models).
            keep_alive: Keepalive duration for the model. Defaults to "5m".
            http_client: Optional pre-configured HTTP client.

        Raises:
            LLMConfigurationError: If model is not configured.
        """
        self._base_url = base_url or DEFAULT_OLLAMA_BASE_URL
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._keep_alive = keep_alive
        self._http_client = http_client

        # Validate required fields
        if not self._model:
            raise LLMConfigurationError(
                "Ollama model is not configured. Set 'model' in settings.",
                provider="ollama"
            )

    @property
    def provider_name(self) -> str:
        """Return the name of this provider.

        Returns:
            Provider identifier: 'ollama'
        """
        return "ollama"

    def _build_request_payload(
        self,
        messages: list[ChatMessage],
        **kwargs: Any
    ) -> dict[str, Any]:
        """Build the request payload for Ollama API.

        Args:
            messages: List of chat messages.
            **kwargs: Additional API parameters.

        Returns:
            Request payload dictionary.
        """
        # Convert ChatMessage objects to Ollama format
        # Ollama expects messages in a specific format
        messages_data = [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in messages
        ]

        # Build payload (Ollama v1 API is OpenAI-compatible)
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages_data,
            "options": {
                "temperature": kwargs.get("temperature", self._temperature),
                "num_predict": kwargs.get("max_tokens", self._max_tokens),
            },
        }

        # Add stream if specified
        if kwargs.get("stream"):
            payload["stream"] = True

        # Ollama-specific: keep_alive
        if "keep_alive" in kwargs:
            payload["keep_alive"] = kwargs["keep_alive"]
        else:
            payload["keep_alive"] = self._keep_alive

        return payload

    def _parse_response(self, response_data: dict[str, Any]) -> LLMResponse:
        """Parse Ollama API response into LLMResponse.

        Args:
            response_data: Raw API response dictionary.

        Returns:
            Parsed LLMResponse object.
        """
        # Ollama v1 API response format (OpenAI-compatible)
        message = response_data.get("message", {})
        content = message.get("content", "")

        # Ollama doesn't always return usage in the same way
        usage = response_data.get("usage", {})
        usage_info: dict[str, int] | None = None

        if usage:
            usage_info = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }

        # Ollama also provides raw response with additional info
        raw_response = {
            "response": response_data,
            "model": response_data.get("model", self._model),
        }

        return LLMResponse(
            content=content,
            raw_response=raw_response,
            usage=usage_info
        )

    def _handle_http_error(
        self,
        error: httpx.HTTPStatusError,
        trace: TraceContext | None = None
    ) -> None:
        """Handle HTTP errors from Ollama API.

        Args:
            error: HTTP status error.
            trace: Tracing context.

        Raises:
            LLMError: Structured error with provider and details.
        """
        try:
            error_data = error.response.json()
            error_message = error_data.get("error", str(error))
            error_type = error_data.get("error_type", "api_error")

            raise LLMError(
                f"Ollama API error: {error_message}",
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
                f"Ollama API error: HTTP {error.response.status_code}",
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
        """Send a chat request to Ollama.

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
            f"Ollama chat request: model={self._model}, "
            f"message_count={len(messages)}"
        )

        if trace:
            trace.record_stage(
                "llm_request",
                {
                    "provider": self.provider_name,
                    "model": self._model,
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
        url = f"{self._base_url}/v1/chat/completions"

        headers = {
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
                f"Ollama chat response: content_length={len(llm_response.content)}"
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
                f"Failed to connect to Ollama API: {e}",
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
        """Stream a chat response from Ollama.

        Args:
            messages: List of chat messages.
            trace: Tracing context for observability.
            **kwargs: Additional arguments.

        Yields:
            str: Chunks of the response text.
        """
        logger.info(
            f"Ollama streaming chat request: model={self._model}"
        )

        payload = self._build_request_payload(messages, stream=True, **kwargs)
        url = f"{self._base_url}/v1/chat/completions"

        headers = {
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
                f"Failed to connect to Ollama API: {e}",
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
            f"OllamaLLM(provider={self.provider_name}, "
            f"model={self._model}, "
            f"temperature={self._temperature})"
        )

    @classmethod
    def check_availability(
        cls,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        timeout: float = 5.0
    ) -> tuple[bool, str]:
        """Check if Ollama server is available.

        Args:
            base_url: Ollama server base URL.
            timeout: Request timeout.

        Returns:
            Tuple of (is_available, message).
        """
        try:
            with httpx.Client(timeout=timeout) as client:
                # Try to get model list (Ollama API)
                response = client.get(f"{base_url}/api/tags")
                if response.status_code == 200:
                    return True, "Ollama server is available"

                # Fallback: try chat completions health check
                response = client.get(f"{base_url}/v1/models")
                if response.status_code == 200:
                    return True, "Ollama server is available"

                return False, f"Ollama returned status {response.status_code}"

        except httpx.RequestError as e:
            return False, f"Cannot connect to Ollama: {e}"
