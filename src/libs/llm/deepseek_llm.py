"""DeepSeek LLM implementation.

This module provides DeepSeek LLM implementation that follows the BaseLLM interface.
DeepSeek provides an OpenAI-compatible API with their own models.

Design Principles:
    - OpenAI-compatible: Uses same API format as OpenAI
    - Type Safe: Full type hints for all methods
    - Observable: trace parameter for tracing integration
    - Cost-effective: DeepSeek offers competitive pricing

Example Configuration:
    llm:
      provider: deepseek
      api_key: your-deepseek-api-key
      base_url: https://api.deepseek.com
      model: deepseek-chat
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

# Default DeepSeek API base URL
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Default DeepSeek model
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"


class DeepSeekLLM(BaseLLM):
    """DeepSeek LLM implementation.

    This class implements the BaseLLM interface for DeepSeek's chat completions API.

    Attributes:
        api_key: DeepSeek API key
        base_url: Base URL for the API endpoint
        model: Model name to use
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds

    Example:
        >>> llm = DeepSeekLLM(
        ...     api_key="...",
        ...     base_url="https://api.deepseek.com",
        ...     model="deepseek-chat"
        ... )
        >>> response = llm.chat([ChatMessage(role="user", content="Hello")])
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
        """Initialize the DeepSeek LLM.

        Args:
            api_key: DeepSeek API key. If None, reads from DEEPSEEK_API_KEY env var.
            base_url: Base URL for the API. Defaults to DeepSeek's official API.
            model: Model name. Defaults to deepseek-chat.
            temperature: Sampling temperature (0.0-2.0).
            max_tokens: Maximum tokens to generate.
            timeout: Request timeout in seconds.
            http_client: Optional pre-configured HTTP client.

        Raises:
            LLMConfigurationError: If API key is not configured.
        """
        self._api_key = api_key
        self._base_url = base_url or DEFAULT_DEEPSEEK_BASE_URL
        self._model = model or DEFAULT_DEEPSEEK_MODEL
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._http_client = http_client

        # Get API key from environment if not provided
        if not self._api_key:
            import os
            self._api_key = os.getenv("DEEPSEEK_API_KEY")
            if not self._api_key:
                raise LLMConfigurationError(
                    "DeepSeek API key is not configured. Set 'api_key' in settings or "
                    "DEEPSEEK_API_KEY env var.",
                    provider="deepseek"
                )

    @property
    def provider_name(self) -> str:
        """Return the name of this provider.

        Returns:
            Provider identifier: 'deepseek'
        """
        return "deepseek"

    def _build_request_payload(
        self,
        messages: list[ChatMessage],
        **kwargs: Any
    ) -> dict[str, Any]:
        """Build the request payload for DeepSeek API.

        Args:
            messages: List of chat messages.
            **kwargs: Additional API parameters.

        Returns:
            Request payload dictionary.
        """
        messages_data = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages_data,
            "temperature": kwargs.get("temperature", self._temperature),
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
        }

        if kwargs.get("stream"):
            payload["stream"] = True

        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]

        if "presence_penalty" in kwargs:
            payload["presence_penalty"] = kwargs["presence_penalty"]

        if "frequency_penalty" in kwargs:
            payload["frequency_penalty"] = kwargs["frequency_penalty"]

        return payload

    def _parse_response(self, response_data: dict[str, Any]) -> LLMResponse:
        """Parse DeepSeek API response into LLMResponse.

        Args:
            response_data: Raw API response dictionary.

        Returns:
            Parsed LLMResponse object.
        """
        choices = response_data.get("choices", [])
        if not choices:
            raise LLMError(
                "Empty response from DeepSeek API",
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
        """Handle HTTP errors from DeepSeek API.

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
                f"DeepSeek API error: {error_message}",
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
                f"DeepSeek API error: HTTP {error.response.status_code}",
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
        """Send a chat request to DeepSeek.

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
            f"DeepSeek chat request: model={self._model}, "
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

        payload = self._build_request_payload(messages, **kwargs)
        url = f"{self._base_url}/chat/completions"

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
                f"DeepSeek chat response: content_length={len(llm_response.content)}"
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
                f"Failed to connect to DeepSeek API: {e}",
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
        """Stream a chat response from DeepSeek.

        Args:
            messages: List of chat messages.
            trace: Tracing context for observability.
            **kwargs: Additional arguments.

        Yields:
            str: Chunks of the response text.
        """
        logger.info(
            f"DeepSeek streaming chat request: model={self._model}"
        )

        payload = self._build_request_payload(messages, stream=True, **kwargs)
        url = f"{self._base_url}/chat/completions"

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
                f"Failed to connect to DeepSeek API: {e}",
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
            f"DeepSeekLLM(provider={self.provider_name}, "
            f"model={self._model}, "
            f"temperature={self._temperature})"
        )
