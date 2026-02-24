"""Qwen LLM implementation.

This module provides Qwen LLM implementation that follows the BaseLLM interface.
Alibaba's Qwen provides powerful language models via DashScope API.

Design Principles:
    - DashScope Compatible: Uses Alibaba's DashScope API
    - Type Safe: Full type hints for all methods
    - Observable: trace parameter for tracing integration
    - High Performance: Supports long context and fast inference

Example Configuration:
    llm:
      provider: qwen
      api_key: your-dashscope-api-key
      base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
      model: qwen-turbo
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

# Default Qwen API base URL
DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# Default Qwen models
DEFAULT_QWEN_MODEL = "qwen-turbo"


class QwenLLM(BaseLLM):
    """Qwen LLM implementation.

    This class implements the BaseLLM interface for Alibaba's Qwen chat completions API.

    Attributes:
        api_key: DashScope API key
        base_url: Base URL for the API endpoint
        model: Model name to use
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds

    Example:
        >>> llm = QwenLLM(
        ...     api_key="...",
        ...     model="qwen-turbo"
        ... )
        >>> response = llm.chat([ChatMessage(role="user", content="Hello")])
    """

    SUPPORTED_MODELS = [
        "qwen-turbo",
        "qwen-plus",
        "qwen-max",
        "qwen-max-longcontext",
    ]

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: float = 60.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        """Initialize the Qwen LLM.

        Args:
            api_key: DashScope API key. If None, reads from DASHSCOPE_API_KEY env var.
            base_url: Base URL for the API. Defaults to DashScope official API.
            model: Model name. Defaults to qwen-turbo.
            temperature: Sampling temperature (0.0-2.0).
            max_tokens: Maximum tokens to generate.
            timeout: Request timeout in seconds.
            http_client: Optional pre-configured HTTP client.

        Raises:
            LLMConfigurationError: If API key is not configured.
        """
        self._api_key = api_key
        self._base_url = base_url or DEFAULT_QWEN_BASE_URL
        self._model = model or DEFAULT_QWEN_MODEL
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._http_client = http_client

        # Get API key from environment if not provided
        if not self._api_key:
            import os
            self._api_key = os.getenv("DASHSCOPE_API_KEY")

        if not self._api_key:
            raise LLMConfigurationError(
                "Qwen API key is required. Set DASHSCOPE_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Use provided client or create new one
        self._client = http_client or httpx.Client(
            base_url=self._base_url,
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )

        logger.info(f"Initialized Qwen LLM: model={self._model}, base_url={self._base_url}")

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "qwen"

    def chat(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        trace: TraceContext | None = None,
    ) -> LLMResponse:
        """Send a chat request to Qwen.

        Args:
            messages: List of chat messages
            temperature: Override default temperature
            max_tokens: Override default max tokens
            trace: Optional trace context

        Returns:
            LLMResponse with the model's reply

        Raises:
            LLMError: If the request fails
        """
        payload = {
            "model": self._model,
            "messages": [msg.model_dump() for msg in messages],
            "temperature": temperature if temperature is not None else self._temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
        }

        try:
            response = self._client.post("/chat/completions", json=payload)
            response.raise_for_status()

            data = response.json()

            if "choices" not in data or len(data["choices"]) == 0:
                raise LLMError("Invalid response: no choices in response")

            choice = data["choices"][0]
            content = choice["message"]["content"]
            finish_reason = choice.get("finish_reason")

            # Extract usage info if available
            usage = data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

            logger.info(
                f"Qwen chat completed: {output_tokens} output tokens, "
                f"finish_reason={finish_reason}"
            )

            return LLMResponse(
                content=content,
                model=self._model,
                provider=self.provider_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                finish_reason=finish_reason,
            )

        except httpx.HTTPStatusError as e:
            error_msg = f"Qwen API error: {e.response.status_code}"
            try:
                error_data = e.response.json()
                error_msg += f" - {error_data.get('message', error_data)}"
            except Exception:
                pass
            raise LLMError(error_msg) from e

        except httpx.RequestError as e:
            raise LLMError(f"Request failed: {e}") from e

    def chat_stream(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        trace: TraceContext | None = None,
    ):
        """Send a streaming chat request to Qwen.

        Args:
            messages: List of chat messages
            temperature: Override default temperature
            max_tokens: Override default max tokens
            trace: Optional trace context

        Yields:
            LLMResponse fragments for streaming

        Raises:
            LLMError: If the request fails
        """
        payload = {
            "model": self._model,
            "messages": [msg.model_dump() for msg in messages],
            "temperature": temperature if temperature is not None else self._temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
            "stream": True,
        }

        try:
            with self._client.stream("POST", "/chat/completions", json=payload) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line.strip() or not line.startswith("data:"):
                        continue

                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if "choices" not in data or len(data["choices"]) == 0:
                        continue

                    choice = data["choices"][0]
                    delta = choice.get("delta", {})
                    content = delta.get("content", "")
                    finish_reason = choice.get("finish_reason")

                    if content:
                        yield LLMResponse(
                            content=content,
                            model=self._model,
                            provider=self.provider_name,
                            finish_reason=finish_reason,
                        )

        except httpx.HTTPStatusError as e:
            error_msg = f"Qwen API error: {e.response.status_code}"
            try:
                error_data = e.response.json()
                error_msg += f" - {error_data.get('message', error_data)}"
            except Exception:
                pass
            raise LLMError(error_msg) from e

        except httpx.RequestError as e:
            raise LLMError(f"Request failed: {e}") from e

    def close(self) -> None:
        """Close the HTTP client."""
        if self._http_client is None and self._client:
            self._client.close()

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"QwenLLM(model={self._model}, "
            f"base_url={self._base_url}, temperature={self._temperature})"
        )
