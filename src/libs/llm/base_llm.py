"""Abstract base class for LLM providers.

This module defines the BaseLLM interface that all LLM implementations
must follow. This enables pluggable LLM providers (OpenAI, Azure, Ollama, etc.).

Design Principles:
    - Pluggable: All providers implement this interface
    - Type Safe: Full type hints for all methods
    - Observable: trace parameter for tracing integration
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from observability.logger import TraceContext


@dataclass
class LLMResponse:
    """Response from an LLM chat completion.

    Attributes:
        content: The text content of the response
        raw_response: The raw response from the provider (if available)
        usage: Token usage information (if available)
    """
    content: str
    raw_response: Any | None = None
    usage: dict[str, int] | None = None


@dataclass
class ChatMessage:
    """A single message in a chat conversation.

    Attributes:
        role: Message role (user, assistant, system)
        content: Message content
    """
    role: str  # "user", "assistant", "system"
    content: str


class BaseLLM(ABC):
    """Abstract base class for LLM providers.

    All LLM implementations (OpenAI, Azure, Ollama, etc.) must
    inherit from this class and implement the chat() method.

    Example:
        >>> class OpenAILLM(BaseLLM):
        ...     def chat(self, messages: list[ChatMessage]) -> LLMResponse:
        ...         # Implementation here
        ...         pass
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this provider.

        Returns:
            Provider identifier (e.g., 'openai', 'azure', 'ollama')
        """
        ...

    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        trace: TraceContext | None = None,
        **kwargs: Any
    ) -> LLMResponse:
        """Send a chat request to the LLM.

        Args:
            messages: List of chat messages in conversation order
            trace: Tracing context for observability
            **kwargs: Additional provider-specific arguments
                - temperature: Sampling temperature (0.0-2.0)
                - max_tokens: Maximum tokens to generate
                - stream: Whether to stream the response

        Returns:
            LLMResponse containing the generated text

        Raises:
            LLMError: If the request fails
        """
        ...

    def chat_stream(
        self,
        messages: list[ChatMessage],
        trace: TraceContext | None = None,
        **kwargs: Any
    ):
        """Stream a chat response from the LLM.

        This is an optional method for providers that support streaming.

        Args:
            messages: List of chat messages
            trace: Tracing context for observability
            **kwargs: Additional provider-specific arguments

        Yields:
            str: Chunks of the response text

        Raises:
            LLMError: If the request fails
        """
        # Default implementation: not supported
        raise NotImplementedError(
            f"Provider '{self.provider_name}' does not support streaming"
        )


class LLMError(Exception):
    """Base exception for LLM-related errors."""

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        code: int | None = None,
        details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.code = code
        self.details = details or {}


class UnknownLLMProviderError(LLMError):
    """Raised when an unknown LLM provider is specified."""

    pass


class LLMConfigurationError(LLMError):
    """Raised when LLM configuration is invalid."""

    pass
