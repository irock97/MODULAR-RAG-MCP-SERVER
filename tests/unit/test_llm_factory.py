"""Tests for LLM Factory and BaseLLM.

These tests verify:
1. BaseLLM interface is correctly defined
2. LLMFactory dynamic provider registration
3. FakeLLM works as expected for testing
"""

from typing import Any

import pytest

from libs.llm.base_llm import (
    BaseLLM,
    ChatMessage,
    LLMResponse,
    LLMError,
    UnknownLLMProviderError,
    LLMConfigurationError,
)
from libs.llm.llm_factory import LLMFactory
from core.settings import Settings, LLMConfig


class FakeLLM(BaseLLM):
    """Fake LLM for testing and development.

    This implementation returns deterministic responses for testing
    without making actual API calls.
    """

    def __init__(
        self,
        response_content: str = "This is a fake LLM response.",
        delay: float = 0.0,
        **kwargs: Any
    ) -> None:
        """Initialize the Fake LLM.

        Args:
            response_content: Content to return for all requests
            delay: Simulated delay in seconds
            **kwargs: Extra arguments (ignored for compatibility)
        """
        self._response_content = response_content
        self._delay = delay
        self.call_count = 0

    @property
    def provider_name(self) -> str:
        return "fake"

    def chat(
        self,
        messages: list[ChatMessage],
        **kwargs: Any
    ) -> LLMResponse:
        """Return a fake response."""
        self.call_count += 1

        if self._delay > 0:
            import time
            time.sleep(self._delay)

        if messages:
            last_msg = messages[-1]
            response = f"Echo: {last_msg.content}"
        else:
            response = self._response_content

        return LLMResponse(
            content=response,
            usage={"prompt_tokens": 10, "completion_tokens": 20}
        )

    def chat_stream(
        self,
        messages: list[ChatMessage],
        **kwargs: Any
    ):
        """Yield fake response chunks."""
        response = self.chat(messages, **kwargs)
        content = response.content
        for char in content:
            yield char


class TestChatMessage:
    """Test ChatMessage dataclass."""

    def test_create_user_message(self) -> None:
        """Test creating a user message."""
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_create_system_message(self) -> None:
        """Test creating a system message."""
        msg = ChatMessage(role="system", content="You are helpful.")
        assert msg.role == "system"
        assert msg.content == "You are helpful."


class TestLLMResponse:
    """Test LLMResponse dataclass."""

    def test_create_response(self) -> None:
        """Test creating a basic response."""
        response = LLMResponse(content="Hello, world!")
        assert response.content == "Hello, world!"
        assert response.raw_response is None
        assert response.usage is None

    def test_create_response_with_usage(self) -> None:
        """Test creating a response with token usage."""
        response = LLMResponse(
            content="Hello!",
            usage={"prompt_tokens": 5, "completion_tokens": 3}
        )
        assert response.usage == {"prompt_tokens": 5, "completion_tokens": 3}


class TestFakeLLM:
    """Test FakeLLM implementation."""

    def test_provider_name(self) -> None:
        """Test that FakeLLM returns correct provider name."""
        fake = FakeLLM()
        assert fake.provider_name == "fake"

    def test_chat_returns_response(self) -> None:
        """Test that chat returns an LLMResponse."""
        fake = FakeLLM(response_content="Test response")
        messages = [ChatMessage(role="user", content="Hi there")]

        response = fake.chat(messages)

        assert isinstance(response, LLMResponse)
        assert response.content == "Echo: Hi there"

    def test_chat_with_no_messages(self) -> None:
        """Test chat with empty message list."""
        fake = FakeLLM(response_content="Default response")

        response = fake.chat([])

        assert response.content == "Default response"

    def test_chat_call_count(self) -> None:
        """Test that call count is tracked."""
        fake = FakeLLM()
        assert fake.call_count == 0

        fake.chat([ChatMessage(role="user", content="Test 1")])
        assert fake.call_count == 1

        fake.chat([ChatMessage(role="user", content="Test 2")])
        assert fake.call_count == 2

    def test_chat_stream(self) -> None:
        """Test streaming response."""
        fake = FakeLLM(response_content="Hello")
        messages = [ChatMessage(role="user", content="Hi")]

        chunks = list(fake.chat_stream(messages))

        # chat_stream calls chat() which prepends "Echo: "
        expected = [c for c in "Echo: Hi"]
        assert chunks == expected


class TestLLMFactoryRegistration:
    """Test LLMFactory dynamic provider registration."""

    def setup_method(self) -> None:
        """Clear registry before each test."""
        LLMFactory.clear()

    def teardown_method(self) -> None:
        """Clear registry after each test."""
        LLMFactory.clear()

    def test_no_providers_registered_by_default(self) -> None:
        """Test that no providers are registered by default."""
        assert LLMFactory.get_provider_names() == []

    def test_register_provider(self) -> None:
        """Test registering a new provider."""

        class CustomLLM(BaseLLM):
            def __init__(self, **kwargs: Any) -> None:
                pass

            @property
            def provider_name(self) -> str:
                return "custom"

            def chat(self, messages, **kwargs):
                return LLMResponse(content="Custom response")

        LLMFactory.register("custom", CustomLLM)

        assert "custom" in LLMFactory.get_provider_names()
        assert LLMFactory.has_provider("custom")

    def test_register_fake(self) -> None:
        """Test registering FakeLLM."""
        LLMFactory.register("fake", FakeLLM)

        assert "fake" in LLMFactory.get_provider_names()
        assert LLMFactory.has_provider("fake")

    def test_unregister_provider(self) -> None:
        """Test unregistering a provider."""
        LLMFactory.register("test", FakeLLM)
        assert LLMFactory.has_provider("test")

        result = LLMFactory.unregister("test")
        assert result is True
        assert not LLMFactory.has_provider("test")

    def test_unregister_nonexistent(self) -> None:
        """Test unregistering a provider that doesn't exist."""
        result = LLMFactory.unregister("nonexistent")
        assert result is False

    def test_provider_case_insensitive(self) -> None:
        """Test that provider names are case-insensitive."""

        class TestProvider(BaseLLM):
            def __init__(self, **kwargs: Any) -> None:
                pass

            @property
            def provider_name(self) -> str:
                return "test"

            def chat(self, messages, **kwargs):
                return LLMResponse(content="test")

        LLMFactory.register("TestProvider", TestProvider)
        assert LLMFactory.has_provider("testprovider")
        assert LLMFactory.has_provider("TESTPROVIDER")
        assert LLMFactory.has_provider("TestProvider")

    def test_clear_all_providers(self) -> None:
        """Test clearing all providers."""
        LLMFactory.register("a", FakeLLM)
        LLMFactory.register("b", FakeLLM)
        LLMFactory.register("c", FakeLLM)

        assert len(LLMFactory.get_provider_names()) == 3

        LLMFactory.clear()

        assert LLMFactory.get_provider_names() == []


class TestLLMFactoryCreate:
    """Test LLMFactory.create() method."""

    def setup_method(self) -> None:
        """Clear and register fake before each test."""
        LLMFactory.clear()
        LLMFactory.register("fake", FakeLLM)

    def teardown_method(self) -> None:
        """Clear after each test."""
        LLMFactory.clear()

    def test_create_fake_provider(self) -> None:
        """Test creating a FakeLLM instance."""
        settings = Settings(
            llm=LLMConfig(provider="fake", model="test-model")
        )

        llm = LLMFactory.create(settings)

        assert isinstance(llm, FakeLLM)
        assert llm.provider_name == "fake"

    def test_create_unknown_provider(self) -> None:
        """Test that unknown provider raises error."""
        settings = Settings(
            llm=LLMConfig(provider="unknown-provider", model="test")
        )

        with pytest.raises(UnknownLLMProviderError) as exc_info:
            LLMFactory.create(settings)

        assert "unknown-provider" in str(exc_info.value)

    def test_create_with_kwargs_override(self) -> None:
        """Test that kwargs override settings."""
        settings = Settings(
            llm=LLMConfig(provider="fake", model="original-model")
        )

        llm = LLMFactory.create(settings, model="override-model")

        assert isinstance(llm, FakeLLM)

    def test_create_missing_provider_config(self) -> None:
        """Test that missing provider config raises error."""
        settings = Settings(
            llm=LLMConfig(provider=None, model="test")
        )

        with pytest.raises(LLMConfigurationError):
            LLMFactory.create(settings)


class TestLLMFactoryDynamicRegistration:
    """Test dynamic provider registration workflow."""

    def setup_method(self) -> None:
        """Clear before each test."""
        LLMFactory.clear()

    def teardown_method(self) -> None:
        """Clear after each test."""
        LLMFactory.clear()

    def test_register_and_create_custom_provider(self) -> None:
        """Test registering and creating a custom provider."""

        class OpenAIProvider(BaseLLM):
            def __init__(
                self,
                api_key: str | None = None,
                model: str = "gpt-4o-mini",
                **kwargs: Any
            ) -> None:
                self._api_key = api_key
                self._model = model

            @property
            def provider_name(self) -> str:
                return "openai"

            def chat(self, messages, **kwargs) -> LLMResponse:
                return LLMResponse(content=f"[OpenAI {self._model}] Response")

        # Register the provider
        LLMFactory.register("openai", OpenAIProvider)

        # Create an instance
        settings = Settings(
            llm=LLMConfig(
                provider="openai",
                model="gpt-4o",
                api_key="sk-xxx"
            )
        )

        llm = LLMFactory.create(settings)

        assert isinstance(llm, OpenAIProvider)
        assert llm.provider_name == "openai"

        # Verify the LLM works
        response = llm.chat([ChatMessage(role="user", content="Hello")])
        assert "gpt-4o" in response.content

    def test_multiple_providers(self) -> None:
        """Test registering multiple providers."""

        class OpenAIProvider(BaseLLM):
            def __init__(self, **kwargs: Any) -> None:
                pass

            @property
            def provider_name(self) -> str:
                return "openai"

            def chat(self, messages, **kwargs):
                return LLMResponse(content="OpenAI")

        class OllamaProvider(BaseLLM):
            def __init__(self, **kwargs: Any) -> None:
                pass

            @property
            def provider_name(self) -> str:
                return "ollama"

            def chat(self, messages, **kwargs):
                return LLMResponse(content="Ollama")

        class AnthropicProvider(BaseLLM):
            def __init__(self, **kwargs: Any) -> None:
                pass

            @property
            def provider_name(self) -> str:
                return "anthropic"

            def chat(self, messages, **kwargs):
                return LLMResponse(content="Anthropic")

        # Register all providers
        LLMFactory.register("openai", OpenAIProvider)
        LLMFactory.register("ollama", OllamaProvider)
        LLMFactory.register("anthropic", AnthropicProvider)

        # Verify all are registered
        assert len(LLMFactory.get_provider_names()) == 3
        assert "openai" in LLMFactory.get_provider_names()
        assert "ollama" in LLMFactory.get_provider_names()
        assert "anthropic" in LLMFactory.get_provider_names()

    def test_provider_override(self) -> None:
        """Test that registering same provider twice overrides."""
        LLMFactory.register("test", FakeLLM)

        class NewTestLLM(BaseLLM):
            def __init__(self, **kwargs: Any) -> None:
                pass

            @property
            def provider_name(self) -> str:
                return "test"

            def chat(self, messages, **kwargs):
                return LLMResponse(content="New")

        # Override the provider
        LLMFactory.register("test", NewTestLLM)

        settings = Settings(llm=LLMConfig(provider="test"))
        llm = LLMFactory.create(settings)

        assert isinstance(llm, NewTestLLM)
        response = llm.chat([])
        assert response.content == "New"


class TestBaseLLMInterface:
    """Test that BaseLLM is properly abstract."""

    def test_cannot_instantiate_base(self) -> None:
        """Test that BaseLLM cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseLLM()

    def test_subclass_must_implement_chat(self) -> None:
        """Test that subclasses must implement chat."""

        class IncompleteLLM(BaseLLM):
            @property
            def provider_name(self) -> str:
                return "incomplete"

            # Missing chat() implementation

        with pytest.raises(TypeError):
            IncompleteLLM()

    def test_subclass_must_implement_provider_name(self) -> None:
        """Test that subclasses must implement provider_name."""

        class IncompleteLLM(BaseLLM):
            def chat(self, messages, **kwargs):
                pass

        with pytest.raises(TypeError):
            IncompleteLLM()


class TestLLMErrors:
    """Test LLM error classes."""

    def test_llm_error_basic(self) -> None:
        """Test basic LLMError."""
        error = LLMError("Test error")
        assert str(error) == "Test error"
        assert error.provider is None
        assert error.code is None

    def test_llm_error_with_details(self) -> None:
        """Test LLMError with provider and code."""
        error = LLMError(
            "API error",
            provider="openai",
            code=429,
            details={"retry_after": 60}
        )
        assert error.provider == "openai"
        assert error.code == 429
        assert error.details["retry_after"] == 60

    def test_unknown_provider_error(self) -> None:
        """Test UnknownLLMProviderError."""
        error = UnknownLLMProviderError(
            "Unknown provider: test",
            provider="test"
        )
        assert error.provider == "test"
