"""LLM Factory for creating LLM and Vision LLM instances based on configuration.

This module provides the LLMFactory class that creates the appropriate
LLM implementation based on the provider setting in the configuration.
It also provides the create_vision_llm method for vision-capable providers.

Design Principles:
    - Factory Pattern: Creates the right implementation based on config
    - Configuration-Driven: Provider selection via settings.llm.provider
    - Extensible: Providers are registered at runtime, no hardcoded list

Usage:
    # Register text LLM providers
    from libs.llm.providers import OpenAIProvider
    LLMFactory.register("openai", OpenAIProvider)

    # Register vision LLM providers
    from libs.llm.providers import OpenAIVisionProvider
    LLMFactory.register_vision("openai-vision", OpenAIVisionProvider)

    # Create instances
    settings = load_settings()
    llm = LLMFactory.create(settings)
    vision_llm = LLMFactory.create_vision_llm(settings)
"""

from typing import Any

from core.settings import Settings
from libs.llm.base_llm import (
    BaseLLM,
    LLMConfigurationError,
    UnknownLLMProviderError,
)
from libs.llm.base_vision_llm import (
    BaseVisionLLM,
    VisionLLMConfigurationError,
    UnknownVisionLLMProviderError,
)
from observability.logger import get_logger

logger = get_logger(__name__)


class LLMFactory:
    """Factory for creating LLM instances based on configuration.

    This factory uses runtime provider registration. No providers are
    hardcoded - all must be registered before use.

    Usage:
        # Register providers
        from libs.llm.providers import OpenAIProvider, OllamaProvider
        LLMFactory.register("openai", OpenAIProvider)
        LLMFactory.register("ollama", OllamaProvider)

        # Create instances
        settings = load_settings()
        llm = LLMFactory.create(settings)
    """

    # Registry of provider names to implementation classes
    # Empty by default - providers must be registered
    _providers: dict[str, type[BaseLLM]] = {}
    _vision_providers: dict[str, type[BaseVisionLLM]] = {}

    @classmethod
    def register(
        cls,
        provider_name: str,
        implementation_class: type[BaseLLM]
    ) -> None:
        """Register an LLM provider.

        Args:
            provider_name: Provider identifier (e.g., 'openai', 'ollama')
            implementation_class: Class that implements BaseLLM

        Example:
            >>> from libs.llm.providers import OpenAIProvider
            >>> LLMFactory.register("openai", OpenAIProvider)
        """
        cls._providers[provider_name.lower()] = implementation_class
        logger.info(f"Registered LLM provider: {provider_name}")

    @classmethod
    def unregister(cls, provider_name: str) -> bool:
        """Unregister an LLM provider.

        Args:
            provider_name: Provider identifier to remove

        Returns:
            True if removed, False if not found
        """
        provider = provider_name.lower()
        if provider in cls._providers:
            del cls._providers[provider]
            logger.info(f"Unregistered LLM provider: {provider_name}")
            return True
        return False

    @classmethod
    def get_provider_names(cls) -> list[str]:
        """Get list of available provider names.

        Returns:
            List of registered provider names
        """
        return list(cls._providers.keys())

    @classmethod
    def has_provider(cls, provider_name: str) -> bool:
        """Check if a provider is registered.

        Args:
            provider_name: Provider identifier

        Returns:
            True if provider is registered
        """
        return provider_name.lower() in cls._providers

    @classmethod
    def clear(cls) -> None:
        """Clear all registered providers."""
        cls._providers.clear()
        logger.info("Cleared all registered LLM providers")

    @classmethod
    def create(
        cls,
        settings: Settings,
        **kwargs: Any
    ) -> BaseLLM:
        """Create an LLM instance based on configuration.

        Args:
            settings: Settings object containing LLM configuration
            **kwargs: Additional arguments to pass to the LLM constructor
                - Can override settings values (e.g., model="gpt-4o")

        Returns:
            BaseLLM implementation instance

        Raises:
            UnknownLLMProviderError: If the provider is not registered
            LLMConfigurationError: If configuration is invalid

        Example:
            >>> settings = load_settings()
            >>> llm = LLMFactory.create(settings)
        """
        llm_config = settings.llm

        # Get provider name
        provider = (llm_config.provider or "").lower()
        if not provider:
            raise LLMConfigurationError(
                "LLM provider is not configured. "
                "Set 'llm.provider' in settings.yaml"
            )

        # Check if provider is registered
        if provider not in cls._providers:
            available = ", ".join(cls._providers.keys())
            if not available:
                available = "(no providers registered - register your own)"
            raise UnknownLLMProviderError(
                f"Unknown LLM provider: '{provider}'. "
                f"Available providers: {available}",
                provider=provider
            )

        # Get the implementation class
        implementation_class = cls._providers[provider]

        # Build constructor arguments from settings
        init_kwargs: dict[str, Any] = {
            "api_key": getattr(llm_config, "api_key", None),
            "base_url": getattr(llm_config, "base_url", None),
            "model": getattr(llm_config, "model", None),
            "temperature": getattr(llm_config, "temperature", None),
            "max_tokens": getattr(llm_config, "max_tokens", None),
        }

        # Apply overrides
        for key, value in kwargs.items():
            if value is not None:
                init_kwargs[key] = value

        # Remove None values
        init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

        # Create the instance
        logger.info(
            f"Creating LLM instance: provider={provider}, "
            f"model={getattr(llm_config, 'model', 'N/A')}"
        )

        return implementation_class(**init_kwargs)

    # ============ Vision LLM Methods ============

    @classmethod
    def register_vision(
        cls,
        provider_name: str,
        implementation_class: type[BaseVisionLLM]
    ) -> None:
        """Register a Vision LLM provider.

        Args:
            provider_name: Provider identifier (e.g., 'openai-vision', 'azure-vision')
            implementation_class: Class that implements BaseVisionLLM

        Example:
            >>> from libs.llm.providers import OpenAIVisionProvider
            >>> LLMFactory.register_vision("openai-vision", OpenAIVisionProvider)
        """
        cls._vision_providers[provider_name.lower()] = implementation_class
        logger.info(f"Registered Vision LLM provider: {provider_name}")

    @classmethod
    def unregister_vision(cls, provider_name: str) -> bool:
        """Unregister a Vision LLM provider.

        Args:
            provider_name: Provider identifier to remove

        Returns:
            True if removed, False if not found
        """
        provider = provider_name.lower()
        if provider in cls._vision_providers:
            del cls._vision_providers[provider]
            logger.info(f"Unregistered Vision LLM provider: {provider_name}")
            return True
        return False

    @classmethod
    def get_vision_provider_names(cls) -> list[str]:
        """Get list of available Vision LLM provider names.

        Returns:
            List of registered Vision provider names
        """
        return list(cls._vision_providers.keys())

    @classmethod
    def has_vision_provider(cls, provider_name: str) -> bool:
        """Check if a Vision LLM provider is registered.

        Args:
            provider_name: Provider identifier

        Returns:
            True if provider is registered
        """
        return provider_name.lower() in cls._vision_providers

    @classmethod
    def clear_vision(cls) -> None:
        """Clear all registered Vision LLM providers."""
        cls._vision_providers.clear()
        logger.info("Cleared all registered Vision LLM providers")

    @classmethod
    def create_vision_llm(
        cls,
        settings: Settings,
        **kwargs: Any
    ) -> BaseVisionLLM:
        """Create a Vision LLM instance based on configuration.

        Args:
            settings: Settings object containing LLM configuration
            **kwargs: Additional arguments to pass to the Vision LLM constructor
                - Can override settings values (e.g., model="gpt-4o")

        Returns:
            BaseVisionLLM implementation instance

        Raises:
            UnknownVisionLLMProviderError: If the provider is not registered
            VisionLLMConfigurationError: If configuration is invalid

        Example:
            >>> settings = load_settings()
            >>> vision_llm = LLMFactory.create_vision_llm(settings)
        """
        llm_config = settings.llm

        # Get provider name
        provider = (llm_config.provider or "").lower()
        if not provider:
            raise VisionLLMConfigurationError(
                "LLM provider is not configured. "
                "Set 'llm.provider' in settings.yaml"
            )

        # Check if provider is registered for vision
        if provider not in cls._vision_providers:
            available = ", ".join(cls._vision_providers.keys())
            if not available:
                available = "(no vision providers registered)"
            raise UnknownVisionLLMProviderError(
                f"Unknown Vision LLM provider: '{provider}'. "
                f"Available vision providers: {available}",
                provider=provider
            )

        # Get the implementation class
        implementation_class = cls._vision_providers[provider]

        # Build constructor arguments from settings
        init_kwargs: dict[str, Any] = {
            "api_key": getattr(llm_config, "api_key", None),
            "base_url": getattr(llm_config, "base_url", None),
            "model": getattr(llm_config, "model", None),
            "temperature": getattr(llm_config, "temperature", None),
            "max_tokens": getattr(llm_config, "max_tokens", None),
        }

        # Apply overrides
        for key, value in kwargs.items():
            if value is not None:
                init_kwargs[key] = value

        # Remove None values
        init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

        # Create the instance
        logger.info(
            f"Creating Vision LLM instance: provider={provider}, "
            f"model={getattr(llm_config, 'model', 'N/A')}"
        )

        return implementation_class(**init_kwargs)
