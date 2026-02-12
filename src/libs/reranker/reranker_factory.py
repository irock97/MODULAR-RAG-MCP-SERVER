"""Reranker Factory for creating Reranker instances based on configuration.

This module provides the RerankerFactory class that creates the appropriate
reranking implementation based on the configuration.

Design Principles:
    - Factory Pattern: Creates the right implementation based on config
    - Configuration-Driven: Provider selection via settings.rerank.provider
    - Fallback: NoneReranker as default when reranking is disabled
    - Extensible: Providers are registered at runtime, no hardcoded list

Usage:
    # Register a provider
    from libs.reranker.providers import CrossEncoderReranker
    RerankerFactory.register("cross_encoder", CrossEncoderReranker)

    # Create an instance
    settings = load_settings()
    reranker = RerankerFactory.create(settings)
"""

from typing import Any

from core.settings import Settings
from libs.reranker.base_reranker import (
    BaseReranker,
    NoneReranker,
    RerankerConfigurationError,
    RerankResult,
    RerankerError,
    UnknownRerankerProviderError,
)
from observability.logger import get_logger

logger = get_logger(__name__)


class RerankerFactory:
    """Factory for creating Reranker instances based on configuration.

    This factory uses runtime provider registration. No providers are
    hardcoded - all must be registered before use.

    Usage:
        # Register providers
        from libs.reranker.providers import CrossEncoderReranker, LLMReranker
        RerankerFactory.register("cross_encoder", CrossEncoderReranker)
        RerankerFactory.register("llm", LLMReranker)

        # Create instances
        settings = load_settings()
        reranker = RerankerFactory.create(settings)
    """

    # Registry of provider names to implementation classes
    # Empty by default - providers must be registered
    _providers: dict[str, type[BaseReranker]] = {}

    @classmethod
    def register(
        cls,
        provider_name: str,
        implementation_class: type[BaseReranker]
    ) -> None:
        """Register a reranker provider.

        Args:
            provider_name: Provider identifier (e.g., 'cross_encoder', 'llm')
            implementation_class: Class that implements BaseReranker

        Example:
            >>> from libs.reranker.providers import CrossEncoderReranker
            >>> RerankerFactory.register("cross_encoder", CrossEncoderReranker)
        """
        cls._providers[provider_name.lower()] = implementation_class
        logger.info(f"Registered reranker provider: {provider_name}")

    @classmethod
    def unregister(cls, provider_name: str) -> bool:
        """Unregister a reranker provider.

        Args:
            provider_name: Provider identifier to remove

        Returns:
            True if removed, False if not found
        """
        provider = provider_name.lower()
        if provider in cls._providers:
            del cls._providers[provider]
            logger.info(f"Unregistered reranker provider: {provider_name}")
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
        logger.info("Cleared all registered reranker providers")

    @classmethod
    def create(
        cls,
        settings: Settings,
        **kwargs: Any
    ) -> BaseReranker:
        """Create a Reranker instance based on configuration.

        Args:
            settings: Settings object containing rerank configuration
            **kwargs: Additional arguments to pass to the reranker constructor
                - Can override settings values (e.g., top_k=10)

        Returns:
            BaseReranker implementation instance

        Raises:
            UnknownRerankerProviderError: If the provider is not registered
            RerankerConfigurationError: If configuration is invalid

        Example:
            >>> settings = load_settings()
            >>> reranker = RerankerFactory.create(settings)
        """
        rerank_config = settings.rerank

        # Check if reranking is enabled
        if not rerank_config.enabled:
            logger.info("Reranking is disabled, using NoneReranker")
            return NoneReranker()

        # Get provider name
        provider = (rerank_config.provider or "").lower()
        if not provider:
            raise RerankerConfigurationError(
                "Reranker provider is not configured. "
                "Set 'rerank.provider' in settings.yaml"
            )

        # Special case: none provider returns NoneReranker
        if provider == "none":
            return NoneReranker()

        # Check if provider is registered
        if provider not in cls._providers:
            available = ", ".join(cls._providers.keys())
            if not available:
                available = "(no providers registered - register your own)"
            raise UnknownRerankerProviderError(
                f"Unknown reranker provider: '{provider}'. "
                f"Available providers: {available}",
                provider=provider
            )

        # Get the implementation class
        implementation_class = cls._providers[provider]

        # Build constructor arguments from settings
        init_kwargs: dict[str, Any] = {
            "model": getattr(rerank_config, "model", None),
            "top_k": getattr(rerank_config, "top_k", None),
        }

        # Apply overrides
        for key, value in kwargs.items():
            if value is not None:
                init_kwargs[key] = value

        # Remove None values
        init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

        # Create the instance
        logger.info(
            f"Creating reranker instance: provider={provider}, "
            f"model={init_kwargs.get('model', 'N/A')}"
        )

        return implementation_class(**init_kwargs)
