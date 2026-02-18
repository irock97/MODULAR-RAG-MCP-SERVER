"""Embedding Factory for creating Embedding instances based on configuration.

This module provides the EmbeddingFactory class that creates the appropriate
embedding implementation based on the provider setting in the configuration.

Design Principles:
    - Factory Pattern: Creates the right implementation based on config
    - Configuration-Driven: Provider selection via settings.embedding.provider
    - Extensible: Providers are registered at runtime, no hardcoded list

Usage:
    # Register a provider
    from libs.embedding.providers import OpenAIEmbedding
    EmbeddingFactory.register("openai", OpenAIEmbedding)

    # Create an instance
    settings = load_settings()
    embedding = EmbeddingFactory.create(settings)
"""

from typing import Any

from core.settings import Settings
from libs.embedding.base_embedding import (
    BaseEmbedding,
    EmbeddingConfigurationError,
    EmbeddingResult,
    UnknownEmbeddingProviderError,
)
from observability.logger import get_logger

logger = get_logger(__name__)


class EmbeddingFactory:
    """Factory for creating Embedding instances based on configuration.

    This factory uses runtime provider registration. No providers are
    hardcoded - all must be registered before use.

    Usage:
        # Register providers
        from libs.embedding.providers import OpenAIEmbedding, LocalEmbedding
        EmbeddingFactory.register("openai", OpenAIEmbedding)
        EmbeddingFactory.register("local", LocalEmbedding)

        # Create instances
        settings = load_settings()
        embedding = EmbeddingFactory.create(settings)
    """

    # Registry of provider names to implementation classes
    # Empty by default - providers must be registered
    _providers: dict[str, type[BaseEmbedding]] = {}

    @classmethod
    def register(
        cls,
        provider_name: str,
        implementation_class: type[BaseEmbedding]
    ) -> None:
        """Register an embedding provider.

        Args:
            provider_name: Provider identifier (e.g., 'openai', 'local')
            implementation_class: Class that implements BaseEmbedding

        Example:
            >>> from libs.embedding.providers import OpenAIEmbedding
            >>> EmbeddingFactory.register("openai", OpenAIEmbedding)
        """
        cls._providers[provider_name.lower()] = implementation_class
        logger.info(f"Registered embedding provider: {provider_name}")

    @classmethod
    def unregister(cls, provider_name: str) -> bool:
        """Unregister an embedding provider.

        Args:
            provider_name: Provider identifier to remove

        Returns:
            True if removed, False if not found
        """
        provider = provider_name.lower()
        if provider in cls._providers:
            del cls._providers[provider]
            logger.info(f"Unregistered embedding provider: {provider_name}")
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
        logger.info("Cleared all registered embedding providers")

    @classmethod
    def create(
        cls,
        settings: Settings,
        **kwargs: Any
    ) -> BaseEmbedding:
        """Create an Embedding instance based on configuration.

        Args:
            settings: Settings object containing embedding configuration
            **kwargs: Additional arguments to pass to the embedding constructor
                - Can override settings values (e.g., model="text-embedding-3-small")

        Returns:
            BaseEmbedding implementation instance

        Raises:
            UnknownEmbeddingProviderError: If the provider is not registered
            EmbeddingConfigurationError: If configuration is invalid

        Example:
            >>> settings = load_settings()
            >>> embedding = EmbeddingFactory.create(settings)
        """
        embed_config = settings.embedding

        # Get provider name
        provider = (embed_config.provider or "").lower()
        if not provider:
            raise EmbeddingConfigurationError(
                "Embedding provider is not configured. "
                "Set 'embedding.provider' in settings.yaml"
            )

        # Check if provider is registered
        if provider not in cls._providers:
            available = ", ".join(cls._providers.keys())
            if not available:
                available = "(no providers registered - register your own)"
            raise UnknownEmbeddingProviderError(
                f"Unknown embedding provider: '{provider}'. "
                f"Available providers: {available}",
                provider=provider
            )

        # Get the implementation class
        implementation_class = cls._providers[provider]

        # Build constructor arguments from settings
        # Common kwargs for all providers
        init_kwargs: dict[str, Any] = {
            "api_key": getattr(embed_config, "api_key", None),
            "base_url": getattr(embed_config, "base_url", None),
            "model": getattr(embed_config, "model", None),
            "dimensions": getattr(embed_config, "dimensions", None),
        }

        # Azure-specific kwargs
        if provider == "azure":
            init_kwargs["deployment"] = getattr(embed_config, "deployment", None)
            init_kwargs["api_version"] = getattr(embed_config, "api_version", None)

        # Apply overrides
        for key, value in kwargs.items():
            if value is not None:
                init_kwargs[key] = value

        # Remove None values
        init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

        # Create the instance
        logger.info(
            f"Creating embedding instance: provider={provider}, "
            f"model={getattr(embed_config, 'model', 'N/A')}"
        )

        return implementation_class(**init_kwargs)


# Register default providers (safe to import - will only register if dependencies available)
def _register_default_providers() -> None:
    """Register default embedding providers.

    This function is called when the module is imported.
    It safely handles missing dependencies.
    """
    # Register Qwen provider
    try:
        from libs.embedding.qwen_embedding import DashScopeEmbedding
        EmbeddingFactory.register("qwen", DashScopeEmbedding)
    except ImportError:
        pass

    # Register Ollama provider
    try:
        from libs.embedding.ollama_embedding import OllamaEmbedding
        EmbeddingFactory.register("ollama", OllamaEmbedding)
    except ImportError:
        pass


# Register default providers on module import
_register_default_providers()
