"""VectorStore Factory for creating VectorStore instances based on configuration.

This module provides the VectorStoreFactory class that creates the appropriate
vector store implementation based on the configuration.

Design Principles:
    - Factory Pattern: Creates the right implementation based on config
    - Configuration-Driven: Provider selection via settings.vector_store.provider
    - Extensible: Providers are registered at runtime, no hardcoded list

Usage:
    # Register a provider
    from libs.vector_store.providers import ChromaVectorStore
    VectorStoreFactory.register("chroma", ChromaVectorStore)

    # Create an instance
    settings = load_settings()
    vector_store = VectorStoreFactory.create(settings)
"""

from typing import Any

from core.settings import Settings
from libs.vector_store.base_vector_store import (
    BaseVectorStore,
    VectorStoreConfigurationError,
    VectorStoreError,
    UnknownVectorStoreProviderError,
)
from observability.logger import get_logger

logger = get_logger(__name__)


class VectorStoreFactory:
    """Factory for creating VectorStore instances based on configuration.

    This factory uses runtime provider registration. No providers are
    hardcoded - all must be registered before use.

    Usage:
        # Register providers
        from libs.vector_store.providers import ChromaVectorStore, QdrantVectorStore
        VectorStoreFactory.register("chroma", ChromaVectorStore)
        VectorStoreFactory.register("qdrant", QdrantVectorStore)

        # Create instances
        settings = load_settings()
        vector_store = VectorStoreFactory.create(settings)
    """

    # Registry of provider names to implementation classes
    # Empty by default - providers must be registered
    _providers: dict[str, type[BaseVectorStore]] = {}

    @classmethod
    def register(
        cls,
        provider_name: str,
        implementation_class: type[BaseVectorStore]
    ) -> None:
        """Register a vector store provider.

        Args:
            provider_name: Provider identifier (e.g., 'chroma', 'qdrant')
            implementation_class: Class that implements BaseVectorStore

        Example:
            >>> from libs.vector_store.providers import ChromaVectorStore
            >>> VectorStoreFactory.register("chroma", ChromaVectorStore)
        """
        cls._providers[provider_name.lower()] = implementation_class
        logger.info(f"Registered vector store provider: {provider_name}")

    @classmethod
    def unregister(cls, provider_name: str) -> bool:
        """Unregister a vector store provider.

        Args:
            provider_name: Provider identifier to remove

        Returns:
            True if removed, False if not found
        """
        provider = provider_name.lower()
        if provider in cls._providers:
            del cls._providers[provider]
            logger.info(f"Unregistered vector store provider: {provider_name}")
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
        logger.info("Cleared all registered vector store providers")

    @classmethod
    def create(
        cls,
        settings: Settings,
        **kwargs: Any
    ) -> BaseVectorStore:
        """Create a VectorStore instance based on configuration.

        Args:
            settings: Settings object containing vector store configuration
            **kwargs: Additional arguments to pass to the vector store constructor
                - Can override settings values (e.g., collection_name="custom")

        Returns:
            BaseVectorStore implementation instance

        Raises:
            UnknownVectorStoreProviderError: If the provider is not registered
            VectorStoreConfigurationError: If configuration is invalid

        Example:
            >>> settings = load_settings()
            >>> vector_store = VectorStoreFactory.create(settings)
        """
        vs_config = settings.vector_store

        # Get provider name
        provider = (vs_config.provider or "").lower()
        if not provider:
            raise VectorStoreConfigurationError(
                "Vector store provider is not configured. "
                "Set 'vector_store.provider' in settings.yaml"
            )

        # Check if provider is registered
        if provider not in cls._providers:
            available = ", ".join(cls._providers.keys())
            if not available:
                available = "(no providers registered - register your own)"
            raise UnknownVectorStoreProviderError(
                f"Unknown vector store provider: '{provider}'. "
                f"Available providers: {available}",
                provider=provider
            )

        # Get the implementation class
        implementation_class = cls._providers[provider]

        # Build constructor arguments from settings
        init_kwargs: dict[str, Any] = {
            "persist_directory": getattr(vs_config, "persist_directory", None),
            "collection_name": getattr(vs_config, "collection_name", None),
        }

        # Apply overrides
        for key, value in kwargs.items():
            if value is not None:
                init_kwargs[key] = value

        # Remove None values
        init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

        # Create the instance
        logger.info(
            f"Creating vector store instance: provider={provider}, "
            f"collection={init_kwargs.get('collection_name', 'N/A')}"
        )

        return implementation_class(**init_kwargs)
