"""Splitter Factory for creating Splitter instances based on configuration.

This module provides the SplitterFactory class that creates the appropriate
splitter implementation based on the configuration.

Design Principles:
    - Factory Pattern: Creates the right implementation based on config
    - Configuration-Driven: Provider selection via settings.ingestion.splitter
    - Extensible: Providers are registered at runtime, no hardcoded list

Usage:
    # Register a provider
    from libs.splitter.providers import RecursiveSplitter
    SplitterFactory.register("recursive", RecursiveSplitter)

    # Create an instance
    settings = load_settings()
    splitter = SplitterFactory.create(settings)
"""

from typing import Any

from core.settings import Settings
from libs.splitter.base_splitter import (
    BaseSplitter,
    SplitResult,
    SplitterConfigurationError,
    SplitterError,
    UnknownSplitterProviderError,
)
from observability.logger import get_logger

logger = get_logger(__name__)


class SplitterFactory:
    """Factory for creating Splitter instances based on configuration.

    This factory uses runtime provider registration. No providers are
    hardcoded - all must be registered before use.

    Usage:
        # Register providers
        from libs.splitter.providers import RecursiveSplitter, FixedSplitter
        SplitterFactory.register("recursive", RecursiveSplitter)
        SplitterFactory.register("fixed", FixedSplitter)

        # Create instances
        settings = load_settings()
        splitter = SplitterFactory.create(settings)
    """

    # Registry of provider names to implementation classes
    # Empty by default - providers must be registered
    _providers: dict[str, type[BaseSplitter]] = {}

    @classmethod
    def register(
        cls,
        provider_name: str,
        implementation_class: type[BaseSplitter]
    ) -> None:
        """Register a splitter provider.

        Args:
            provider_name: Provider identifier (e.g., 'recursive', 'fixed')
            implementation_class: Class that implements BaseSplitter

        Example:
            >>> from libs.splitter.providers import RecursiveSplitter
            >>> SplitterFactory.register("recursive", RecursiveSplitter)
        """
        cls._providers[provider_name.lower()] = implementation_class
        logger.info(f"Registered splitter provider: {provider_name}")

    @classmethod
    def unregister(cls, provider_name: str) -> bool:
        """Unregister a splitter provider.

        Args:
            provider_name: Provider identifier to remove

        Returns:
            True if removed, False if not found
        """
        provider = provider_name.lower()
        if provider in cls._providers:
            del cls._providers[provider]
            logger.info(f"Unregistered splitter provider: {provider_name}")
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
        logger.info("Cleared all registered splitter providers")

    @classmethod
    def create(
        cls,
        settings: Settings,
        **kwargs: Any
    ) -> BaseSplitter:
        """Create a Splitter instance based on configuration.

        Args:
            settings: Settings object containing ingestion configuration
            **kwargs: Additional arguments to pass to the splitter constructor
                - Can override settings values (e.g., chunk_size=500)

        Returns:
            BaseSplitter implementation instance

        Raises:
            UnknownSplitterProviderError: If the provider is not registered
            SplitterConfigurationError: If configuration is invalid

        Example:
            >>> settings = load_settings()
            >>> splitter = SplitterFactory.create(settings)
        """
        ingest_config = settings.ingestion

        # Get provider name
        provider = (ingest_config.splitter or "").lower()
        if not provider:
            raise SplitterConfigurationError(
                "Splitter provider is not configured. "
                "Set 'ingestion.splitter' in settings.yaml"
            )

        # Check if provider is registered
        if provider not in cls._providers:
            available = ", ".join(cls._providers.keys())
            if not available:
                available = "(no providers registered - register your own)"
            raise UnknownSplitterProviderError(
                f"Unknown splitter provider: '{provider}'. "
                f"Available providers: {available}",
                provider=provider
            )

        # Get the implementation class
        implementation_class = cls._providers[provider]

        # Build constructor arguments from settings
        init_kwargs: dict[str, Any] = {
            "chunk_size": getattr(ingest_config, "chunk_size", None),
            "chunk_overlap": getattr(ingest_config, "chunk_overlap", None),
        }

        # Apply overrides
        for key, value in kwargs.items():
            if value is not None:
                init_kwargs[key] = value

        # Remove None values
        init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

        # Create the instance
        logger.info(
            f"Creating splitter instance: provider={provider}, "
            f"chunk_size={init_kwargs.get('chunk_size', 'N/A')}"
        )

        return implementation_class(**init_kwargs)
