"""Evaluator Factory for creating evaluator instances based on configuration.

This module provides the EvaluatorFactory class that creates the appropriate
evaluator implementation based on the provider setting in the configuration.

Design Principles:
    - Factory Pattern: Creates the right implementation based on config
    - Configuration-Driven: Provider selection via settings.evaluation.provider
    - Extensible: Providers are registered at runtime, no hardcoded list

Usage:
    # Register a provider
    from libs.evaluator.providers import CustomEvaluator
    EvaluatorFactory.register("custom", CustomEvaluator)

    # Create an evaluator instance
    settings = load_settings()
    evaluator = EvaluatorFactory.create(settings)
"""

from typing import Any

from core.settings import Settings
from libs.evaluator.base_evaluator import (
    BaseEvaluator,
    EvaluatorConfigurationError,
    UnknownEvaluatorProviderError,
)
from observability.logger import get_logger

logger = get_logger(__name__)


class EvaluatorFactory:
    """Factory for creating evaluator instances based on configuration.

    This factory uses runtime provider registration. No providers are
    hardcoded - all must be registered before use.

    Usage:
        # Register providers
        from libs.evaluator import CustomEvaluator, RagasEvaluator
        EvaluatorFactory.register("custom", CustomEvaluator)
        EvaluatorFactory.register("ragas", RagasEvaluator)

        # Create instances
        settings = load_settings()
        evaluator = EvaluatorFactory.create(settings)
    """

    # Registry of provider names to implementation classes
    # Empty by default - providers must be registered
    _providers: dict[str, type[BaseEvaluator]] = {}

    @classmethod
    def register(
        cls,
        provider_name: str,
        implementation_class: type[BaseEvaluator]
    ) -> None:
        """Register an evaluator provider.

        Args:
            provider_name: Provider identifier (e.g., 'custom', 'ragas')
            implementation_class: Class that implements BaseEvaluator

        Example:
            >>> from libs.evaluator import CustomEvaluator
            >>> EvaluatorFactory.register("custom", CustomEvaluator)
        """
        cls._providers[provider_name.lower()] = implementation_class
        logger.info(f"Registered evaluator provider: {provider_name}")

    @classmethod
    def unregister(cls, provider_name: str) -> bool:
        """Unregister an evaluator provider.

        Args:
            provider_name: Provider identifier to remove

        Returns:
            True if removed, False if not found
        """
        provider = provider_name.lower()
        if provider in cls._providers:
            del cls._providers[provider]
            logger.info(f"Unregistered evaluator provider: {provider_name}")
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
        logger.info("Cleared all registered evaluator providers")

    @classmethod
    def create(
        cls,
        settings: Settings,
        **kwargs: Any
    ) -> BaseEvaluator:
        """Create an evaluator instance based on configuration.

        Args:
            settings: Settings object containing evaluation configuration
            **kwargs: Additional arguments to pass to the evaluator constructor
                - Can override settings values (e.g., metrics=["hit_rate"])

        Returns:
            BaseEvaluator implementation instance

        Raises:
            UnknownEvaluatorProviderError: If the provider is not registered
            EvaluatorConfigurationError: If configuration is invalid

        Example:
            >>> settings = load_settings()
            >>> evaluator = EvaluatorFactory.create(settings)
        """
        eval_config = settings.evaluation

        # Get provider name
        provider = (eval_config.provider or "").lower()
        if not provider:
            raise EvaluatorConfigurationError(
                "Evaluator provider is not configured. "
                "Set 'evaluation.provider' in settings.yaml"
            )

        # Check if provider is registered
        if provider not in cls._providers:
            available = ", ".join(cls._providers.keys())
            if not available:
                available = "(no providers registered - register your own)"
            raise UnknownEvaluatorProviderError(
                f"Unknown evaluator provider: '{provider}'. "
                f"Available providers: {available}",
                provider=provider
            )

        # Get the implementation class
        implementation_class = cls._providers[provider]

        # Build constructor arguments from settings
        init_kwargs: dict[str, Any] = {
            "metrics": getattr(eval_config, "metrics", None),
        }

        # Apply overrides
        for key, value in kwargs.items():
            if value is not None:
                init_kwargs[key] = value

        # Remove None values
        init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

        # Create the instance
        logger.info(
            f"Creating evaluator instance: provider={provider}"
        )

        return implementation_class(**init_kwargs)
