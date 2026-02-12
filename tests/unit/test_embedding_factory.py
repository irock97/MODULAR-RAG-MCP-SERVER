"""Tests for Embedding Factory and BaseEmbedding.

These tests verify:
1. BaseEmbedding interface is correctly defined
2. EmbeddingFactory dynamic provider registration
3. FakeEmbedding works as expected for testing
"""

from typing import Any

import pytest

from libs.embedding.base_embedding import (
    BaseEmbedding,
    EmbeddingResult,
    EmbeddingError,
    UnknownEmbeddingProviderError,
    EmbeddingConfigurationError,
)
from libs.embedding.embedding_factory import EmbeddingFactory
from core.settings import Settings, EmbeddingConfig


class FakeEmbedding(BaseEmbedding):
    """Fake embedding provider for testing.

    This implementation returns deterministic vectors for testing
    without making actual API calls.
    """

    def __init__(
        self,
        vector_dim: int = 384,
        response_vectors: list[list[float]] | None = None,
        **kwargs: Any
    ) -> None:
        """Initialize the Fake Embedding.

        Args:
            vector_dim: Dimension of embedding vectors
            response_vectors: Predefined vectors to return
            **kwargs: Extra arguments (ignored for compatibility)
        """
        self._vector_dim = vector_dim
        self._response_vectors = response_vectors
        self.call_count = 0

    @property
    def provider_name(self) -> str:
        return "fake"

    def _get_vector(self, index: int) -> list[float]:
        """Get a deterministic vector based on index."""
        if self._response_vectors and index < len(self._response_vectors):
            return self._response_vectors[index]
        # Return deterministic vector based on index
        return [float(i + index * 0.1) for i in range(self._vector_dim)]

    def embed(
        self,
        texts: list[str],
        trace: Any = None,
        **kwargs: Any
    ) -> EmbeddingResult:
        """Return fake embeddings for texts."""
        self.call_count += 1

        vectors = [self._get_vector(i) for i in range(len(texts))]
        return EmbeddingResult(
            vectors=vectors,
            usage={"prompt_tokens": len(texts) * 10, "total_tokens": len(texts) * 10}
        )

    def embed_single(self, text: str, trace: Any = None, **kwargs: Any) -> list[float]:
        """Return fake embedding for single text."""
        self.call_count += 1
        return self._get_vector(0)


class TestEmbeddingResult:
    """Test EmbeddingResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating a basic result."""
        result = EmbeddingResult(vectors=[[0.1, 0.2, 0.3]])
        assert len(result.vectors) == 1
        assert result.usage is None

    def test_create_result_with_usage(self) -> None:
        """Test creating a result with token usage."""
        result = EmbeddingResult(
            vectors=[[0.1, 0.2], [0.3, 0.4]],
            usage={"prompt_tokens": 10}
        )
        assert len(result.vectors) == 2
        assert result.usage["prompt_tokens"] == 10


class TestFakeEmbedding:
    """Test FakeEmbedding implementation."""

    def test_provider_name(self) -> None:
        """Test that FakeEmbedding returns correct provider name."""
        fake = FakeEmbedding()
        assert fake.provider_name == "fake"

    def test_embed_returns_result(self) -> None:
        """Test that embed returns an EmbeddingResult."""
        fake = FakeEmbedding(vector_dim=4)
        texts = ["Hello world", "Test text"]

        result = fake.embed(texts)

        assert isinstance(result, EmbeddingResult)
        assert len(result.vectors) == 2
        assert len(result.vectors[0]) == 4

    def test_embed_single(self) -> None:
        """Test embed_single returns a single vector."""
        fake = FakeEmbedding(vector_dim=4)
        vector = fake.embed_single("Hello")

        assert isinstance(vector, list)
        assert len(vector) == 4

    def test_embed_call_count(self) -> None:
        """Test that call count is tracked."""
        fake = FakeEmbedding()
        assert fake.call_count == 0

        fake.embed(["Text 1"])
        assert fake.call_count == 1

        fake.embed_single("Text 2")
        assert fake.call_count == 2

    def test_embed_with_custom_vectors(self) -> None:
        """Test embed with predefined vectors."""
        custom_vectors = [[1.0, 2.0], [3.0, 4.0]]
        fake = FakeEmbedding(response_vectors=custom_vectors)

        result = fake.embed(["A", "B"])

        assert result.vectors == custom_vectors


class TestEmbeddingFactoryRegistration:
    """Test EmbeddingFactory dynamic provider registration."""

    def setup_method(self) -> None:
        """Clear registry before each test."""
        EmbeddingFactory.clear()

    def teardown_method(self) -> None:
        """Clear registry after each test."""
        EmbeddingFactory.clear()

    def test_no_providers_registered_by_default(self) -> None:
        """Test that no providers are registered by default."""
        assert EmbeddingFactory.get_provider_names() == []

    def test_register_provider(self) -> None:
        """Test registering a new provider."""

        class CustomEmbedding(BaseEmbedding):
            def __init__(self, **kwargs: Any) -> None:
                pass

            @property
            def provider_name(self) -> str:
                return "custom"

            def embed(self, texts, **kwargs):
                return EmbeddingResult(vectors=[])

            def embed_single(self, text, **kwargs):
                return [0.0]

        EmbeddingFactory.register("custom", CustomEmbedding)

        assert "custom" in EmbeddingFactory.get_provider_names()
        assert EmbeddingFactory.has_provider("custom")

    def test_register_fake(self) -> None:
        """Test registering FakeEmbedding."""
        EmbeddingFactory.register("fake", FakeEmbedding)

        assert "fake" in EmbeddingFactory.get_provider_names()
        assert EmbeddingFactory.has_provider("fake")

    def test_unregister_provider(self) -> None:
        """Test unregistering a provider."""
        EmbeddingFactory.register("test", FakeEmbedding)
        assert EmbeddingFactory.has_provider("test")

        result = EmbeddingFactory.unregister("test")
        assert result is True
        assert not EmbeddingFactory.has_provider("test")

    def test_unregister_nonexistent(self) -> None:
        """Test unregistering a provider that doesn't exist."""
        result = EmbeddingFactory.unregister("nonexistent")
        assert result is False

    def test_provider_case_insensitive(self) -> None:
        """Test that provider names are case-insensitive."""

        class TestProvider(BaseEmbedding):
            def __init__(self, **kwargs: Any) -> None:
                pass

            @property
            def provider_name(self) -> str:
                return "test"

            def embed(self, texts, **kwargs):
                return EmbeddingResult(vectors=[])

            def embed_single(self, text, **kwargs):
                return [0.0]

        EmbeddingFactory.register("TestProvider", TestProvider)
        assert EmbeddingFactory.has_provider("testprovider")
        assert EmbeddingFactory.has_provider("TESTPROVIDER")
        assert EmbeddingFactory.has_provider("TestProvider")

    def test_clear_all_providers(self) -> None:
        """Test clearing all providers."""
        EmbeddingFactory.register("a", FakeEmbedding)
        EmbeddingFactory.register("b", FakeEmbedding)
        EmbeddingFactory.register("c", FakeEmbedding)

        assert len(EmbeddingFactory.get_provider_names()) == 3

        EmbeddingFactory.clear()

        assert EmbeddingFactory.get_provider_names() == []


class TestEmbeddingFactoryCreate:
    """Test EmbeddingFactory.create() method."""

    def setup_method(self) -> None:
        """Clear and register fake before each test."""
        EmbeddingFactory.clear()
        EmbeddingFactory.register("fake", FakeEmbedding)

    def teardown_method(self) -> None:
        """Clear after each test."""
        EmbeddingFactory.clear()

    def test_create_fake_provider(self) -> None:
        """Test creating a FakeEmbedding instance."""
        settings = Settings(
            embedding=EmbeddingConfig(provider="fake", model="test-model")
        )

        embedding = EmbeddingFactory.create(settings)

        assert isinstance(embedding, FakeEmbedding)
        assert embedding.provider_name == "fake"

    def test_create_unknown_provider(self) -> None:
        """Test that unknown provider raises error."""
        settings = Settings(
            embedding=EmbeddingConfig(provider="unknown-provider", model="test")
        )

        with pytest.raises(UnknownEmbeddingProviderError) as exc_info:
            EmbeddingFactory.create(settings)

        assert "unknown-provider" in str(exc_info.value)

    def test_create_with_kwargs_override(self) -> None:
        """Test that kwargs override settings."""
        settings = Settings(
            embedding=EmbeddingConfig(provider="fake", model="original-model")
        )

        embedding = EmbeddingFactory.create(settings, model="override-model")

        assert isinstance(embedding, FakeEmbedding)

    def test_create_missing_provider_config(self) -> None:
        """Test that missing provider config raises error."""
        settings = Settings(
            embedding=EmbeddingConfig(provider=None, model="test")
        )

        with pytest.raises(EmbeddingConfigurationError):
            EmbeddingFactory.create(settings)


class TestEmbeddingFactoryDynamicRegistration:
    """Test dynamic provider registration workflow."""

    def setup_method(self) -> None:
        """Clear before each test."""
        EmbeddingFactory.clear()

    def teardown_method(self) -> None:
        """Clear after each test."""
        EmbeddingFactory.clear()

    def test_register_and_create_custom_provider(self) -> None:
        """Test registering and creating a custom provider."""

        class OpenAIEmbedding(BaseEmbedding):
            def __init__(
                self,
                api_key: str | None = None,
                model: str = "text-embedding-3-small",
                dimensions: int | None = None,
                **kwargs: Any
            ) -> None:
                self._api_key = api_key
                self._model = model
                self._dimensions = dimensions

            @property
            def provider_name(self) -> str:
                return "openai"

            def embed(self, texts, **kwargs) -> EmbeddingResult:
                dim = self._dimensions or 1536
                return EmbeddingResult(
                    vectors=[[float(i) for i in range(dim)] for _ in texts]
                )

            def embed_single(self, text, **kwargs) -> list[float]:
                dim = self._dimensions or 1536
                return [float(i) for i in range(dim)]

        # Register the provider
        EmbeddingFactory.register("openai", OpenAIEmbedding)

        # Create an instance
        settings = Settings(
            embedding=EmbeddingConfig(
                provider="openai",
                model="text-embedding-3-large",
                dimensions=1024
            )
        )

        embedding = EmbeddingFactory.create(settings)

        assert isinstance(embedding, OpenAIEmbedding)
        assert embedding.provider_name == "openai"

        # Verify the embedding works
        result = embedding.embed(["Hello"])
        assert len(result.vectors) == 1
        assert len(result.vectors[0]) == 1024  # Custom dimensions

    def test_multiple_providers(self) -> None:
        """Test registering multiple providers."""

        class OpenAIEmbedding(BaseEmbedding):
            def __init__(self, **kwargs: Any) -> None:
                pass

            @property
            def provider_name(self) -> str:
                return "openai"

            def embed(self, texts, **kwargs):
                return EmbeddingResult(vectors=[[0.1] for _ in texts])

            def embed_single(self, text, **kwargs):
                return [0.1]

        class LocalEmbedding(BaseEmbedding):
            def __init__(self, **kwargs: Any) -> None:
                pass

            @property
            def provider_name(self) -> str:
                return "local"

            def embed(self, texts, **kwargs):
                return EmbeddingResult(vectors=[[0.2] for _ in texts])

            def embed_single(self, text, **kwargs):
                return [0.2]

        class CohereEmbedding(BaseEmbedding):
            def __init__(self, **kwargs: Any) -> None:
                pass

            @property
            def provider_name(self) -> str:
                return "cohere"

            def embed(self, texts, **kwargs):
                return EmbeddingResult(vectors=[[0.3] for _ in texts])

            def embed_single(self, text, **kwargs):
                return [0.3]

        # Register all providers
        EmbeddingFactory.register("openai", OpenAIEmbedding)
        EmbeddingFactory.register("local", LocalEmbedding)
        EmbeddingFactory.register("cohere", CohereEmbedding)

        # Verify all are registered
        assert len(EmbeddingFactory.get_provider_names()) == 3
        assert "openai" in EmbeddingFactory.get_provider_names()
        assert "local" in EmbeddingFactory.get_provider_names()
        assert "cohere" in EmbeddingFactory.get_provider_names()

    def test_provider_override(self) -> None:
        """Test that registering same provider twice overrides."""
        EmbeddingFactory.register("test", FakeEmbedding)

        class NewTestEmbedding(BaseEmbedding):
            def __init__(self, **kwargs: Any) -> None:
                pass

            @property
            def provider_name(self) -> str:
                return "test"

            def embed(self, texts, **kwargs):
                return EmbeddingResult(vectors=[[0.9]])

            def embed_single(self, text, **kwargs):
                return [0.9]

        # Override the provider
        EmbeddingFactory.register("test", NewTestEmbedding)

        settings = Settings(embedding=EmbeddingConfig(provider="test"))
        embedding = EmbeddingFactory.create(settings)

        assert isinstance(embedding, NewTestEmbedding)
        result = embedding.embed(["Test"])
        assert result.vectors[0][0] == 0.9


class TestBaseEmbeddingInterface:
    """Test that BaseEmbedding is properly abstract."""

    def test_cannot_instantiate_base(self) -> None:
        """Test that BaseEmbedding cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseEmbedding()

    def test_subclass_must_implement_embed(self) -> None:
        """Test that subclasses must implement embed."""

        class IncompleteEmbedding(BaseEmbedding):
            @property
            def provider_name(self) -> str:
                return "incomplete"

            def embed_single(self, text, **kwargs):
                pass

            # Missing embed() implementation

        with pytest.raises(TypeError):
            IncompleteEmbedding()

    def test_subclass_must_implement_embed_single(self) -> None:
        """Test that subclasses must implement embed_single."""

        class IncompleteEmbedding(BaseEmbedding):
            @property
            def provider_name(self) -> str:
                return "incomplete"

            def embed(self, texts, **kwargs):
                pass

            # Missing embed_single() implementation

        with pytest.raises(TypeError):
            IncompleteEmbedding()


class TestEmbeddingErrors:
    """Test embedding error classes."""

    def test_embedding_error_basic(self) -> None:
        """Test basic EmbeddingError."""
        error = EmbeddingError("Test error")
        assert str(error) == "Test error"
        assert error.provider is None
        assert error.code is None

    def test_embedding_error_with_details(self) -> None:
        """Test EmbeddingError with provider and code."""
        error = EmbeddingError(
            "API error",
            provider="openai",
            code=429,
            details={"retry_after": 60}
        )
        assert error.provider == "openai"
        assert error.code == 429
        assert error.details["retry_after"] == 60

    def test_unknown_provider_error(self) -> None:
        """Test UnknownEmbeddingProviderError."""
        error = UnknownEmbeddingProviderError(
            "Unknown provider: test",
            provider="test"
        )
        assert error.provider == "test"
