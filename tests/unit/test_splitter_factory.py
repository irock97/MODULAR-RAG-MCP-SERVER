"""Tests for Splitter Factory and BaseSplitter.

These tests verify:
1. BaseSplitter interface is correctly defined
2. SplitterFactory dynamic provider registration
3. FakeSplitter works as expected for testing
"""

from typing import Any

import pytest

from libs.splitter.base_splitter import (
    BaseSplitter,
    SplitResult,
    SplitterError,
    UnknownSplitterProviderError,
    SplitterConfigurationError,
)
from libs.splitter.splitter_factory import SplitterFactory
from core.settings import Settings, IngestionConfig


class FakeSplitter(BaseSplitter):
    """Fake splitter for testing.

    This implementation returns deterministic chunks for testing
    without implementing actual splitting logic.
    """

    def __init__(
        self,
        chunk_size: int = 100,
        chunk_overlap: int = 20,
        response_chunks: list[str] | None = None,
        **kwargs: Any
    ) -> None:
        """Initialize the Fake Splitter.

        Args:
            chunk_size: Target chunk size
            chunk_overlap: Overlap between chunks
            response_chunks: Predefined chunks to return
            **kwargs: Extra arguments (ignored for compatibility)
        """
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._response_chunks = response_chunks or ["chunk1", "chunk2", "chunk3"]
        self.call_count = 0

    @property
    def provider_name(self) -> str:
        return "fake"

    def split_text(self, text: str, trace: Any = None, **kwargs: Any) -> SplitResult:
        """Return fake split result."""
        self.call_count += 1

        chunks = self._response_chunks.copy()
        return SplitResult(
            chunks=chunks,
            metadata={"original_length": len(text), "chunk_count": len(chunks)}
        )

    def split_documents(
        self,
        documents: list[str],
        trace: Any = None,
        **kwargs: Any
    ) -> list[SplitResult]:
        """Return fake split results for multiple documents."""
        self.call_count += 1

        return [self.split_text(doc, trace=trace) for doc in documents]


class TestSplitResult:
    """Test SplitResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating a basic result."""
        result = SplitResult(chunks=["a", "b", "c"])
        assert result.chunks == ["a", "b", "c"]
        assert result.metadata is None

    def test_create_result_with_metadata(self) -> None:
        """Test creating a result with metadata."""
        result = SplitResult(
            chunks=["a", "b"],
            metadata={"chunk_count": 2}
        )
        assert result.metadata["chunk_count"] == 2


class TestFakeSplitter:
    """Test FakeSplitter implementation."""

    def test_provider_name(self) -> None:
        """Test that FakeSplitter returns correct provider name."""
        fake = FakeSplitter()
        assert fake.provider_name == "fake"

    def test_split_text_returns_result(self) -> None:
        """Test that split_text returns a SplitResult."""
        fake = FakeSplitter()
        text = "Hello world test"

        result = fake.split_text(text)

        assert isinstance(result, SplitResult)
        assert len(result.chunks) == 3
        assert result.metadata["original_length"] == len(text)

    def test_split_documents(self) -> None:
        """Test split_documents returns list of SplitResult."""
        fake = FakeSplitter()
        docs = ["Doc1", "Doc2"]

        results = fake.split_documents(docs)

        assert len(results) == 2
        assert isinstance(results[0], SplitResult)
        assert isinstance(results[1], SplitResult)

    def test_split_call_count(self) -> None:
        """Test that call count is tracked."""
        fake = FakeSplitter()
        assert fake.call_count == 0

        fake.split_text("Text 1")
        assert fake.call_count == 1

        # split_documents increments call_count AND calls split_text per document
        # So 1 document = 1 (split_documents) + 1 (split_text) = 2 increments
        fake.split_documents(["A"])
        assert fake.call_count == 3

    def test_split_with_custom_chunks(self) -> None:
        """Test split with custom predefined chunks."""
        custom_chunks = ["custom1", "custom2"]
        fake = FakeSplitter(response_chunks=custom_chunks)

        result = fake.split_text("Test")

        assert result.chunks == custom_chunks


class TestSplitterFactoryRegistration:
    """Test SplitterFactory dynamic provider registration."""

    def setup_method(self) -> None:
        """Clear registry before each test."""
        SplitterFactory.clear()

    def teardown_method(self) -> None:
        """Clear registry after each test."""
        SplitterFactory.clear()

    def test_no_providers_registered_by_default(self) -> None:
        """Test that no providers are registered by default."""
        assert SplitterFactory.get_provider_names() == []

    def test_register_provider(self) -> None:
        """Test registering a new provider."""

        class CustomSplitter(BaseSplitter):
            def __init__(self, **kwargs: Any) -> None:
                pass

            @property
            def provider_name(self) -> str:
                return "custom"

            def split_text(self, text, **kwargs):
                return SplitResult(chunks=[])

            def split_documents(self, documents, **kwargs):
                return [SplitResult(chunks=[]) for _ in documents]

        SplitterFactory.register("custom", CustomSplitter)

        assert "custom" in SplitterFactory.get_provider_names()
        assert SplitterFactory.has_provider("custom")

    def test_register_fake(self) -> None:
        """Test registering FakeSplitter."""
        SplitterFactory.register("fake", FakeSplitter)

        assert "fake" in SplitterFactory.get_provider_names()
        assert SplitterFactory.has_provider("fake")

    def test_unregister_provider(self) -> None:
        """Test unregistering a provider."""
        SplitterFactory.register("test", FakeSplitter)
        assert SplitterFactory.has_provider("test")

        result = SplitterFactory.unregister("test")
        assert result is True
        assert not SplitterFactory.has_provider("test")

    def test_unregister_nonexistent(self) -> None:
        """Test unregistering a provider that doesn't exist."""
        result = SplitterFactory.unregister("nonexistent")
        assert result is False

    def test_provider_case_insensitive(self) -> None:
        """Test that provider names are case-insensitive."""

        class TestProvider(BaseSplitter):
            def __init__(self, **kwargs: Any) -> None:
                pass

            @property
            def provider_name(self) -> str:
                return "test"

            def split_text(self, text, **kwargs):
                return SplitResult(chunks=[])

            def split_documents(self, documents, **kwargs):
                return [SplitResult(chunks=[]) for _ in documents]

        SplitterFactory.register("TestProvider", TestProvider)
        assert SplitterFactory.has_provider("testprovider")
        assert SplitterFactory.has_provider("TESTPROVIDER")
        assert SplitterFactory.has_provider("TestProvider")

    def test_clear_all_providers(self) -> None:
        """Test clearing all providers."""
        SplitterFactory.register("a", FakeSplitter)
        SplitterFactory.register("b", FakeSplitter)
        SplitterFactory.register("c", FakeSplitter)

        assert len(SplitterFactory.get_provider_names()) == 3

        SplitterFactory.clear()

        assert SplitterFactory.get_provider_names() == []


class TestSplitterFactoryCreate:
    """Test SplitterFactory.create() method."""

    def setup_method(self) -> None:
        """Clear and register fake before each test."""
        SplitterFactory.clear()
        SplitterFactory.register("fake", FakeSplitter)

    def teardown_method(self) -> None:
        """Clear after each test."""
        SplitterFactory.clear()

    def test_create_fake_provider(self) -> None:
        """Test creating a FakeSplitter instance."""
        settings = Settings(
            ingestion=IngestionConfig(splitter="fake", chunk_size=100)
        )

        splitter = SplitterFactory.create(settings)

        assert isinstance(splitter, FakeSplitter)
        assert splitter.provider_name == "fake"
        assert splitter._chunk_size == 100

    def test_create_unknown_provider(self) -> None:
        """Test that unknown provider raises error."""
        settings = Settings(
            ingestion=IngestionConfig(splitter="unknown-splitter")
        )

        with pytest.raises(UnknownSplitterProviderError) as exc_info:
            SplitterFactory.create(settings)

        assert "unknown-splitter" in str(exc_info.value)

    def test_create_with_kwargs_override(self) -> None:
        """Test that kwargs override settings."""
        settings = Settings(
            ingestion=IngestionConfig(splitter="fake", chunk_size=500)
        )

        splitter = SplitterFactory.create(settings, chunk_size=1000)

        assert isinstance(splitter, FakeSplitter)
        assert splitter._chunk_size == 1000

    def test_create_missing_splitter_config(self) -> None:
        """Test that missing splitter config raises error."""
        settings = Settings(
            ingestion=IngestionConfig(splitter=None)
        )

        with pytest.raises(SplitterConfigurationError):
            SplitterFactory.create(settings)

    def test_create_with_chunk_overlap(self) -> None:
        """Test creating splitter with chunk_overlap."""
        settings = Settings(
            ingestion=IngestionConfig(splitter="fake", chunk_size=500, chunk_overlap=50)
        )

        splitter = SplitterFactory.create(settings)

        assert isinstance(splitter, FakeSplitter)
        assert splitter._chunk_overlap == 50


class TestSplitterFactoryDynamicRegistration:
    """Test dynamic provider registration workflow."""

    def setup_method(self) -> None:
        """Clear before each test."""
        SplitterFactory.clear()

    def teardown_method(self) -> None:
        """Clear after each test."""
        SplitterFactory.clear()

    def test_register_and_create_custom_provider(self) -> None:
        """Test registering and creating a custom provider."""

        class RecursiveSplitter(BaseSplitter):
            def __init__(
                self,
                chunk_size: int = 1000,
                chunk_overlap: int = 200,
                **kwargs: Any
            ) -> None:
                self._chunk_size = chunk_size
                self._chunk_overlap = chunk_overlap

            @property
            def provider_name(self) -> str:
                return "recursive"

            def split_text(self, text, **kwargs) -> SplitResult:
                # Simple split for testing
                words = text.split()
                chunks = []
                for i in range(0, len(words), self._chunk_size):
                    chunk = " ".join(words[i:i + self._chunk_size])
                    chunks.append(chunk)
                return SplitResult(chunks=chunks)

            def split_documents(self, documents, **kwargs) -> list[SplitResult]:
                return [self.split_text(doc) for doc in documents]

        # Register the provider
        SplitterFactory.register("recursive", RecursiveSplitter)

        # Create an instance
        settings = Settings(
            ingestion=IngestionConfig(
                splitter="recursive",
                chunk_size=50
            )
        )

        splitter = SplitterFactory.create(settings)

        assert isinstance(splitter, RecursiveSplitter)
        assert splitter.provider_name == "recursive"
        assert splitter._chunk_size == 50

        # Verify the splitter works
        result = splitter.split_text("one two three four five six seven eight nine ten")
        assert len(result.chunks) > 0

    def test_multiple_providers(self) -> None:
        """Test registering multiple providers."""

        class RecursiveSplitter(BaseSplitter):
            def __init__(self, **kwargs: Any) -> None:
                pass

            @property
            def provider_name(self) -> str:
                return "recursive"

            def split_text(self, text, **kwargs):
                return SplitResult(chunks=["recursive-chunk"])

            def split_documents(self, documents, **kwargs):
                return [SplitResult(chunks=["recursive-chunk"]) for _ in documents]

        class FixedSplitter(BaseSplitter):
            def __init__(self, **kwargs: Any) -> None:
                pass

            @property
            def provider_name(self) -> str:
                return "fixed"

            def split_text(self, text, **kwargs):
                return SplitResult(chunks=["fixed-chunk"])

            def split_documents(self, documents, **kwargs):
                return [SplitResult(chunks=["fixed-chunk"]) for _ in documents]

        class SemanticSplitter(BaseSplitter):
            def __init__(self, **kwargs: Any) -> None:
                pass

            @property
            def provider_name(self) -> str:
                return "semantic"

            def split_text(self, text, **kwargs):
                return SplitResult(chunks=["semantic-chunk"])

            def split_documents(self, documents, **kwargs):
                return [SplitResult(chunks=["semantic-chunk"]) for _ in documents]

        # Register all providers
        SplitterFactory.register("recursive", RecursiveSplitter)
        SplitterFactory.register("fixed", FixedSplitter)
        SplitterFactory.register("semantic", SemanticSplitter)

        # Verify all are registered
        assert len(SplitterFactory.get_provider_names()) == 3
        assert "recursive" in SplitterFactory.get_provider_names()
        assert "fixed" in SplitterFactory.get_provider_names()
        assert "semantic" in SplitterFactory.get_provider_names()

    def test_provider_override(self) -> None:
        """Test that registering same provider twice overrides."""
        SplitterFactory.register("test", FakeSplitter)

        class NewTestSplitter(BaseSplitter):
            def __init__(self, **kwargs: Any) -> None:
                pass

            @property
            def provider_name(self) -> str:
                return "test"

            def split_text(self, text, **kwargs):
                return SplitResult(chunks=["new-chunk"])

            def split_documents(self, documents, **kwargs):
                return [SplitResult(chunks=["new-chunk"]) for _ in documents]

        # Override the provider
        SplitterFactory.register("test", NewTestSplitter)

        settings = Settings(ingestion=IngestionConfig(splitter="test"))
        splitter = SplitterFactory.create(settings)

        assert isinstance(splitter, NewTestSplitter)
        result = splitter.split_text("Test")
        assert result.chunks == ["new-chunk"]


class TestBaseSplitterInterface:
    """Test that BaseSplitter is properly abstract."""

    def test_cannot_instantiate_base(self) -> None:
        """Test that BaseSplitter cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseSplitter()

    def test_subclass_must_implement_split_text(self) -> None:
        """Test that subclasses must implement split_text."""

        class IncompleteSplitter(BaseSplitter):
            @property
            def provider_name(self) -> str:
                return "incomplete"

            def split_documents(self, documents, **kwargs):
                pass

            # Missing split_text() implementation

        with pytest.raises(TypeError):
            IncompleteSplitter()

    def test_subclass_must_implement_split_documents(self) -> None:
        """Test that subclasses must implement split_documents."""

        class IncompleteSplitter(BaseSplitter):
            @property
            def provider_name(self) -> str:
                return "incomplete"

            def split_text(self, text, **kwargs):
                pass

            # Missing split_documents() implementation

        with pytest.raises(TypeError):
            IncompleteSplitter()


class TestSplitterErrors:
    """Test splitter error classes."""

    def test_splitter_error_basic(self) -> None:
        """Test basic SplitterError."""
        error = SplitterError("Test error")
        assert str(error) == "Test error"
        assert error.provider is None

    def test_splitter_error_with_details(self) -> None:
        """Test SplitterError with provider and details."""
        error = SplitterError(
            "Split failed",
            provider="recursive",
            details={"chunk_index": 5}
        )
        assert error.provider == "recursive"
        assert error.details["chunk_index"] == 5

    def test_unknown_provider_error(self) -> None:
        """Test UnknownSplitterProviderError."""
        error = UnknownSplitterProviderError(
            "Unknown provider: test",
            provider="test"
        )
        assert error.provider == "test"
