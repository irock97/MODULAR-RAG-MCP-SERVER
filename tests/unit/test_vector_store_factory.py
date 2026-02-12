"""Tests for VectorStore Factory and BaseVectorStore.

These tests verify:
1. BaseVectorStore interface is correctly defined
2. VectorStoreFactory dynamic provider registration
3. FakeVectorStore works as expected for testing
"""

from typing import Any

import pytest

from libs.vector_store.base_vector_store import (
    BaseVectorStore,
    VectorRecord,
    QueryResult,
    VectorStoreError,
    UnknownVectorStoreProviderError,
    VectorStoreConfigurationError,
)
from libs.vector_store.vector_store_factory import VectorStoreFactory
from core.settings import Settings, VectorStoreConfig


class FakeVectorStore(BaseVectorStore):
    """Fake vector store for testing.

    This implementation returns deterministic results for testing
    without connecting to actual databases.
    """

    def __init__(
        self,
        persist_directory: str = "./data/db/fake",
        collection_name: str = "test_collection",
        **kwargs: Any
    ) -> None:
        """Initialize the Fake VectorStore.

        Args:
            persist_directory: Directory for persistence
            collection_name: Collection name
            **kwargs: Extra arguments (ignored for compatibility)
        """
        self._persist_directory = persist_directory
        self._collection_name = collection_name
        self._records: dict[str, VectorRecord] = {}
        self.call_count = 0

    @property
    def provider_name(self) -> str:
        return "fake"

    def upsert(
        self,
        records: list[VectorRecord],
        **kwargs: Any
    ) -> list[str]:
        """Store records (fake implementation)."""
        self.call_count += 1
        ids = []
        for record in records:
            self._records[record.id] = record
            ids.append(record.id)
        return ids

    def query(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        **kwargs: Any
    ) -> QueryResult:
        """Query records (fake implementation)."""
        self.call_count += 1
        # Return mock results
        ids = [rid for rid in self._records.keys()][:top_k]
        scores = [0.9 - i * 0.1 for i in range(len(ids))]
        metadata = [self._records[rid].metadata for rid in ids]
        return QueryResult(ids=ids, scores=scores, metadata=metadata)

    def delete(self, ids: list[str], **kwargs: Any) -> bool:
        """Delete records (fake implementation)."""
        self.call_count += 1
        for rid in ids:
            if rid in self._records:
                del self._records[rid]
        return True

    def count(self, **kwargs: Any) -> int:
        """Count records (fake implementation)."""
        return len(self._records)

    def clear(self, **kwargs: Any) -> bool:
        """Clear all records (fake implementation)."""
        self.call_count += 1
        self._records.clear()
        return True


class TestVectorRecord:
    """Test VectorRecord dataclass."""

    def test_create_record(self) -> None:
        """Test creating a basic record."""
        record = VectorRecord(id="test1", vector=[0.1, 0.2, 0.3])
        assert record.id == "test1"
        assert record.vector == [0.1, 0.2, 0.3]
        assert record.metadata is None

    def test_create_record_with_metadata(self) -> None:
        """Test creating a record with metadata."""
        record = VectorRecord(
            id="test1",
            vector=[0.1, 0.2],
            metadata={"source": "test"}
        )
        assert record.metadata["source"] == "test"


class TestQueryResult:
    """Test QueryResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating a basic result."""
        result = QueryResult(ids=["id1", "id2"], scores=[0.9, 0.8])
        assert result.ids == ["id1", "id2"]
        assert result.scores == [0.9, 0.8]

    def test_create_result_with_metadata(self) -> None:
        """Test creating a result with metadata."""
        metadata = [{"source": "doc1"}, {"source": "doc2"}]
        result = QueryResult(
            ids=["id1", "id2"],
            scores=[0.9, 0.8],
            metadata=metadata
        )
        assert result.metadata == metadata


class TestFakeVectorStore:
    """Test FakeVectorStore implementation."""

    def test_provider_name(self) -> None:
        """Test that FakeVectorStore returns correct provider name."""
        fake = FakeVectorStore()
        assert fake.provider_name == "fake"

    def test_upsert_returns_ids(self) -> None:
        """Test that upsert returns record IDs."""
        fake = FakeVectorStore()
        records = [
            VectorRecord(id="id1", vector=[0.1, 0.2]),
            VectorRecord(id="id2", vector=[0.3, 0.4])
        ]

        ids = fake.upsert(records)

        assert len(ids) == 2
        assert "id1" in fake._records
        assert "id2" in fake._records

    def test_query_returns_result(self) -> None:
        """Test that query returns QueryResult."""
        fake = FakeVectorStore()
        fake.upsert([VectorRecord(id="id1", vector=[0.1, 0.2])])

        result = fake.query(query_vector=[0.1, 0.2], top_k=5)

        assert isinstance(result, QueryResult)
        assert "id1" in result.ids

    def test_delete_returns_bool(self) -> None:
        """Test that delete returns success status."""
        fake = FakeVectorStore()
        fake.upsert([VectorRecord(id="id1", vector=[0.1])])

        result = fake.delete(ids=["id1"])

        assert result is True
        assert "id1" not in fake._records

    def test_count(self) -> None:
        """Test counting records."""
        fake = FakeVectorStore()
        assert fake.count() == 0

        fake.upsert([VectorRecord(id="id1", vector=[0.1])])
        assert fake.count() == 1

        fake.upsert([VectorRecord(id="id2", vector=[0.2])])
        assert fake.count() == 2

    def test_clear(self) -> None:
        """Test clearing all records."""
        fake = FakeVectorStore()
        fake.upsert([
            VectorRecord(id="id1", vector=[0.1]),
            VectorRecord(id="id2", vector=[0.2])
        ])
        assert fake.count() == 2

        fake.clear()
        assert fake.count() == 0


class TestVectorStoreFactoryRegistration:
    """Test VectorStoreFactory dynamic provider registration."""

    def setup_method(self) -> None:
        """Clear registry before each test."""
        VectorStoreFactory.clear()

    def teardown_method(self) -> None:
        """Clear registry after each test."""
        VectorStoreFactory.clear()

    def test_no_providers_registered_by_default(self) -> None:
        """Test that no providers are registered by default."""
        assert VectorStoreFactory.get_provider_names() == []

    def test_register_provider(self) -> None:
        """Test registering a new provider."""

        class CustomVectorStore(BaseVectorStore):
            def __init__(self, **kwargs: Any) -> None:
                pass

            @property
            def provider_name(self) -> str:
                return "custom"

            def upsert(self, records, **kwargs):
                return []

            def query(self, vector, top_k=5, filters=None, **kwargs):
                return QueryResult(ids=[], scores=[])

            def delete(self, ids, **kwargs):
                return True

            def count(self, **kwargs):
                return 0

            def clear(self, **kwargs):
                return True

        VectorStoreFactory.register("custom", CustomVectorStore)

        assert "custom" in VectorStoreFactory.get_provider_names()
        assert VectorStoreFactory.has_provider("custom")

    def test_register_fake(self) -> None:
        """Test registering FakeVectorStore."""
        VectorStoreFactory.register("fake", FakeVectorStore)

        assert "fake" in VectorStoreFactory.get_provider_names()
        assert VectorStoreFactory.has_provider("fake")

    def test_unregister_provider(self) -> None:
        """Test unregistering a provider."""
        VectorStoreFactory.register("test", FakeVectorStore)
        assert VectorStoreFactory.has_provider("test")

        result = VectorStoreFactory.unregister("test")
        assert result is True
        assert not VectorStoreFactory.has_provider("test")

    def test_unregister_nonexistent(self) -> None:
        """Test unregistering a provider that doesn't exist."""
        result = VectorStoreFactory.unregister("nonexistent")
        assert result is False

    def test_provider_case_insensitive(self) -> None:
        """Test that provider names are case-insensitive."""

        class TestProvider(BaseVectorStore):
            def __init__(self, **kwargs: Any) -> None:
                pass

            @property
            def provider_name(self) -> str:
                return "test"

            def upsert(self, records, **kwargs):
                return []

            def query(self, vector, top_k=5, filters=None, **kwargs):
                return QueryResult(ids=[], scores=[])

            def delete(self, ids, **kwargs):
                return True

            def count(self, **kwargs):
                return 0

            def clear(self, **kwargs):
                return True

        VectorStoreFactory.register("TestProvider", TestProvider)
        assert VectorStoreFactory.has_provider("testprovider")
        assert VectorStoreFactory.has_provider("TESTPROVIDER")
        assert VectorStoreFactory.has_provider("TestProvider")

    def test_clear_all_providers(self) -> None:
        """Test clearing all providers."""
        VectorStoreFactory.register("a", FakeVectorStore)
        VectorStoreFactory.register("b", FakeVectorStore)
        VectorStoreFactory.register("c", FakeVectorStore)

        assert len(VectorStoreFactory.get_provider_names()) == 3

        VectorStoreFactory.clear()

        assert VectorStoreFactory.get_provider_names() == []


class TestVectorStoreFactoryCreate:
    """Test VectorStoreFactory.create() method."""

    def setup_method(self) -> None:
        """Clear and register fake before each test."""
        VectorStoreFactory.clear()
        VectorStoreFactory.register("fake", FakeVectorStore)

    def teardown_method(self) -> None:
        """Clear after each test."""
        VectorStoreFactory.clear()

    def test_create_fake_provider(self) -> None:
        """Test creating a FakeVectorStore instance."""
        settings = Settings(
            vector_store=VectorStoreConfig(
                provider="fake",
                persist_directory="./data/db/test",
                collection_name="test_collection"
            )
        )

        vs = VectorStoreFactory.create(settings)

        assert isinstance(vs, FakeVectorStore)
        assert vs.provider_name == "fake"
        assert vs._persist_directory == "./data/db/test"
        assert vs._collection_name == "test_collection"

    def test_create_unknown_provider(self) -> None:
        """Test that unknown provider raises error."""
        settings = Settings(
            vector_store=VectorStoreConfig(provider="unknown-vs")
        )

        with pytest.raises(UnknownVectorStoreProviderError) as exc_info:
            VectorStoreFactory.create(settings)

        assert "unknown-vs" in str(exc_info.value)

    def test_create_with_kwargs_override(self) -> None:
        """Test that kwargs override settings."""
        settings = Settings(
            vector_store=VectorStoreConfig(
                provider="fake",
                collection_name="original"
            )
        )

        vs = VectorStoreFactory.create(settings, collection_name="override")

        assert isinstance(vs, FakeVectorStore)
        assert vs._collection_name == "override"

    def test_create_missing_provider_config(self) -> None:
        """Test that missing provider config raises error."""
        settings = Settings(
            vector_store=VectorStoreConfig(provider=None)
        )

        with pytest.raises(VectorStoreConfigurationError):
            VectorStoreFactory.create(settings)


class TestVectorStoreFactoryDynamicRegistration:
    """Test dynamic provider registration workflow."""

    def setup_method(self) -> None:
        """Clear before each test."""
        VectorStoreFactory.clear()

    def teardown_method(self) -> None:
        """Clear after each test."""
        VectorStoreFactory.clear()

    def test_register_and_create_custom_provider(self) -> None:
        """Test registering and creating a custom provider."""

        class ChromaVectorStore(BaseVectorStore):
            def __init__(
                self,
                persist_directory: str = "./data/db/chroma",
                collection_name: str = "knowledge_hub",
                **kwargs: Any
            ) -> None:
                self._persist_directory = persist_directory
                self._collection_name = collection_name

            @property
            def provider_name(self) -> str:
                return "chroma"

            def upsert(self, records, **kwargs) -> list[str]:
                return [r.id for r in records]

            def query(self, vector, top_k=5, filters=None, **kwargs) -> QueryResult:
                return QueryResult(ids=["mock1"], scores=[0.95], metadata=[None])

            def delete(self, ids, **kwargs) -> bool:
                return True

            def count(self, **kwargs) -> int:
                return 0

            def clear(self, **kwargs) -> bool:
                return True

        # Register the provider
        VectorStoreFactory.register("chroma", ChromaVectorStore)

        # Create an instance
        settings = Settings(
            vector_store=VectorStoreConfig(
                provider="chroma",
                persist_directory="./data/db/custom",
                collection_name="my_collection"
            )
        )

        vs = VectorStoreFactory.create(settings)

        assert isinstance(vs, ChromaVectorStore)
        assert vs.provider_name == "chroma"

        # Verify the store works
        result = vs.query([0.1, 0.2, 0.3])
        assert len(result.ids) == 1

    def test_multiple_providers(self) -> None:
        """Test registering multiple providers."""

        class ChromaVectorStore(BaseVectorStore):
            def __init__(self, **kwargs: Any) -> None:
                pass

            @property
            def provider_name(self) -> str:
                return "chroma"

            def upsert(self, records, **kwargs):
                return []

            def query(self, vector, top_k=5, filters=None, **kwargs):
                return QueryResult(ids=[], scores=[])

            def delete(self, ids, **kwargs):
                return True

            def count(self, **kwargs):
                return 0

            def clear(self, **kwargs):
                return True

        class QdrantVectorStore(BaseVectorStore):
            def __init__(self, **kwargs: Any) -> None:
                pass

            @property
            def provider_name(self) -> str:
                return "qdrant"

            def upsert(self, records, **kwargs):
                return []

            def query(self, vector, top_k=5, filters=None, **kwargs):
                return QueryResult(ids=[], scores=[])

            def delete(self, ids, **kwargs):
                return True

            def count(self, **kwargs):
                return 0

            def clear(self, **kwargs):
                return True

        class PineconeVectorStore(BaseVectorStore):
            def __init__(self, **kwargs: Any) -> None:
                pass

            @property
            def provider_name(self) -> str:
                return "pinecone"

            def upsert(self, records, **kwargs):
                return []

            def query(self, vector, top_k=5, filters=None, **kwargs):
                return QueryResult(ids=[], scores=[])

            def delete(self, ids, **kwargs):
                return True

            def count(self, **kwargs):
                return 0

            def clear(self, **kwargs):
                return True

        # Register all providers
        VectorStoreFactory.register("chroma", ChromaVectorStore)
        VectorStoreFactory.register("qdrant", QdrantVectorStore)
        VectorStoreFactory.register("pinecone", PineconeVectorStore)

        # Verify all are registered
        assert len(VectorStoreFactory.get_provider_names()) == 3
        assert "chroma" in VectorStoreFactory.get_provider_names()
        assert "qdrant" in VectorStoreFactory.get_provider_names()
        assert "pinecone" in VectorStoreFactory.get_provider_names()


class TestBaseVectorStoreInterface:
    """Test that BaseVectorStore is properly abstract."""

    def test_cannot_instantiate_base(self) -> None:
        """Test that BaseVectorStore cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseVectorStore()

    def test_subclass_must_implement_all_methods(self) -> None:
        """Test that subclasses must implement all abstract methods."""

        class IncompleteVectorStore(BaseVectorStore):
            @property
            def provider_name(self) -> str:
                return "incomplete"

            def upsert(self, records, **kwargs):
                pass

            def query(self, vector, top_k=5, filters=None, **kwargs):
                pass

            def delete(self, ids, **kwargs):
                pass

            def count(self, **kwargs):
                pass

            # Missing clear() implementation

        with pytest.raises(TypeError):
            IncompleteVectorStore()


class TestVectorStoreErrors:
    """Test vector store error classes."""

    def test_vector_store_error_basic(self) -> None:
        """Test basic VectorStoreError."""
        error = VectorStoreError("Test error")
        assert str(error) == "Test error"
        assert error.provider is None
        assert error.code is None

    def test_vector_store_error_with_details(self) -> None:
        """Test VectorStoreError with provider and code."""
        error = VectorStoreError(
            "Connection failed",
            provider="chroma",
            code=500,
            details={"retry_after": 60}
        )
        assert error.provider == "chroma"
        assert error.code == 500
        assert error.details["retry_after"] == 60

    def test_unknown_provider_error(self) -> None:
        """Test UnknownVectorStoreProviderError."""
        error = UnknownVectorStoreProviderError(
            "Unknown provider: test",
            provider="test"
        )
        assert error.provider == "test"
