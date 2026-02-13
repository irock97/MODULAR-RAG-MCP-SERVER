"""Integration tests for ChromaStore.

This module contains integration tests for ChromaStore that require
an actual ChromaDB instance to run. These tests are marked as integration
tests and can be skipped with: pytest -m "not integration"

Design Principles:
    - Integration tests: Require actual ChromaDB instance
    - Deterministic: Tests use deterministic data for reproducibility
    - Cleanup: Tests clean up their data after completion
"""

import pytest
import os
import tempfile
from unittest.mock import MagicMock

from libs.vector_store.chroma_store import ChromaStore
from libs.vector_store.base_vector_store import VectorRecord, QueryResult


# Mark all tests as integration tests
pytestmark = pytest.mark.integration


class TestChromaStoreIntegration:
    """Integration tests for ChromaStore with real ChromaDB."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for ChromaDB persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def store(self, temp_dir):
        """Create a ChromaStore with temporary directory."""
        store = ChromaStore(
            persist_directory=temp_dir,
            collection_name="test_collection"
        )
        yield store
        # Cleanup is handled by the store's clear method or temp dir deletion

    def test_upsert_and_query_roundtrip(self, store):
        """Test basic upsert and query roundtrip."""
        # Clear any existing data
        store.clear()

        # Create test records
        records = [
            VectorRecord(
                id="doc1",
                vector=[0.1, 0.2, 0.3],
                metadata={"source": "test1", "category": "A"}
            ),
            VectorRecord(
                id="doc2",
                vector=[0.4, 0.5, 0.6],
                metadata={"source": "test2", "category": "B"}
            ),
            VectorRecord(
                id="doc3",
                vector=[0.1, 0.2, 0.3],  # Same as doc1
                metadata={"source": "test3", "category": "A"}
            ),
        ]

        # Upsert records
        ids = store.upsert(records)
        assert len(ids) == 3
        assert "doc1" in ids
        assert "doc2" in ids
        assert "doc3" in ids

        # Query for similar vectors
        query_vec = [0.1, 0.2, 0.3]
        result = store.query(query_vector=query_vec, top_k=3)

        # Should return doc1 and doc3 (same vector)
        assert len(result.ids) >= 2
        assert "doc1" in result.ids or "doc3" in result.ids

    def test_query_with_filters(self, store):
        """Test query with metadata filters."""
        # Clear any existing data
        store.clear()

        # Create test records with different categories
        records = [
            VectorRecord(
                id="cat_a_1",
                vector=[0.1, 0.2, 0.3],
                metadata={"category": "A", "priority": 1}
            ),
            VectorRecord(
                id="cat_a_2",
                vector=[0.1, 0.2, 0.3],
                metadata={"category": "A", "priority": 2}
            ),
            VectorRecord(
                id="cat_b_1",
                vector=[0.1, 0.2, 0.3],
                metadata={"category": "B", "priority": 1}
            ),
        ]

        store.upsert(records)

        # Query with filter for category A
        result = store.query(
            query_vector=[0.1, 0.2, 0.3],
            top_k=10,
            filters={"category": "A"}
        )

        # Should only return category A records
        assert len(result.ids) == 2
        assert "cat_a_1" in result.ids
        assert "cat_a_2" in result.ids
        assert "cat_b_1" not in result.ids

    def test_delete_records(self, store):
        """Test deleting records."""
        # Clear any existing data
        store.clear()

        # Create and upsert a record
        record = VectorRecord(
            id="to_delete",
            vector=[0.1, 0.2, 0.3],
            metadata={"source": "test"}
        )
        store.upsert([record])

        # Verify it exists
        result = store.query(query_vector=[0.1, 0.2, 0.3], top_k=10)
        assert "to_delete" in result.ids

        # Delete it
        success = store.delete(ids=["to_delete"])
        assert success is True

        # Verify it's gone
        result = store.query(query_vector=[0.1, 0.2, 0.3], top_k=10)
        assert "to_delete" not in result.ids

    def test_count_records(self, store):
        """Test counting records."""
        # Clear any existing data
        store.clear()

        # Initially should be empty
        count = store.count()
        assert count == 0

        # Add some records
        records = [
            VectorRecord(id="doc1", vector=[0.1, 0.2, 0.3]),
            VectorRecord(id="doc2", vector=[0.4, 0.5, 0.6]),
            VectorRecord(id="doc3", vector=[0.7, 0.8, 0.9]),
        ]
        store.upsert(records)

        # Count should be 3
        count = store.count()
        assert count == 3

    def test_clear_collection(self, store):
        """Test clearing the collection."""
        # Add some records
        records = [
            VectorRecord(id="doc1", vector=[0.1, 0.2, 0.3]),
            VectorRecord(id="doc2", vector=[0.4, 0.5, 0.6]),
        ]
        store.upsert(records)

        # Verify records exist
        count = store.count()
        assert count == 2

        # Clear collection
        success = store.clear()
        assert success is True

        # Should be empty now
        count = store.count()
        assert count == 0

    def test_reopen_persisted_store(self, temp_dir):
        """Test that data persists across store reopen."""
        collection_name = "persist_test"

        # Create first store and add data
        store1 = ChromaStore(
            persist_directory=temp_dir,
            collection_name=collection_name
        )
        records = [
            VectorRecord(id="persistent_doc", vector=[0.1, 0.2, 0.3]),
        ]
        store1.upsert(records)

        # Create new store with same directory
        store2 = ChromaStore(
            persist_directory=temp_dir,
            collection_name=collection_name
        )

        # Data should be available in the new store
        result = store2.query(query_vector=[0.1, 0.2, 0.3], top_k=10)
        assert "persistent_doc" in result.ids


class TestChromaStoreFactory:
    """Tests for VectorStoreFactory with ChromaStore."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for ChromaDB persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_factory_create_chroma(self, temp_dir):
        """Test VectorStoreFactory can create ChromaStore."""
        from libs.vector_store.vector_store_factory import VectorStoreFactory
        from libs.vector_store.chroma_store import ChromaStore

        # Register provider
        VectorStoreFactory.register("chroma", ChromaStore)

        # Create mock settings
        mock_settings = MagicMock()
        mock_settings.vector_store.provider = "chroma"
        mock_settings.vector_store.persist_directory = temp_dir
        mock_settings.vector_store.collection_name = "factory_test"

        # Test factory creation
        store = VectorStoreFactory.create(mock_settings)

        assert isinstance(store, ChromaStore)
        assert store._collection_name == "factory_test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
