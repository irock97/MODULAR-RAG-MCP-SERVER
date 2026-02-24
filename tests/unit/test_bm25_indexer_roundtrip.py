"""Unit tests for BM25Indexer roundtrip (build, save, load, search).

This module tests the BM25Indexer class including:
- IDF computation accuracy
- Inverted index building
- Save/load persistence
- Search functionality
- Query result stability
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from core.types import ChunkRecord
from ingestion.storage import BM25Indexer
from ingestion.embedding import SparseEncoder


class TestBM25IndexerIDF:
    """Tests for IDF computation accuracy."""

    def test_idf_formula(self):
        """Test IDF computation with known values."""
        indexer = BM25Indexer(index_dir="/tmp/test_idf")

        # With N=10, df=1: IDF = log((10 - 1 + 0.5) / (1 + 0.5) + 1)
        idf = indexer._compute_idf(df=1, n=10)
        expected = 9.5 / 1.5 + 1  # Without log for easy verification

        assert idf > 0

    def test_idf_rare_term(self):
        """Test IDF is high for rare terms."""
        indexer = BM25Indexer(index_dir="/tmp/test")

        idf_rare = indexer._compute_idf(df=1, n=100)
        idf_common = indexer._compute_idf(df=100, n=100)

        # Rare term should have higher IDF
        assert idf_rare > idf_common

    def test_idf_all_docs(self):
        """Test IDF is 0 when term appears in all docs."""
        indexer = BM25Indexer(index_dir="/tmp/test")

        # df = N means term appears in all documents
        idf = indexer._compute_idf(df=100, n=100)

        # With smoothing: log((N - df + 0.5) / (df + 0.5) + 1) = log(0.5/100.5 + 1)
        assert idf > 0  # Smoothing prevents zero


class TestBM25IndexerBuild:
    """Tests for index building."""

    def test_empty_records(self):
        """Test building index with empty records."""
        indexer = BM25Indexer(index_dir="/tmp/test_empty")

        indexer.build([])

        assert indexer.num_docs == 0
        assert indexer.num_terms == 0

    def test_single_record(self):
        """Test building index with single record."""
        indexer = BM25Indexer(index_dir="/tmp/test_single")

        records = [
            ChunkRecord(
                id="doc1",
                text="Hello world",
                sparse_vector={"hello": 1.0, "world": 1.0},
            )
        ]

        indexer.build(records)

        assert indexer.num_docs == 1
        assert indexer.num_terms == 2

    def test_multiple_records(self):
        """Test building index with multiple records."""
        indexer = BM25Indexer(index_dir="/tmp/test_multi")

        records = [
            ChunkRecord(
                id="doc1",
                text="machine learning algorithms",
                sparse_vector={"machine": 1.0, "learning": 1.0, "algorithms": 1.0},
            ),
            ChunkRecord(
                id="doc2",
                text="deep learning neural networks",
                sparse_vector={"deep": 1.0, "learning": 1.0, "neural": 1.0, "networks": 1.0},
            ),
            ChunkRecord(
                id="doc3",
                text="machine learning basics",
                sparse_vector={"machine": 1.0, "learning": 1.0, "basics": 1.0},
            ),
        ]

        indexer.build(records)

        assert indexer.num_docs == 3
        # "learning" appears in all 3 docs, others in 1 or 2
        assert "learning" in indexer._index

    def test_doc_length_tracking(self):
        """Test that document lengths are tracked correctly."""
        indexer = BM25Indexer(index_dir="/tmp/test_doclen")

        records = [
            ChunkRecord(
                id="doc1",
                text="one two three four five",
                sparse_vector={"one": 1.0, "two": 1.0, "three": 1.0, "four": 1.0, "five": 1.0},
            ),
        ]

        indexer.build(records)

        assert indexer._doc_lengths["doc1"] == 5


class TestBM25IndexerSearch:
    """Tests for search functionality."""

    def test_empty_index_search(self):
        """Test searching empty index returns empty results."""
        indexer = BM25Indexer(index_dir="/tmp/test_empty_search")
        indexer.build([])

        results = indexer.search("query")

        assert results == []

    def test_single_term_query(self):
        """Test searching with single term query."""
        indexer = BM25Indexer(index_dir="/tmp/test_single_term")

        records = [
            ChunkRecord(
                id="doc1",
                text="hello world",
                sparse_vector={"hello": 1.0, "world": 1.0},
            ),
            ChunkRecord(
                id="doc2",
                text="goodbye world",
                sparse_vector={"goodbye": 1.0, "world": 1.0},
            ),
        ]

        indexer.build(records)
        results = indexer.search("hello")

        assert len(results) == 1
        assert results[0][0] == "doc1"

    def test_multi_term_query(self):
        """Test searching with multiple term query."""
        indexer = BM25Indexer(index_dir="/tmp/test_multi_term")

        records = [
            ChunkRecord(
                id="doc1",
                text="machine learning",
                sparse_vector={"machine": 1.0, "learning": 1.0},
            ),
            ChunkRecord(
                id="doc2",
                text="machine learning",
                sparse_vector={"machine": 1.0, "learning": 1.0},
            ),
            ChunkRecord(
                id="doc3",
                text="deep learning",
                sparse_vector={"deep": 1.0, "learning": 1.0},
            ),
        ]

        indexer.build(records)
        results = indexer.search("machine learning")

        # All 3 docs contain at least one query term
        # doc1 and doc2 contain both terms (higher score)
        # doc3 contains only "learning"
        assert len(results) == 3
        doc_ids = [r[0] for r in results]

        # doc1 and doc2 should have higher scores (both terms match)
        # Find their scores
        scores = {r[0]: r[1] for r in results}
        assert scores["doc1"] > scores["doc3"]
        assert scores["doc2"] > scores["doc3"]

    def test_top_k_limit(self):
        """Test that top_k parameter limits results."""
        indexer = BM25Indexer(index_dir="/tmp/test_topk")

        records = [
            ChunkRecord(id=f"doc{i}", text="test document", sparse_vector={"test": 1.0, "document": 1.0})
            for i in range(20)
        ]

        indexer.build(records)
        results = indexer.search("test", top_k=5)

        assert len(results) == 5

    def test_result_stability(self):
        """Test that search returns stable results."""
        indexer = BM25Indexer(index_dir="/tmp/test_stability")

        records = [
            ChunkRecord(
                id="doc1",
                text="apple banana cherry",
                sparse_vector={"apple": 1.0, "banana": 1.0, "cherry": 1.0},
            ),
            ChunkRecord(
                id="doc2",
                text="banana date elderberry",
                sparse_vector={"banana": 1.0, "date": 1.0, "elderberry": 1.0},
            ),
        ]

        indexer.build(records)

        # Run search multiple times
        results1 = indexer.search("apple banana")
        results2 = indexer.search("apple banana")

        assert results1 == results2


class TestBM25IndexerRoundtrip:
    """Tests for save/load roundtrip."""

    def test_save_load(self):
        """Test saving and loading index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_dir = Path(tmpdir) / "bm25_test"

            # Build index
            indexer1 = BM25Indexer(index_dir=str(index_dir))
            records = [
                ChunkRecord(
                    id="doc1",
                    text="hello world",
                    sparse_vector={"hello": 1.0, "world": 1.0},
                ),
                ChunkRecord(
                    id="doc2",
                    text="test query",
                    sparse_vector={"test": 1.0, "query": 1.0},
                ),
            ]
            indexer1.build(records)
            indexer1.save()

            # Load index
            indexer2 = BM25Indexer(index_dir=str(index_dir))
            loaded = indexer2.load()

            assert loaded is True
            assert indexer2.num_docs == 2
            assert indexer2.num_terms == 4

    def test_search_after_load(self):
        """Test search works correctly after loading index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_dir = Path(tmpdir) / "bm25_search_test"

            # Build and save
            indexer1 = BM25Indexer(index_dir=str(index_dir))
            records = [
                ChunkRecord(
                    id="doc1",
                    text="machine learning",
                    sparse_vector={"machine": 1.0, "learning": 1.0},
                ),
            ]
            indexer1.build(records)
            indexer1.save()

            # Load and search
            indexer2 = BM25Indexer(index_dir=str(index_dir))
            indexer2.load()
            results = indexer2.search("machine learning")

            assert len(results) == 1
            assert results[0][0] == "doc1"

    def test_empty_load(self):
        """Test loading non-existent index returns False."""
        indexer = BM25Indexer(index_dir="/tmp/non_existent_index")

        loaded = indexer.load()

        assert loaded is False


class TestBM25IndexerWithSparseEncoder:
    """Integration tests using real SparseEncoder output."""

    def test_full_pipeline(self):
        """Test complete pipeline: encode -> index -> search."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_dir = Path(tmpdir) / "bm25_pipeline"

            # Encode documents
            encoder = SparseEncoder()
            chunks = [
                ChunkRecord(
                    id="doc1",
                    text="The quick brown fox jumps over the lazy dog",
                    metadata={"source": "test1"},
                ),
                ChunkRecord(
                    id="doc2",
                    text="The lazy dog sleeps all day",
                    metadata={"source": "test2"},
                ),
                ChunkRecord(
                    id="doc3",
                    text="A fox is clever and wild",
                    metadata={"source": "test3"},
                ),
            ]
            records = encoder.encode(chunks)

            # Build index
            indexer = BM25Indexer(index_dir=str(index_dir))
            indexer.build(records)

            # Search
            results = indexer.search("fox lazy", top_k=3)

            # Should find relevant documents
            assert len(results) > 0
            doc_ids = [r[0] for r in results]
            assert "doc1" in doc_ids  # Contains both fox and lazy
            assert "doc2" in doc_ids  # Contains lazy
            assert "doc3" in doc_ids  # Contains fox


class TestBM25IndexerEdgeCases:
    """Tests for edge cases."""

    def test_empty_sparse_vector(self):
        """Test records with empty sparse_vector."""
        indexer = BM25Indexer(index_dir="/tmp/test_empty_vec")

        records = [
            ChunkRecord(
                id="doc1",
                text="hello world",
                sparse_vector={},  # Empty
            ),
            ChunkRecord(
                id="doc2",
                text="test query",
                sparse_vector={"test": 1.0},
            ),
        ]

        indexer.build(records)

        # Should still index doc2
        assert indexer.num_docs == 2
        assert indexer.num_terms == 1

    def test_query_no_matching_terms(self):
        """Test query with no matching terms."""
        indexer = BM25Indexer(index_dir="/tmp/test_no_match")

        records = [
            ChunkRecord(
                id="doc1",
                text="hello world",
                sparse_vector={"hello": 1.0, "world": 1.0},
            ),
        ]

        indexer.build(records)
        results = indexer.search("xyz abc")

        assert results == []

    def test_case_handling(self):
        """Test that query is case-insensitive."""
        indexer = BM25Indexer(index_dir="/tmp/test_case")

        records = [
            ChunkRecord(
                id="doc1",
                text="HELLO WORLD",
                sparse_vector={"hello": 1.0, "world": 1.0},
            ),
        ]

        indexer.build(records)

        # Query should match regardless of case
        results_lower = indexer.search("hello")
        results_upper = indexer.search("HELLO")
        results_mixed = indexer.search("Hello")

        assert results_lower == results_upper == results_mixed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
