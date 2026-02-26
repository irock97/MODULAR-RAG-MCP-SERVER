"""Unit tests for RRFFusion."""

import pytest

from core.query_engine.fusion import RRFFusion
from core.types import RetrievalResult


class TestRRFFusion:
    """Tests for RRFFusion."""

    def test_fuse_basic(self):
        """Test basic fusion of two result lists."""
        # Dense results: doc1 > doc2 > doc3
        dense_results = [
            RetrievalResult(chunk_id="doc1", score=0.9, text="Text 1"),
            RetrievalResult(chunk_id="doc2", score=0.8, text="Text 2"),
            RetrievalResult(chunk_id="doc3", score=0.7, text="Text 3"),
        ]

        # Sparse results: doc2 > doc4 > doc1
        sparse_results = [
            RetrievalResult(chunk_id="doc2", score=0.95, text="Sparse 2"),
            RetrievalResult(chunk_id="doc4", score=0.85, text="Sparse 4"),
            RetrievalResult(chunk_id="doc1", score=0.75, text="Sparse 1"),
        ]

        fusion = RRFFusion(k=60)
        fused = fusion.fuse([dense_results, sparse_results])

        # Verify all documents are present
        fused_ids = [r.chunk_id for r in fused]
        assert "doc1" in fused_ids
        assert "doc2" in fused_ids
        assert "doc3" in fused_ids
        assert "doc4" in fused_ids

        # Verify doc2 ranks highest (present in both lists at top positions)
        assert fused[0].chunk_id == "doc2"

        # Verify doc1 ranks second (present in both, but lower ranks)
        assert fused[1].chunk_id == "doc1"

    def test_fuse_empty_lists(self):
        """Test fusion with empty result lists."""
        fusion = RRFFusion()

        # All empty
        fused = fusion.fuse([[], []])
        assert fused == []

        # One empty, one non-empty
        non_empty = [RetrievalResult(chunk_id="doc1", score=0.9, text="Text")]
        fused = fusion.fuse([non_empty, []])
        assert len(fused) == 1
        assert fused[0].chunk_id == "doc1"

    def test_fuse_single_list(self):
        """Test fusion with a single result list."""
        results = [
            RetrievalResult(chunk_id="doc1", score=0.9, text="Text 1"),
            RetrievalResult(chunk_id="doc2", score=0.8, text="Text 2"),
        ]

        fusion = RRFFusion()
        fused = fusion.fuse([results])

        assert len(fused) == 2
        assert fused[0].chunk_id == "doc1"
        assert fused[1].chunk_id == "doc2"

    def test_fuse_top_k(self):
        """Test fusion with top_k limit."""
        results1 = [
            RetrievalResult(chunk_id="doc1", score=0.9, text="Text 1"),
            RetrievalResult(chunk_id="doc2", score=0.8, text="Text 2"),
            RetrievalResult(chunk_id="doc3", score=0.7, text="Text 3"),
        ]
        results2 = [
            RetrievalResult(chunk_id="doc3", score=0.6, text="Text 3"),
            RetrievalResult(chunk_id="doc4", score=0.5, text="Text 4"),
            RetrievalResult(chunk_id="doc5", score=0.4, text="Text 5"),
        ]

        fusion = RRFFusion()
        fused = fusion.fuse([results1, results2], top_k=3)

        assert len(fused) == 3

    def test_fuse_deterministic(self):
        """Test that fusion is deterministic."""
        results1 = [
            RetrievalResult(chunk_id="doc1", score=0.9, text="Text 1"),
            RetrievalResult(chunk_id="doc2", score=0.8, text="Text 2"),
        ]
        results2 = [
            RetrievalResult(chunk_id="doc2", score=0.7, text="Text 2"),
            RetrievalResult(chunk_id="doc1", score=0.6, text="Text 1"),
        ]

        fusion = RRFFusion()

        # Run multiple times
        fused1 = fusion.fuse([results1, results2])
        fused2 = fusion.fuse([results1, results2])

        # Should produce same results
        assert [r.chunk_id for r in fused1] == [r.chunk_id for r in fused2]

    def test_fuse_k_parameter(self):
        """Test that k parameter can be configured."""
        results = [RetrievalResult(chunk_id="doc1", score=0.9, text="Text")]

        # Test with custom k
        fusion = RRFFusion(k=50)
        fused = fusion.fuse([results])
        assert len(fused) == 1
        assert fusion.k == 50

    def test_fuse_default_k(self):
        """Test that default k is 60."""
        fusion = RRFFusion()
        assert fusion.k == 60

    def test_fuse_with_trace(self):
        """Test fusion with trace context."""
        from core.trace.trace_context import TraceContext

        results = [RetrievalResult(chunk_id="doc1", score=0.9, text="Text")]

        fusion = RRFFusion()
        trace = TraceContext()

        fused = fusion.fuse([results], trace=trace)

        assert "fusion_start" in trace.get_all_stages()
        assert "fusion_complete" in trace.get_all_stages()

    def test_fuse_preserves_text_and_metadata(self):
        """Test that fusion preserves text and metadata from first occurrence."""
        results1 = [
            RetrievalResult(
                chunk_id="doc1",
                score=0.9,
                text="Text from dense",
                metadata={"source": "dense"}
            ),
        ]
        results2 = [
            RetrievalResult(
                chunk_id="doc1",
                score=0.8,
                text="Text from sparse",
                metadata={"source": "sparse"}
            ),
        ]

        fusion = RRFFusion()
        fused = fusion.fuse([results1, results2])

        # Should preserve text and metadata from first occurrence (results1)
        assert fused[0].text == "Text from dense"
        assert fused[0].metadata == {"source": "dense"}


class TestRRFFusionValidation:
    """Tests for RRFFusion validation."""

    def test_invalid_k_raises(self):
        """Test that invalid k raises ValueError."""
        with pytest.raises(ValueError, match="k must be > 0"):
            RRFFusion(k=0)

        with pytest.raises(ValueError, match="k must be > 0"):
            RRFFusion(k=-1)
