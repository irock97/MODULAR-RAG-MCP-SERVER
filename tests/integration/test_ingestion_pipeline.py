"""Integration tests for IngestionPipeline.

These tests verify the pipeline orchestration works correctly.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.settings import Settings
from ingestion.pipeline import (
    IngestionPipeline,
    PipelineResult,
)


class TestPipelineResult:
    """Tests for PipelineResult."""

    def test_default_result(self):
        """Test default result values."""
        result = PipelineResult(success=True, file_path="/test/file.pdf")

        assert result.success is True
        assert result.file_path == "/test/file.pdf"
        assert result.doc_id is None
        assert result.chunk_count == 0
        assert result.image_count == 0
        assert result.vector_ids == []
        assert result.error is None
        assert result.stages == {}

    def test_result_with_data(self):
        """Test result with data."""
        result = PipelineResult(
            success=True,
            file_path="/test/file.pdf",
            doc_id="abc123",
            chunk_count=10,
            image_count=5,
            vector_ids=["id1", "id2", "id3"],
            error=None,
            stages={"loading": {"text_length": 1000}},
        )

        assert result.success is True
        assert result.doc_id == "abc123"
        assert result.chunk_count == 10
        assert result.image_count == 5
        assert len(result.vector_ids) == 3
        assert "loading" in result.stages

    def test_to_dict(self):
        """Test to_dict conversion."""
        result = PipelineResult(
            success=True,
            file_path="/test/file.pdf",
            chunk_count=5,
        )

        d = result.to_dict()
        assert d["success"] is True
        assert d["file_path"] == "/test/file.pdf"
        assert d["chunk_count"] == 5
        assert d["vector_ids_count"] == 0


class TestIngestionPipeline:
    """Tests for IngestionPipeline attributes."""

    def test_pipeline_attributes(self):
        """Test pipeline has correct attributes."""
        # Create mock settings that work for initialization
        with patch('ingestion.pipeline.DocumentChunker'):
            with patch('ingestion.pipeline.ChunkRefiner'):
                with patch('ingestion.pipeline.MetadataEnricher'):
                    with patch('ingestion.pipeline.ImageCaptioner'):
                        with patch('ingestion.pipeline.DenseEncoder'):
                            with patch('ingestion.pipeline.SparseEncoder'):
                                with patch('ingestion.pipeline.BatchProcessor'):
                                    with patch('ingestion.pipeline.VectorUpserter'):
                                        with patch('ingestion.pipeline.BM25Indexer'):
                                            with patch('ingestion.pipeline.ImageStorage'):
                                                with patch('ingestion.pipeline.SqliteFileIntegrityChecker'):
                                                    with patch('ingestion.pipeline.PdfLoader'):
                                                        with patch('ingestion.pipeline.EmbeddingFactory'):
                                                            settings = MagicMock(spec=Settings)
                                                            settings.ingestion = MagicMock()
                                                            settings.ingestion.batch_size = 100
                                                            settings.embedding = MagicMock()
                                                            settings.embedding.provider = "ollama"
                                                            settings.vector_store = MagicMock()
                                                            settings.vector_store.provider = "chroma"

                                                            pipeline = IngestionPipeline(
                                                                settings,
                                                                collection="test"
                                                            )

                                                            assert pipeline.collection == "test"
                                                            assert pipeline.force is False
                                                            assert pipeline.settings is settings


class TestPipelineEdgeCases:
    """Tests for pipeline edge cases."""

    def test_pipeline_result_error_handling(self):
        """Test PipelineResult with error."""
        result = PipelineResult(
            success=False,
            file_path="/test/file.pdf",
            error="Something went wrong",
        )

        assert result.success is False
        assert result.error == "Something went wrong"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
