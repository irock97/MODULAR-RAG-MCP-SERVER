"""Integration tests for IngestionPipeline.

These tests verify the pipeline orchestration works correctly.
"""

import os
import tempfile
from pathlib import Path

import pytest

from core.settings import load_settings
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


class TestIngestionPipelineWithRealComponents:
    """Integration tests using real components from settings.yaml."""

    @pytest.fixture
    def project_root(self) -> Path:
        """Return project root."""
        return Path(__file__).parent.parent.parent

    @pytest.fixture
    def complex_pdf_path(self, project_root: Path) -> Path:
        """Return path to complex_technical_doc.pdf."""
        pdf_path = project_root / "tests" / "fixtures" / "sample_documents" / "complex_technical_doc.pdf"
        assert pdf_path.exists(), f"Test PDF not found: {pdf_path}"
        return pdf_path

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for test databases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            # Create necessary directories
            (tmppath / "data" / "db").mkdir(parents=True, exist_ok=True)
            (tmppath / "images").mkdir(exist_ok=True)
            (tmppath / "chroma").mkdir(exist_ok=True)
            yield tmppath

    @pytest.fixture
    def real_settings(self, project_root, temp_dirs):
        """Load real settings from settings.yaml and configure for testing."""
        # Register splitter providers first
        from libs.splitter.splitter_factory import SplitterFactory
        from libs.splitter.recursive_splitter import RecursiveSplitter

        SplitterFactory.register("recursive", RecursiveSplitter)

        # Register vector store providers
        from libs.vector_store.vector_store_factory import VectorStoreFactory
        from libs.vector_store.chroma_store import ChromaStore

        VectorStoreFactory.register("chroma", ChromaStore)

        # Load settings
        settings = load_settings(str(project_root / "config" / "settings.yaml"))

        # Override directories to use temp directories
        # settings.vector_store.persist_directory = str(temp_dirs / "chroma")

        # Disable vision LLM for testing (would require API call)
        # settings.vision_llm.enabled = False

        return settings

    def test_pipeline_with_complex_pdf_real_components(self, real_settings, complex_pdf_path, temp_dirs):
        """Test complete pipeline with real components.

        This test runs the full ingestion pipeline using real components
        from settings.yaml and verifies that:
        1. Pipeline executes without errors
        2. Chunks are generated from the real PDF
        3. Document is processed (not skipped)
        4. Trace context records all stages correctly
        """
        from core.trace.trace_context import TraceContext

        # Create pipeline with real settings
        # Note: DenseEncoder 需要真实的 API 调用，这里会调用 Qwen embedding API
        pipeline = IngestionPipeline(
            real_settings,
            collection="collection",
            force=True,
        )

        # Create trace context for testing
        trace = TraceContext()

        # Run pipeline on real PDF
        result = pipeline.run(str(complex_pdf_path), trace=trace)

        # Verify results
        assert result.success is True, f"Pipeline failed: {result.error}"
        assert result.doc_id is not None

        # Verify stages completed
        assert "loading" in result.stages
        assert "chunking" in result.stages
        assert result.stages["chunking"]["chunk_count"] > 0, "No chunks generated in chunking stage"
        assert "encoding" in result.stages
        assert "storage" in result.stages

        # Verify trace context
        all_stages = trace.get_all_stages()
        assert len(all_stages) > 0, "No trace stages recorded"

        # Check expected trace stages
        expected_stages = [
            "document_loading_complete",
            "chunking_complete",
            "transform_complete",
            "dense_encoding_complete",
            "storage_complete",
        ]
        for stage_name in expected_stages:
            assert stage_name in all_stages, f"Expected trace stage '{stage_name}' not found"

        print(f"\n✓ Pipeline completed successfully with real components")
        print(f"  - Document ID: {result.doc_id}")
        print(f"  - Chunk count: {result.chunk_count}")
        print(f"  - Chunk stage: {result.stages.get('chunking', {})}")
        print(f"  - Image count: {result.image_count}")
        print(f"  - Vector IDs: {len(result.vector_ids)}")
        print(f"  - Trace stages: {list(all_stages.keys())}")

        # Cleanup
        pipeline.close()

    def test_pipeline_skip_already_processed(self, real_settings, complex_pdf_path, temp_dirs):
        """Test pipeline skips already processed files.

        This test verifies that when force=False, the pipeline correctly
        skips files that have already been processed (based on SHA256 hash).
        """
        # First run to process the file
        pipeline = IngestionPipeline(
            real_settings,
            collection="test_skip",
            force=True
        )

        result1 = pipeline.run(str(complex_pdf_path))
        assert result1.success is True
        assert result1.stages.get("integrity", {}).get("skipped") is False

        pipeline.close()

        # Second run should skip (force=False)
        pipeline2 = IngestionPipeline(
            real_settings,
            collection="test_skip",
            force=False
        )

        result2 = pipeline2.run(str(complex_pdf_path))

        # Should skip due to already processed
        assert result2.success is True
        assert result2.stages.get("integrity", {}).get("skipped") is True

        pipeline2.close()

    def test_pipeline_with_images(self, project_root, complex_pdf_path, temp_dirs):
        """Test pipeline with a PDF that contains images."""
        # Register splitter
        from libs.splitter.splitter_factory import SplitterFactory
        from libs.splitter.recursive_splitter import RecursiveSplitter
        SplitterFactory.register("recursive", RecursiveSplitter)

        # Register vector store
        from libs.vector_store.vector_store_factory import VectorStoreFactory
        from libs.vector_store.chroma_store import ChromaStore
        VectorStoreFactory.register("chroma", ChromaStore)

        # Load settings and configure
        settings = load_settings(str(project_root / "config" / "settings.yaml"))
        settings.vector_store.persist_directory = str(temp_dirs / "chroma_images")
        settings.vision_llm.enabled = False

        pipeline = IngestionPipeline(
            settings,
            collection="test_images",
            force=True
        )

        result = pipeline.run(str(complex_pdf_path))

        assert result.success is True, f"Pipeline failed: {result.error}"

        # Verify we got some content
        assert result.stages.get("chunking", {}).get("chunk_count", 0) > 0

        print(f"\n✓ Pipeline with images completed")
        print(f"  - Chunks: {result.chunk_count}")
        print(f"  - Images: {result.image_count}")

        pipeline.close()


class TestIngestionPipelineAttributes:
    """Tests for IngestionPipeline attributes with real settings."""

    def test_pipeline_has_real_settings(self, project_root):
        """Test pipeline initializes with real settings."""
        # Register splitter
        from libs.splitter.splitter_factory import SplitterFactory
        from libs.splitter.recursive_splitter import RecursiveSplitter
        SplitterFactory.register("recursive", RecursiveSplitter)

        # Register vector store
        from libs.vector_store.vector_store_factory import VectorStoreFactory
        from libs.vector_store.chroma_store import ChromaStore
        VectorStoreFactory.register("chroma", ChromaStore)

        settings = load_settings(str(project_root / "config" / "settings.yaml"))
        settings.vision_llm.enabled = False

        pipeline = IngestionPipeline(
            settings,
            collection="test"
        )

        assert pipeline.collection == "test"
        assert pipeline.settings is settings
        assert pipeline.force is False

        pipeline.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
