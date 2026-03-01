"""Integration tests for IngestionPipeline.

These tests verify the pipeline orchestration works correctly.
"""

import os
import sys
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


class TestIngestScriptExecution:
    """Integration tests that run the actual ingest.py script as subprocess.

    These tests allow modifying commands in the test file and get the same
    effect as running in the terminal. Uses real config (no mocks).

    Usage:
        Modify the command parameters below to test different scenarios:
        - CHANGE_PATH: Path to file or directory to ingest
        - CHANGE_COLLECTION: Collection name
        - CHANGE_FORCE: Force re-processing
        - CHANGE_CONFIG: Config file path

    Example:
        # Test with different collection
        python -m pytest tests/integration/test_ingestion_pipeline.py::TestIngestScriptExecution::test_ingest_script_single_pdf -v -s

        # Test with verbose output
        python -m pytest tests/integration/test_ingestion_pipeline.py::TestIngestScriptExecution::test_ingest_script_single_pdf -v -s --capture=no
    """

    # ===== MODIFY THESE PARAMETERS TO TEST DIFFERENT SCENARIOS =====
    # These can be changed to test different ingest scenarios
    CHANGE_PATH = "tests/fixtures/sample_documents/complex_technical_doc.pdf"
    CHANGE_COLLECTION = "test_cli_execution"
    CHANGE_FORCE = True  # Set to True to force re-processing
    CHANGE_CONFIG = "config/settings.yaml"
    # =============================================================

    @pytest.fixture
    def project_root(self) -> Path:
        """Return project root."""
        return Path(__file__).parent.parent.parent

    @pytest.fixture
    def test_pdf_path(self, project_root: Path) -> Path:
        """Return path to test PDF."""
        pdf_path = project_root / self.CHANGE_PATH
        assert pdf_path.exists(), f"Test PDF not found: {pdf_path}"
        return pdf_path

    @pytest.fixture
    def config_path(self, project_root: Path) -> Path:
        """Return path to config file."""
        config = project_root / self.CHANGE_CONFIG
        assert config.exists(), f"Config file not found: {config}"
        return config

    def test_ingest_script_single_pdf(self, project_root, test_pdf_path, config_path):
        """Test running ingest.py as subprocess for a single PDF file.

        This test executes the exact same command as would be run in terminal:
            python scripts/ingest.py --path <path> --collection <collection> --force --config <config>

        Args:
            Modify class parameters CHANGE_PATH, CHANGE_COLLECTION, CHANGE_FORCE, CHANGE_CONFIG
            to test different scenarios.
        """
        import subprocess

        # Build the command - same as terminal
        cmd = [
            sys.executable,  # Use current Python interpreter
            str(project_root / "scripts" / "ingest.py"),
            "--path", str(test_pdf_path),
            "--collection", self.CHANGE_COLLECTION,
            "--config", str(config_path),
        ]

        if self.CHANGE_FORCE:
            cmd.append("--force")

        print(f"\n[TEST] Executing command:")
        print(f"  $ {' '.join(cmd)}")

        # Run the command - same effect as terminal
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(project_root),
        )

        # Print output (same as terminal)
        print("\n[STDOUT]")
        print(result.stdout)
        if result.stderr:
            print("\n[STDERR]")
            print(result.stderr)

        print(f"\n[EXIT CODE] {result.returncode}")

        # Verify results
        # Exit code 0 = success
        assert result.returncode == 0, (
            f"Ingest script failed with exit code {result.returncode}\n"
            f"STDOUT: {result.stdout}\n"
            f"STDERR: {result.stderr}"
        )

        # Verify success message in output
        assert "[OK]" in result.stdout or "Successful:" in result.stdout, (
            "Expected success message not found in output"
        )

        # Verify chunks were generated
        assert "chunk" in result.stdout.lower(), (
            "Expected chunk count not found in output"
        )

        print("\n[PASS] Ingest script executed successfully!")

    def test_ingest_script_dry_run(self, project_root, test_pdf_path, config_path):
        """Test dry-run mode - lists files without processing."""
        import subprocess

        cmd = [
            sys.executable,
            str(project_root / "scripts" / "ingest.py"),
            "--path", str(test_pdf_path),
            "--collection", self.CHANGE_COLLECTION,
            "--config", str(config_path),
            "--dry-run",
        ]

        print(f"\n[TEST] Executing dry-run command:")
        print(f"  $ {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(project_root),
        )

        print("\n[STDOUT]")
        print(result.stdout)

        # Dry-run should always succeed
        assert result.returncode == 0, f"Dry-run failed: {result.stderr}"
        assert "Dry run mode" in result.stdout or "Found" in result.stdout

        print("\n[PASS] Dry-run executed successfully!")

    def test_ingest_script_verbose_mode(self, project_root, test_pdf_path, config_path):
        """Test verbose mode - shows detailed trace information."""
        import subprocess

        cmd = [
            sys.executable,
            str(project_root / "scripts" / "ingest.py"),
            "--path", str(test_pdf_path),
            "--collection", f"{self.CHANGE_COLLECTION}_verbose",
            "--config", str(config_path),
            "--force",
            "--verbose",
        ]

        print(f"\n[TEST] Executing verbose command:")
        print(f"  $ {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(project_root),
        )

        print("\n[STDOUT]")
        print(result.stdout)

        assert result.returncode == 0, f"Verbose ingest failed: {result.stderr}"

        # Verbose should show trace stages
        assert "Trace stages:" in result.stdout or "chunking" in result.stdout.lower()

        print("\n[PASS] Verbose mode executed successfully!")

    def test_ingest_script_with_custom_collection(self, project_root, test_pdf_path, config_path):
        """Test ingesting with a specific collection name."""
        import subprocess

        custom_collection = "test_collection"
        cmd = [
            sys.executable,
            str(project_root / "scripts" / "ingest.py"),
            "--path", str(test_pdf_path),
            "--collection", custom_collection,
            "--config", str(config_path),
            "--force",
        ]

        print(f"\n[TEST] Executing with custom collection:")
        print(f"  $ {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(project_root),
        )

        print("\n[STDOUT]")
        print(result.stdout)

        assert result.returncode == 0, f"Custom collection ingest failed: {result.stderr}"

        # Verify collection name appears in output
        assert custom_collection in result.stdout or "Collection:" in result.stdout

        print("\n[PASS] Custom collection test executed successfully!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
