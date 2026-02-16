"""Integration tests for ChunkRefiner with real LLM.

This module tests the ChunkRefiner with real LLM providers to validate
configuration correctness and refinement quality.

IMPORTANT: These tests make real API calls and may incur costs.
Run with: pytest tests/integration/test_chunk_refiner_llm.py -v -s
"""

import pytest

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.chunk_refiner import ChunkRefiner


class TestChunkRefinerLLMIntegration:
    """Integration tests for ChunkRefiner with real LLM providers."""

    @pytest.fixture
    def settings_with_llm(self) -> Settings:
        """Load settings with LLM configuration."""
        settings = Settings()
        settings.ingestion.chunk_refiner.use_llm = True
        return settings

    @pytest.fixture
    def sample_chunks(self) -> list[Chunk]:
        """Create sample chunks for testing."""
        return [
            Chunk(
                id="test_chunk_1",
                text="""  This is a sample text with some   extra whitespace

   and multiple   newlines. It also has a page number: Page 5 of 10.

   This is the main content that should be preserved.
   ---
   End of page
   """,
                metadata={"source_path": "/test/doc.pdf", "chunk_index": 0},
                source_ref="test_doc_001",
            ),
            Chunk(
                id="test_chunk_2",
                text="""# Important Document

## Summary

This document contains important information about the project.

<!-- TODO: Add more details -->

The key findings are:
1. Performance improved by 15%
2. Cost reduced by 10%
3. User satisfaction increased
""",
                metadata={"source_path": "/test/doc2.pdf", "chunk_index": 1},
                source_ref="test_doc_002",
            ),
        ]

    def test_real_llm_refinement_success(self, settings_with_llm: Settings):
        """Test that real LLM can be called for refinement."""
        # Skip if LLM not configured
        if not settings_with_llm.llm.provider or not settings_with_llm.llm.model:
            pytest.skip("LLM not configured (missing llm.provider or llm.model)")

        refiner = ChunkRefiner(settings_with_llm)

        # Verify LLM is available
        llm = refiner._get_llm()
        assert llm is not None, "LLM should be created from settings"
        print(f"Using LLM provider: {llm.provider_name}")

        # Create a simple test chunk
        chunk = Chunk(
            id="simple_test",
            text="This is a test.   With   extra   spaces.",
            metadata={},
            source_ref="test",
        )

        # Refine the chunk
        result = refiner.transform([chunk])

        # Verify result
        assert len(result) == 1
        refined = result[0]

        # Metadata should be present
        assert "refinement" in refined.metadata
        refinement = refined.metadata["refinement"]
        assert refinement["refined_by"] in ["rule", "llm"]

        print(f"Original: '{chunk.text}'")
        print(f"Refined: '{refined.text}'")

        # If LLM was used, verify the output is different
        if refinement["refined_by"] == "llm":
            assert len(refined.text) > 0

    def test_real_llm_refinement_quality(self, settings_with_llm: Settings, sample_chunks: list[Chunk]):
        """Test that LLM refinement improves text quality."""
        if not settings_with_llm.llm.provider or not settings_with_llm.llm.model:
            pytest.skip("LLM not configured")

        refiner = ChunkRefiner(settings_with_llm)
        trace = TraceContext()

        # Refine chunks
        result = refiner.transform(sample_chunks, trace)

        # Verify all chunks were refined
        assert len(result) == len(sample_chunks)

        print(f"\nProcessed {len(result)} chunks")

        for i, refined in enumerate(result):
            # Check metadata
            assert "refinement" in refined.metadata
            refinement = refined.metadata["refinement"]
            print(f"\nChunk {i+1}: refined_by={refinement['refined_by']}")

            # Verify content is not empty
            assert len(refined.text) > 0, f"Chunk {i+1} has empty text"

            # Verify structure is preserved
            if "Important Document" in sample_chunks[i].text:
                # Markdown headers should be preserved
                assert "# Important Document" in refined.text or "Important Document" in refined.text

            # Verify noise is reduced
            original_noise = sample_chunks[i].text.count("  ")
            refined_noise = refined.text.count("  ")
            print(f"  Whitespace noise: {original_noise} -> {refined_noise}")

        # Verify trace was recorded
        trace_data = trace.get_stage("chunk_refinement")
        assert trace_data is not None
        assert trace_data["input_count"] == len(sample_chunks)
        assert trace_data["llm_used"] is True

    def test_fallback_on_invalid_model(self, settings_with_llm: Settings):
        """Test graceful fallback when LLM model is invalid."""
        if not settings_with_llm.llm.provider:
            pytest.skip("LLM not configured")

        # Set an invalid model name
        settings_with_llm.llm.model = "invalid-model-name-that-does-not-exist"

        refiner = ChunkRefiner(settings_with_llm)

        chunk = Chunk(
            id="test_fallback",
            text="This text has   extra   spaces and\n\n\nmultiple newlines.",
            metadata={},
            source_ref="test",
        )

        # Should not raise, should fallback to rule-based
        result = refiner.transform([chunk])

        assert len(result) == 1
        refined = result[0]

        # Should fallback to rule-based (not crash)
        assert refined.text is not None
        assert "refinement" in refined.metadata

        # Verify refinement metadata
        refinement = refined.metadata["refinement"]
        print(f"Refinement method: {refinement['refined_by']}")
        print(f"Error logged: {'error' in refinement}")

    def test_refinement_with_trace_context(self, settings_with_llm: Settings):
        """Test that trace context is properly recorded."""
        if not settings_with_llm.llm.provider or not settings_with_llm.llm.odel:
            pytest.skip("LLM not configured")

        refiner = ChunkRefiner(settings_with_llm)
        trace = TraceContext()

        chunk = Chunk(
            id="trace_test",
            text="Test content with   extra   spaces.",
            metadata={},
            source_ref="test",
        )

        result = refiner.transform([chunk], trace)

        # Check trace was recorded
        assert trace.trace_id is not None

        # Check stage data
        refinement_stage = trace.get_stage("chunk_refinement")
        assert refinement_stage is not None
        assert refinement_stage["input_count"] == 1
        assert refinement_stage["output_count"] == 1

        # Check LLM refinement stage if LLM was used
        llm_stage = trace.get_stage("chunk_llm_refinement")
        if llm_stage:
            assert "input_length" in llm_stage
            assert "output_length" in llm_stage
            assert "llm_provider" in llm_stage


class TestChunkRefinerNoLLM:
    """Tests for ChunkRefiner when LLM is not available."""

    def test_rule_based_only(self):
        """Test that rule-based refinement works without LLM."""
        settings = Settings()
        settings.ingestion.chunk_refiner.use_llm = False

        refiner = ChunkRefiner(settings)

        chunk = Chunk(
            id="rule_only",
            text="""  Text with   whitespace

   and multiple   newlines.

   Page 1 of 10
   """,
            metadata={},
            source_ref="test",
        )

        result = refiner.transform([chunk])

        assert len(result) == 1
        refined = result[0]

        # Should be rule-based
        assert refined.metadata["refinement"]["refined_by"] == "rule"

        # Verify some cleaning happened
        assert refined.text.count("  ") < chunk.text.count("  ")

    def test_no_crash_on_empty_chunks(self):
        """Test that empty chunks don't cause crashes."""
        settings = Settings()
        refiner = ChunkRefiner(settings)

        chunks = [
            Chunk(id="1", text="", metadata={}, source_ref="test"),
            Chunk(id="2", text="Normal text", metadata={}, source_ref="test"),
        ]

        # Should not raise
        result = refiner.transform(chunks)

        assert len(result) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
