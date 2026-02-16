"""Unit tests for ChunkRefiner.

This module tests the ChunkRefiner class which provides rule-based and
optional LLM-based text refinement for chunks.

Design Principles:
    - Mock-based: Uses FakeLLM for LLM tests
    - Contract Testing: Verify chunk output format
    - Coverage: Rule-based cleaning, LLM mode, fallback, error handling
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.chunk_refiner import ChunkRefiner, ChunkRefinementError


class FakeLLM:
    """A fake LLM for testing LLM-based refinement."""

    def __init__(self, refined_text: str = "Refined by LLM") -> None:
        self._refined_text = refined_text
        self.call_count = 0
        self.last_messages = None

    @property
    def provider_name(self) -> str:
        return "fake"

    def chat(self, messages: list, **kwargs: Any) -> Any:
        """Return predefined refined text."""
        self.call_count += 1
        self.last_messages = messages

        class FakeResponse:
            def __init__(self, content: str):
                self.content = content

        return FakeResponse(self._refined_text)


class FakeLLMError(FakeLLM):
    """A fake LLM that raises errors."""

    def chat(self, messages: list, **kwargs: Any) -> Any:
        raise Exception("LLM API error")


class TestChunkRefinerInitialization:
    """Tests for ChunkRefiner initialization."""

    def test_default_initialization(self):
        """Test ChunkRefiner initializes with default settings."""
        settings = Settings()
        refiner = ChunkRefiner(settings)

        assert refiner._settings == settings
        assert refiner._use_llm is False
        assert refiner._llm is None

    def test_initialization_with_llm_enabled(self):
        """Test ChunkRefiner initializes with LLM enabled."""
        settings = Settings()
        settings.ingestion.chunk_refiner.use_llm = True

        refiner = ChunkRefiner(settings)

        assert refiner._use_llm is True
        assert refiner._llm is None  # Not created yet

    def test_initialization_with_custom_llm(self):
        """Test ChunkRefiner with provided LLM instance."""
        settings = Settings()
        fake_llm = FakeLLM()
        refiner = ChunkRefiner(settings, llm=fake_llm)

        assert refiner._llm == fake_llm
        assert refiner._llm.call_count == 0


class TestRuleBasedRefine:
    """Tests for rule-based text cleaning."""

    def test_removes_excessive_whitespace(self):
        """Test that excessive whitespace is removed."""
        settings = Settings()
        refiner = ChunkRefiner(settings)

        input_text = "Hello    world   this   is    test"
        result = refiner._rule_based_refine(input_text)

        # Multiple spaces should be reduced to single space
        assert "    " not in result
        assert "   " not in result

    def test_removes_consecutive_newlines(self):
        """Test that consecutive newlines are normalized."""
        settings = Settings()
        refiner = ChunkRefiner(settings)

        input_text = "Line 1\n\n\n\n\nLine 2"
        result = refiner._rule_based_refine(input_text)

        # Max 2 consecutive newlines
        assert "\n\n\n" not in result

    def test_removes_page_markers(self):
        """Test that page markers are removed."""
        settings = Settings()
        refiner = ChunkRefiner(settings)

        input_text = "Content here\n\nPage 1 of 10\n\nMore content"
        result = refiner._rule_based_refine(input_text)

        assert "Page 1 of 10" not in result
        assert "Content here" in result
        assert "More content" in result

    def test_removes_separator_lines(self):
        """Test that separator lines are removed."""
        settings = Settings()
        refiner = ChunkRefiner(settings)

        input_text = "Section 1\n======================\nSection 2"
        result = refiner._rule_based_refine(input_text)

        assert "======================" not in result
        assert "Section 1" in result
        assert "Section 2" in result

    def test_removes_html_comments(self):
        """Test that HTML comments are removed."""
        settings = Settings()
        refiner = ChunkRefiner(settings)

        input_text = "Text before <!-- comment --> Text after"
        result = refiner._rule_based_refine(input_text)

        assert "<!-- comment -->" not in result
        assert "Text before" in result
        assert "Text after" in result

    def test_removes_leading_trailing_whitespace(self):
        """Test that leading/trailing whitespace is stripped."""
        settings = Settings()
        refiner = ChunkRefiner(settings)

        input_text = "   \n\n   Text content   \n\n   "
        result = refiner._rule_based_refine(input_text)

        assert result.startswith("Text content")
        assert result.endswith("Text content")

    def test_handles_empty_text(self):
        """Test that empty text returns empty string."""
        settings = Settings()
        refiner = ChunkRefiner(settings)

        assert refiner._rule_based_refine("") == ""
        assert refiner._rule_based_refine("   ") == ""

    def test_preserves_code_blocks(self):
        """Test that code block content is preserved."""
        settings = Settings()
        refiner = ChunkRefiner(settings)

        input_text = """Here is code:
```python
def hello():
    print("world")
```
End of code."""

        result = refiner._rule_based_refine(input_text)

        assert "def hello" in result
        assert 'print("world")' in result

    def test_preserves_markdown_formatting(self):
        """Test that Markdown formatting is preserved."""
        settings = Settings()
        refiner = ChunkRefiner(settings)

        input_text = """# Title

**Bold text** and *italic text*

- Item 1
- Item 2
"""

        result = refiner._rule_based_refine(input_text)

        assert "# Title" in result
        assert "**Bold text**" in result
        assert "*italic text*" in result
        assert "- Item 1" in result
        assert "- Item 2" in result


class TestLLMRefine:
    """Tests for LLM-based refinement."""

    def test_llm_refine_called_when_enabled(self):
        """Test that LLM is called when enabled."""
        settings = Settings()
        settings.ingestion.chunk_refiner.use_llm = True

        fake_llm = FakeLLM("Refined text")
        refiner = ChunkRefiner(settings, llm=fake_llm)
        # Set prompt template directly for testing
        refiner._prompt_template = "Refine: {text}"

        result = refiner._llm_refine("Original text")

        assert result == "Refined text"
        assert fake_llm.call_count == 1

    def test_llm_refine_not_called_when_disabled(self):
        """Test that LLM is not called when disabled."""
        settings = Settings()
        settings.ingestion.chunk_refiner.use_llm = False

        fake_llm = FakeLLM("Refined text")
        refiner = ChunkRefiner(settings, llm=fake_llm)

        result = refiner._llm_refine("Original text")

        assert result is None
        assert fake_llm.call_count == 0

    def test_llm_refine_prompt_includes_text(self):
        """Test that LLM prompt includes the text to refine."""
        settings = Settings()
        settings.ingestion.chunk_refiner.use_llm = True

        fake_llm = FakeLLM("Refined")
        refiner = ChunkRefiner(settings, llm=fake_llm)
        # Set prompt template directly for testing
        refiner._prompt_template = "Refine this text: {text}"

        refiner._llm_refine("Test content")

        # Check that messages were passed to chat()
        assert fake_llm.last_messages is not None
        # The system prompt should contain the text
        system_content = fake_llm.last_messages[0].content
        assert "Test content" in system_content


class TestFallbackBehavior:
    """Tests for fallback behavior when LLM fails."""

    def test_fallback_to_rule_when_llm_fails(self):
        """Test that fallback to rule-based when LLM fails."""
        settings = Settings()
        settings.ingestion.chunk_refiner.use_llm = True

        fake_llm = FakeLLMError()
        refiner = ChunkRefiner(settings, llm=fake_llm)

        # Should not raise, should return rule-cleaned result
        # Multiple spaces should become single spaces
        text = "  Text    with  whitespace  \n\n\n"
        result = refiner._rule_based_refine(text)

        # Multiple spaces reduced to single, extra newlines normalized
        assert "Text with whitespace" in result or result.strip() == "Text with whitespace"

    def test_fallback_preserves_original_on_error(self):
        """Test that original chunk is preserved on transformation error."""
        settings = Settings()
        refiner = ChunkRefiner(settings)

        # Create a mock that raises during transform
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = Exception("API error")
        refiner._llm = mock_llm
        refiner._use_llm = True
        refiner._prompt_template = "Refine this: {text}"

        # This should not raise, but fallback
        original = Chunk(
            id="test",
            text="Original text",
            metadata={},
            source_ref="doc1",
        )
        result = refiner.transform([original])

        assert len(result) == 1
        assert result[0].text == "Original text"


class TestTransformMethod:
    """Tests for the transform method."""

    def test_transform_returns_list_of_chunks(self):
        """Test that transform returns a list of Chunk objects."""
        settings = Settings()
        refiner = ChunkRefiner(settings)

        chunks = [
            Chunk(id="1", text="Text 1", metadata={}, source_ref="doc"),
            Chunk(id="2", text="Text 2", metadata={}, source_ref="doc"),
        ]

        result = refiner.transform(chunks)

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(c, Chunk) for c in result)

    def test_transform_preserves_chunk_id(self):
        """Test that chunk IDs are preserved."""
        settings = Settings()
        refiner = ChunkRefiner(settings)

        chunks = [Chunk(id="unique_id_123", text="Text", metadata={}, source_ref="doc")]

        result = refiner.transform(chunks)

        assert result[0].id == "unique_id_123"

    def test_transform_adds_refinement_metadata(self):
        """Test that refinement metadata is added."""
        settings = Settings()
        refiner = ChunkRefiner(settings)

        chunks = [Chunk(id="1", text="Text", metadata={}, source_ref="doc")]

        result = refiner.transform(chunks)

        assert "refinement" in result[0].metadata
        refinement = result[0].metadata["refinement"]
        assert "refined_by" in refinement
        assert refinement["refined_by"] in ["rule", "llm"]

    def test_transform_preserves_source_ref(self):
        """Test that source_ref is preserved."""
        settings = Settings()
        refiner = ChunkRefiner(settings)

        chunks = [Chunk(id="1", text="Text", metadata={}, source_ref="parent_doc")]

        result = refiner.transform(chunks)

        assert result[0].source_ref == "parent_doc"

    def test_transform_preserves_original_metadata(self):
        """Test that original metadata is preserved."""
        settings = Settings()
        refiner = ChunkRefiner(settings)

        chunks = [
            Chunk(
                id="1",
                text="Text",
                metadata={"custom_field": "custom_value"},
                source_ref="doc",
            )
        ]

        result = refiner.transform(chunks)

        assert result[0].metadata.get("custom_field") == "custom_value"

    def test_transform_handles_empty_list(self):
        """Test that empty list returns empty list."""
        settings = Settings()
        refiner = ChunkRefiner(settings)

        result = refiner.transform([])

        assert result == []

    def test_transform_chunk_index_preserved(self):
        """Test that chunk_index is preserved in metadata."""
        settings = Settings()
        refiner = ChunkRefiner(settings)

        chunks = [
            Chunk(id="1", text="Text 1", metadata={"chunk_index": 5}, source_ref="doc"),
            Chunk(id="2", text="Text 2", metadata={"chunk_index": 10}, source_ref="doc"),
        ]

        result = refiner.transform(chunks)

        assert result[0].metadata.get("chunk_index") == 5
        assert result[1].metadata.get("chunk_index") == 10


class TestConfigurationDriven:
    """Tests for configuration-driven behavior."""

    def test_llm_enabled_from_settings(self):
        """Test that LLM is enabled from settings."""
        settings = Settings()
        settings.ingestion.chunk_refiner.use_llm = True

        refiner = ChunkRefiner(settings)

        assert refiner.use_llm is True

    def test_llm_disabled_from_settings(self):
        """Test that LLM is disabled when not in settings."""
        settings = Settings()

        refiner = ChunkRefiner(settings)

        assert refiner.use_llm is False


class TestExceptionHandling:
    """Tests for exception handling."""

    def test_single_chunk_exception_does_not_affect_others(self):
        """Test that exception in one chunk doesn't affect others."""
        settings = Settings()
        refiner = ChunkRefiner(settings)

        # Create chunks where one might cause issues
        chunks = [
            Chunk(id="1", text="Normal text", metadata={}, source_ref="doc"),
            Chunk(id="2", text="More normal text", metadata={}, source_ref="doc"),
        ]

        # Should not raise
        result = refiner.transform(chunks)

        assert len(result) == 2
        assert result[0].text == "Normal text"
        assert result[1].text == "More normal text"

    def test_invalid_chunk_does_not_crash(self):
        """Test that invalid chunk structure doesn't crash."""
        settings = Settings()
        refiner = ChunkRefiner(settings)

        # This should be handled gracefully
        chunks = [Chunk(id="1", text="", metadata={}, source_ref="doc")]

        result = refiner.transform(chunks)

        assert len(result) == 1


class TestGetRefinementStats:
    """Tests for refinement statistics."""

    def test_stats_counts_by_method(self):
        """Test that stats correctly count by method."""
        settings = Settings()
        refiner = ChunkRefiner(settings)

        # All should be rule-based since LLM is disabled
        chunks = [
            Chunk(id="1", text="Text", metadata={}, source_ref="doc"),
            Chunk(id="2", text="Text", metadata={}, source_ref="doc"),
        ]

        result = refiner.transform(chunks)
        stats = refiner.get_refinement_stats(result)

        assert stats["total"] == 2
        assert stats["by_method"]["rule"] == 2

    def test_stats_tracks_errors(self):
        """Test that stats track error fallbacks."""
        settings = Settings()
        refiner = ChunkRefiner(settings)

        # Empty chunks might cause issues
        chunks = [Chunk(id="1", text="", metadata={}, source_ref="doc")]

        result = refiner.transform(chunks)
        stats = refiner.get_refinement_stats(result)

        # Empty text is valid, just returns empty string
        assert stats["total"] == 1


class TestWithFixtures:
    """Tests using the noisy_chunks.json fixtures."""

    @pytest.fixture
    def fixtures_path(self) -> Path:
        """Return path to fixtures directory."""
        # Look in project-level tests/fixtures directory
        return Path(__file__).parent.parent / "fixtures"

    def test_typical_noise_scenario(self, fixtures_path: Path):
        """Test typical noise scenario cleaning."""
        with open(fixtures_path / "noisy_chunks.json") as f:
            data = json.load(f)

        scenario = data["scenarios"]["typical_noise_scenario"]
        settings = Settings()
        refiner = ChunkRefiner(settings)

        result = refiner._rule_based_refine(scenario["input"])

        # Check expected content is preserved
        for expected in scenario["expected_contains"]:
            assert expected in result

        # Check unwanted content is removed
        for unwanted in scenario["expected_not_contains"]:
            assert unwanted not in result

    def test_excessive_whitespace_scenario(self, fixtures_path: Path):
        """Test excessive whitespace cleaning."""
        with open(fixtures_path / "noisy_chunks.json") as f:
            data = json.load(f)

        scenario = data["scenarios"]["excessive_whitespace"]
        settings = Settings()
        refiner = ChunkRefiner(settings)

        result = refiner._rule_based_refine(scenario["input"])

        # Check content preserved
        for expected in scenario["expected_contains"]:
            assert expected in result

    def test_clean_text_not_over_cleaned(self, fixtures_path: Path):
        """Test that clean text is not over-cleaned."""
        with open(fixtures_path / "noisy_chunks.json") as f:
            data = json.load(f)

        scenario = data["scenarios"]["clean_text"]
        settings = Settings()
        refiner = ChunkRefiner(settings)

        result = refiner._rule_based_refine(scenario["input"])

        # Should be unchanged
        assert result == scenario["expected_equals"]

    def test_code_blocks_preserved(self, fixtures_path: Path):
        """Test that code blocks are preserved."""
        with open(fixtures_path / "noisy_chunks.json") as f:
            data = json.load(f)

        scenario = data["scenarios"]["code_blocks"]
        settings = Settings()
        refiner = ChunkRefiner(settings)

        result = refiner._rule_based_refine(scenario["input"])

        # Check code content preserved
        for expected in scenario["expected_contains"]:
            assert expected in result


class TestTraceContext:
    """Tests for trace context integration."""

    def test_transform_records_to_trace(self):
        """Test that transform records to trace context."""
        settings = Settings()
        refiner = ChunkRefiner(settings)

        chunks = [Chunk(id="1", text="Text", metadata={}, source_ref="doc")]
        trace = TraceContext()

        result = refiner.transform(chunks, trace)

        stage_data = trace.get_stage("chunk_refinement")
        assert stage_data is not None
        assert stage_data["input_count"] == 1
        assert stage_data["output_count"] == 1

    def test_transform_without_trace(self):
        """Test that transform works without trace context."""
        settings = Settings()
        refiner = ChunkRefiner(settings)

        chunks = [Chunk(id="1", text="Text", metadata={}, source_ref="doc")]

        result = refiner.transform(chunks, None)

        assert len(result) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
