"""Unit tests for MetadataEnricher.

This module tests the MetadataEnricher class with mock LLM responses.
It covers rule-based extraction, LLM enhancement, and fallback scenarios.

Design Principles:
    - Mock-based: Uses unittest.mock for LLM client
    - Contract Testing: Verify interface compliance
    - Coverage: Rule extraction, LLM enhancement, fallback, error handling
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.metadata_enricher import MetadataEnricher


class MockLLM:
    """A mock LLM for testing."""

    def __init__(self, response_content: str = "") -> None:
        self._response_content = response_content
        self.call_count = 0
        self.last_messages = None

    @property
    def provider_name(self) -> str:
        return "mock-llm"

    def chat(self, messages: Any, trace: Any = None, **kwargs: Any) -> Any:
        self.call_count += 1
        self.last_messages = messages

        class MockResponse:
            def __init__(self, content: str) -> None:
                self.content = content

        return MockResponse(self._response_content)


class TestMetadataEnricher:
    """Tests for MetadataEnricher class."""

    def test_initialization_without_llm(self):
        """Test initialization without LLM (rule-only mode)."""
        settings = Settings()
        settings.ingestion = MagicMock()
        settings.ingestion.metadata_enricher = MagicMock()
        settings.ingestion.metadata_enricher.use_llm = False

        enricher = MetadataEnricher(settings)

        assert enricher.use_llm is False

    def test_initialization_with_llm_enabled(self):
        """Test initialization with LLM enabled."""
        settings = Settings()
        settings.ingestion = MagicMock()
        settings.ingestion.metadata_enricher = MagicMock()
        settings.ingestion.metadata_enricher.use_llm = True

        mock_llm = MockLLM()
        enricher = MetadataEnricher(settings, llm=mock_llm)

        assert enricher.use_llm is True

    def test_initialization_missing_metadata_enricher_config(self):
        """Test initialization when metadata_enricher config doesn't exist."""
        settings = Settings()
        # No ingestion.metadata_enricher attribute

        enricher = MetadataEnricher(settings)

        assert enricher.use_llm is False


class TestMetadataEnricherRuleBased:
    """Tests for rule-based metadata extraction."""

    def test_rule_extract_title_from_heading(self):
        """Test extracting title from markdown heading."""
        settings = Settings()
        enricher = MetadataEnricher(settings)

        text = """# Introduction to Python

Python is a popular programming language.
It is easy to learn and use.
"""

        metadata = enricher._rule_based_extract(text)

        assert metadata["title"] == "Introduction to Python"
        assert len(metadata["summary"]) > 0
        assert "language:python" in metadata["tags"]

    def test_rule_extract_title_from_first_line(self):
        """Test extracting title from first non-heading line."""
        settings = Settings()
        enricher = MetadataEnricher(settings)

        text = """Getting Started Guide

This is the beginning of the document.
More content here.
"""

        metadata = enricher._rule_based_extract(text)

        assert metadata["title"] == "Getting Started Guide"

    def test_rule_extract_summary(self):
        """Test extracting summary from text."""
        settings = Settings()
        enricher = MetadataEnricher(settings)

        text = """# Document Title

This is the first sentence of the summary. This is the second sentence.
And this is the third sentence that might extend beyond our limit.
"""

        metadata = enricher._rule_based_extract(text)

        # Summary should contain first sentence(s)
        assert "first sentence" in metadata["summary"].lower()

    def test_rule_extract_tags_programming_language(self):
        """Test extracting programming language tags."""
        settings = Settings()
        enricher = MetadataEnricher(settings)

        text = """# Python Tutorial

This guide covers Python programming with Django and FastAPI.
"""

        metadata = enricher._rule_based_extract(text)

        tags = metadata["tags"]
        assert any("language:python" in tag for tag in tags)

    def test_rule_extract_tags_framework(self):
        """Test extracting framework tags."""
        settings = Settings()
        enricher = MetadataEnricher(settings)

        text = """# Building REST APIs

Using FastAPI with Python to create REST APIs.
Authentication via OAuth and JWT.
"""

        metadata = enricher._rule_based_extract(text)

        tags = metadata["tags"]
        assert any("framework:fastapi" in tag for tag in tags)

    def test_rule_extract_empty_text(self):
        """Test handling empty text."""
        settings = Settings()
        enricher = MetadataEnricher(settings)

        metadata = enricher._rule_based_extract("")

        assert metadata["title"] == ""
        assert metadata["summary"] == ""
        assert metadata["tags"] == []

    def test_rule_extract_with_concepts(self):
        """Test extracting concept tags."""
        settings = Settings()
        enricher = MetadataEnricher(settings)

        text = """# API Design Best Practices

This document covers REST API design with proper authentication using OAuth and JWT tokens.
"""

        metadata = enricher._rule_based_extract(text)

        tags = metadata["tags"]
        assert any("concept:api" in tag for tag in tags) or any("concept:rest" in tag for tag in tags)


class TestMetadataEnricherTransform:
    """Tests for transform method."""

    def test_transform_empty_list(self):
        """Test transforming empty chunk list."""
        settings = Settings()
        enricher = MetadataEnricher(settings)

        result = enricher.transform([])

        assert result == []

    def test_transform_single_chunk(self):
        """Test enriching a single chunk."""
        settings = Settings()
        enricher = MetadataEnricher(settings)

        chunk = Chunk(
            id="test_0001_abc12345",
            text="# Getting Started\n\nWelcome to this guide.",
            metadata={"source_path": "/test/doc.txt"},
            source_ref="doc_001",
        )

        result = enricher.transform([chunk])

        assert len(result) == 1
        enriched = result[0]
        assert enriched.id == chunk.id
        assert enriched.text == chunk.text
        assert "enrichment" in enriched.metadata
        assert enriched.metadata["enrichment"]["enriched_by"] == "rule"
        assert enriched.metadata["title"] == "Getting Started"
        assert enriched.metadata["summary"] != ""

    def test_transform_preserves_original_metadata(self):
        """Test that original metadata is preserved."""
        settings = Settings()
        enricher = MetadataEnricher(settings)

        original_metadata = {"source_path": "/test/doc.txt", "custom_field": "value"}
        chunk = Chunk(
            id="test_0001_abc12345",
            text="# Test Document\n\nSome content here.",
            metadata=original_metadata,
            source_ref="doc_001",
        )

        result = enricher.transform([chunk])

        assert result[0].metadata["source_path"] == "/test/doc.txt"
        assert result[0].metadata["custom_field"] == "value"

    def test_transform_multiple_chunks(self):
        """Test enriching multiple chunks."""
        settings = Settings()
        enricher = MetadataEnricher(settings)

        chunks = [
            Chunk(
                id=f"test_{i:04d}_abc12345",
                text=f"# Section {i}\n\nContent for section {i}.",
                metadata={"chunk_index": i},
                source_ref="doc_001",
            )
            for i in range(3)
        ]

        result = enricher.transform(chunks)

        assert len(result) == 3
        for i, enriched in enumerate(result):
            assert enriched.metadata["chunk_index"] == i
            assert enriched.metadata["enrichment"]["enriched_by"] == "rule"


class TestMetadataEnricherLLM:
    """Tests for LLM-based metadata extraction."""

    def test_llm_extract_success(self):
        """Test successful LLM metadata extraction."""
        settings = Settings()
        settings.ingestion = MagicMock()
        settings.ingestion.metadata_enricher = MagicMock()
        settings.ingestion.metadata_enricher.use_llm = True

        llm_response = json.dumps({
            "title": "LLM Generated Title",
            "summary": "This is a summary generated by LLM.",
            "tags": ["llm", "generated", "test"]
        })
        mock_llm = MockLLM(llm_response)
        enricher = MetadataEnricher(settings, llm=mock_llm)

        chunk = Chunk(
            id="test_0001_abc12345",
            text="Some text to analyze.",
            metadata={},
            source_ref="doc_001",
        )

        result = enricher.transform([chunk])

        assert len(result) == 1
        enriched = result[0]
        assert enriched.metadata["title"] == "LLM Generated Title"
        assert enriched.metadata["summary"] == "This is a summary generated by LLM."
        assert "llm" in [t.lower() for t in enriched.metadata["tags"]]
        assert enriched.metadata["enrichment"]["enriched_by"] == "llm"

    def test_llm_extract_fallback_to_rule(self):
        """Test fallback to rule-based extraction when LLM fails."""
        settings = Settings()
        settings.ingestion = MagicMock()
        settings.ingestion.metadata_enricher = MagicMock()
        settings.ingestion.metadata_enricher.use_llm = True

        # LLM that raises an exception
        mock_llm = MagicMock()
        mock_llm.provider_name = "mock"
        mock_llm.chat.side_effect = Exception("LLM error")

        enricher = MetadataEnricher(settings, llm=mock_llm)

        chunk = Chunk(
            id="test_0001_abc12345",
            text="# Fallback Test\n\nThis should use rule-based extraction.",
            metadata={},
            source_ref="doc_001",
        )

        result = enricher.transform([chunk])

        assert len(result) == 1
        enriched = result[0]
        # Should have rule-based title
        assert "Fallback Test" in enriched.metadata.get("title", "")
        # Should be marked as fallback
        assert enriched.metadata["enrichment"]["fallback"] is True

    def test_llm_extract_with_trace(self):
        """Test LLM extraction with trace context."""
        settings = Settings()
        settings.ingestion = MagicMock()
        settings.ingestion.metadata_enricher = MagicMock()
        settings.ingestion.metadata_enricher.use_llm = True

        llm_response = json.dumps({
            "title": "Traced Title",
            "summary": "Summary with trace.",
            "tags": ["trace", "test"]
        })
        mock_llm = MockLLM(llm_response)
        enricher = MetadataEnricher(settings, llm=mock_llm)

        trace = TraceContext()
        chunk = Chunk(
            id="test_0001_abc12345",
            text="Text for tracing.",
            metadata={},
            source_ref="doc_001",
        )

        result = enricher.transform([chunk], trace=trace)

        assert len(result) == 1
        # Verify trace was recorded
        assert len(trace._stages) > 0


class TestMetadataEnricherStats:
    """Tests for enrichment statistics."""

    def test_get_enrichment_stats(self):
        """Test getting enrichment statistics."""
        settings = Settings()
        enricher = MetadataEnricher(settings)

        chunks = [
            Chunk(
                id=f"test_{i:04d}_abc12345",
                text=f"# Section {i}\n\nContent {i}.",
                metadata={},
                source_ref="doc_001",
            )
            for i in range(5)
        ]

        result = enricher.transform(chunks)
        stats = enricher.get_enrichment_stats(result)

        assert stats["total"] == 5
        assert stats["by_method"]["rule"] == 5
        assert stats["with_title"] == 5
        assert stats["with_summary"] == 5


class TestMetadataEnricherEdgeCases:
    """Tests for edge cases."""

    def test_transform_with_none_text(self):
        """Test handling chunks with None text (should not happen but be safe)."""
        settings = Settings()
        enricher = MetadataEnricher(settings)

        chunk = Chunk(
            id="test_0001_abc12345",
            text="",
            metadata={},
            source_ref="doc_001",
        )

        result = enricher.transform([chunk])

        assert len(result) == 1
        # Should still have enrichment metadata
        assert "enrichment" in result[0].metadata

    def test_transform_preserves_chunk_index(self):
        """Test that chunk_index is preserved from original metadata."""
        settings = Settings()
        enricher = MetadataEnricher(settings)

        chunk = Chunk(
            id="test_0001_abc12345",
            text="# Test\n\nContent.",
            metadata={"chunk_index": 5},
            source_ref="doc_001",
        )

        result = enricher.transform([chunk])

        assert result[0].metadata["chunk_index"] == 5

    def test_merge_metadata_llm_precedence(self):
        """Test that LLM metadata takes precedence over rule metadata."""
        settings = Settings()
        settings.ingestion = MagicMock()
        settings.ingestion.metadata_enricher = MagicMock()
        settings.ingestion.metadata_enricher.use_llm = True

        llm_response = json.dumps({
            "title": "LLM Title",
            "summary": "LLM Summary",
            "tags": ["llm-tag"]
        })
        mock_llm = MockLLM(llm_response)
        enricher = MetadataEnricher(settings, llm=mock_llm)

        rule_metadata = {"title": "Rule Title", "summary": "Rule Summary", "tags": ["rule-tag"]}
        llm_metadata = {"title": "LLM Title", "summary": "LLM Summary", "tags": ["llm-tag"]}

        merged = enricher._merge_metadata(rule_metadata, llm_metadata)

        # LLM values should take precedence
        assert merged["title"] == "LLM Title"
        assert merged["summary"] == "LLM Summary"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
