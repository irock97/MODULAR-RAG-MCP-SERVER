"""Unit tests for Response Builder and Citation Generator.

These tests verify the citation generation and response building functionality.
"""

import pytest

from core.response import (
    Citation,
    CitationConfig,
    CitationGenerator,
    ResponseBuilder,
)
from core.types import RetrievalResult


class TestCitationGenerator:
    """Tests for CitationGenerator."""

    def test_citation_generator_creation(self):
        """Test creating a CitationGenerator instance."""
        generator = CitationGenerator()
        assert generator is not None
        assert generator.config.snippet_max_length == 200

    def test_citation_generator_with_config(self):
        """Test creating a CitationGenerator with custom config."""
        generator = CitationGenerator(snippet_max_length=100)
        assert generator.config.snippet_max_length == 100

    def test_generate_citations_from_results(self):
        """Test generating citations from retrieval results."""
        results = [
            RetrievalResult(
                chunk_id="doc1#1",
                score=0.95,
                text="This is test content for citation 1",
                metadata={"source": "test.pdf", "page": 1},
            ),
            RetrievalResult(
                chunk_id="doc1#2",
                score=0.85,
                text="This is test content for citation 2",
                metadata={"source": "test.pdf", "page": 2},
            ),
        ]

        generator = CitationGenerator()
        citations = generator.generate(results)

        assert len(citations) == 2
        assert citations[0].index == 1
        assert citations[0].chunk_id == "doc1#1"
        assert citations[0].source == "test.pdf"
        assert citations[0].page == 1
        assert citations[0].score == 0.95
        assert citations[1].index == 2
        assert citations[1].page == 2

    def test_generate_citations_empty_results(self):
        """Test generating citations from empty results."""
        generator = CitationGenerator()
        citations = generator.generate([])
        assert citations == []

    def test_extract_source_from_metadata(self):
        """Test source extraction from various metadata keys."""
        result = RetrievalResult(
            chunk_id="test#1",
            score=0.9,
            text="test",
            metadata={"source": "document.pdf"},
        )
        generator = CitationGenerator()
        citations = generator.generate([result])
        assert citations[0].source == "document.pdf"

    def test_extract_page_from_metadata(self):
        """Test page extraction from various metadata keys."""
        result = RetrievalResult(
            chunk_id="test#1",
            score=0.9,
            text="test",
            metadata={"page_number": 5},
        )
        generator = CitationGenerator()
        citations = generator.generate([result])
        assert citations[0].page == 5

    def test_citation_to_dict(self):
        """Test Citation to_dict method."""
        citation = Citation(
            index=1,
            chunk_id="doc1#1",
            source="test.pdf",
            page=1,
            score=0.95,
            text_snippet="Preview text",
        )
        result = citation.to_dict()
        assert result["index"] == 1
        assert result["chunk_id"] == "doc1#1"
        assert result["source"] == "test.pdf"
        assert result["page"] == 1
        assert result["score"] == 0.95
        assert result["text_snippet"] == "Preview text"


class TestResponseBuilder:
    """Tests for ResponseBuilder."""

    def test_response_builder_creation(self):
        """Test creating a ResponseBuilder instance."""
        builder = ResponseBuilder()
        assert builder is not None

    def test_response_builder_with_params(self):
        """Test creating a ResponseBuilder with custom params."""
        builder = ResponseBuilder(max_results_in_content=3, snippet_max_length=100)
        assert builder.max_results_in_content == 3
        assert builder.snippet_max_length == 100

    def test_build_response_with_results(self):
        """Test building response with retrieval results."""
        results = [
            RetrievalResult(
                chunk_id="doc1#1",
                score=0.95,
                text="This is test content for citation 1",
                metadata={"source": "test.pdf", "page": 1},
            ),
        ]

        builder = ResponseBuilder()
        response = builder.build(results=results, query="test query", collection="default")

        assert response.content is not None
        assert "test query" in response.content
        assert len(response.citations) == 1
        assert response.citations[0].chunk_id == "doc1#1"

    def test_build_response_empty_results(self):
        """Test building response with no results."""
        builder = ResponseBuilder()
        response = builder.build(results=[], query="test query", collection="default")

        assert response.content is not None
        assert "未找到" in response.content or "no results" in response.content.lower()
        assert response.citations == []
        assert response.metadata["result_count"] == 0

    def test_build_response_metadata(self):
        """Test that response metadata is correctly built."""
        results = [
            RetrievalResult(chunk_id="doc1#1", score=0.95, text="test"),
            RetrievalResult(chunk_id="doc1#2", score=0.85, text="test"),
        ]

        builder = ResponseBuilder()
        response = builder.build(results=results, query="test", collection="default")

        assert "query" in response.metadata
        assert response.metadata["result_count"] == 2


class TestIntegration:
    """Integration tests for response building."""

    def test_full_pipeline(self):
        """Test full citation and response pipeline."""
        # Create test results
        results = [
            RetrievalResult(
                chunk_id="doc1#1",
                score=0.95,
                text="First document content about machine learning",
                metadata={"source": "ml-intro.pdf", "page": 1},
            ),
            RetrievalResult(
                chunk_id="doc2#1",
                score=0.88,
                text="Second document about deep learning",
                metadata={"source": "dl-guide.pdf", "page": 5},
            ),
            RetrievalResult(
                chunk_id="doc3#1",
                score=0.75,
                text="Third document about neural networks",
                metadata={"source": "nn-basics.pdf", "page": 10},
            ),
        ]

        # Build response
        builder = ResponseBuilder()
        response = builder.build(results=results, query="learning", collection="default")

        # Verify
        assert response.metadata["query"] == "learning"
        assert response.metadata["result_count"] == 3
        assert len(response.citations) == 3

        # Check first citation (Citation objects, not dicts)
        assert response.citations[0].source == "ml-intro.pdf"
        assert response.citations[0].page == 1
        assert response.citations[0].score == 0.95

        # Check second citation
        assert response.citations[1].source == "dl-guide.pdf"
        assert response.citations[1].page == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
