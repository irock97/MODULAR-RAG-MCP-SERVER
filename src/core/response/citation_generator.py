"""Citation Generator - Generate structured citations from retrieval results.

This module provides the CitationGenerator class that extracts citation
information from retrieval results for MCP tool responses.

Design Principles:
    - Idempotent: Same results always produce same citations
    - Configurable: Citation format can be customized
    - Fallback: Graceful handling when metadata is missing
"""

from dataclasses import dataclass
from typing import Any

from core.types import RetrievalResult
from observability.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Data Types
# =============================================================================


@dataclass
class Citation:
    """Structured citation for a single retrieval result.

    Attributes:
        index: Citation number (1, 2, 3, ...)
        chunk_id: Unique identifier for the chunk
        source: Source document name
        page: Page number (if available)
        score: Relevance score
        text_snippet: First N characters of text for context
    """

    index: int
    chunk_id: str
    source: str = ""
    page: int | None = None
    score: float = 0.0
    text_snippet: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for structured response."""
        return {
            "index": self.index,
            "chunk_id": self.chunk_id,
            "source": self.source,
            "page": self.page,
            "score": round(self.score, 4),
            "text_snippet": self.text_snippet,
        }


@dataclass
class CitationConfig:
    """Configuration for citation generation.

    Attributes:
        snippet_max_length: Number of characters to include in text snippet
        include_metadata_fields: Additional metadata fields to include
    """

    snippet_max_length: int = 200
    include_metadata_fields: list[str] | None = None


# =============================================================================
# CitationGenerator
# =============================================================================


class CitationGenerator:
    """Generate structured citations from retrieval results.

    This class extracts citation information from RetrievalResult objects
    and formats them for MCP tool responses.

    Example:
        >>> from core.response import CitationGenerator, CitationConfig
        >>> from core.types import RetrievalResult
        >>>
        >>> config = CitationConfig(snippet_max_length=150)
        >>> generator = CitationGenerator(config)
        >>>
        >>> results = [
        ...     RetrievalResult(chunk_id="doc1#1", score=0.95, text="...", metadata={"source": "doc.pdf", "page": 1}),
        ... ]
        >>>
        >>> citations = generator.generate(results)
        >>> print(citations[0].to_dict())
        {'index': 1, 'chunk_id': 'doc1#1', 'source': 'doc.pdf', 'page': 1, 'score': 0.95, ...}
    """

    def __init__(
        self,
        snippet_max_length: int = 200,
        include_metadata_fields: list[str] | None = None,
    ) -> None:
        """Initialize the CitationGenerator.

        Args:
            snippet_max_length: Maximum characters for text snippet.
            include_metadata_fields: Additional metadata fields to include.
        """
        self._config = CitationConfig(
            snippet_max_length=snippet_max_length,
            include_metadata_fields=include_metadata_fields,
        )

    @property
    def config(self) -> CitationConfig:
        """Get the citation configuration."""
        return self._config

    def generate(
        self,
        results: list[RetrievalResult],
    ) -> list[Citation]:
        """Generate citations from retrieval results.

        Args:
            results: List of retrieval results to generate citations from.

        Returns:
            List of Citation objects with index starting from 1.
        """
        if not results:
            return []

        citations = []
        for idx, result in enumerate(results, start=1):
            citation = self._extract_citation(result, idx)
            citations.append(citation)

        logger.debug(f"Generated {len(citations)} citations from {len(results)} results")
        return citations

    def _extract_citation(
        self,
        result: RetrievalResult,
        index: int,
    ) -> Citation:
        """Extract citation from a single retrieval result.

        Args:
            result: Retrieval result to extract from
            index: Citation index (1-based)

        Returns:
            Citation object with extracted information
        """
        # Extract source from metadata with fallbacks
        source = self._extract_source(result.metadata)
        page = self._extract_page(result.metadata)

        # Generate text snippet
        text_snippet = self._generate_snippet(result.text)

        return Citation(
            index=index,
            chunk_id=result.chunk_id,
            source=source,
            page=page,
            score=result.score,
            text_snippet=text_snippet,
        )

    def _extract_source(self, metadata: dict[str, Any]) -> str:
        """Extract source document name from metadata.

        Args:
            metadata: Result metadata

        Returns:
            Source document name or empty string
        """
        # Try multiple possible keys
        for key in ("source", "source_path", "doc_name", "filename", "file_name"):
            if value := metadata.get(key):
                # Extract just the filename if it's a path
                if isinstance(value, str) and "/" in value:
                    return value.split("/")[-1]
                return str(value)
        return ""

    def _extract_page(self, metadata: dict[str, Any]) -> int | None:
        """Extract page number from metadata.

        Args:
            metadata: Result metadata

        Returns:
            Page number or None if not available
        """
        for key in ("page", "page_number", "page_num"):
            if (value := metadata.get(key)) is not None:
                try:
                    return int(value)
                except (ValueError, TypeError):
                    pass
        return None

    def _generate_snippet(self, text: str) -> str:
        """Generate text snippet for citation.

        Args:
            text: Full text content

        Returns:
            Snippet text truncated to configured length
        """
        if not text:
            return ""

        # Truncate and add ellipsis if needed
        max_len = self._config.snippet_max_length
        if len(text) <= max_len:
            return text.strip()

        return text[:max_len].strip() + "..."

    def format_citations_markdown(
        self,
        citations: list[Citation],
    ) -> str:
        """Format citations as Markdown reference list.

        Args:
            citations: List of citations to format

        Returns:
            Markdown formatted citation list
        """
        if not citations:
            return ""

        lines = ["\n\n## References\n"]
        for citation in citations:
            line = f"[{citation.index}] "
            if citation.source:
                line += f"**{citation.source}**"
                if citation.page:
                    line += f" (page {citation.page})"
                line += ": "
            line += citation.text_snippet
            lines.append(line)

        return "\n".join(lines)

    def format_citation_marker(self, index: int) -> str:
        """Format citation marker (e.g., [1], [2]).

        Args:
            index: Citation index (1-based)

        Returns:
            Formatted citation marker string
        """
        return f"[{index}]"


# =============================================================================
# Factory Function
# =============================================================================


def create_citation_generator(
    snippet_max_length: int = 200,
    include_metadata_fields: list[str] | None = None,
) -> CitationGenerator:
    """Create a CitationGenerator with optional configuration.

    Args:
        snippet_max_length: Maximum characters for text snippet.
        include_metadata_fields: Additional metadata fields to include.

    Returns:
        Configured CitationGenerator instance
    """
    return CitationGenerator(
        snippet_max_length=snippet_max_length,
        include_metadata_fields=include_metadata_fields,
    )
