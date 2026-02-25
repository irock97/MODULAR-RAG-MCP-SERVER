"""Query Processor - Keyword extraction and filters parsing.

This module provides the QueryProcessor class for extracting keywords
from user queries and parsing filter structures.

Design Principles:
    - Keyword extraction using rule-based tokenization
    - Configurable stopwords with dynamic management
    - Filter parsing from query string
    - Query normalization
    - Smart Chinese tokenization

Features:
    - Filter parsing: Extract filters like "collection:financial" from query
    - Smart Chinese tokenization: Preserve complete words, handle mixed text
    - Configuration class: QueryProcessorConfig for all settings
    - Dynamic stopwords: Add/remove stopwords at runtime
    - Query normalization: Clean up whitespace
    - Case-insensitive deduplication
    - Max keywords limit
"""

import re
from dataclasses import dataclass, field
from typing import Any

from core.types import ProcessedQuery
from observability.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Stopwords - Organized by category for better maintainability
# =============================================================================

# English stopwords (common functional words)
ENGLISH_STOPWORDS = frozenset({
    # Articles
    "a", "an", "the",
    # Prepositions
    "in", "on", "at", "by", "for", "with", "from", "to", "of", "about",
    # Conjunctions
    "and", "or", "but", "so", "yet", "nor",
    # Pronouns
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "its", "our", "their",
    # Verbs (common auxiliary)
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can",
    # Adverbs
    "not", "no", "yes", "very", "too", "also", "just", "only",
    # Determiners
    "this", "that", "these", "those", "some", "any", "all",
    "each", "every", "both", "few", "more", "most", "other",
    "such", "no", "own", "same", "than",
    # Question words
    "what", "when", "where", "who", "which", "why", "how",
    # Others
    "as", "if", "then", "there", "here", "now", "up", "down",
    "out", "over", "under", "again", "further", "once"
})

# Chinese stopwords (organized by category)
CHINESE_STOPWORDS = frozenset({
    # 助词 (Particles) - Structural words
    "的", "了", "着", "过", "的", "地", "得",
    # 代词 (Pronouns)
    "我", "你", "他", "她", "它", "我们", "你们", "他们", "她们", "它们",
    "这", "那", "这个", "那个", "这些", "那些",
    "哪", "哪个", "哪里", "谁", "什么", "怎么", "怎样", "如何", "为什么",
    # 介词 (Prepositions)
    "在", "从", "到", "向", "对", "把", "被", "给", "以", "为", "由", "于",
    # 连词 (Conjunctions)
    "和", "与", "或", "而", "但", "因", "所以", "如果", "因为", "虽然",
    # 动词 (Common verbs)
    "是", "有", "在", "会", "能", "要", "到", "说", "让", "使", "去", "来",
    # 副词 (Adverbs)
    "不", "也", "都", "就", "还", "很", "更", "最", "太", "正", "再", "又",
    "已", "曾", "将", "正在", "可", "可能", "必须", "应该", "应当",
    # 量词 after number
    "个", "些", "点",
    # 否定词
    "没有", "无", "非", "别", "莫", "勿",
    # 疑问词
    "吗", "呢", "吧", "呀", "啊", "哪", "怎",
    # 其他虚词
    "等", "以及", "则", "如", "且", "并", "只", "仅", "不过", "可是",
    # 复合虚词
    "可以", "可能", "必须", "不得不", "不要", "不能", "无法",
    "什么样", "怎么样", "这个", "那个", "这里", "那里",
    "这样", "那样", "这么", "那么", "多少", "几个"
})

# Combined default stopwords
DEFAULT_STOPWORDS = ENGLISH_STOPWORDS | CHINESE_STOPWORDS


# =============================================================================
# Filter Pattern
# =============================================================================

# Regex pattern to match filters in query string: "key:value"
# Examples: "collection:financial", "doc_type:pdf", "year:2024"
FILTER_PATTERN = re.compile(r'(\w+):([^\s]+)')


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class QueryProcessorConfig:
    """Configuration for QueryProcessor.

    Attributes:
        stopwords: Set of stopwords to remove
        min_keyword_length: Minimum length for keywords (default: 1)
        max_keywords: Maximum number of keywords to extract (default: 20)
        enable_filter_parsing: Whether to parse filters from query string
        lowercase: Whether to lowercase English keywords
    """

    stopwords: set[str] = field(default_factory=lambda: set(DEFAULT_STOPWORDS))
    min_keyword_length: int = 1
    max_keywords: int = 20
    enable_filter_parsing: bool = True
    lowercase: bool = True


# =============================================================================
# Query Processor
# =============================================================================

class QueryProcessor:
    """Processor for extracting keywords and parsing filters from queries.

    This class handles:
    - Filter parsing: Extract "key:value" filters from query string
    - Keyword extraction using tokenization and stopword removal
    - Smart Chinese tokenization for mixed-language queries
    - Query normalization
    - Dynamic stopword management

    Example:
        >>> processor = QueryProcessor()
        >>> result = processor.process("collection:financial 2024 revenue")
        >>> result.filters
        {'collection': 'financial'}
        >>> result.keywords
        ['2024', 'revenue']

        >>> # With custom config
        >>> config = QueryProcessorConfig(max_keywords=5, min_keyword_length=2)
        >>> processor = QueryProcessor(config)
        >>> processor.add_stopwords({"test"})
    """

    def __init__(
        self,
        config: QueryProcessorConfig | None = None,
    ) -> None:
        """Initialize the QueryProcessor.

        Args:
            config: QueryProcessorConfig instance. If None, uses defaults.
        """
        self._config = config or QueryProcessorConfig()

        # Convert to frozenset for internal use
        self._stopwords: frozenset[str] = frozenset(self._config.stopwords)

        logger.info(
            f"QueryProcessor initialized: stopwords={len(self._stopwords)}, "
            f"min_length={self._config.min_keyword_length}, "
            f"max_keywords={self._config.max_keywords}, "
            f"filter_parsing={self._config.enable_filter_parsing}"
        )

    @property
    def config(self) -> QueryProcessorConfig:
        """Get the configuration."""
        return self._config

    @property
    def stopwords(self) -> frozenset[str]:
        """Get the stopwords set (read-only)."""
        return self._stopwords

    # -------------------------------------------------------------------------
    # Dynamic Stopword Management
    # -------------------------------------------------------------------------

    def add_stopwords(self, words: set[str]) -> None:
        """Add stopwords to the processor.

        Args:
            words: Set of words to add as stopwords
        """
        self._stopwords = self._stopwords | frozenset(words)
        logger.debug(f"Added stopwords: {words}, total: {len(self._stopwords)}")

    def remove_stopwords(self, words: set[str]) -> None:
        """Remove stopwords from the processor.

        Args:
            words: Set of words to remove from stopwords
        """
        self._stopwords = self._stopwords - frozenset(words)
        logger.debug(f"Removed stopwords: {words}, total: {len(self._stopwords)}")

    def reset_stopwords(self) -> None:
        """Reset stopwords to default."""
        self._stopwords = frozenset(DEFAULT_STOPWORDS)
        logger.debug(f"Reset stopwords to default: {len(self._stopwords)}")

    # -------------------------------------------------------------------------
    # Main Processing
    # -------------------------------------------------------------------------

    def process(self, query: str) -> ProcessedQuery:
        """Process a query to extract keywords and filters.

        Args:
            query: The raw query string (may contain filters like "collection:xyz")

        Returns:
            ProcessedQuery containing keywords, filters, and normalized query

        Raises:
            ValueError: If query is empty
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        raw_query = query

        # Step 1: Normalize query
        normalized = self._normalize(query)

        # Step 2: Extract filters from query string (if enabled)
        filters: dict[str, Any] = {}
        if self._config.enable_filter_parsing:
            query_without_filters, filters = self._extract_filters(normalized)
        else:
            query_without_filters = normalized

        # Step 3: Extract keywords
        keywords = self._extract_keywords(query_without_filters)

        logger.debug(
            f"Processed query: keywords={keywords}, filters={filters}"
        )

        return ProcessedQuery(
            keywords=keywords,
            filters=filters,
            raw_query=raw_query,
            normalized_query=normalized,
        )

    # -------------------------------------------------------------------------
    # Query Normalization
    # -------------------------------------------------------------------------

    def _normalize(self, query: str) -> str:
        """Normalize query string.

        Steps:
        - Strip leading/trailing whitespace
        - Collapse multiple whitespace to single space

        Args:
            query: Raw query string

        Returns:
            Normalized query string
        """
        # Strip and collapse whitespace
        normalized = query.strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized

    # -------------------------------------------------------------------------
    # Filter Extraction
    # -------------------------------------------------------------------------

    def _extract_filters(
        self,
        query: str
    ) -> tuple[str, dict[str, Any]]:
        """Extract filters from query string.

        Parses filters in format "key:value" from the query.
        Supported filters: collection, doc_type, source_path, tags, year, author, etc.

        Args:
            query: Normalized query string

        Returns:
            Tuple of (query_without_filters, filters_dict)
        """
        filters: dict[str, Any] = {}
        query_parts = []

        # Split query into parts (by whitespace)
        parts = query.split()

        for part in parts:
            # Check if part matches filter pattern
            match = FILTER_PATTERN.match(part)
            if match:
                key, value = match.groups()
                filters[key] = value
            else:
                query_parts.append(part)

        # Reconstruct query without filters
        query_without_filters = ' '.join(query_parts)

        logger.debug(f"Extracted filters: {filters}")

        return query_without_filters, filters

    # -------------------------------------------------------------------------
    # Keyword Extraction
    # -------------------------------------------------------------------------

    def _extract_keywords(self, query: str) -> list[str]:
        """Extract keywords from query using smart tokenization.

        Steps:
        1. Extract English tokens (alphanumeric sequences)
        2. Extract Chinese tokens (2-4 character words, try to preserve complete words)
        3. Handle mixed English-Chinese tokens (e.g., "Azure配置")
        4. Filter by length and remove stopwords
        5. Apply lowercase for English
        6. Deduplicate (case-insensitive)
        7. Limit to max_keywords

        Args:
            query: Query string (filters already removed)

        Returns:
            List of extracted keywords
        """
        # Extract English tokens
        english_tokens = re.findall(r'[a-zA-Z0-9]+', query)

        # Extract Chinese tokens - try to get complete words
        # First try 4-char, then 3-char, then 2-char for remaining
        chinese_tokens = []
        remaining = query

        # Priority: try to extract complete Chinese words (2-4 chars)
        while remaining:
            # Try 4-char first
            match = re.match(r'[\u4e00-\u9fff]{4}', remaining)
            if match:
                chinese_tokens.append(match.group(0))
                remaining = remaining[4:]
                continue

            # Try 3-char
            match = re.match(r'[\u4e00-\u9fff]{3}', remaining)
            if match:
                chinese_tokens.append(match.group(0))
                remaining = remaining[3:]
                continue

            # Try 2-char
            match = re.match(r'[\u4e00-\u9fff]{2}', remaining)
            if match:
                chinese_tokens.append(match.group(0))
                remaining = remaining[2:]
                continue

            # Skip non-Chinese character
            remaining = remaining[1:]

        # Handle mixed tokens (English + Chinese, e.g., "Azure配置")
        mixed_tokens = re.findall(r'[a-zA-Z0-9]+[\u4e00-\u9fff]+|[\u4e00-\u9fff]+[a-zA-Z0-9]+', query)

        # Combine all tokens
        all_tokens = english_tokens + chinese_tokens + mixed_tokens

        # Process tokens
        keywords = []
        seen_lower: set[str] = set()  # For case-insensitive deduplication

        for token in all_tokens:
            if not token:
                continue

            # Apply lowercase for English tokens
            if token.isascii():
                word = token.lower() if self._config.lowercase else token
            else:
                # Chinese/mixed tokens: keep as is
                word = token

            # Filter by minimum length
            if len(word) < self._config.min_keyword_length:
                continue

            # Remove stopwords
            if word in self._stopwords:
                continue

            # Case-insensitive deduplication
            word_lower = word.lower()
            if word_lower in seen_lower:
                continue
            seen_lower.add(word_lower)

            keywords.append(word)

            # Limit to max_keywords
            if len(keywords) >= self._config.max_keywords:
                break

        return keywords

    def extract_keywords_only(self, query: str) -> list[str]:
        """Extract keywords only, without full processing.

        Args:
            query: Raw query string

        Returns:
            List of extracted keywords
        """
        normalized = self._normalize(query)

        # Extract filters if enabled
        if self._config.enable_filter_parsing:
            query_without_filters, _ = self._extract_filters(normalized)
        else:
            query_without_filters = normalized

        return self._extract_keywords(query_without_filters)


# =============================================================================
# Factory Function
# =============================================================================

def create_query_processor(
    stopwords: list[str] | None = None,
    min_keyword_length: int = 1,
    max_keywords: int = 20,
    enable_filter_parsing: bool = True,
) -> QueryProcessor:
    """Create a QueryProcessor with optional configuration.

    Args:
        stopwords: Optional list of stopwords to add (to default set)
        min_keyword_length: Minimum keyword length (default: 1)
        max_keywords: Maximum keywords to extract (default: 20)
        enable_filter_parsing: Whether to parse filters (default: True)

    Returns:
        Configured QueryProcessor instance

    Example:
        >>> processor = create_query_processor(
        ...     stopwords=["test", "demo"],
        ...     max_keywords=10
        ... )
    """
    # Start with default config
    config = QueryProcessorConfig(
        min_keyword_length=min_keyword_length,
        max_keywords=max_keywords,
        enable_filter_parsing=enable_filter_parsing,
    )

    # Add custom stopwords if provided
    if stopwords:
        config.stopwords = DEFAULT_STOPWORDS | set(stopwords)

    return QueryProcessor(config)


# =============================================================================
# Backward Compatibility
# =============================================================================

# Keep QueryResult for backward compatibility
@dataclass
class QueryResult:
    """Processed query result (legacy, use ProcessedQuery instead).

    Attributes:
        keywords: Extracted keywords from the query
        filters: Parsed filter structure
        raw_query: Original query string
    """

    keywords: list[str]
    filters: dict[str, Any]
    raw_query: str

    @classmethod
    def from_processed_query(cls, pq: ProcessedQuery) -> "QueryResult":
        """Create QueryResult from ProcessedQuery."""
        return cls(
            keywords=pq.keywords,
            filters=pq.filters,
            raw_query=pq.raw_query,
        )
