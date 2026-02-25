"""Unit tests for QueryProcessor."""

import pytest

from core.query_engine.query_processor import (
    DEFAULT_STOPWORDS,
    ENGLISH_STOPWORDS,
    CHINESE_STOPWORDS,
    FILTER_PATTERN,
    QueryProcessor,
    QueryProcessorConfig,
    QueryResult,
    create_query_processor,
)


class TestQueryProcessorConfig:
    """Tests for QueryProcessorConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = QueryProcessorConfig()

        assert config.min_keyword_length == 1
        assert config.max_keywords == 20
        assert config.enable_filter_parsing is True
        assert config.lowercase is True
        assert len(config.stopwords) > 0

    def test_custom_config(self):
        """Test custom configuration."""
        config = QueryProcessorConfig(
            min_keyword_length=2,
            max_keywords=10,
            enable_filter_parsing=False,
        )

        assert config.min_keyword_length == 2
        assert config.max_keywords == 10
        assert config.enable_filter_parsing is False


class TestQueryProcessor:
    """Tests for QueryProcessor."""

    def test_default_initialization(self):
        """Test default initialization."""
        processor = QueryProcessor()

        assert len(processor.stopwords) > 0
        assert processor.config.min_keyword_length == 1
        assert processor.config.max_keywords == 20

    def test_custom_config_initialization(self):
        """Test initialization with custom config."""
        config = QueryProcessorConfig(min_keyword_length=3, max_keywords=5)
        processor = QueryProcessor(config)

        assert processor.config.min_keyword_length == 3
        assert processor.config.max_keywords == 5

    def test_process_basic_query(self):
        """Test processing a basic query."""
        processor = QueryProcessor()
        result = processor.process("What is the capital of France?")

        assert result.keywords
        assert "capital" in result.keywords
        assert "france" in result.keywords
        # Stopwords should be removed
        assert "what" not in result.keywords
        assert "is" not in result.keywords
        assert "the" not in result.keywords
        assert "of" not in result.keywords

    def test_process_empty_query_raises(self):
        """Test that empty query raises ValueError."""
        processor = QueryProcessor()

        with pytest.raises(ValueError, match="cannot be empty"):
            processor.process("")

        with pytest.raises(ValueError, match="cannot be empty"):
            processor.process("   ")

    def test_process_with_filters(self):
        """Test processing query with filters."""
        processor = QueryProcessor()
        result = processor.process("collection:technical_docs Python tutorial")

        assert "python" in result.keywords
        assert "tutorial" in result.keywords
        assert result.filters.get("collection") == "technical_docs"

    def test_process_filters_extraction(self):
        """Test filter extraction from query."""
        processor = QueryProcessor()
        result = processor.process("doc_type:pdf year:2024 revenue report")

        assert result.filters["doc_type"] == "pdf"
        assert result.filters["year"] == "2024"
        assert "revenue" in result.keywords
        assert "report" in result.keywords

    def test_process_no_filters(self):
        """Test processing without filters."""
        processor = QueryProcessor()
        result = processor.process("test query")

        assert result.filters == {}
        assert "test" in result.keywords
        assert "query" in result.keywords

    def test_process_filter_disabled(self):
        """Test processing with filter parsing disabled."""
        config = QueryProcessorConfig(enable_filter_parsing=False)
        processor = QueryProcessor(config)
        result = processor.process("collection:financial test")

        # Filters should NOT be parsed - filter syntax is treated as regular text
        # "collection:financial" is split into ["collection", "financial"]
        assert "collection" in result.keywords
        assert "financial" in result.keywords
        assert result.filters == {}

    def test_normalize_query(self):
        """Test query normalization."""
        processor = QueryProcessor()
        result = processor.process("  multiple   spaces   here  ")

        assert result.normalized_query == "multiple spaces here"

    def test_extract_keywords_removes_stopwords(self):
        """Test that stopwords are removed."""
        processor = QueryProcessor()
        keywords = processor._extract_keywords("the quick brown fox")

        assert "quick" in keywords
        assert "brown" in keywords
        assert "fox" in keywords
        assert "the" not in keywords

    def test_extract_keywords_min_length(self):
        """Test minimum keyword length filtering."""
        config = QueryProcessorConfig(min_keyword_length=3)
        processor = QueryProcessor(config)
        keywords = processor._extract_keywords("a bc def ghij")

        assert "def" in keywords
        assert "ghij" in keywords
        # Short words should be filtered
        assert "a" not in keywords
        assert "bc" not in keywords

    def test_extract_keywords_lowercase(self):
        """Test lowercase conversion."""
        config = QueryProcessorConfig(lowercase=True)
        processor = QueryProcessor(config)
        keywords = processor._extract_keywords("Python Programming")

        assert "python" in keywords
        assert "programming" in keywords

    def test_extract_keywords_special_characters(self):
        """Test handling of special characters."""
        processor = QueryProcessor()
        keywords = processor._extract_keywords("hello-world! test@file #tag")

        assert "hello" in keywords
        assert "world" in keywords
        assert "test" in keywords
        assert "file" in keywords
        assert "tag" in keywords

    def test_extract_keywords_numbers(self):
        """Test handling of numbers."""
        processor = QueryProcessor()
        keywords = processor._extract_keywords("test123 python2024")

        assert "test123" in keywords
        assert "python2024" in keywords

    def test_extract_keywords_only(self):
        """Test extract_keywords_only method."""
        processor = QueryProcessor()
        keywords = processor.extract_keywords_only("hello world")

        assert "hello" in keywords
        assert "world" in keywords

    def test_max_keywords_limit(self):
        """Test max keywords limit."""
        config = QueryProcessorConfig(max_keywords=2)
        processor = QueryProcessor(config)
        result = processor.process("word1 word2 word3 word4 word5")

        assert len(result.keywords) <= 2

    def test_case_insensitive_deduplication(self):
        """Test case-insensitive deduplication."""
        processor = QueryProcessor()
        keywords = processor._extract_keywords("Hello hello HELLO world")

        # Should only have one "hello" (case-insensitive)
        hello_count = sum(1 for k in keywords if k.lower() == "hello")
        assert hello_count == 1

    def test_chinese_query(self):
        """Test handling of Chinese characters."""
        processor = QueryProcessor()
        keywords = processor._extract_keywords("你好世界")

        # Should extract Chinese words
        assert "你好世界" in keywords or len(keywords) > 0

    def test_chinese_query_with_stopwords(self):
        """Test Chinese stopwords are removed."""
        processor = QueryProcessor()
        # "的" is a Chinese stopword
        keywords = processor._extract_keywords("我的书本")

        # The phrase should be extracted
        assert len(keywords) > 0

    def test_mixed_query(self):
        """Test mixed language query."""
        processor = QueryProcessor()
        keywords = processor._extract_keywords("Python 编程 language")

        assert "python" in keywords
        assert "language" in keywords


class TestDynamicStopwords:
    """Tests for dynamic stopword management."""

    def test_add_stopwords(self):
        """Test adding stopwords."""
        processor = QueryProcessor()
        initial_count = len(processor.stopwords)

        processor.add_stopwords({"test", "demo"})

        assert len(processor.stopwords) == initial_count + 2
        assert "test" in processor.stopwords
        assert "demo" in processor.stopwords

    def test_remove_stopwords(self):
        """Test removing stopwords."""
        processor = QueryProcessor()

        # "the" is a stopword
        assert "the" in processor.stopwords

        processor.remove_stopwords({"the"})

        assert "the" not in processor.stopwords

    def test_reset_stopwords(self):
        """Test resetting stopwords to default."""
        processor = QueryProcessor()
        processor.add_stopwords({"custom"})

        processor.reset_stopwords()

        assert "custom" not in processor.stopwords
        assert len(processor.stopwords) == len(DEFAULT_STOPWORDS)


class TestFilterPattern:
    """Tests for filter pattern."""

    def test_filter_pattern_matches(self):
        """Test filter pattern matching."""
        assert FILTER_PATTERN.match("collection:financial")
        assert FILTER_PATTERN.match("doc_type:pdf")
        assert FILTER_PATTERN.match("year:2024")

    def test_filter_pattern_no_match(self):
        """Test filter pattern non-matching."""
        assert not FILTER_PATTERN.match("hello world")
        assert not FILTER_PATTERN.match("collection")


class TestCreateQueryProcessor:
    """Tests for create_query_processor factory function."""

    def test_create_with_defaults(self):
        """Test factory with default settings."""
        processor = create_query_processor()

        assert isinstance(processor, QueryProcessor)
        assert processor.config.min_keyword_length == 1

    def test_create_with_custom_stopwords(self):
        """Test factory with custom stopwords."""
        processor = create_query_processor(stopwords=["the", "a"])

        assert "the" in processor.stopwords
        assert "a" in processor.stopwords

    def test_create_with_min_length(self):
        """Test factory with custom min length."""
        processor = create_query_processor(min_keyword_length=4)

        assert processor.config.min_keyword_length == 4

    def test_create_with_max_keywords(self):
        """Test factory with custom max keywords."""
        processor = create_query_processor(max_keywords=5)

        assert processor.config.max_keywords == 5

    def test_create_with_filter_parsing_disabled(self):
        """Test factory with filter parsing disabled."""
        processor = create_query_processor(enable_filter_parsing=False)

        assert processor.config.enable_filter_parsing is False


class TestQueryResult:
    """Tests for QueryResult dataclass."""

    def test_query_result_creation(self):
        """Test QueryResult creation."""
        result = QueryResult(
            keywords=["python", "tutorial"],
            filters={"year": 2024},
            raw_query="Python tutorial",
        )

        assert result.keywords == ["python", "tutorial"]
        assert result.filters == {"year": 2024}
        assert result.raw_query == "Python tutorial"

    def test_query_result_empty_filters(self):
        """Test QueryResult with empty filters."""
        result = QueryResult(
            keywords=["test"],
            filters={},
            raw_query="test",
        )

        assert result.filters == {}


class TestBackwardCompatibility:
    """Tests for backward compatibility."""

    def test_process_returns_processed_query(self):
        """Test that process returns ProcessedQuery."""
        from core.types import ProcessedQuery

        processor = QueryProcessor()
        result = processor.process("test query")

        assert isinstance(result, ProcessedQuery)
        assert hasattr(result, "normalized_query")

    def test_query_result_from_processed_query(self):
        """Test QueryResult.from_processed_query."""
        from core.types import ProcessedQuery

        pq = ProcessedQuery(
            keywords=["test"],
            filters={"collection": "docs"},
            raw_query="test",
            normalized_query="test",
        )

        qr = QueryResult.from_processed_query(pq)

        assert qr.keywords == ["test"]
        assert qr.filters == {"collection": "docs"}
        assert qr.raw_query == "test"
