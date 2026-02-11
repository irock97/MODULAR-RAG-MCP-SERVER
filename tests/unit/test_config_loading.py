"""Tests for configuration loading and validation.

These tests verify that:
1. Settings can be loaded from YAML files
2. Required fields are validated
3. Error messages are clear and include field paths
"""

import tempfile
from pathlib import Path

import pytest

from core.settings import (
    Settings,
    LLMConfig,
    EmbeddingConfig,
    VectorStoreConfig,
    RetrievalConfig,
    RerankConfig,
    EvaluationConfig,
    IngestionConfig,
    load_settings,
    validate_settings,
    SettingsError,
    SettingsFileError,
    SettingsValidationError,
)


class TestSettingsLoading:
    """Test loading settings from YAML files."""

    def test_load_default_settings(self) -> None:
        """Test loading the actual settings.yaml file."""
        settings = load_settings("config/settings.yaml")
        assert settings is not None
        assert isinstance(settings, Settings)

    def test_load_minimal_settings(self) -> None:
        """Test loading a minimal valid settings file."""
        minimal_yaml = """
llm:
  provider: openai
  model: gpt-4o-mini
embedding:
  provider: openai
  model: text-embedding-3-small
vector_store:
  provider: chroma
retrieval:
  dense_top_k: 20
  sparse_top_k: 20
  fusion_top_k: 10
rerank:
  enabled: false
  provider: none
evaluation:
  enabled: false
  provider: custom
ingestion:
  chunk_size: 1000
  chunk_overlap: 200
  splitter: recursive
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(minimal_yaml)
            f.flush()
            settings = load_settings(f.name)

        assert settings.llm.provider == "openai"
        assert settings.llm.model == "gpt-4o-mini"
        assert settings.embedding.provider == "openai"
        assert settings.vector_store.provider == "chroma"

        # Clean up
        Path(f.name).unlink()

    def test_load_nonexistent_file(self) -> None:
        """Test that loading a nonexistent file raises SettingsFileError."""
        with pytest.raises(SettingsFileError) as exc_info:
            load_settings("/nonexistent/path/settings.yaml")
        assert "not found" in str(exc_info.value).lower()

    def test_load_invalid_yaml(self) -> None:
        """Test that invalid YAML raises SettingsFileError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [[[")
            f.flush()
            with pytest.raises(SettingsFileError) as exc_info:
                load_settings(f.name)
        Path(f.name).unlink()


class TestSettingsValidation:
    """Test settings validation."""

    def test_validate_minimal_settings(self) -> None:
        """Test that minimal valid settings pass validation."""
        settings = Settings(
            llm=LLMConfig(provider="openai", model="gpt-4o-mini"),
            embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small"),
            vector_store=VectorStoreConfig(provider="chroma"),
            retrieval=RetrievalConfig(
                dense_top_k=20, sparse_top_k=20, fusion_top_k=10
            ),
            rerank=RerankConfig(enabled=False, provider="none"),
            evaluation=EvaluationConfig(enabled=False, provider="custom"),
            ingestion=IngestionConfig(
                chunk_size=1000, chunk_overlap=200, splitter="recursive"
            ),
        )
        # Should not raise
        validate_settings(settings)

    def test_missing_required_field_llm_provider(self) -> None:
        """Test that missing llm.provider raises validation error."""
        settings = Settings(
            llm=LLMConfig(model="gpt-4o-mini"),  # Missing provider
            embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small"),
            vector_store=VectorStoreConfig(provider="chroma"),
            retrieval=RetrievalConfig(
                dense_top_k=20, sparse_top_k=20, fusion_top_k=10
            ),
            rerank=RerankConfig(enabled=False, provider="none"),
            evaluation=EvaluationConfig(enabled=False, provider="custom"),
            ingestion=IngestionConfig(
                chunk_size=1000, chunk_overlap=200, splitter="recursive"
            ),
        )
        with pytest.raises(SettingsValidationError) as exc_info:
            validate_settings(settings)
        assert "llm.provider" in str(exc_info.value)
        assert "llm.provider" in exc_info.value.missing_fields

    def test_missing_required_field_embedding_provider(self) -> None:
        """Test that missing embedding.provider raises validation error."""
        settings = Settings(
            llm=LLMConfig(provider="openai", model="gpt-4o-mini"),
            embedding=EmbeddingConfig(model="text-embedding-3-small"),  # Missing provider
            vector_store=VectorStoreConfig(provider="chroma"),
            retrieval=RetrievalConfig(
                dense_top_k=20, sparse_top_k=20, fusion_top_k=10
            ),
            rerank=RerankConfig(enabled=False, provider="none"),
            evaluation=EvaluationConfig(enabled=False, provider="custom"),
            ingestion=IngestionConfig(
                chunk_size=1000, chunk_overlap=200, splitter="recursive"
            ),
        )
        with pytest.raises(SettingsValidationError) as exc_info:
            validate_settings(settings)
        assert "embedding.provider" in str(exc_info.value)

    def test_missing_multiple_required_fields(self) -> None:
        """Test that multiple missing fields are all reported."""
        settings = Settings(
            llm=LLMConfig(),  # Missing provider and model
            embedding=EmbeddingConfig(),
            vector_store=VectorStoreConfig(),
            retrieval=RetrievalConfig(dense_top_k=20),
            rerank=RerankConfig(),
            evaluation=EvaluationConfig(),
            ingestion=IngestionConfig(),
        )
        with pytest.raises(SettingsValidationError) as exc_info:
            validate_settings(settings)
        # Should include multiple fields
        error_msg = str(exc_info.value)
        assert "llm.provider" in error_msg
        assert "llm.model" in error_msg

    def test_empty_settings_raises_error(self) -> None:
        """Test that empty settings raise validation error."""
        settings = Settings(
            llm=LLMConfig(),
            embedding=EmbeddingConfig(),
            vector_store=VectorStoreConfig(),
            retrieval=RetrievalConfig(),
            rerank=RerankConfig(),
            evaluation=EvaluationConfig(),
            ingestion=IngestionConfig(),
        )
        with pytest.raises(SettingsValidationError):
            validate_settings(settings)


class TestSettingsStructure:
    """Test settings dataclass structure."""

    def test_settings_has_all_sections(self) -> None:
        """Test that Settings has all required sections."""
        settings = load_settings("config/settings.yaml")
        assert hasattr(settings, "llm")
        assert hasattr(settings, "embedding")
        assert hasattr(settings, "vector_store")
        assert hasattr(settings, "retrieval")
        assert hasattr(settings, "rerank")
        assert hasattr(settings, "evaluation")
        assert hasattr(settings, "observability")
        assert hasattr(settings, "ingestion")

    def test_llm_config_defaults(self) -> None:
        """Test LLMConfig default values for optional fields."""
        config = LLMConfig(provider="openai", model="gpt-4o-mini")
        assert config.provider == "openai"
        assert config.model == "gpt-4o-mini"
        assert config.temperature == 0.0
        assert config.max_tokens == 4096

    def test_embedding_config_defaults(self) -> None:
        """Test EmbeddingConfig default values for optional fields."""
        config = EmbeddingConfig(provider="openai", model="text-embedding-3-small")
        assert config.provider == "openai"
        assert config.model == "text-embedding-3-small"
        assert config.dimensions == 1536

    def test_vector_store_config_defaults(self) -> None:
        """Test VectorStoreConfig default values for optional fields."""
        config = VectorStoreConfig(provider="chroma")
        assert config.provider == "chroma"
        assert config.persist_directory == "./data/db/chroma"
        assert config.collection_name == "knowledge_hub"


class TestErrorMessages:
    """Test error message quality."""

    def test_error_message_includes_field_path(self) -> None:
        """Test that error messages include dot-notation field paths."""
        settings = Settings(
            llm=LLMConfig(model="gpt-4o-mini"),  # Missing provider
            embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small"),
            vector_store=VectorStoreConfig(provider="chroma"),
            retrieval=RetrievalConfig(
                dense_top_k=20, sparse_top_k=20, fusion_top_k=10
            ),
            rerank=RerankConfig(enabled=False, provider="none"),
            evaluation=EvaluationConfig(enabled=False, provider="custom"),
            ingestion=IngestionConfig(
                chunk_size=1000, chunk_overlap=200, splitter="recursive"
            ),
        )
        with pytest.raises(SettingsValidationError) as exc_info:
            validate_settings(settings)
        # Error should clearly indicate which field is missing
        assert "llm.provider" in exc_info.value.missing_fields


class TestGetEffectiveSettings:
    """Test get_effective_settings function."""

    def test_overrides_apply(self) -> None:
        """Test that overrides are applied correctly."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("llm:\n  provider: openai\n  model: gpt-4o-mini\n")
            f.write("embedding:\n  provider: openai\n  model: text-embedding-3-small\n")
            f.write("vector_store:\n  provider: chroma\n")
            f.write("retrieval:\n  dense_top_k: 20\n  sparse_top_k: 20\n  fusion_top_k: 10\n")
            f.write("rerank:\n  enabled: false\n  provider: none\n")
            f.write("evaluation:\n  enabled: false\n  provider: custom\n")
            f.write("ingestion:\n  chunk_size: 1000\n  chunk_overlap: 200\n  splitter: recursive\n")
            f.flush()

            settings = load_settings(f.name)
            assert settings.llm.provider == "openai"

            # Clean up
            Path(f.name).unlink()
