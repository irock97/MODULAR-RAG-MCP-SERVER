"""Configuration management for Modular RAG MCP Server.

This module provides the Settings dataclass and loading/validation functions.
All configuration values are read from config/settings.yaml.

Design Principles:
    - Config-Driven: All values sourced from settings.yaml
    - Fail-Fast: Missing required fields cause immediate failure
    - Clear Errors: Error messages include field paths (e.g., 'embedding.provider')
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LLMConfig:
    """LLM provider configuration.

    Attributes:
        provider: LLM provider type (openai, azure, ollama, deepseek) - REQUIRED
        model: Model name to use - REQUIRED
        temperature: Sampling temperature (0.0-1.0)
        max_tokens: Maximum tokens to generate
        api_key: API key (should use env var in production)
        base_url: Base URL for API requests
    """
    provider: str | None = None
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int = 4096
    api_key: str | None = None
    base_url: str | None = None


@dataclass
class EmbeddingConfig:
    """Embedding provider configuration.

    Attributes:
        provider: Embedding provider type (openai, local) - REQUIRED
        model: Model name - REQUIRED
        dimensions: Vector dimensions
        api_key: API key for remote providers
    """
    provider: str | None = None
    model: str | None = None
    dimensions: int = 1536
    api_key: str | None = None


@dataclass
class VectorStoreConfig:
    """Vector store configuration.

    Attributes:
        provider: Vector store type (chroma, qdrant, pinecone) - REQUIRED
        persist_directory: Directory for persistent storage
        collection_name: Default collection name
    """
    provider: str | None = None
    persist_directory: str = "./data/db/chroma"
    collection_name: str = "knowledge_hub"


@dataclass
class RetrievalConfig:
    """Retrieval pipeline configuration.

    Attributes:
        dense_top_k: Top-K results from dense retrieval
        sparse_top_k: Top-K results from sparse retrieval
        fusion_top_k: Final top-K after fusion
        rrf_k: RRF ranking constant
    """
    dense_top_k: int = 20
    sparse_top_k: int = 20
    fusion_top_k: int = 10
    rrf_k: int = 60


@dataclass
class RerankConfig:
    """Reranker configuration.

    Attributes:
        enabled: Whether reranking is enabled
        provider: Reranker type (none, cross_encoder, llm)
        model: Model name
        top_k: Number of results to return after reranking
    """
    enabled: bool = False
    provider: str = "none"
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 5


@dataclass
class EvaluationConfig:
    """Evaluation configuration.

    Attributes:
        enabled: Whether evaluation is enabled
        provider: Evaluation provider (ragas, deepeval, custom)
        metrics: List of metrics to compute
    """
    enabled: bool = False
    provider: str = "custom"
    metrics: list[str] = field(default_factory=lambda: ["hit_rate", "mrr"])


@dataclass
class ObservabilityConfig:
    """Observability and logging configuration.

    Attributes:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        trace_enabled: Whether tracing is enabled
        trace_file: Path to trace output file
        structured_logging: Whether to use structured JSON logging
    """
    log_level: str = "INFO"
    trace_enabled: bool = True
    trace_file: str = "./logs/traces.jsonl"
    structured_logging: bool = True


@dataclass
class ChunkRefinerConfig:
    """Chunk refinement configuration.

    Attributes:
        use_llm: Whether to use LLM-based refinement
        prompt_path: Optional path to custom prompt template
    """
    use_llm: bool = False
    prompt_path: str | None = None


@dataclass
class IngestionConfig:
    """Ingestion pipeline configuration.

    Attributes:
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        splitter: Splitter type (recursive, semantic, fixed_length)
        batch_size: Batch size for processing
        chunk_refiner: Chunk refinement configuration
    """
    chunk_size: int = 1000
    chunk_overlap: int = 200
    splitter: str = "recursive"
    batch_size: int = 100
    chunk_refiner: ChunkRefinerConfig = field(default_factory=ChunkRefinerConfig)


@dataclass
class Settings:
    """Application settings container.

    This dataclass holds all configuration for the Modular RAG MCP Server.
    It is designed to be immutable after loading.

    Attributes:
        llm: LLM configuration
        embedding: Embedding configuration
        vector_store: Vector store configuration
        retrieval: Retrieval configuration
        rerank: Reranking configuration
        evaluation: Evaluation configuration
        observability: Observability configuration
        ingestion: Ingestion configuration
    """
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)


class SettingsError(Exception):
    """Base exception for settings-related errors."""

    pass


class SettingsFileError(SettingsError):
    """Raised when settings file cannot be read or parsed."""

    pass


class SettingsValidationError(SettingsError):
    """Raised when settings validation fails."""

    def __init__(self, message: str, missing_fields: list[str] | None = None) -> None:
        super().__init__(message)
        self.missing_fields = missing_fields or []


def _get_field_path(prefix: str, field_name: str) -> str:
    """Generate a dot-notation field path.

    Args:
        prefix: Parent section name (e.g., 'llm', 'embedding')
        field_name: Field name within the section

    Returns:
        Dot-notation path (e.g., 'llm.provider')
    """
    return f"{prefix}.{field_name}"


def _flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dictionary to dot-notation keys.

    Args:
        d: Dictionary to flatten
        prefix: Prefix for nested keys

    Returns:
        Flattened dictionary with dot-notation keys
    """
    result: dict[str, Any] = {}
    for key, value in d.items():
        new_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_flatten_dict(value, new_key))
        else:
            result[new_key] = value
    return result


def _get_required_fields() -> dict[str, tuple[type, Any]]:
    """Define required fields with their types and defaults.

    Returns:
        Dictionary mapping field paths to (type, default) tuples.
        Required fields (without defaults) have default=None.
    """
    return {
        # Truly required - must be provided (no defaults)
        "llm.provider": (str, None),
        "llm.model": (str, None),
        "embedding.provider": (str, None),
        # Optional with defaults - these can be omitted
        "embedding.model": (str, "text-embedding-3-small"),
        "vector_store.provider": (str, "chroma"),
        "retrieval.dense_top_k": (int, 20),
        "retrieval.sparse_top_k": (int, 20),
        "retrieval.fusion_top_k": (int, 10),
        "rerank.enabled": (bool, False),
        "rerank.provider": (str, "none"),
        "evaluation.enabled": (bool, False),
        "evaluation.provider": (str, "custom"),
        "observability.log_level": (str, "INFO"),
        "observability.trace_enabled": (bool, True),
        "ingestion.chunk_size": (int, 1000),
        "ingestion.chunk_overlap": (int, 200),
        "ingestion.splitter": (str, "recursive"),
    }


def validate_settings(settings: Settings) -> None:
    """Validate required fields in settings.

    This function checks that all required configuration fields are present.
    Missing required fields raise a SettingsValidationError with a clear message.

    Args:
        settings: Settings object to validate

    Raises:
        SettingsValidationError: If required fields are missing

    Example:
        >>> settings = load_settings("config/settings.yaml")
        >>> validate_settings(settings)  # May raise if required fields missing
    """
    required_fields = _get_required_fields()
    missing: list[str] = []

    for field_path, (field_type, _) in required_fields.items():
        parts = field_path.split(".")
        current: Any = settings

        # Navigate to the parent object
        for part in parts[:-1]:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                missing.append(field_path)
                break
        else:
            # Check the final field exists and is not None
            final_field = parts[-1]
            if hasattr(current, final_field):
                value = getattr(current, final_field)
                if value is None:
                    missing.append(field_path)
            else:
                missing.append(field_path)

    if missing:
        field_list = ", ".join(missing)
        raise SettingsValidationError(
            f"Missing required configuration fields: {field_list}",
            missing_fields=missing
        )


def _yaml_to_settings(data: dict[str, Any]) -> Settings:
    """Convert YAML dictionary to Settings object.

    Args:
        data: Parsed YAML dictionary

    Returns:
        Settings object
    """
    def _build_llm(data: dict[str, Any]) -> LLMConfig:
        return LLMConfig(
            provider=data.get("provider", "openai"),
            model=data.get("model", "gpt-4o-mini"),
            temperature=data.get("temperature", 0.0),
            max_tokens=data.get("max_tokens", 4096),
            api_key=data.get("api_key"),
            base_url=data.get("base_url"),
        )

    def _build_embedding(data: dict[str, Any]) -> EmbeddingConfig:
        return EmbeddingConfig(
            provider=data.get("provider", "openai"),
            model=data.get("model", "text-embedding-3-small"),
            dimensions=data.get("dimensions", 1536),
            api_key=data.get("api_key"),
        )

    def _build_vector_store(data: dict[str, Any]) -> VectorStoreConfig:
        return VectorStoreConfig(
            provider=data.get("provider", "chroma"),
            persist_directory=data.get("persist_directory", "./data/db/chroma"),
            collection_name=data.get("collection_name", "knowledge_hub"),
        )

    def _build_retrieval(data: dict[str, Any]) -> RetrievalConfig:
        return RetrievalConfig(
            dense_top_k=data.get("dense_top_k", 20),
            sparse_top_k=data.get("sparse_top_k", 20),
            fusion_top_k=data.get("fusion_top_k", 10),
            rrf_k=data.get("rrf_k", 60),
        )

    def _build_rerank(data: dict[str, Any]) -> RerankConfig:
        return RerankConfig(
            enabled=data.get("enabled", False),
            provider=data.get("provider", "none"),
            model=data.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            top_k=data.get("top_k", 5),
        )

    def _build_evaluation(data: dict[str, Any]) -> EvaluationConfig:
        return EvaluationConfig(
            enabled=data.get("enabled", False),
            provider=data.get("provider", "custom"),
            metrics=data.get("metrics", ["hit_rate", "mrr"]),
        )

    def _build_observability(data: dict[str, Any]) -> ObservabilityConfig:
        return ObservabilityConfig(
            log_level=data.get("log_level", "INFO"),
            trace_enabled=data.get("trace_enabled", True),
            trace_file=data.get("trace_file", "./logs/traces.jsonl"),
            structured_logging=data.get("structured_logging", True),
        )

    def _build_chunk_refiner(data: dict[str, Any]) -> ChunkRefinerConfig:
        return ChunkRefinerConfig(
            use_llm=data.get("use_llm", False),
            prompt_path=data.get("prompt_path"),
        )

    def _build_ingestion(data: dict[str, Any]) -> IngestionConfig:
        return IngestionConfig(
            chunk_size=data.get("chunk_size", 1000),
            chunk_overlap=data.get("chunk_overlap", 200),
            splitter=data.get("splitter", "recursive"),
            batch_size=data.get("batch_size", 100),
            chunk_refiner=_build_chunk_refiner(data.get("chunk_refiner", {})),
        )

    return Settings(
        llm=_build_llm(data.get("llm", {})),
        embedding=_build_embedding(data.get("embedding", {})),
        vector_store=_build_vector_store(data.get("vector_store", {})),
        retrieval=_build_retrieval(data.get("retrieval", {})),
        rerank=_build_rerank(data.get("rerank", {})),
        evaluation=_build_evaluation(data.get("evaluation", {})),
        observability=_build_observability(data.get("observability", {})),
        ingestion=_build_ingestion(data.get("ingestion", {})),
    )


def load_settings(path: str | Path = "config/settings.yaml") -> Settings:
    """Load settings from a YAML file.

    This function reads the YAML configuration file, parses it, and returns
    a Settings object. Required fields are validated after loading.

    Args:
        path: Path to the settings YAML file (default: config/settings.yaml)

    Returns:
        Settings object with all configuration loaded

    Raises:
        SettingsFileError: If the file cannot be read or parsed
        SettingsValidationError: If required fields are missing

    Example:
        >>> settings = load_settings()
        >>> print(settings.llm.provider)
        openai
    """
    path = Path(path)

    # Read and parse YAML
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError as e:
        raise SettingsFileError(f"Settings file not found: {path}") from e
    except yaml.YAMLError as e:
        raise SettingsFileError(f"Invalid YAML in settings file: {e}") from e

    # Convert to Settings object
    settings = _yaml_to_settings(data)

    # Validate required fields
    validate_settings(settings)

    return settings


def get_effective_settings(
    path: str | Path = "config/settings.yaml",
    overrides: dict[str, Any] | None = None
) -> Settings:
    """Load settings with optional runtime overrides.

    This is a convenience function for loading settings and applying
    runtime overrides for testing or debugging purposes.

    Args:
        path: Path to the settings YAML file
        overrides: Optional dictionary of field paths to override.
                   Use dot-notation (e.g., {"llm.provider": "ollama"})

    Returns:
        Settings object with overrides applied

    Example:
        >>> settings = get_effective_settings(
        ...     overrides={"llm.provider": "ollama"}
        ... )
        >>> settings.llm.provider
        ollama
    """
    settings = load_settings(path)

    if overrides:
        flat_overrides = _flatten_dict(overrides)
        for field_path, value in flat_overrides.items():
            parts = field_path.split(".")
            obj = settings
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)

    return settings
