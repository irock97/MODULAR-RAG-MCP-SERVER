# Embedding - Embedding interfaces

from libs.embedding.base_embedding import (
    BaseEmbedding,
    EmbeddingResult,
    EmbeddingError,
    EmbeddingConfigurationError,
    UnknownEmbeddingProviderError,
)

from libs.embedding.ollama_embedding import (
    OllamaEmbedding,
)

__all__ = [
    # Base
    "BaseEmbedding",
    "EmbeddingResult",
    "EmbeddingError",
    "EmbeddingConfigurationError",
    "UnknownEmbeddingProviderError",
    # Ollama
    "OllamaEmbedding",
]
