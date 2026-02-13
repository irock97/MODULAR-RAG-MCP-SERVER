# Embedding - Embedding interfaces

from libs.embedding.base_embedding import (
    BaseEmbedding,
    EmbeddingResult,
    EmbeddingError,
    EmbeddingConfigurationError,
    UnknownEmbeddingProviderError,
)

from libs.embedding.local_embedding import (
    BaseLocalEmbedding,
    SentenceTransformersEmbedding,
    OllamaEmbedding,
    FakeEmbedding,
    LocalEmbedding,
)

__all__ = [
    # Base
    "BaseEmbedding",
    "EmbeddingResult",
    "EmbeddingError",
    "EmbeddingConfigurationError",
    "UnknownEmbeddingProviderError",
    # Local
    "BaseLocalEmbedding",
    "SentenceTransformersEmbedding",
    "OllamaEmbedding",
    "FakeEmbedding",
    "LocalEmbedding",
]
