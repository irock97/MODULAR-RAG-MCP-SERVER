"""Local Embedding implementation using sentence-transformers or Ollama.

This module provides local embedding implementations that follow the BaseEmbedding interface.
It supports:
- sentence-transformers models (BGE, etc.)
- Ollama embeddings API
- Fake embeddings for testing

Design Principles:
    - Local-first: Runs models locally without API calls
    - Pluggable: Follows BaseEmbedding interface
    - Testable: Supports fake embeddings for deterministic testing
"""

from abc import ABC, abstractmethod
from typing import Any

from libs.embedding.base_embedding import (
    BaseEmbedding,
    EmbeddingResult,
    EmbeddingConfigurationError,
    EmbeddingError,
)
from observability.logger import TraceContext, get_logger

logger = get_logger(__name__)


class BaseLocalEmbedding(BaseEmbedding, ABC):
    """Abstract base class for local embedding providers.

    Provides common functionality for sentence-transformers and Ollama embeddings.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this provider."""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the embedding dimensions."""
        ...


class SentenceTransformersEmbedding(BaseLocalEmbedding):
    """Sentence-transformers embedding implementation.

    Uses HuggingFace sentence-transformers library to run local embedding models
    such as BGE, E5, and other open-source embedding models.

    Attributes:
        model_name: Name or path of the sentence-transformers model
        device: Device to run the model on ('cpu', 'cuda', 'mps')
        normalize: Whether to normalize embedding vectors
        prompt_name: Optional prompt name for instruction-tuned models
        prompt: Optional prompt template for instruction-tuned models
    """

    # Default model for local embeddings
    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
    # Default dimensions for the default model
    DEFAULT_DIMENSIONS = 384

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        normalize: bool = True,
        prompt_name: str | None = None,
        prompt: str | None = None,
    ) -> None:
        """Initialize the Sentence-transformers Embedding.

        Args:
            model_name: Model name or path. Defaults to BAAI/bge-small-en-v1.5.
            device: Device to run on ('cpu', 'cuda', 'mps'). Auto-detected if None.
            normalize: Whether to normalize output vectors (default: True).
            prompt_name: Prompt name for instruction-tuned models.
            prompt: Custom prompt template.

        Raises:
            EmbeddingConfigurationError: If sentence-transformers is not installed.
        """
        self._model_name = model_name or self.DEFAULT_MODEL
        self._device = device
        self._normalize = normalize
        self._prompt_name = prompt_name
        self._prompt = prompt
        self._model = None

        # Try to import sentence_transformers
        try:
            from sentence_transformers import SentenceTransformer
            self._SentenceTransformer = SentenceTransformer
        except ImportError:
            raise EmbeddingConfigurationError(
                "sentence-transformers is not installed. "
                "Install it with: pip install sentence-transformers",
                provider="sentence-transformers"
            )

    @property
    def provider_name(self) -> str:
        """Return the name of this provider.

        Returns:
            Provider identifier: 'sentence-transformers'
        """
        return "sentence-transformers"

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensions.

        Returns:
            Embedding dimensions based on model configuration.
        """
        return self._get_model().get_sentence_embedding_dimension()

    def _get_model(self):
        """Lazy-load the model.

        Returns:
            Loaded SentenceTransformer model.
        """
        if self._model is None:
            self._model = self._SentenceTransformer(
                self._model_name,
                device=self._device,
                prompts={self._prompt_name: self._prompt} if self._prompt_name else None,
            )
        return self._model

    def embed(
        self,
        texts: list[str],
        trace: TraceContext | None = None,
        **kwargs: Any
    ) -> EmbeddingResult:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.
            trace: Tracing context for observability.
            **kwargs: Additional provider-specific arguments.

        Returns:
            EmbeddingResult containing list of vectors.

        Raises:
            EmbeddingError: If embedding fails.
        """
        if not texts:
            return EmbeddingResult(vectors=[])

        logger.info(
            f"Sentence-transformers embedding request: model={self._model_name}, "
            f"text_count={len(texts)}"
        )

        if trace:
            trace.record_stage(
                "embedding_request",
                {
                    "provider": self.provider_name,
                    "model": self._model_name,
                    "text_count": len(texts)
                }
            )

        try:
            model = self._get_model()

            # Encode texts
            embeddings = model.encode(
                texts,
                normalize_embeddings=self._normalize,
                show_progress_bar=False,
            )

            # Convert to list of lists
            vectors: list[list[float]] = embeddings.tolist()

            logger.info(
                f"Sentence-transformers embedding response: vector_count={len(vectors)}, "
                f"dimensions={self.dimensions}"
            )

            if trace:
                trace.record_stage(
                    "embedding_response",
                    {
                        "vector_count": len(vectors),
                        "dimensions": self.dimensions,
                    }
                )

            return EmbeddingResult(vectors=vectors)

        except Exception as e:
            raise EmbeddingError(
                f"Failed to generate embeddings: {e}",
                provider=self.provider_name,
                details={"model": self._model_name, "error": str(e)}
            )

    def embed_single(
        self,
        text: str,
        trace: TraceContext | None = None,
        **kwargs: Any
    ) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Single text string to embed.
            trace: Tracing context for observability.
            **kwargs: Additional arguments.

        Returns:
            Single embedding vector.
        """
        if not text:
            return []

        result = self.embed([text], trace, **kwargs)

        if result.vectors:
            return result.vectors[0]

        return []

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"SentenceTransformersEmbedding("
            f"provider={self.provider_name}, "
            f"model={self._model_name}, "
            f"dimensions={self.dimensions})"
        )


class OllamaEmbedding(BaseLocalEmbedding):
    """Ollama embeddings API implementation.

    Connects to a local Ollama server to generate embeddings using
    Ollama's embeddings API format.

    Attributes:
        base_url: Base URL of the Ollama server
        model: Model name to use
        timeout: Request timeout in seconds
        http_client: Optional HTTP client for custom configuration
    """

    # Default Ollama embedding model
    DEFAULT_MODEL = "nomic-embed-text"
    # Default dimensions for nomic-embed-text
    DEFAULT_DIMENSIONS = 768

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        http_client: Any | None = None,
    ) -> None:
        """Initialize the Ollama Embedding.

        Args:
            base_url: Ollama server URL. Required if OLLAMA_BASE_URL env var not set.
            model: Model name. Defaults to nomic-embed-text.
            timeout: Request timeout in seconds.
            http_client: Optional HTTP client.

        Raises:
            EmbeddingConfigurationError: If base_url is not configured.
        """
        import os

        # Get base_url from parameter or environment
        env_base_url = os.getenv("OLLAMA_BASE_URL")

        # base_url parameter takes precedence over environment
        if base_url is not None:
            self._base_url = base_url
        elif env_base_url is not None:
            self._base_url = env_base_url
        else:
            self._base_url = None

        self._model = model or self.DEFAULT_MODEL
        self._timeout = timeout
        self._http_client = http_client

        if not self._base_url:
            raise EmbeddingConfigurationError(
                "Ollama base_url is required. Set 'base_url' in settings or "
                "OLLAMA_BASE_URL env var.",
                provider="ollama"
            )

    @property
    def provider_name(self) -> str:
        """Return the name of this provider.

        Returns:
            Provider identifier: 'ollama'
        """
        return "ollama"

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensions.

        Returns:
            Default embedding dimensions.
        """
        return self.DEFAULT_DIMENSIONS

    def _parse_response(self, response_data: dict[str, Any]) -> EmbeddingResult:
        """Parse Ollama API response into EmbeddingResult.

        Args:
            response_data: Raw API response dictionary.

        Returns:
            Parsed EmbeddingResult object.
        """
        embedding = response_data.get("embedding", [])
        if not embedding:
            raise EmbeddingError(
                "Empty response from Ollama Embeddings API",
                provider=self.provider_name,
                details=response_data
            )

        return EmbeddingResult(
            vectors=[embedding],
        )

    def embed(
        self,
        texts: list[str],
        trace: TraceContext | None = None,
        **kwargs: Any
    ) -> EmbeddingResult:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.
            trace: Tracing context for observability.
            **kwargs: Additional provider-specific arguments.

        Returns:
            EmbeddingResult containing list of vectors.

        Raises:
            EmbeddingError: If embedding fails.
        """
        if not texts:
            return EmbeddingResult(vectors=[])

        logger.info(
            f"Ollama embedding request: model={self._model}, "
            f"text_count={len(texts)}"
        )

        if trace:
            trace.record_stage(
                "embedding_request",
                {
                    "provider": self.provider_name,
                    "model": self._model,
                    "text_count": len(texts)
                }
            )

        import httpx

        url = f"{self._base_url}/api/embeddings"
        payload = {
            "model": self._model,
            "input": texts,
        }

        client = self._http_client or httpx.Client(timeout=self._timeout)

        try:
            response = client.post(url, json=payload)
            response.raise_for_status()
            response_data = response.json()

            result = self._parse_response(response_data)

            logger.info(
                f"Ollama embedding response: vector_count={len(result.vectors)}, "
                f"dimensions={self.dimensions}"
            )

            if trace:
                trace.record_stage(
                    "embedding_response",
                    {
                        "vector_count": len(result.vectors),
                        "dimensions": self.dimensions,
                    }
                )

            return result

        except httpx.HTTPStatusError as e:
            try:
                error_data = e.response.json()
                error_message = error_data.get("error", str(e))
            except Exception:
                error_message = str(e)

            raise EmbeddingError(
                f"Ollama API error: {error_message}",
                provider=self.provider_name,
                code=e.response.status_code,
            )
        except httpx.RequestError as e:
            raise EmbeddingError(
                f"Failed to connect to Ollama API: {e}",
                provider=self.provider_name,
                details={"url": url, "error": str(e)}
            )
        finally:
            if self._http_client is None:
                client.close()

    def embed_single(
        self,
        text: str,
        trace: TraceContext | None = None,
        **kwargs: Any
    ) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Single text string to embed.
            trace: Tracing context for observability.
            **kwargs: Additional arguments.

        Returns:
            Single embedding vector.
        """
        if not text:
            return []

        result = self.embed([text], trace, **kwargs)

        if result.vectors:
            return result.vectors[0]

        return []

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"OllamaEmbedding("
            f"provider={self.provider_name}, "
            f"model={self._model}, "
            f"dimensions={self.dimensions})"
        )


class FakeEmbedding(BaseLocalEmbedding):
    """Fake embedding for testing.

    Generates deterministic fake embeddings for testing purposes.
    All embeddings are normalized vectors of the specified dimension.

    Attributes:
        dimensions: Dimensions of the fake embeddings
        seed: Random seed for reproducibility
    """

    DEFAULT_DIMENSIONS = 384

    def __init__(
        self,
        dimensions: int | None = None,
        seed: int = 42,
    ) -> None:
        """Initialize the Fake Embedding.

        Args:
            dimensions: Embedding dimensions. Defaults to 384.
            seed: Random seed for reproducibility.
        """
        self._dimensions = dimensions or self.DEFAULT_DIMENSIONS
        self._seed = seed
        self._random = __import__("random")
        self._random.seed(seed)
        self._math = __import__("math")

    @property
    def provider_name(self) -> str:
        """Return the name of this provider.

        Returns:
            Provider identifier: 'fake'
        """
        return "fake"

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensions.

        Returns:
            Configured embedding dimensions.
        """
        return self._dimensions

    def _generate_deterministic_vector(self, text: str) -> list[float]:
        """Generate a deterministic fake vector based on text content.

        Args:
            text: Input text to generate vector for.

        Returns:
            Deterministic fake embedding vector.
        """
        # Use text hash to generate reproducible vectors
        hash_val = hash(text) % (2**31)
        self._random.seed(hash_val)

        # Generate random values
        vector = [self._random.uniform(-1, 1) for _ in range(self._dimensions)]

        # Normalize
        norm = self._math.sqrt(sum(x * x for x in vector))
        if norm > 0:
            vector = [x / norm for x in vector]

        return vector

    def embed(
        self,
        texts: list[str],
        trace: TraceContext | None = None,
        **kwargs: Any
    ) -> EmbeddingResult:
        """Generate fake embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.
            trace: Tracing context for observability.
            **kwargs: Additional provider-specific arguments.

        Returns:
            EmbeddingResult containing fake vectors.
        """
        if not texts:
            return EmbeddingResult(vectors=[])

        logger.debug(
            f"Fake embedding request: text_count={len(texts)}, "
            f"dimensions={self._dimensions}"
        )

        vectors = [self._generate_deterministic_vector(text) for text in texts]

        return EmbeddingResult(vectors=vectors)

    def embed_single(
        self,
        text: str,
        trace: TraceContext | None = None,
        **kwargs: Any
    ) -> list[float]:
        """Generate fake embedding for a single text.

        Args:
            text: Single text string to embed.
            trace: Tracing context for observability.
            **kwargs: Additional arguments.

        Returns:
            Single fake embedding vector.
        """
        if not text:
            return []

        return self._generate_deterministic_vector(text)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"FakeEmbedding("
            f"provider={self.provider_name}, "
            f"dimensions={self.dimensions}, "
            f"seed={self._seed})"
        )


# Alias for backward compatibility
LocalEmbedding = FakeEmbedding
