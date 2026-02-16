"""Abstract base class for Vision LLM providers.

This module defines the BaseVisionLLM interface for multi-modal LLM providers
that can process both text and images. This enables pluggable vision-capable
LLM providers (OpenAI GPT-4V, Azure Vision, Ollama with vision models, etc.).

Design Principles:
    - Multi-modal: Supports text + image input (path or base64)
    - Extensible: Image preprocessing extension points
    - Type Safe: Full type hints for all methods
    - Observable: trace parameter for tracing integration

Usage:
    >>> class OpenAIVisionLLM(BaseVisionLLM):
    ...     def chat_with_image(
    ...         self,
    ...         text: str,
    ...         image: str | bytes,
    ...         trace: TraceContext | None = None
    ...     ) -> VisionResponse:
    ...         # Implementation here
    ...         pass
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from core.trace.trace_context import TraceContext


@dataclass
class VisionResponse:
    """Response from a Vision LLM.

    Attributes:
        content: The text content of the response
        raw_response: The raw response from the provider (if available)
        usage: Token usage information (if available)
        image_size: Tuple of (width, height) of processed image (if available)
    """
    content: str
    raw_response: Any | None = None
    usage: dict[str, int] | None = None
    image_size: tuple[int, int] | None = None


@dataclass
class ImageInput:
    """Wrapper for image input supporting both path and base64.

    Attributes:
        path: File path to the image (preferred for large images)
        base64: Base64 encoded image data (preferred for URLs/memory)
        mime_type: MIME type of the image (e.g., 'image/png', 'image/jpeg')
    """
    path: str | None = None
    base64: str | None = None
    mime_type: str = "image/png"

    def __post_init__(self) -> None:
        """Validate that either path or base64 is provided."""
        if not self.path and not self.base64:
            raise ValueError("Either path or base64 must be provided for ImageInput")

    def get_bytes(self) -> bytes:
        """Get image bytes from path or base64.

        Returns:
            Raw image bytes

        Raises:
            FileNotFoundError: If path doesn't exist
            ValueError: If both path and base64 are invalid
        """
        if self.base64:
            import base64
            return base64.b64decode(self.base64)
        elif self.path:
            with open(self.path, "rb") as f:
                return f.read()
        raise ValueError("No image data available")


class ImagePreprocessor(ABC):
    """Abstract base class for image preprocessing.

    This defines the extension point for image preprocessing such as
    compression, format conversion, resizing, etc.

    Usage:
        >>> class CompressPreprocessor(ImagePreprocessor):
        ...     def process(self, image_data: bytes) -> bytes:
        ...         # Compress image
        ...         return compressed_data
    """

    @abstractmethod
    def process(self, image_data: bytes) -> bytes:
        """Process image data.

        Args:
            image_data: Raw image bytes

        Returns:
            Processed image bytes
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this preprocessor."""
        ...


class BaseVisionLLM(ABC):
    """Abstract base class for Vision LLM providers.

    All vision-capable LLM implementations (OpenAI GPT-4V, Azure Vision,
    Ollama with vision support, etc.) must inherit from this class and
    implement the chat_with_image() method.

    Example:
        >>> class OpenAIVisionLLM(BaseVisionLLM):
        ...     def chat_with_image(
        ...         self,
        ...         text: str,
        ...         image: str | bytes,
        ...         trace: TraceContext | None = None
        ...     ) -> VisionResponse:
        ...         # Implementation here
        ...         pass
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this provider.

        Returns:
            Provider identifier (e.g., 'openai-vision', 'azure-vision')
        """
        ...

    @property
    def supported_formats(self) -> list[str]:
        """Return list of supported image formats.

        Returns:
            List of supported MIME types (e.g., ['image/png', 'image/jpeg'])
        """
        return ["image/png", "image/jpeg", "image/gif", "image/webp"]

    @property
    def max_image_size(self) -> tuple[int, int]:
        """Return maximum supported image dimensions (width, height).

        Returns:
            Tuple of (max_width, max_height)
        """
        return (2048, 2048)

    @property
    def max_token_limit(self) -> int:
        """Return maximum number of images per request.

        Returns:
            Maximum number of images
        """
        return 10

    def preprocess_image(
        self,
        image: ImageInput | str | bytes,
        preprocessor: ImagePreprocessor | None = None
    ) -> bytes:
        """Preprocess image data.

        This is a convenience method that handles image input conversion
        and optional preprocessing.

        Args:
            image: Image as path (str), base64 (str), bytes, or ImageInput
            preprocessor: Optional preprocessor for image processing

        Returns:
            Raw image bytes (preprocessed if preprocessor provided)
        """
        # Convert to ImageInput if needed
        if isinstance(image, ImageInput):
            img_input = image
        elif isinstance(image, str):
            # String could be path or base64 - detect which one
            if self._is_likely_base64(image):
                # Treat as base64
                import base64
                img_input = ImageInput(base64=image)
            else:
                # Treat as file path
                img_input = ImageInput(path=image)
        elif isinstance(image, bytes):
            # Bytes need to be converted to base64 for ImageInput
            import base64
            img_input = ImageInput(base64=base64.b64encode(image).decode())
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Get image bytes
        image_data = img_input.get_bytes()

        # Apply preprocessing if provided
        if preprocessor:
            image_data = preprocessor.process(image_data)

        return image_data

    def _is_likely_base64(self, text: str) -> bool:
        """Check if a string is likely base64 encoded data.

        Args:
            text: String to check

        Returns:
            True if likely base64, False if likely a file path
        """
        import re
        # Base64 strings have a specific character set and length
        # They should match: [A-Za-z0-9+/]+=?
        # File paths typically contain /, \, :, ., - etc.
        if len(text) < 8:
            return False  # Too short to be meaningful base64

        # Check for common file path patterns
        if "/" in text or "\\" in text or ":" in text or text.startswith("."):
            return False  # Likely a file path

        # Check for base64 character pattern
        # Valid: A-Z, a-z, 0-9, +, /, =
        if not re.match(r'^[A-Za-z0-9+/=]+$', text):
            return False

        # Check padding - base64 strings may end with = or ==
        return True

    @abstractmethod
    def chat_with_image(
        self,
        text: str,
        image: ImageInput | str | bytes,
        trace: TraceContext | None = None,
        **kwargs: Any
    ) -> VisionResponse:
        """Send a text + image request to the Vision LLM.

        Args:
            text: The text prompt to send with the image
            image: Image to process (path, base64, or ImageInput)
            trace: Tracing context for observability
            **kwargs: Additional provider-specific arguments
                - temperature: Sampling temperature (0.0-2.0)
                - max_tokens: Maximum tokens to generate
                - preprocessor: ImagePreprocessor instance

        Returns:
            VisionResponse containing the generated text

        Raises:
            VisionLLMError: If the request fails
            UnsupportedFormatError: If image format is not supported
            ImageTooLargeError: If image exceeds size limits
        """
        ...

    def chat_with_images(
        self,
        text: str,
        images: list[ImageInput | str | bytes],
        trace: TraceContext | None = None,
        **kwargs: Any
    ) -> VisionResponse:
        """Send a text + multiple images request to the Vision LLM.

        Args:
            text: The text prompt to send with the images
            images: List of images to process
            trace: Tracing context for observability
            **kwargs: Additional provider-specific arguments

        Returns:
            VisionResponse containing the generated text

        Raises:
            VisionLLMError: If the request fails
        """
        if len(images) > self.max_token_limit:
            raise ValueError(
                f"Too many images: {len(images)} > {self.max_token_limit}"
            )

        # Default implementation: process first image only
        # Override in provider for full multi-image support
        if images:
            return self.chat_with_image(text, images[0], trace, **kwargs)

        raise ValueError("No images provided")


class VisionLLMError(Exception):
    """Base exception for Vision LLM-related errors."""

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        code: int | None = None,
        details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.code = code
        self.details = details or {}


class UnknownVisionLLMProviderError(VisionLLMError):
    """Raised when an unknown Vision LLM provider is specified."""
    pass


class VisionLLMConfigurationError(VisionLLMError):
    """Raised when Vision LLM configuration is invalid."""
    pass


class UnsupportedImageFormatError(VisionLLMError):
    """Raised when image format is not supported."""

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        supported_formats: list[str] | None = None
    ) -> None:
        super().__init__(message, provider)
        self.supported_formats = supported_formats or []


class ImageTooLargeError(VisionLLMError):
    """Raised when image exceeds size limits."""

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        image_size: tuple[int, int] | None = None,
        max_size: tuple[int, int] | None = None
    ) -> None:
        super().__init__(message, provider)
        self.image_size = image_size
        self.max_size = max_size
