"""Azure Vision LLM implementation.

This module provides the AzureVisionLLM class that implements the BaseVisionLLM
interface for Azure OpenAI's GPT-4 Vision capabilities.

Design Principles:
    - Azure-compatible: Follows Azure OpenAI API conventions
    - Type Safe: Full type hints for all methods
    - Observable: trace parameter for tracing integration
    - Configurable: Supports all Azure API parameters
    - Auto-compression: Images are automatically compressed to max_image_size
"""

import base64
import importlib
from io import BytesIO
from typing import Any

import httpx
from PIL import Image

from libs.llm.base_vision_llm import (
    BaseVisionLLM,
    ImageInput,
    VisionLLMConfigurationError,
    VisionLLMError,
)
from observability.logger import TraceContext, get_logger

logger = get_logger(__name__)


class AzureVisionLLM(BaseVisionLLM):
    """Azure OpenAI Vision LLM implementation.

    This class implements the BaseVisionLLM interface for Azure OpenAI's
    GPT-4 Vision API. It supports both image paths and base64-encoded images.

    Attributes:
        azure_endpoint: Azure OpenAI endpoint URL
        api_version: Azure API version
        deployment_name: Deployment name for the vision model
        api_key: Azure API key
        max_image_size: Maximum image dimension (width, height) for compression
        timeout: Request timeout in seconds

    Example:
        >>> llm = AzureVisionLLM(
        ...     azure_endpoint="https://my-resource.openai.azure.com/",
        ...     api_version="2024-02-15-preview",
        ...     deployment_name="gpt-4o",
        ...     api_key="..."
        ... )
        >>> response = llm.chat_with_image("Describe this image", "image.png")
        >>> print(response.content)
    """

    def __init__(
        self,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        deployment_name: str | None = None,
        api_key: str | None = None,
        max_image_size: int = 2048,
        timeout: float = 30.0,
        http_client: httpx.Client | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """Initialize the Azure Vision LLM.

        Args:
            azure_endpoint: Azure OpenAI endpoint URL
            api_version: Azure API version string
            deployment_name: Deployment name for the vision model
            api_key: Azure API key. If None, reads from AZURE_OPENAI_API_KEY env var.
            max_image_size: Maximum image dimension for auto-compression (default: 2048)
            timeout: Request timeout in seconds
            http_client: Optional pre-configured HTTP client
            model: Model name (for compatibility with LLMFactory)
            temperature: Sampling temperature (for compatibility)
            max_tokens: Maximum tokens (for compatibility)

        Raises:
            VisionLLMConfigurationError: If required Azure config is missing
        """
        self._azure_endpoint = azure_endpoint
        self._api_version = api_version or "2024-02-15-preview"
        self._deployment_name = deployment_name
        self._api_key = api_key
        self._max_image_size = max_image_size
        self._timeout = timeout
        self._http_client = http_client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

        # Validate required Azure config
        if not self._azure_endpoint:
            raise VisionLLMConfigurationError(
                "Azure endpoint is not configured. Set 'azure_endpoint' or AZURE_OPENAI_ENDPOINT env var.",
                provider="azure"
            )

        if not self._deployment_name:
            raise VisionLLMConfigurationError(
                "Azure deployment name is not configured. Set 'deployment_name' in settings.",
                provider="azure"
            )

        # Get API key from environment if not provided
        if not self._api_key:
            import os
            self._api_key = os.getenv("AZURE_OPENAI_API_KEY")
            if not self._api_key:
                raise VisionLLMConfigurationError(
                    "Azure API key is not configured. Set 'api_key' or AZURE_OPENAI_API_KEY env var.",
                    provider="azure"
                )

    @property
    def provider_name(self) -> str:
        """Return the name of this provider.

        Returns:
            Provider identifier: 'azure-vision'
        """
        return "azure-vision"

    @property
    def supported_formats(self) -> list[str]:
        """Return list of supported image formats.

        Returns:
            List of supported MIME types
        """
        return ["image/png", "image/jpeg", "image/gif", "image/webp"]

    @property
    def max_image_size(self) -> tuple[int, int]:
        """Return maximum supported image dimensions.

        Returns:
            Tuple of (max_width, max_height)
        """
        return (self._max_image_size, self._max_image_size)

    def _compress_image(self, image_data: bytes) -> bytes:
        """Compress image to max_image_size dimensions.

        Args:
            image_data: Raw image bytes

        Returns:
            Compressed image bytes in PNG format

        Raises:
            ValueError: If image format is not supported
        """
        try:
            # Open and process image
            img = Image.open(BytesIO(image_data))

            # Convert to RGB if necessary (for JPEG compatibility)
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            # Calculate scaling factor to fit within max_image_size
            original_width, original_height = img.size
            max_dim = self._max_image_size

            if original_width > max_dim or original_height > max_dim:
                # Scale down to fit within max_image_size while maintaining aspect ratio
                scaling_factor = min(max_dim / original_width, max_dim / original_height)
                new_width = int(original_width * scaling_factor)
                new_height = int(original_height * scaling_factor)

                img = img.resize((new_width, new_height), Image.LANCZOS)
                logger.info(
                    f"Compressed image: {original_width}x{original_height} -> "
                    f"{new_width}x{new_height}"
                )

            # Save to PNG bytes
            output = BytesIO()
            img.save(output, format="PNG", optimize=True)
            compressed_data = output.getvalue()

            logger.info(
                f"Image compression: {len(image_data)} bytes -> {len(compressed_data)} bytes"
            )
            return compressed_data

        except Exception as e:
            logger.error(f"Failed to compress image: {e}")
            # Return original data if compression fails
            return image_data

    def _encode_image(self, image: ImageInput | str | bytes) -> str:
        """Encode image to base64 string.

        Args:
            image: Image as ImageInput, path (str), or bytes

        Returns:
            Base64-encoded image string with data URI prefix

        Raises:
            UnsupportedImageFormatError: If image format is not supported
            FileNotFoundError: If image path doesn't exist
        """
        # Convert to ImageInput if needed
        if isinstance(image, str):
            # String could be path or base64
            if self._is_likely_base64(image):
                image_input = ImageInput(base64=image)
            else:
                image_input = ImageInput(path=image)
        elif isinstance(image, bytes):
            image_input = ImageInput(base64=base64.b64encode(image).decode())
        elif isinstance(image, ImageInput):
            image_input = image
        else:
            raise VisionLLMError(
                f"Unsupported image type: {type(image)}",
                provider=self.provider_name
            )

        # Get image bytes
        image_data = image_input.get_bytes()

        # Compress if needed
        compressed_data = self._compress_image(image_data)

        # Encode to base64 with PNG data URI
        b64_data = base64.b64encode(compressed_data).decode("utf-8")
        mime_type = image_input.mime_type if image_input.mime_type else "image/png"

        return f"data:{mime_type};base64,{b64_data}"

    def _build_request_payload(
        self,
        text: str,
        image_url: str,
    ) -> dict[str, Any]:
        """Build the request payload for Azure Vision API.

        Args:
            text: The text prompt
            image_url: Base64-encoded image URL

        Returns:
            Request payload dictionary
        """
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                    ],
                }
            ],
            "max_tokens": 4096,
        }

    def _parse_response(self, response_data: dict[str, Any]) -> str:
        """Parse Azure Vision API response.

        Args:
            response_data: Raw API response dictionary

        Returns:
            Extracted text content from response

        Raises:
            VisionLLMError: If response is invalid or empty
        """
        try:
            choices = response_data.get("choices", [])
            if not choices:
                raise VisionLLMError(
                    "Empty response from Azure Vision API",
                    provider=self.provider_name,
                    details=response_data
                )

            message = choices[0].get("message", {})
            content = message.get("content", "")

            if not content:
                raise VisionLLMError(
                    "No content in Azure Vision API response",
                    provider=self.provider_name,
                    details=response_data
                )

            return content

        except (KeyError, TypeError) as e:
            raise VisionLLMError(
                f"Failed to parse Azure Vision response: {e}",
                provider=self.provider_name,
                details={"response": response_data}
            )

    def _handle_http_error(self, error: httpx.HTTPStatusError) -> None:
        """Handle HTTP errors from Azure API.

        Args:
            error: HTTP status error

        Raises:
            VisionLLMError: Structured error with Azure-specific details
        """
        try:
            error_data = error.response.json()
            error_message = error_data.get("error", {}).get("message", str(error))
            error_code = error_data.get("error", {}).get("code", "unknown")

            # Map common Azure error codes
            azure_error_messages = {
                "invalid_api_key": "Invalid Azure API key. Please check your credentials.",
                "rate_limited": "Azure API rate limit exceeded. Please try again later.",
                "quota_exceeded": "Azure quota exceeded. Please check your subscription.",
                "deployment_not_found": f"Deployment '{self._deployment_name}' not found. Please verify deployment name.",
                "invalid_image": "Invalid image format or corrupted image file.",
            }

            user_message = azure_error_messages.get(error_code, error_message)

            raise VisionLLMError(
                f"Azure Vision API error: {user_message}",
                provider=self.provider_name,
                code=error.response.status_code,
                details={
                    "azure_error_code": error_code,
                    "status_code": error.response.status_code,
                    "response_body": error_data
                }
            )

        except Exception:
            raise VisionLLMError(
                f"Azure Vision API error: HTTP {error.response.status_code}",
                provider=self.provider_name,
                code=error.response.status_code,
                details={"status_code": error.response.status_code}
            )

    def chat_with_image(
        self,
        text: str,
        image: ImageInput | str | bytes,
        trace: TraceContext | None = None,
        **kwargs: Any
    ) -> "VisionResponse":
        """Send a text + image request to Azure Vision API.

        Args:
            text: The text prompt to send with the image
            image: Image to process (path, base64, or ImageInput)
            trace: Tracing context for observability
            **kwargs: Additional provider-specific arguments
                - temperature: Sampling temperature (0.0-2.0)
                - max_tokens: Maximum tokens to generate

        Returns:
            VisionResponse containing the generated text

        Raises:
            VisionLLMError: On API errors with Azure-specific codes
            UnsupportedImageFormatError: If image format is not supported
            ImageTooLargeError: If image processing fails
        """
        # Log request start
        logger.info(
            f"Azure Vision request: deployment={self._deployment_name}"
        )

        # Record trace if provided
        if trace:
            trace.record_stage(
                "vision_request",
                {
                    "provider": self.provider_name,
                    "deployment": self._deployment_name
                }
            )

        try:
            # Encode image
            image_url = self._encode_image(image)

            # Build payload
            payload = self._build_request_payload(text, image_url)

            # Build URL
            url = (
                f"{self._azure_endpoint}/openai/deployments/{self._deployment_name}"
                f"/chat/completions?api-version={self._api_version}"
            )

            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "User-Agent": "ModularRAG-MCP-Server/1.0",
            }

            # Make request - use http_client from kwargs or from self
            http_client_param = kwargs.get("http_client")
            client = http_client_param or self._http_client or httpx.Client(timeout=self._timeout)

            try:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                response_data = response.json()

                # Parse response
                content = self._parse_response(response_data)

                # Extract usage if available
                usage = response_data.get("usage", {})
                usage_info = {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }

                # Record trace
                if trace:
                    trace.record_stage(
                        "vision_response",
                        {"content_length": len(content)}
                    )

                logger.info(
                    f"Azure Vision response: content_length={len(content)}, "
                    f"tokens={usage_info}"
                )

                from libs.llm.base_vision_llm import VisionResponse
                return VisionResponse(
                    content=content,
                    raw_response=response_data,
                    usage=usage_info
                )

            except httpx.HTTPStatusError as e:
                self._handle_http_error(e)
            except httpx.RequestError as e:
                raise VisionLLMError(
                    f"Failed to connect to Azure Vision API: {e}",
                    provider=self.provider_name,
                    details={"url": url, "error": str(e)}
                )
            finally:
                if self._http_client is None:
                    client.close()

        except VisionLLMError:
            raise
        except Exception as e:
            raise VisionLLMError(
                f"Unexpected error in Azure Vision request: {e}",
                provider=self.provider_name,
                details={"error": str(e)}
            )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"AzureVisionLLM("
            f"provider={self.provider_name}, "
            f"deployment={self._deployment_name}, "
            f"endpoint={self._azure_endpoint})"
        )
