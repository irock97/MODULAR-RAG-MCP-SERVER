"""Qwen Vision LLM implementation.

This module provides the QwenVisionLLM class that implements the BaseVisionLLM
interface for Alibaba DashScope's Qwen Vision models (qwen-vl-max, qwen-vl-plus).

Design Principles:
    - DashScope-compatible: Follows DashScope API conventions
    - Type Safe: Full type hints for all methods
    - Observable: trace parameter for tracing integration
    - Configurable: Supports all DashScope API parameters
"""

import base64
import os
from typing import Any

import httpx

from libs.llm.base_vision_llm import (
    BaseVisionLLM,
    ImageInput,
    VisionLLMConfigurationError,
    VisionLLMError,
)
from observability.logger import TraceContext, get_logger

logger = get_logger(__name__)


class QwenVisionLLM(BaseVisionLLM):
    """Qwen Vision LLM implementation using DashScope API.

    This class implements the BaseVisionLLM interface for Alibaba Cloud's
    DashScope service with Qwen Vision models. It supports both image paths
    and base64-encoded images.

    Attributes:
        api_key: DashScope API key
        model: Model name (qwen-vl-max, qwen-vl-plus)
        base_url: DashScope API base URL
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds

    Example:
        >>> llm = QwenVisionLLM(
        ...     api_key="sk-...",
        ...     model="qwen-vl-max"
        ... )
        >>> response = llm.chat_with_image("Describe this image", "image.png")
        >>> print(response.content)
    """

    # DashScope API base URL (compatible mode for OpenAI-style API)
    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # Supported models
    SUPPORTED_MODELS = ["qwen-vl-max", "qwen-vl-plus"]

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: float = 60.0,
        max_image_size: int | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        """Initialize the Qwen Vision LLM.

        Args:
            api_key: DashScope API key. If None, reads from DASHSCOPE_API_KEY env var.
            model: Model name (qwen-vl-max, qwen-vl-plus). Defaults to qwen-vl-max.
            base_url: DashScope API base URL. Defaults to compatible mode endpoint.
            max_tokens: Maximum tokens to generate. Defaults to 2000.
            temperature: Sampling temperature (for compatibility).
            timeout: Request timeout in seconds.
            max_image_size: Maximum image dimension (width, height). Defaults to 4096.
            http_client: Optional pre-configured HTTP client.

        Raises:
            VisionLLMConfigurationError: If required config is missing.
        """
        self._api_key = api_key
        self._model = model or "qwen-vl-max"
        self._base_url = base_url or self.DEFAULT_BASE_URL
        self._max_tokens = max_tokens or 2000
        self._temperature = temperature
        self._timeout = timeout
        self._max_image_size = max_image_size or 4096
        self._http_client = http_client

        # Validate model
        if self._model not in self.SUPPORTED_MODELS:
            logger.warning(
                f"Model {self._model} not in explicitly supported models: "
                f"{self.SUPPORTED_MODELS}. Proceeding anyway."
            )

        # Get API key from environment if not provided
        if not self._api_key:
            self._api_key = os.getenv("DASHSCOPE_API_KEY")
            if not self._api_key:
                raise VisionLLMConfigurationError(
                    "DashScope API key is not configured. Set 'api_key' or "
                    "DASHSCOPE_API_KEY env var.",
                    provider="qwen"
                )

    @property
    def provider_name(self) -> str:
        """Return the name of this provider.

        Returns:
            Provider identifier: 'qwen-vision'
        """
        return "qwen-vision"

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
        # Qwen VL supports up to 4096x4096 by default
        return (self._max_image_size, self._max_image_size)

    def _encode_image(self, image: ImageInput | str | bytes) -> str:
        """Encode image to base64 string.

        Args:
            image: Image as ImageInput, path (str), or bytes

        Returns:
            Base64-encoded image string with data URI prefix
        """
        # Convert to ImageInput if needed
        if isinstance(image, str):
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

        # Encode to base64
        b64_data = base64.b64encode(image_data).decode("utf-8")
        mime_type = image_input.mime_type if image_input.mime_type else "image/png"

        return f"data:{mime_type};base64,{b64_data}"

    def _build_request_payload(
        self,
        text: str,
        image_url: str,
    ) -> dict[str, Any]:
        """Build the request payload for DashScope Vision API.

        Args:
            text: The text prompt
            image_url: Base64-encoded image URL

        Returns:
            Request payload dictionary (OpenAI-compatible format)
        """
        payload: dict[str, Any] = {
            "model": self._model,
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
        }

        # Add optional parameters
        if self._max_tokens:
            payload["max_tokens"] = self._max_tokens
        if self._temperature is not None:
            payload["temperature"] = self._temperature

        return payload

    def _parse_response(self, response_data: dict[str, Any]) -> str:
        """Parse DashScope Vision API response.

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
                    "Empty response from DashScope Vision API",
                    provider=self.provider_name,
                    details=response_data
                )

            message = choices[0].get("message", {})
            content = message.get("content", "")

            if not content:
                raise VisionLLMError(
                    "No content in DashScope Vision API response",
                    provider=self.provider_name,
                    details=response_data
                )

            return content

        except (KeyError, TypeError) as e:
            raise VisionLLMError(
                f"Failed to parse DashScope Vision response: {e}",
                provider=self.provider_name,
                details={"response": response_data}
            )

    def _handle_http_error(self, error: httpx.HTTPStatusError) -> None:
        """Handle HTTP errors from DashScope API.

        Args:
            error: HTTP status error

        Raises:
            VisionLLMError: Structured error with DashScope-specific details
        """
        try:
            error_data = error.response.json()
            error_message = error_data.get("message", str(error))
            error_type = error_data.get("type", "unknown")

            # Map common DashScope error types
            dashscope_error_messages = {
                "invalid_api_key": "Invalid DashScope API key. Please check your credentials.",
                "rate_limited": "DashScope API rate limit exceeded. Please try again later.",
                "quota_exceeded": "DashScope quota exceeded. Please check your subscription.",
                "invalid_model": f"Model '{self._model}' is not available or you don't have access.",
                "invalid_image": "Invalid image format or corrupted image file.",
                "image_too_large": "Image file size exceeds the limit.",
            }

            user_message = dashscope_error_messages.get(error_type, error_message)

            raise VisionLLMError(
                f"DashScope Vision API error: {user_message}",
                provider=self.provider_name,
                code=error.response.status_code,
                details={
                    "dashscope_error_type": error_type,
                    "status_code": error.response.status_code,
                    "response_body": error_data
                }
            )

        except Exception:
            raise VisionLLMError(
                f"DashScope Vision API error: HTTP {error.response.status_code}",
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
        """Send a text + image request to DashScope Vision API.

        Args:
            text: The text prompt to send with the image
            image: Image to process (path, base64, or ImageInput)
            trace: Tracing context for observability
            **kwargs: Additional provider-specific arguments
                - temperature: Sampling temperature (0.0-1.0)
                - max_tokens: Maximum tokens to generate

        Returns:
            VisionResponse containing the generated text

        Raises:
            VisionLLMError: On API errors with DashScope-specific codes
            UnsupportedImageFormatError: If image format is not supported
            ImageTooLargeError: If image processing fails
        """
        # Log request start
        logger.info(
            f"DashScope Vision request: model={self._model}"
        )

        # Record trace if provided
        if trace:
            trace.record_stage(
                "vision_request",
                {
                    "provider": self.provider_name,
                    "model": self._model
                }
            )

        try:
            # Encode image
            image_url = self._encode_image(image)

            # Build payload
            payload = self._build_request_payload(text, image_url)

            # Build URL (compatible mode endpoint)
            url = f"{self._base_url}/chat/completions"

            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "User-Agent": "ModularRAG-MCP-Server/1.0",
            }

            # Make request
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
                    f"DashScope Vision response: content_length={len(content)}, "
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
                    f"Failed to connect to DashScope Vision API: {e}",
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
                f"Unexpected error in DashScope Vision request: {e}",
                provider=self.provider_name,
                details={"error": str(e)}
            )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"QwenVisionLLM("
            f"provider={self.provider_name}, "
            f"model={self._model}, "
            f"endpoint={self._base_url})"
        )
