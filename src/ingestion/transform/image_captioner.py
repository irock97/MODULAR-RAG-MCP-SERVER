"""ImageCaptioner - Vision LLM-based image caption generation.

This module provides the ImageCaptioner class that generates descriptive
captions for images in chunks using Vision LLM.

Design Principles:
    - Fail-Safe: Vision LLM failures gracefully fall back to no captions
    - Configurable: Captioning controlled via settings.yaml
    - Traceable: All caption generations logged to trace context
    - Metadata Preserved: Original and caption metadata tracked

Pipeline:
    1. Check if Vision LLM is enabled
    2. For each chunk with images:
       a. Generate caption using Vision LLM
       b. Store caption in chunk.metadata.image_captions
    3. Fallback: If Vision LLM fails/disabled, mark has_unprocessed_images
"""

"""
处理后的chunk格式：
Chunk(
    id="chunk_001",
    text="这是一段文本 [IMAGE: img_123]",  # 文本不变
    metadata={
        "images": [
            {"id": "img_123", "path": "/images/chart.png"},
            {"id": "img_456", "path": "/images/broken.jpg"}  # 失败的图片
        ],
        "image_captioning": {  # 详细配置信息
            "enabled": True,
            "vision_provider": "azure-vision",
            "captions_count": 1,
            "failed_count": 1,
            "failed_images": ["img_456"],
            "reason": None
        },
        "image_captions": {  # 字典格式的图片描述
            "img_123": "一张显示数据增长的折线图"
        },
        "has_unprocessed_images": True,
        "unprocessed_images": ["img_456"]
    }
)
"""

import re
from pathlib import Path
from typing import Any

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.base_transform import BaseTransform
from libs.llm.base_vision_llm import BaseVisionLLM, ImageInput, VisionResponse
from libs.llm.llm_factory import LLMFactory
from observability.logger import get_logger

logger = get_logger(__name__)


class ImageCaptioningError(Exception):
    """Error during image caption generation."""

    pass


class ImageCaptioner(BaseTransform):
    """Generates captions for images in chunks using Vision LLM.

    This class processes chunks that contain image references and generates
    descriptive captions using a Vision LLM. The captions are stored in
    chunk.metadata for retrieval purposes.

    Configuration is controlled by `settings.ingestion.image_captioner.enabled`.
    If enabled but Vision LLM fails, the captioner gracefully falls back
    and marks the chunk with `has_unprocessed_images`.

    Attributes:
        image_captions: Dictionary mapping image_id -> caption
        enabled: Whether caption generation is enabled
    """

    def __init__(
        self,
        settings: Settings,
        vision_llm: BaseVisionLLM | None = None,
        prompt_path: str | Path | None = None,
    ) -> None:
        """Initialize the ImageCaptioner.

        Args:
            settings: Settings object with configuration.
            vision_llm: Optional Vision LLM instance. If None, created from settings.
            prompt_path: Optional path to prompt template file.
        """
        super().__init__(settings)

        self._vision_llm = vision_llm
        self._prompt_path = Path(prompt_path) if prompt_path else None

        # Check if captioning is enabled via config
        captioner_config = settings.ingestion.image_captioner
        self._enabled = captioner_config.enabled

        self._prompt_template: str | None = None

        # Load prompt template if enabled
        if self._enabled:
            self._load_prompt()

    @property
    def enabled(self) -> bool:
        """Check if image captioning is enabled."""
        return self._enabled

    def _load_prompt(self) -> None:
        """Load the captioning prompt template from file or use default."""
        if self._prompt_path and self._prompt_path.exists():
            self._prompt_template = self._prompt_path.read_text()
            logger.info(f"Loaded captioning prompt from: {self._prompt_path}")
        else:
            # Default prompt path
            default_prompt = (
                Path(__file__).parent.parent.parent.parent
                / "config"
                / "prompts"
                / "image_captioning.txt"
            )
            if default_prompt.exists():
                self._prompt_template = default_prompt.read_text()
                logger.info(f"Loaded default captioning prompt from: {default_prompt}")
            else:
                logger.warning(
                    "No captioning prompt template found, "
                    "will use default captioning approach"
                )
                self._prompt_template = None

    def _get_vision_llm(self) -> BaseVisionLLM | None:
        """Get or create the Vision LLM instance."""
        if self._vision_llm is None and self._enabled:
            try:
                self._vision_llm = LLMFactory.create_vision_llm(self._settings)
                logger.info(
                    f"Created Vision LLM for captioning: {self._vision_llm.provider_name}"
                )
            except Exception as e:
                logger.warning(f"Failed to create Vision LLM for captioning: {e}")
                self._vision_llm = None
        return self._vision_llm

    def transform(
        self,
        chunks: list[Chunk],
        trace: TraceContext | None = None,
    ) -> list[Chunk]:
        """Generate captions for images in chunks.

        This method processes chunks and generates captions for any images
        referenced in the chunk metadata. If Vision LLM is not available
        or fails, it marks chunks with `has_unprocessed_images`.

        Args:
            chunks: List of chunks to process.
            trace: Optional trace context for observability.

        Returns:
            List of chunks with image captions added to metadata.
        """
        if not chunks:
            return []

        logger.info(f"Processing {len(chunks)} chunks for image captioning")

        # Check if captioning is enabled
        if not self._enabled:
            logger.info("Image captioning is disabled, skipping")
            # Still mark chunks with image_captioning metadata
            result_chunks: list[Chunk] = []
            for chunk in chunks:
                if self._has_images(chunk):
                    marked_chunk = self._mark_unprocessed(chunk, "disabled")
                    result_chunks.append(marked_chunk)
                else:
                    # Add empty captioning metadata for chunks without images
                    chunk_with_meta = Chunk(
                        id=chunk.id,
                        text=chunk.text,
                        metadata=dict(chunk.metadata),
                        source_ref=chunk.source_ref,
                    )
                    chunk_with_meta.metadata["image_captioning"] = {
                        "enabled": False,
                        "captions_count": 0,
                    }
                    result_chunks.append(chunk_with_meta)
            return result_chunks

        # Get Vision LLM
        vision_llm = self._get_vision_llm()
        if vision_llm is None:
            logger.warning("Vision LLM not available, skipping caption generation")
            result_chunks = []
            for chunk in chunks:
                if self._has_images(chunk):
                    marked_chunk = self._mark_unprocessed(
                        chunk, "vision_llm_unavailable"
                    )
                    result_chunks.append(marked_chunk)
                else:
                    # Add empty captioning metadata for chunks without images
                    chunk_with_meta = Chunk(
                        id=chunk.id,
                        text=chunk.text,
                        metadata=dict(chunk.metadata),
                        source_ref=chunk.source_ref,
                    )
                    chunk_with_meta.metadata["image_captioning"] = {
                        "enabled": True,
                        "vision_provider": None,
                        "captions_count": 0,
                    }
                    result_chunks.append(chunk_with_meta)
            return result_chunks

        processed_chunks: list[Chunk] = []
        errors: list[dict[str, Any]] = []

        for chunk in chunks:
            try:
                processed_chunk = self._caption_chunk(chunk, vision_llm, trace)
                processed_chunks.append(processed_chunk)
            except Exception as e:
                logger.warning(f"Failed to caption chunk {chunk.id}: {e}")
                marked_chunk = self._mark_unprocessed(chunk, str(e))
                processed_chunks.append(marked_chunk)
                errors.append({"chunk_id": chunk.id, "error": str(e)})

        # Record stage in trace
        if trace:
            trace.record_stage(
                "image_captioning",
                {
                    "input_count": len(chunks),
                    "output_count": len(processed_chunks),
                    "enabled": self._enabled,
                    "errors": errors,
                },
            )

        logger.info(
            f"Captioned {len(processed_chunks)} chunks, {len(errors)} errors"
        )
        return processed_chunks

    def _has_images(self, chunk: Chunk) -> bool:
        """Check if chunk has image references."""
        images = chunk.metadata.get("images", [])
        return bool(images)

    def _mark_unprocessed(
        self,
        chunk: Chunk,
        reason: str,
        vision_provider: str | None = None,
    ) -> Chunk:
        """Mark chunk as having unprocessed images."""
        marked = Chunk(
            id=chunk.id,
            text=chunk.text,
            metadata=dict(chunk.metadata),
            source_ref=chunk.source_ref,
        )
        # Always add captioning metadata
        marked.metadata["image_captioning"] = {
            "enabled": self._enabled,
            "vision_provider": vision_provider,
            "captions_count": 0,
            "failed_count": len(chunk.metadata.get("images", [])),
            "reason": reason,
        }
        marked.metadata["has_unprocessed_images"] = True
        return marked

    def _caption_chunk(
        self,
        chunk: Chunk,
        vision_llm: BaseVisionLLM,
        trace: TraceContext | None = None,
    ) -> Chunk:
        """Generate captions for a single chunk's images.

        Args:
            chunk: The chunk to process.
            vision_llm: The Vision LLM instance.
            trace: Optional trace context.

        Returns:
            Chunk with captions added to metadata.
        """
        images = chunk.metadata.get("images", [])

        # Create chunk with metadata even if no images
        enriched_chunk = Chunk(
            id=chunk.id,
            text=chunk.text,
            metadata=dict(chunk.metadata),
            source_ref=chunk.source_ref,
        )

        if not images:
            # Add empty captioning metadata
            enriched_chunk.metadata["image_captioning"] = {
                "enabled": self._enabled,
                "vision_provider": vision_llm.provider_name,
                "captions_count": 0,
            }
            return enriched_chunk

        # Build captioning prompt
        prompt = self._build_caption_prompt(chunk)

        captions: dict[str, str] = {}
        failed_images: list[str] = []

        for image_info in images:
            image_id = image_info.get("id")
            image_path = image_info.get("path")

            if not image_id:
                logger.warning(f"Image id not found: {image_id}")
                continue

            if not image_path or not Path(image_path).exists():
                logger.warning(f"Image path not found: {image_path}")
                continue

            try:
                # Generate caption
                caption = self._generate_caption(
                    image_id, image_path, prompt, vision_llm, trace
                )
                if caption:
                    captions[image_id] = caption
                else:
                    failed_images.append(image_id)
            except Exception as e:
                logger.warning(f"Failed to caption image {image_id}: {e}")
                failed_images.append(image_id)

        # Add captioning metadata
        captioning_info: dict[str, Any] = {
            "enabled": self._enabled,
            "vision_provider": vision_llm.provider_name,
            "captions_count": len(captions),
            "failed_count": len(failed_images),
        }
        if failed_images:
            captioning_info["failed_images"] = failed_images

        enriched_chunk.metadata["image_captioning"] = captioning_info

        # Add captions
        if captions:
            existing_captions = enriched_chunk.metadata.get("image_captions", {})
            existing_captions.update(captions)
            enriched_chunk.metadata["image_captions"] = existing_captions

        # Mark as unprocessed if any images failed
        if failed_images:
            enriched_chunk.metadata["has_unprocessed_images"] = True
            enriched_chunk.metadata["unprocessed_images"] = failed_images

        # Record to trace
        if trace:
            trace.record_stage(
                "image_chunk_captioning",
                {
                    "chunk_id": chunk.id,
                    "image_count": len(images),
                    "captioned_count": len(captions),
                    "failed_count": len(failed_images),
                    "vision_provider": vision_llm.provider_name,
                },
            )

        return enriched_chunk

    def _build_caption_prompt(self, chunk: Chunk) -> str:
        """Build the captioning prompt for a chunk.

        Args:
            chunk: The chunk being processed.

        Returns:
            Prompt string for caption generation.
        """
        # Get context from surrounding text if available
        context = chunk.text[:500] if chunk.text else ""

        # Use template if available
        if self._prompt_template:
            # Replace placeholders if present
            prompt = self._prompt_template
            prompt = prompt.replace("{context}", context)
            prompt = prompt.replace("{chunk_id}", chunk.id)
            return prompt

        # Default prompt
        default_prompt = (
            "You are analyzing an image extracted from a document. "
            "Generate a concise, descriptive caption (2-4 sentences) that:\n"
            "1. Describes the main content and subject matter\n"
            "2. Notes any text, labels, or annotations visible\n"
            "3. Identifies the image type (diagram, chart, screenshot, etc.)\n"
            "4. Explains relationships or flows depicted\n\n"
            "Context from document (first 500 chars):\n"
            f"{context}\n\n"
            "Provide only the caption text, no explanations."
        )

        return default_prompt

    def _generate_caption(
        self,
        image_id: str,
        image_path: str | None,
        prompt: str,
        vision_llm: BaseVisionLLM,
        trace: TraceContext | None = None,
    ) -> str | None:
        """Generate a single caption for an image.

        Args:
            image_id: The image identifier.
            image_path: Path to the image file.
            prompt: The captioning prompt.
            vision_llm: The Vision LLM instance.
            trace: Optional trace context.

        Returns:
            Generated caption or None if generation failed.
        """
        if not image_path:
            logger.warning(f"No image path for {image_id}, skipping caption")
            return None

        # Check if image file exists
        path = Path(image_path)
        if not path.exists():
            logger.warning(f"Image file not found: {image_path}")
            return None

        try:
            # Call Vision LLM with ImageInput
            image_input = ImageInput(path=str(path))
            response = vision_llm.chat_with_image(
                text=prompt,
                image=image_input,
                trace=trace,
            )

            # Extract caption from response
            caption = self._extract_caption(response)
            logger.debug(f"Generated caption for {image_id}: {caption[:50]}...")

            return caption

        except Exception as e:
            logger.warning(f"Vision LLM caption failed for {image_id}: {e}")
            return None

    def _extract_caption(self, response: VisionResponse) -> str:
        """Extract clean caption text from Vision LLM response.

        Args:
            response: The Vision LLM response.

        Returns:
            Cleaned caption text.
        """
        content = response.content.strip()

        # Remove markdown formatting if present
        content = re.sub(r"^```[\s\S]*?```$", "", content, flags=re.MULTILINE)
        content = re.sub(r"\*\*([^*]+)\*\*", r"\1", content)
        content = re.sub(r"__([^_]+)__", r"\1", content)

        # Clean up whitespace
        content = re.sub(r"\s+", " ", content).strip()

        return content

    def get_captioning_stats(self, chunks: list[Chunk]) -> dict[str, Any]:
        """Get statistics about image captioning.

        Args:
            chunks: List of processed chunks.

        Returns:
            Dictionary with captioning statistics.
        """
        stats = {
            "total": len(chunks),
            "with_images": 0,
            "with_captions": 0,
            "unprocessed": 0,
            "total_images": 0,
            "total_captions": 0,
        }

        for chunk in chunks:
            captioning = chunk.metadata.get("image_captioning", {})
            images = chunk.metadata.get("images", [])
            captions = chunk.metadata.get("image_captions", {})

            if images:
                stats["with_images"] += 1
                stats["total_images"] += len(images)

            if captions:
                stats["with_captions"] += 1
                stats["total_captions"] += len(captions)

            if chunk.metadata.get("has_unprocessed_images"):
                stats["unprocessed"] += 1

        return stats
