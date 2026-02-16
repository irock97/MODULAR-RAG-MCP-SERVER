"""MetadataEnricher - Rule-based and LLM-based metadata enrichment.

This module provides the MetadataEnricher class that enhances chunk metadata
with title, summary, and tags extracted through rules and optional LLM.

Design Principles:
    - Fail-Safe: LLM failures gracefully fall back to rule-based results
    - Configurable: LLM usage controlled via settings.yaml
    - Traceable: All enrichments logged to trace context
    - Metadata Preserved: Original and enriched metadata tracked

Enrichment Pipeline:
    1. Rule-based extraction (always runs as fallback)
    2. Optional LLM enhancement (if enabled and available)
    3. Fallback to rule-only if LLM fails
"""

import re
from pathlib import Path
from typing import Any

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.base_transform import BaseTransform
from libs.llm.base_llm import ChatMessage
from libs.llm.llm_factory import LLMFactory
from observability.logger import get_logger

logger = get_logger(__name__)


class MetadataEnrichmentError(Exception):
    """Error during metadata enrichment."""

    pass


class MetadataEnricher(BaseTransform):
    """Enriches chunk metadata using rules and optional LLM.

    This class applies two-stage metadata enrichment:
    1. **Rule-based extraction**: Extracts title, summary, tags from text
    2. **LLM enhancement**: Optionally generates semantic metadata via LLM

    The LLM enhancement is controlled by `settings.ingestion.metadata_enricher.use_llm`.
    If LLM is enabled but fails, the enricher falls back to rule-based results.
    """

    def __init__(
        self,
        settings: Settings,
        llm: Any | None = None,
        prompt_path: str | Path | None = None,
    ) -> None:
        """Initialize the MetadataEnricher.

        Args:
            settings: Settings object with configuration.
            llm: Optional LLM instance. If None, created from settings.
            prompt_path: Optional path to prompt template file.
        """
        super().__init__(settings)

        self._llm = llm
        self._prompt_path = Path(prompt_path) if prompt_path else None

        # Check if LLM is enabled
        metadata_enricher_config = getattr(settings, "ingestion", None)
        if metadata_enricher_config is not None:
            enricher_config = getattr(metadata_enricher_config, "metadata_enricher", None)
            if enricher_config is not None:
                self._use_llm = getattr(enricher_config, "use_llm", False)
            else:
                self._use_llm = False
        else:
            self._use_llm = False

        self._prompt_template: str | None = None

        # Load prompt template if LLM is enabled
        if self._use_llm:
            self._load_prompt()

    @property
    def use_llm(self) -> bool:
        """Check if LLM enhancement is enabled."""
        return self._use_llm

    def _load_prompt(self) -> None:
        """Load the prompt template from file or use default."""
        if self._prompt_path and self._prompt_path.exists():
            self._prompt_template = self._prompt_path.read_text()
            logger.info(f"Loaded prompt from: {self._prompt_path}")
        else:
            # Default prompt path - look in config/prompts relative to project root
            # Use the standard location for prompts
            default_prompt = Path(__file__).parent.parent.parent.parent / "config" / "prompts" / "metadata_enrichment.txt"
            if default_prompt.exists():
                self._prompt_template = default_prompt.read_text()
                logger.info(f"Loaded default prompt from: {default_prompt}")
            else:
                logger.warning(
                    "No prompt template found for metadata enrichment, "
                    "LLM mode may not work properly"
                )
                self._prompt_template = None

    def _get_llm(self) -> Any | None:
        """Get or create the LLM instance."""
        if self._llm is None and self._use_llm:
            try:
                self._llm = LLMFactory.create(self._settings)
                logger.info(
                    f"Created LLM for metadata enrichment: {self._llm.provider_name}"
                )
            except Exception as e:
                logger.warning(f"Failed to create LLM for enrichment: {e}")
                self._llm = None
        return self._llm

    def transform(
        self,
        chunks: list[Chunk],
        trace: TraceContext | None = None,
    ) -> list[Chunk]:
        """Enrich chunks with metadata (title, summary, tags).

        This method applies rule-based metadata extraction to all chunks, and
        optionally applies LLM enhancement if enabled.

        Args:
            chunks: List of chunks to enrich.
            trace: Optional trace context for observability.

        Returns:
            List of enriched chunks with additional metadata.

        Raises:
            MetadataEnrichmentError: If all enrichments fail unexpectedly.
        """
        if not chunks:
            return []

        logger.info(f"Enriching {len(chunks)} chunks (LLM: {self._use_llm})")

        enriched_chunks: list[Chunk] = []
        errors: list[dict[str, Any]] = []

        for index, chunk in enumerate(chunks):
            try:
                # Step 1: Rule-based extraction
                rule_metadata = self._rule_based_extract(chunk.text)
                enriched_by = "rule"
                llm_failed = False

                # Step 2: Optional LLM enhancement
                llm_metadata: dict[str, Any] | None = None
                if self._use_llm:
                    llm_metadata = self._llm_extract(chunk.text, trace)
                    if llm_metadata is not None:
                        # Merge: LLM takes precedence but keep rule values as fallback
                        rule_metadata = self._merge_metadata(rule_metadata, llm_metadata)
                        enriched_by = "llm"
                    else:
                        # LLM was enabled but failed
                        llm_failed = True

                # Create enriched chunk with metadata
                enriched_chunk = Chunk(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=dict(chunk.metadata),
                    source_ref=chunk.source_ref,
                )

                # Add enrichment metadata
                enrichment_info: dict[str, Any] = {
                    "enriched_by": enriched_by,
                    "rule_based_applied": True,
                }
                if llm_failed:
                    enrichment_info["fallback"] = True
                enriched_chunk.metadata["enrichment"] = enrichment_info

                # Add extracted metadata
                if "title" in rule_metadata:
                    enriched_chunk.metadata["title"] = rule_metadata["title"]
                if "summary" in rule_metadata:
                    enriched_chunk.metadata["summary"] = rule_metadata["summary"]
                if "tags" in rule_metadata:
                    enriched_chunk.metadata["tags"] = rule_metadata["tags"]

                enriched_chunk.metadata["chunk_index"] = chunk.metadata.get(
                    "chunk_index", index
                )

                enriched_chunks.append(enriched_chunk)

            except Exception as e:
                logger.warning(f"Failed to enrich chunk {chunk.id}: {e}")
                # Fallback: return original chunk with error metadata
                error_chunk = Chunk(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=dict(chunk.metadata),
                    source_ref=chunk.source_ref,
                )
                error_chunk.metadata["enrichment"] = {
                    "enriched_by": "rule",
                    "error": str(e),
                    "fallback": True,
                }
                # Apply rule-based extraction as fallback
                rule_metadata = self._rule_based_extract(chunk.text)
                if "title" in rule_metadata:
                    error_chunk.metadata["title"] = rule_metadata["title"]
                if "summary" in rule_metadata:
                    error_chunk.metadata["summary"] = rule_metadata["summary"]
                if "tags" in rule_metadata:
                    error_chunk.metadata["tags"] = rule_metadata["tags"]

                errors.append({"chunk_id": chunk.id, "error": str(e)})
                enriched_chunks.append(error_chunk)

        # Record stage in trace
        if trace:
            trace.record_stage(
                "metadata_enrichment",
                {
                    "input_count": len(chunks),
                    "output_count": len(enriched_chunks),
                    "llm_used": self._use_llm,
                    "errors": errors,
                },
            )

        logger.info(
            f"Enriched {len(enriched_chunks)} chunks, {len(errors)} errors"
        )
        return enriched_chunks

    def _merge_metadata(
        self,
        rule_metadata: dict[str, Any],
        llm_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge rule-based and LLM metadata.

        LLM values take precedence, but rule values are kept as fallback.

        Args:
            rule_metadata: Rule-based extracted metadata.
            llm_metadata: LLM-generated metadata.

        Returns:
            Merged metadata dictionary.
        """
        merged = dict(rule_metadata)

        for key, llm_value in llm_metadata.items():
            if llm_value and str(llm_value).strip():
                # LLM has a valid value, use it
                merged[key] = llm_value
            elif key in rule_metadata and rule_metadata[key]:
                # LLM value is empty/invalid, fall back to rule
                pass

        return merged

    def _rule_based_extract(self, text: str) -> dict[str, Any]:
        """Apply rule-based metadata extraction.

        Extracts title, summary, and tags from text using heuristics:
        - Title: First non-empty line, or first markdown heading
        - Summary: First few sentences (up to 200 chars)
        - Tags: Extracted from content patterns

        Args:
            text: Raw text from chunk.

        Returns:
            Dictionary with extracted metadata (title, summary, tags).
        """
        if not text:
            return {"title": "", "summary": "", "tags": []}

        result: dict[str, Any] = {
            "title": "",
            "summary": "",
            "tags": [],
        }

        lines = text.strip().split("\n")
        non_empty_lines = [line.strip() for line in lines if line.strip()]

        # Extract title
        # Priority: 1. Markdown heading, 2. First non-empty line
        for line in non_empty_lines:
            if line.startswith("#"):
                # Markdown heading
                result["title"] = line.lstrip("#").strip()
                break
            elif len(line) < 100 and not line.endswith("."):
                # Likely a title (short, not a sentence)
                result["title"] = line
                break
        else:
            # Fallback: first line
            result["title"] = non_empty_lines[0] if non_empty_lines else ""

        # Extract summary: first few sentences
        # Clean the text for sentence splitting
        clean_text = re.sub(r"\s+", " ", text.strip())

        # Find end of first sentence or take first 200 chars
        sentence_end = re.search(r"[.!?]\s", clean_text)
        if sentence_end:
            end_pos = min(sentence_end.end(), 200)
            result["summary"] = clean_text[:end_pos].strip()
        else:
            result["summary"] = clean_text[:200].strip()

        # Extract tags from patterns
        tags: list[str] = []

        # Common tag patterns in technical documents
        tag_patterns = [
            # Programming languages
            (r"\b(Python|JavaScript|TypeScript|Java|Kotlin|Swift|Rust|Go|C\+\+|C#)\b", "language"),
            # Frameworks
            (r"\b(React|Vue|Angular|Django|Flask|FastAPI|Spring|SwiftUI)\b", "framework"),
            # Concepts
            (r"\b(API|REST|GraphQL|OAuth|JWT|TLS|SSL|CI/CD|DevOps)\b", "concept"),
            # File types in content
            (r"\.(py|js|ts|java|cpp|h|json|yaml|yml|md|txt|pdf)\b", "filetype"),
        ]

        for pattern, tag_type in tag_patterns:
            matches = re.findall(pattern, clean_text, re.IGNORECASE)
            for match in matches:
                tag = f"{tag_type}:{match.lower()}"
                if tag not in tags and len(tags) < 10:  # Limit tags
                    tags.append(tag)

        result["tags"] = tags

        return result

    def _llm_extract(
        self,
        text: str,
        trace: TraceContext | None = None,
    ) -> dict[str, Any] | None:
        """Apply LLM-based metadata extraction.

        This method sends the text to an LLM for intelligent metadata generation
        if an LLM is configured and available.

        Args:
            text: Text to extract metadata from.
            trace: Optional trace context for observability.

        Returns:
            Dictionary with LLM-generated metadata, or None if LLM fails.

        Raises:
            MetadataEnrichmentError: If LLM call fails unexpectedly.
        """
        if not self._use_llm or not self._prompt_template:
            return None

        llm = self._get_llm()
        if llm is None:
            logger.warning("LLM not available, skipping LLM metadata extraction")
            return None

        try:
            # Format prompt with the text
            if "{text}" in self._prompt_template:
                prompt = self._prompt_template.replace("{text}", text[:3000])  # Limit text length
            else:
                prompt = self._prompt_template + f"\n\nText to analyze:\n{text[:3000]}"

            # Use chat() with system and user messages
            messages = [
                ChatMessage(role="system", content=prompt),
                ChatMessage(
                    role="user",
                    content=(
                        "Extract metadata from the above text. "
                        "Return a JSON object with 'title', 'summary', and 'tags' fields. "
                        "Example: {\"title\": \"...\", \"summary\": \"...\", \"tags\": [\"tag1\", \"tag2\"]}"
                    ),
                ),
            ]

            # Call LLM
            response = llm.chat(messages=messages, trace=trace)

            # Parse response - try to extract JSON
            content = response.content.strip()

            # Try to find JSON in response
            json_match = re.search(r"\{[\s\S]*\}", content)
            if json_match:
                import json
                metadata = json.loads(json_match.group())
            else:
                # Fallback: parse manually
                metadata = self._parse_llm_response(content)

            logger.debug(
                f"LLM extracted metadata: title={metadata.get('title', '')[:50]}..., "
                f"tags_count={len(metadata.get('tags', []))}"
            )

            # Record to trace
            if trace:
                trace.record_stage(
                    "metadata_llm_extraction",
                    {
                        "input_length": len(text),
                        "output_keys": list(metadata.keys()),
                        "llm_provider": llm.provider_name,
                    },
                )

            return metadata

        except Exception as e:
            logger.warning(f"LLM metadata extraction failed: {e}")
            # Return None to trigger fallback
            return None

    def _parse_llm_response(self, content: str) -> dict[str, Any]:
        """Parse LLM response when JSON parsing fails.

        Args:
            content: Raw LLM response content.

        Returns:
            Parsed metadata dictionary.
        """
        metadata: dict[str, Any] = {
            "title": "",
            "summary": "",
            "tags": [],
        }

        # Try to extract title (often in quotes or after "title:")
        title_match = re.search(
            r'(?:title["\']?\s*[:=]\s*["\']?)([^"\n\'\[\]{}]+)', content, re.IGNORECASE
        )
        if title_match:
            metadata["title"] = title_match.group(1).strip()

        # Try to extract summary
        summary_match = re.search(
            r'(?:summary["\']?\s*[:=]\s*["\']?)([^"\n\'\[\]{}]+)', content, re.IGNORECASE
        )
        if summary_match:
            metadata["summary"] = summary_match.group(1).strip()

        # Try to extract tags
        tags_match = re.search(r"tags\s*:\s*\[([^\]]+)\]", content)
        if tags_match:
            tags_str = tags_match.group(1)
            # Parse comma-separated or quoted tags
            tags = re.findall(r'"([^"]+)"', tags_str)
            if tags:
                metadata["tags"] = tags
            else:
                metadata["tags"] = [t.strip() for t in tags_str.split(",") if t.strip()]

        return metadata

    def get_enrichment_stats(self, chunks: list[Chunk]) -> dict[str, Any]:
        """Get statistics about metadata enrichment.

        Args:
            chunks: List of enriched chunks.

        Returns:
            Dictionary with enrichment statistics.
        """
        stats = {
            "total": len(chunks),
            "by_method": {"rule": 0, "llm": 0},
            "with_title": 0,
            "with_summary": 0,
            "with_tags": 0,
            "errors": 0,
        }

        for chunk in chunks:
            enrichment = chunk.metadata.get("enrichment", {})
            enriched_by = enrichment.get("enriched_by", "rule")
            stats["by_method"][enriched_by] = stats["by_method"].get(enriched_by, 0) + 1

            if chunk.metadata.get("title"):
                stats["with_title"] += 1
            if chunk.metadata.get("summary"):
                stats["with_summary"] += 1
            if chunk.metadata.get("tags"):
                stats["with_tags"] += 1

            if enrichment.get("fallback"):
                stats["errors"] += 1

        return stats
