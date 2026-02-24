"""Ingestion Pipeline orchestrator for the Modular RAG MCP Server.

This module implements the main pipeline that orchestrates the complete
document ingestion flow:
    1. File Integrity Check (SHA256 skip check)
    2. Document Loading (PDF → Document)
    3. Chunking (Document → Chunks)
    4. Transform (Refine + Enrich + Caption)
    5. Encoding (Dense + Sparse vectors)
    6. Storage (VectorStore + BM25 Index + ImageStorage)

Design Principles:
- Config-Driven: All components configured via settings.yaml
- Observable: Logs progress and stage completion
- Graceful Degradation: LLM failures don't block pipeline
- Idempotent: SHA256-based skip for unchanged files
"""

from pathlib import Path
from typing import List, Optional, Dict, Any

from core.settings import Settings, load_settings
from core.types import Document, Chunk
from core.trace.trace_context import TraceContext
from observability.logger import get_logger

# Libs layer imports
from libs.loader.file_integrity import (
    SqliteFileIntegrityChecker,
    compute_sha256,
)
from libs.loader.pdf_loader import PdfLoader
from libs.embedding.embedding_factory import EmbeddingFactory

# Ingestion layer imports
from ingestion.chunking.document_chunker import DocumentChunker
from ingestion.transform.chunk_refiner import ChunkRefiner
from ingestion.transform.metadata_enricher import MetadataEnricher
from ingestion.transform.image_captioner import ImageCaptioner
from ingestion.embedding.dense_encoder import DenseEncoder
from ingestion.embedding.sparse_encoder import SparseEncoder
from ingestion.embedding.batch_processor import BatchProcessor
from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.vector_upserter import VectorUpserter
from ingestion.storage.image_storage import ImageStorage

logger = get_logger(__name__)


class PipelineResult:
    """Result of pipeline execution with detailed statistics.

    Attributes:
        success: Whether pipeline completed successfully
        file_path: Path to the processed file
        doc_id: Document ID (SHA256 hash)
        chunk_count: Number of chunks generated
        image_count: Number of images processed
        vector_ids: List of vector IDs stored
        error: Error message if pipeline failed
        stages: Dict of stage names to their individual results
    """

    def __init__(
        self,
        success: bool,
        file_path: str,
        doc_id: Optional[str] = None,
        chunk_count: int = 0,
        image_count: int = 0,
        vector_ids: Optional[List[str]] = None,
        error: Optional[str] = None,
        stages: Optional[Dict[str, Any]] = None
    ):
        self.success = success
        self.file_path = file_path
        self.doc_id = doc_id
        self.chunk_count = chunk_count
        self.image_count = image_count
        self.vector_ids = vector_ids or []
        self.error = error
        self.stages = stages or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "file_path": self.file_path,
            "doc_id": self.doc_id,
            "chunk_count": self.chunk_count,
            "image_count": self.image_count,
            "vector_ids_count": len(self.vector_ids),
            "error": self.error,
            "stages": self.stages
        }


class IngestionPipeline:
    """Main pipeline orchestrator for document ingestion.

    This class coordinates all stages of the ingestion process:
    - File integrity checking for incremental processing
    - Document loading (PDF with image extraction)
    - Text chunking with configurable splitter
    - Chunk refinement (rule-based + LLM)
    - Metadata enrichment (rule-based + LLM)
    - Image captioning (Vision LLM)
    - Dense embedding (Azure text-embedding-ada-002)
    - Sparse encoding (BM25 term statistics)
    - Vector storage (ChromaDB)
    - BM25 index building

    Example:
        >>> from core.settings import load_settings
        >>> settings = load_settings("config/settings.yaml")
        >>> pipeline = IngestionPipeline(settings)
        >>> result = pipeline.run("documents/report.pdf", collection="contracts")
        >>> print(f"Processed {result.chunk_count} chunks")
    """

    def __init__(
        self,
        settings: Settings,
        collection: str = "default",
        force: bool = False
    ):
        """Initialize pipeline with all components.

        Args:
            settings: Application settings from settings.yaml
            collection: Collection name for organizing documents
            force: If True, re-process even if file was previously processed
        """
        self.settings = settings
        self.collection = collection
        self.force = force

        # Initialize all components
        logger.info("Initializing Ingestion Pipeline components...")

        # Stage 1: File Integrity
        self.integrity_checker = SqliteFileIntegrityChecker(db_path="data/db/ingestion_history.db")
        logger.info("  ✓ FileIntegrityChecker initialized")

        # Stage 2: Loader
        self.loader = PdfLoader(
            extract_images=True,
            image_storage_dir=f"data/images/{collection}"
        )
        logger.info("  ✓ PdfLoader initialized")

        # Stage 3: Chunker
        self.chunker = DocumentChunker(settings)
        logger.info("  ✓ DocumentChunker initialized")

        # Stage 4: Transforms
        self.chunk_refiner = ChunkRefiner(settings)
        logger.info(f"  ✓ ChunkRefiner initialized (use_llm={self.chunk_refiner.use_llm})")

        self.metadata_enricher = MetadataEnricher(settings)
        logger.info(f"  ✓ MetadataEnricher initialized (use_llm={self.metadata_enricher.use_llm})")

        self.image_captioner = ImageCaptioner(settings)
        has_vision = self.image_captioner.llm is not None
        logger.info(f"  ✓ ImageCaptioner initialized (vision_enabled={has_vision})")

        # Stage 5: Encoders
        embedding = EmbeddingFactory.create(settings)
        batch_size = settings.ingestion.batch_size if settings.ingestion else 100
        self.dense_encoder = DenseEncoder(embedding, batch_size=batch_size)
        logger.info(f"  ✓ DenseEncoder initialized (provider={settings.embedding.provider})")

        self.sparse_encoder = SparseEncoder()
        logger.info("  ✓ SparseEncoder initialized")

        self.batch_processor = BatchProcessor(
            dense_encoder=self.dense_encoder,
            sparse_encoder=self.sparse_encoder,
            batch_size=batch_size
        )
        logger.info(f"  ✓ BatchProcessor initialized (batch_size={batch_size})")

        # Stage 6: Storage
        self.vector_upserter = VectorUpserter(settings)
        logger.info(f"  ✓ VectorUpserter initialized (provider={settings.vector_store.provider})")

        self.bm25_indexer = BM25Indexer(index_dir=f"data/db/bm25/{collection}")
        logger.info("  ✓ BM25Indexer initialized")

        self.image_storage = ImageStorage(
            db_path="data/db/image_index.db",
            images_root="data/images"
        )
        logger.info("  ✓ ImageStorage initialized")

        logger.info("Pipeline initialization complete!")

    def run(
        self,
        file_path: str,
        trace: Optional[TraceContext] = None
    ) -> PipelineResult:
        """Execute the full ingestion pipeline on a file.

        Args:
            file_path: Path to the file to process (e.g., PDF)
            trace: Optional trace context for observability

        Returns:
            PipelineResult with success status and statistics
        """
        file_path = Path(file_path)
        stages: Dict[str, Any] = {}

        logger.info(f"=" * 60)
        logger.info(f"Starting Ingestion Pipeline for: {file_path}")
        logger.info(f"Collection: {self.collection}")
        logger.info(f"=" * 60)

        try:
            # ─────────────────────────────────────────────────────────────
            # Stage 1: File Integrity Check
            # ─────────────────────────────────────────────────────────────
            logger.info("\n📋 Stage 1: File Integrity Check")

            file_hash = compute_sha256(str(file_path))
            logger.info(f"  File hash: {file_hash[:16]}...")

            if not self.force and self.integrity_checker.should_skip(file_hash):
                logger.info(f"  ⏭️  File already processed, skipping (use force=True to reprocess)")
                return PipelineResult(
                    success=True,
                    file_path=str(file_path),
                    doc_id=file_hash,
                    stages={"integrity": {"skipped": True, "reason": "already_processed"}}
                )

            stages["integrity"] = {"file_hash": file_hash, "skipped": False}
            logger.info("  ✓ File needs processing")

            # ─────────────────────────────────────────────────────────────
            # Stage 2: Document Loading
            # ─────────────────────────────────────────────────────────────
            logger.info("\n📄 Stage 2: Document Loading")

            document = self.loader.load(str(file_path))

            text_preview = document.text[:200].replace('\n', ' ') + "..." if len(document.text) > 200 else document.text
            image_count = len(document.metadata.get("images", []))

            logger.info(f"  Document ID: {document.id}")
            logger.info(f"  Text length: {len(document.text)} chars")
            logger.info(f"  Images extracted: {image_count}")
            logger.info(f"  Preview: {text_preview[:100]}...")

            stages["loading"] = {
                "doc_id": document.id,
                "text_length": len(document.text),
                "image_count": image_count
            }

            # ─────────────────────────────────────────────────────────────
            # Stage 3: Chunking
            # ─────────────────────────────────────────────────────────────
            logger.info("\n✂️  Stage 3: Document Chunking")

            chunks = self.chunker.split_document(document)

            logger.info(f"  Chunks generated: {len(chunks)}")
            if chunks:
                logger.info(f"  First chunk ID: {chunks[0].id}")
                logger.info(f"  First chunk preview: {chunks[0].text[:100]}...")

            stages["chunking"] = {
                "chunk_count": len(chunks),
                "avg_chunk_size": sum(len(c.text) for c in chunks) // len(chunks) if chunks else 0
            }

            # ─────────────────────────────────────────────────────────────
            # Stage 4: Transform Pipeline
            # ─────────────────────────────────────────────────────────────
            logger.info("\n🔄 Stage 4: Transform Pipeline")

            # 4a: Chunk Refinement
            logger.info("  4a. Chunk Refinement...")
            chunks = self.chunk_refiner.transform(chunks, trace)
            refined_by_llm = sum(1 for c in chunks if c.metadata.get("refined_by") == "llm")
            refined_by_rule = sum(1 for c in chunks if c.metadata.get("refined_by") == "rule")
            logger.info(f"      LLM refined: {refined_by_llm}, Rule refined: {refined_by_rule}")

            # 4b: Metadata Enrichment
            logger.info("  4b. Metadata Enrichment...")
            chunks = self.metadata_enricher.transform(chunks, trace)
            enriched_by_llm = sum(1 for c in chunks if c.metadata.get("enriched_by") == "llm")
            enriched_by_rule = sum(1 for c in chunks if c.metadata.get("enriched_by") == "rule")
            logger.info(f"      LLM enriched: {enriched_by_llm}, Rule enriched: {enriched_by_rule}")

            # 4c: Image Captioning
            logger.info("  4c. Image Captioning...")
            chunks = self.image_captioner.transform(chunks, trace)
            captioned = sum(1 for c in chunks if c.metadata.get("image_captions"))
            logger.info(f"      Chunks with captions: {captioned}")

            stages["transform"] = {
                "chunk_refiner": {"llm": refined_by_llm, "rule": refined_by_rule},
                "metadata_enricher": {"llm": enriched_by_llm, "rule": enriched_by_rule},
                "image_captioner": {"captioned_chunks": captioned}
            }

            # ─────────────────────────────────────────────────────────────
            # Stage 5: Encoding
            # ─────────────────────────────────────────────────────────────
            logger.info("\n🔢 Stage 5: Encoding")

            # Process through BatchProcessor
            records = self.batch_processor.process(chunks, trace)

            # Extract dense vectors and sparse stats from records
            dense_vectors = [r.dense_vector for r in records if r.dense_vector is not None]
            sparse_stats = [r.sparse_vector for r in records if r.sparse_vector is not None]

            logger.info(f"  Dense vectors: {len(dense_vectors)} (dim={len(dense_vectors[0]) if dense_vectors else 0})")
            logger.info(f"  Sparse stats: {len(sparse_stats)} documents")

            stages["encoding"] = {
                "dense_vector_count": len(dense_vectors),
                "dense_dimension": len(dense_vectors[0]) if dense_vectors else 0,
                "sparse_doc_count": len(sparse_stats)
            }

            # ─────────────────────────────────────────────────────────────
            # Stage 6: Storage
            # ─────────────────────────────────────────────────────────────
            logger.info("\n💾 Stage 6: Storage")

            # 6a: Vector Upsert
            logger.info("  6a. Vector Storage (ChromaDB)...")
            vector_ids = self.vector_upserter.upsert(records, trace)
            logger.info(f"      Stored {len(vector_ids)} vectors")

            # 6b: BM25 Index
            logger.info("  6b. BM25 Index...")
            self.bm25_indexer.build(records, collection=self.collection, trace=trace)
            logger.info(f"      Index built for {len(sparse_stats)} documents")

            # 6c: Store images in image storage index
            logger.info("  6c. Image Storage Index...")
            images = document.metadata.get("images", [])
            for img in images:
                self.image_storage.save_image(
                    source_path=img["path"],
                    collection=self.collection,
                    doc_hash=file_hash,
                    page_num=img.get("page", 0)
                )
            logger.info(f"      Indexed {len(images)} images")

            stages["storage"] = {
                "vector_count": len(vector_ids),
                "bm25_docs": len(sparse_stats),
                "images_indexed": len(images)
            }

            # ─────────────────────────────────────────────────────────────
            # Mark Success
            # ─────────────────────────────────────────────────────────────
            self.integrity_checker.mark_success(file_hash, str(file_path))

            logger.info("\n" + "=" * 60)
            logger.info("✅ Pipeline completed successfully!")
            logger.info(f"   Chunks: {len(chunks)}")
            logger.info(f"   Vectors: {len(vector_ids)}")
            logger.info(f"   Images: {len(images)}")
            logger.info("=" * 60)

            return PipelineResult(
                success=True,
                file_path=str(file_path),
                doc_id=file_hash,
                chunk_count=len(chunks),
                image_count=len(images),
                vector_ids=vector_ids,
                stages=stages
            )

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
            self.integrity_checker.mark_failed(file_hash, str(e), str(file_path))

            return PipelineResult(
                success=False,
                file_path=str(file_path),
                doc_id=file_hash if 'file_hash' in locals() else None,
                error=str(e),
                stages=stages
            )

    def close(self) -> None:
        """Clean up resources."""
        self.image_storage.close()


def run_pipeline(
    file_path: str,
    settings_path: str = "config/settings.yaml",
    collection: str = "default",
    force: bool = False
) -> PipelineResult:
    """Convenience function to run the pipeline.

    Args:
        file_path: Path to file to process
        settings_path: Path to settings.yaml
        collection: Collection name
        force: Force reprocessing

    Returns:
        PipelineResult with execution details
    """
    settings = load_settings(settings_path)
    pipeline = IngestionPipeline(settings, collection=collection, force=force)

    try:
        return pipeline.run(file_path)
    finally:
        pipeline.close()
