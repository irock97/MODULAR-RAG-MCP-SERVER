# Ingestion Pipeline - Offline data processing
"""Ingestion Pipeline提供文档摄取和向量化功能。"""
from ingestion.embedding import BatchProcessor, DenseEncoder, SparseEncoder
from ingestion.pipeline import IngestionPipeline, PipelineResult

__all__ = [
    "BatchProcessor",
    "DenseEncoder",
    "IngestionPipeline",
    "PipelineResult",
    "SparseEncoder",
]
