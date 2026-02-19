# Ingestion Pipeline - Offline data processing
"""Ingestion Pipeline提供文档摄取和向量化功能。"""
from ingestion.embedding import BatchProcessor, DenseEncoder, SparseEncoder

__all__ = ["BatchProcessor", "DenseEncoder", "SparseEncoder"]
