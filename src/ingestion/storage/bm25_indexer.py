"""BM25 Indexer - Build inverted index for sparse retrieval.

This module provides the BM25Indexer class that receives sparse vectors
from SparseEncoder, computes IDF weights, and builds an inverted index
for keyword-based retrieval.

Design Principles:
    - IDF Computation: BM25 IDF formula with smoothing
    - Inverted Index: term -> {idf, postings: [{chunk_id, tf, doc_length}]}
    - Persistence: Serialize/deserialize index to/from files
    - Incremental Support: Support rebuild and incremental updates

Example:
    >>> from ingestion.storage import BM25Indexer
    >>>
    >>> indexer = BM25Indexer(index_dir="data/db/bm25/my_collection")
    >>> indexer.build(records)  # list of ChunkRecord with sparse_vector
    >>> results = indexer.search("query terms", top_k=10)
"""

import json
import math
from pathlib import Path
from typing import Any

from core.trace.trace_context import TraceContext
from core.types import ChunkRecord
from observability.logger import get_logger

logger = get_logger(__name__)


class BM25Indexer:
    """BM25 Inverted Index for sparse retrieval.

    This class builds and manages an inverted index for BM25-based
    keyword search. It receives sparse vectors from SparseEncoder,
    computes IDF weights, and builds the inverted index structure.

    Attributes:
        index_dir: Directory to store index files
        k1: BM25 k1 parameter (term frequency saturation)
        b: BM25 b parameter (document length normalization)

    Example:
        >>> indexer = BM25Indexer(index_dir="data/db/bm25/docs")
        >>> indexer.build(records)
        >>> results = indexer.search("machine learning", top_k=5)

    _index:
    {
        "hello": {
            "idf": 0.8,
            "postings": [
                {"chunk_id": "doc1", "tf": 2, "doc_length": 10},
                {"chunk_id": "doc2", "tf": 1, "doc_length": 8}
            ]
        },
        "world": {
            "idf": 0.6,
            "postings": [
                {"chunk_id": "doc1", "tf": 1, "doc_length": 10},
                {"chunk_id": "doc3", "tf": 3, "doc_length": 12}
            ]
        }
    }

    """

    # BM25 default parameters
    DEFAULT_K1 = 1.5
    DEFAULT_B = 0.75

    # Index file name template (collection will be inserted)
    INDEX_FILENAME_TEMPLATE = "{collection}_bm25.json"
    METADATA_FILENAME_TEMPLATE = "{collection}_bm25_meta.json"

    def __init__(
        self,
        index_dir: str = "data/db/bm25/docs",
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> None:
        """Initialize the BM25Indexer.

        Args:
            index_dir: Directory to store index files.
            k1: BM25 k1 parameter (controls term frequency saturation).
            b: BM25 b parameter (controls document length normalization).
        """

        if k1 <= 0:
            raise ValueError(f"k1 must be > 0, got {k1}")
        if not 0 <= b <= 1:
            raise ValueError(f"b must be in [0, 1], got {b}")
        self._index_dir = Path(index_dir)
        self._k1 = k1
        self._b = b

        # Index data structures
        self._index: dict[str, dict[str, Any]] = {}  # term -> {idf, postings}
        self._doc_lengths: dict[str, int] = {}  # chunk_id -> document length
        self._avg_doc_length: float = 0.0
        self._num_docs: int = 0
        self._collection: str = "default"

        # Ensure index directory exists
        self._index_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"BM25Indexer initialized: index_dir={index_dir}, k1={k1}, b={b}"
        )

    @property
    def num_docs(self) -> int:
        """Get the number of documents in the index.

        Returns:
            Number of indexed documents.
        """
        return self._num_docs

    @property
    def num_terms(self) -> int:
        """Get the number of unique terms in the index.

        Returns:
            Number of unique indexed terms.
        """
        return len(self._index)

    @property
    def collection(self) -> str:
        """Get the collection name.

        Returns:
            Collection name.
        """
        return self._collection

    def _compute_idf(self, df: int, n: int) -> float:
        """Compute IDF weight using BM25 formula with smoothing.

        IDF = log((N - df + 0.5) / (df + 0.5) + 1)

        Args:
            df: Document frequency (number of docs containing the term).
            n: Total number of documents.

        Returns:
            IDF weight.
        """
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    def _compute_bm25_score(
        self,
        tf: int,
        doc_length: int,
        idf: float,
    ) -> float:
        """Compute BM25 score for a single term.

        BM25 formula:
        score = IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / avg_doc_length)))

        Args:
            tf: Term frequency in the document.
            doc_length: Length of the document (number of terms).
            idf: IDF weight for the term.

        Returns:
            BM25 score for this term.
        """
        # Avoid division by zero
        if self._avg_doc_length == 0:
            return 0.0

        numerator = tf * (self._k1 + 1)
        denominator = tf + self._k1 * (1 - self._b + self._b * (doc_length / self._avg_doc_length))

        if denominator == 0:
            return 0.0

        return idf * numerator / denominator

    def _compute_doc_length(self, text: str) -> int:
        """Compute document length (number of terms).

        Args:
            text: Document text.

        Returns:
            Number of terms in the document.
        """
        # Count whitespace-separated tokens
        return len(text.split())

    def build(
        self,
        records: list[ChunkRecord],
        collection: str = "default",
        trace: TraceContext | None = None,
    ) -> None:
        """Build the inverted index from chunk records.

        This method processes the sparse vectors from SparseEncoder,
        computes IDF weights, and builds the inverted index structure.

        Args:
            records: List of ChunkRecords with sparse_vector populated.
            collection: Collection name for managing multiple indices.
            trace: Optional trace context for observability.

        Example:
            >>> from ingestion.embedding import SparseEncoder
            >>> encoder = SparseEncoder()
            >>> records = encoder.encode(chunks)
            >>> indexer = BM25Indexer("data/db/bm25/my_index")
            >>> indexer.build(records)
        """
        if not records:
            logger.warning("No records to index")
            return

        logger.info(
            f"Building BM25 index from {len(records)} records, collection={collection}"
        )

        if trace:
            trace.record_stage(
                "bm25_indexing_start",
                {"record_count": len(records), "collection": collection},
            )

        # Reset index
        self._index = {}
        self._doc_lengths = {}
        self._num_docs = len(records)
        self._collection = collection

        # First pass: collect document frequencies and document lengths
        doc_frequencies: dict[str, int] = {}  # term -> df

        for record in records:
            # sparse_vector: 每个文档的词频
            #{"hello": 2, "world": 2}  # doc1: hello出现2次

            # doc_frequencies: 全局文档频率
            #{"hello": 3, "world": 5}  # hello在3个文档中出现
            if record.sparse_vector:  # sparse_vector: {term, tf_weights}
                # Count document frequency for each term
                for term in record.sparse_vector.keys():
                    doc_frequencies[term] = doc_frequencies.get(term, 0) + 1

                # Compute document length
                doc_length = self._compute_doc_length(record.text)
                self._doc_lengths[record.id] = doc_length

        # Compute average document length
        if self._doc_lengths:
            self._avg_doc_length = sum(self._doc_lengths.values()) / self._num_docs

        # Second pass: compute IDF and build inverted index
        for record in records:
            if not record.sparse_vector:
                continue

            for term, tf in record.sparse_vector.items():
                # Get or create term entry
                if term not in self._index:
                    df = doc_frequencies.get(term, 0)
                    idf = self._compute_idf(df, self._num_docs)
                    self._index[term] = {
                        "idf": idf,
                        "postings": [],
                    }

                # Add to postings list
                self._index[term]["postings"].append({
                    "chunk_id": record.id,
                    "tf": tf,
                    "doc_length": self._doc_lengths[record.id],
                })

        logger.info(
            f"BM25 index built: {self._num_docs} docs, "
            f"{len(self._index)} terms"
        )

        if trace:
            trace.record_stage(
                "bm25_indexing_complete",
                {
                    "num_docs": self._num_docs,
                    "num_terms": len(self._index),
                },
            )

    def search(
        self,
        query: str,
        top_k: int = 10,
        trace: TraceContext | None = None,
    ) -> list[tuple[str, float]]:
        """Search the index for matching documents.

        This method computes BM25 scores for the query and returns
        the top-k matching documents.

        Args:
            query: Search query string.
            top_k: Number of top results to return.
            trace: Optional trace context.

        Returns:
            List of (chunk_id, score) tuples, sorted by score descending.
            [('doc1', 0.8), ('doc2', 0.6), ('doc3', 0.9)]

        Example:
            >>> results = indexer.search("machine learning", top_k=5)
            >>> for chunk_id, score in results:
            ...     print(f"{chunk_id}: {score:.4f}")
        """
        if not self._index:
            logger.warning("Index is empty, returning empty results")
            return []

        # Tokenize query
        query_terms = self._tokenize_query(query)

        if not query_terms:
            logger.info("Empty query, returning empty results")
            return []

        logger.info(
            f"Searching index for query with {len(query_terms)} terms, top_k={top_k}"
        )

        if trace:
            trace.record_stage(
                "bm25_search_start",
                {
                    "query": query,
                    "query_terms": query_terms,
                    "top_k": top_k,
                },
            )

        # Compute BM25 scores for each document
        scores: dict[str, float] = {}

        for term in query_terms:
            if term not in self._index:
                continue

            term_info = self._index[term]
            idf = term_info["idf"]

            for posting in term_info["postings"]:
                chunk_id = posting["chunk_id"]
                tf = posting["tf"]
                doc_length = posting["doc_length"]

                # Compute BM25 score for this term
                term_score = self._compute_bm25_score(tf, doc_length, idf)

                scores[chunk_id] = scores.get(chunk_id, 0.0) + term_score

        # Sort by score and return top-k
        results = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:top_k]

        logger.info(f"Search returned {len(results)} results")

        if trace:
            trace.record_stage(
                "bm25_search_complete",
                {"result_count": len(results)},
            )

        return results

    def _tokenize_query(self, query: str) -> list[str]:
        """Tokenize a query string.

        Args:
            query: Query string to tokenize.

        Returns:
            List of lowercase tokens.
        """
        # Simple whitespace tokenization and lowercasing
        return [t.lower() for t in query.split() if t.strip()]

    def save(self) -> None:
        """Serialize the index to disk.

        Saves the inverted index and metadata to {index_dir}/{collection}_bm25.json.
        """
        if not self._index:
            logger.warning("Index is empty, nothing to save")
            return

        # Save index: {index_dir}/{collection}_bm25.json
        index_filename = self.INDEX_FILENAME_TEMPLATE.format(collection=self._collection)
        index_path = self._index_dir / index_filename
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(self._index, f, indent=2, ensure_ascii=False)

        # Save metadata: {index_dir}/{collection}_bm25_meta.json
        meta_filename = self.METADATA_FILENAME_TEMPLATE.format(collection=self._collection)
        meta = {
            "collection": self._collection,
            "num_docs": self._num_docs,
            "num_terms": len(self._index),
            "avg_doc_length": self._avg_doc_length,
            "doc_lengths": self._doc_lengths,
            "k1": self._k1,
            "b": self._b,
        }
        meta_path = self._index_dir / meta_filename
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info(f"Index saved to {self._index_dir}/{index_filename}")

    def load(self, collection: str = "default") -> bool:
        """Load the index from disk.

        Args:
            collection: Collection name to load.

        Returns:
            True if index was loaded successfully, False otherwise.
        """
        self._collection = collection

        # Load from {index_dir}/{collection}_bm25.json
        index_filename = self.INDEX_FILENAME_TEMPLATE.format(collection=collection)
        meta_filename = self.METADATA_FILENAME_TEMPLATE.format(collection=collection)

        index_path = self._index_dir / index_filename
        meta_path = self._index_dir / meta_filename

        if not index_path.exists() or not meta_path.exists():
            logger.warning(f"Index files not found: {index_path}, {meta_path}")
            return False

        # Load index
        with open(index_path, "r", encoding="utf-8") as f:
            self._index = json.load(f)

        # Load metadata
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self._num_docs = meta["num_docs"]
        self._avg_doc_length = meta["avg_doc_length"]
        self._doc_lengths = meta["doc_lengths"]
        self._k1 = meta["k1"]
        self._b = meta["b"]

        logger.info(
            f"Index loaded from {self._index_dir}/{index_filename}: "
            f"{self._num_docs} docs, {len(self._index)} terms"
        )

        return True

    def clear(self) -> None:
        """Clear the index and reset state."""
        self._index = {}
        self._doc_lengths = {}
        self._num_docs = 0
        self._avg_doc_length = 0.0

        logger.info("Index cleared")

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"BM25Indexer("
            f"index_dir={self._index_dir}, "
            f"docs={self._num_docs}, "
            f"terms={len(self._index)})"
        )
