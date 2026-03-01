#!/usr/bin/env python
"""Query script for the Modular RAG MCP Server.

This script provides a command-line interface for querying the knowledge hub
using hybrid search (dense + sparse) with optional reranking.

Usage:
    # Basic query
    python scripts/query.py --query "如何配置 Azure？"

    # Query with custom top-k
    python scripts/query.py --query "机器学习" --top-k 5

    # Query specific collection
    python scripts/query.py --query "配置问题" --collection contracts

    # Verbose mode (show intermediate results)
    python scripts/query.py --query "配置问题" --verbose

    # Skip reranking
    python scripts/query.py --query "配置问题" --no-rerank

Exit codes:
    0 - Success
    1 - Query execution error
    2 - Configuration error
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is in path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.query_engine import (
    DenseRetriever,
    HybridSearch,
    QueryProcessor,
    RRFFusion,
    SparseRetriever,
    create_hybrid_search,
    create_query_processor,
)
from core.query_engine.reranker import CoreReranker, CoreRerankerConfig, create_core_reranker
from core.settings import RetrievalConfig, RerankConfig, Settings, load_settings
from core.trace.trace_context import TraceContext
from core.types import RetrievalResult
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.vector_store.chroma_store import ChromaStore
from observability.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Query the Modular RAG knowledge hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--query", "-q",
        required=True,
        help="Query text to search for"
    )

    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=None,
        help="Number of results to return (default: from config or 10)"
    )

    parser.add_argument(
        "--collection", "-c",
        default=None,
        help="Collection name to query (default: from config)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show intermediate results (dense, sparse, fusion, rerank)"
    )

    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip reranking step"
    )

    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Path to configuration file (default: config/settings.yaml)"
    )

    return parser.parse_args()


def load_components(
    settings: Settings,
    collection: Optional[str] = None,
    enable_rerank: bool = True,
) -> tuple[
    QueryProcessor,
    DenseRetriever,
    SparseRetriever,
    HybridSearch,
    Optional[CoreReranker],
]:
    """Load and initialize query components.

    Args:
        settings: Settings object
        collection: Optional collection name override
        enable_rerank: Whether to enable reranking

    Returns:
        Tuple of (QueryProcessor, DenseRetriever, SparseRetriever, HybridSearch, Reranker)
    """
    # Create embedding client
    embedding_client = EmbeddingFactory.create(settings)

    # Determine collection name: CLI > settings.vector_store.collection_name > default
    if collection:
        collection_name = collection
    else:
        # Try to get from settings.vector_store.collection_name
        vector_store = getattr(settings, "vector_store", None)
        if vector_store:
            collection_name = getattr(vector_store, "collection_name", None) or "default"
        else:
            collection_name = "default"

    # Create vector store
    vector_store = ChromaStore(
        persist_directory=getattr(settings, "persist_directory", None),
        collection_name=collection_name,
    )

    # Create DenseRetriever
    dense_retriever = DenseRetriever(
        settings=settings,
        embedding_client=embedding_client,
        vector_store=vector_store,
    )

    # Create SparseRetriever
    # bm25_index_dir = getattr(settings, "bm25_index_dir", None)
    # if bm25_index_dir:
        # Use collection-specific subdirectory
    bm25_path = f"data/db/bm25/{collection_name}"


    # Try to create BM25 indexer (may not exist if not indexed yet)
    try:
        from ingestion.storage.bm25_indexer import BM25Indexer

        bm25_indexer = BM25Indexer(index_dir=bm25_path)
    except Exception:
        bm25_indexer = None

    sparse_retriever = SparseRetriever(
        settings=settings,
        bm25_indexer=bm25_indexer,
        vector_store=vector_store,
    )

    # Create QueryProcessor using factory function
    query_processor = create_query_processor()

    # Create HybridSearch using factory function to load config from settings
    fusion = RRFFusion()
    hybrid_search = create_hybrid_search(
        settings=settings,
        query_processor=query_processor,
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        fusion=fusion,
    )

    # Create Reranker (optional) using factory function
    reranker = None
    if enable_rerank:
        try:
            from libs.reranker.reranker_factory import RerankerFactory

            # Register reranker providers
            from libs.reranker.cross_encoder_reranker import CrossEncoderReranker
            from libs.reranker.llm_reranker import LLMReranker
            RerankerFactory.register("cross_encoder", CrossEncoderReranker)
            RerankerFactory.register("llm", LLMReranker)

            # Use factory function to create CoreReranker with settings
            reranker = create_core_reranker(settings=settings)
        except Exception as e:
            logger.warning(f"Failed to create reranker: {e}, continuing without reranking")

    return query_processor, dense_retriever, sparse_retriever, hybrid_search, reranker


def format_result(result: RetrievalResult, index: int) -> str:
    """Format a single retrieval result.

    Args:
        result: RetrievalResult to format
        index: Result index (1-based)

    Returns:
        Formatted string
    """
    # Extract source and page from metadata
    metadata = result.metadata or {}
    source = metadata.get("source_path", metadata.get("source", "unknown"))
    page = metadata.get("page", metadata.get("page_num", "-"))

    # Truncate text for display
    text_preview = result.text[:200]
    if len(result.text) > 200:
        text_preview += "..."

    return (
        f"{index}. [score={result.score:.4f}] {text_preview}\n"
        f"   Source: {source}, Page: {page}"
    )


def print_results(
    results: list[RetrievalResult],
    title: str = "RESULTS",
    show_index: bool = True,
) -> None:
    """Print formatted retrieval results.

    Args:
        results: List of RetrievalResult
        title: Title for the output section
        show_index: Whether to show result index
    """
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print('=' * 60)

    if not results:
        print("No results found.")
        return

    for i, result in enumerate(results, 1):
        if show_index:
            print(format_result(result, i))
        else:
            print(format_result(result, i - 1))
        print()


def print_intermediate_results(
    dense_results: list[RetrievalResult],
    sparse_results: list[RetrievalResult],
    fusion_results: list[RetrievalResult],
    rerank_results: Optional[list[RetrievalResult]],
    used_rerank: bool,
) -> None:
    """Print intermediate results in verbose mode.

    Args:
        dense_results: Results from dense retrieval
        sparse_results: Results from sparse retrieval
        fusion_results: Results from fusion
        rerank_results: Results from reranking (if enabled)
        used_rerank: Whether reranking was used
    """
    print_results(dense_results, "DENSE RETRIEVAL RESULTS")
    print_results(sparse_results, "SPARSE RETRIEVAL RESULTS")
    print_results(fusion_results, "FUSION RESULTS")

    if used_rerank and rerank_results:
        print_results(rerank_results, "RERANKED RESULTS")
    else:
        print("\n[Reranking disabled or skipped]")


def resolve_top_k(
    cli_top_k: Optional[int],
    settings: Settings,
    default: int = 10,
) -> tuple[int, int]:
    """Resolve fusion_top_k and rerank_top_k with unified priority: CLI > settings > default.

    Args:
        cli_top_k: Top-K from command line (None if not specified)
        settings: Settings object
        default: Default value if neither CLI nor settings provided

    Returns:
        Tuple of (fusion_top_k, rerank_top_k)
    """
    # CLI has highest priority - controls both fusion and rerank
    if cli_top_k is not None:
        return cli_top_k, cli_top_k

    # Get from settings
    retrieval: Optional[RetrievalConfig] = getattr(settings, "retrieval", None)
    rerank: Optional[RerankConfig] = getattr(settings, "rerank", None)

    # fusion_top_k: try retrieval.fusion_top_k, fall back to default
    fusion_top_k = getattr(retrieval, "fusion_top_k", None) if retrieval else None
    if fusion_top_k is None:
        fusion_top_k = default

    # rerank_top_k: try rerank.top_k, fall back to fusion_top_k or default
    rerank_top_k = getattr(rerank, "top_k", None) if rerank else None
    if rerank_top_k is None:
        rerank_top_k = fusion_top_k

    return fusion_top_k, rerank_top_k


def main() -> int:
    """Main entry point for the query script.

    Returns:
        Exit code (0=success, 1=error)
    """
    args = parse_args()

    # Load configuration first to resolve top_k priority
    try:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"[FAIL] Configuration file not found: {config_path}")
            return 2

        settings = load_settings(str(config_path))
        print(f"[OK] Configuration loaded from: {config_path}")
    except Exception as e:
        print(f"[FAIL] Failed to load configuration: {e}")
        return 2

    # Resolve top_k with unified priority: CLI > settings > default
    fusion_top_k, rerank_top_k = resolve_top_k(args.top_k, settings)

    # Resolve collection: CLI > settings.vector_store.collection_name > default
    if args.collection:
        collection_name = args.collection
    else:
        vector_store = getattr(settings, "vector_store", None)
        if vector_store:
            collection_name = getattr(vector_store, "collection_name", None) or "default"
        else:
            collection_name = "default"

    print("[*] Modular RAG Query Script")
    print("=" * 60)
    print(f"Query: {args.query}")
    print(f"Top-K: fusion={fusion_top_k}, rerank={rerank_top_k} (CLI: {args.top_k})")
    print(f"Rerank: {'enabled' if not args.no_rerank else 'disabled'}")
    print(f"Collection: {collection_name} (CLI: {args.collection})")
    print("=" * 60)

    # Load components
    try:
        print("\n[INFO] Initializing components...")
        _, _, _, hybrid_search, reranker = load_components(
            settings=settings,
            collection=collection_name,
            enable_rerank=not args.no_rerank,
        )
        print("[OK] Components initialized")
    except Exception as e:
        print(f"[FAIL] Failed to initialize components: {e}")
        logger.exception("Component initialization failed")
        return 1

    # Create trace context for verbose mode
    trace = TraceContext() if args.verbose else None

    # Execute hybrid search
    try:
        print(f"\n[INFO] Executing hybrid search...")
        # Use return_details=True to get intermediate results for verbose mode
        detailed_results = hybrid_search.search(
            query=args.query,
            top_k=fusion_top_k,
            collection=collection_name,
            trace=trace,
            return_details=args.verbose,
        )

        # Extract results based on return type
        if hasattr(detailed_results, 'results'):
            # HybridSearchResult with details
            fusion_results = detailed_results.results
            dense_results = detailed_results.dense_results or []
            sparse_results = detailed_results.sparse_results or []
        else:
            # Simple list result
            fusion_results = detailed_results
            dense_results = detailed_results
            sparse_results = detailed_results

        print(f"[OK] Hybrid search complete: {len(fusion_results)} results")
    except Exception as e:
        print(f"[FAIL] Hybrid search failed: {e}")
        logger.exception("Hybrid search failed")
        return 1

    # Apply reranking
    final_results = fusion_results
    used_rerank = False

    if reranker and not args.no_rerank and fusion_results:
        try:
            print(f"[INFO] Applying reranking...")
            rerank_result = reranker.rerank(
                query=args.query,
                candidates=fusion_results,
                top_k=rerank_top_k,
                trace=trace,
            )
            final_results = rerank_result.results
            used_rerank = True

            if rerank_result.used_fallback:
                print(f"[WARN] Reranking fell back to fusion results: {rerank_result.fallback_reason}")
            else:
                print(f"[OK] Reranking complete: {len(final_results)} results")
        except Exception as e:
            print(f"[WARN] Reranking failed, using fusion results: {e}")
            final_results = fusion_results

    # Handle empty results
    if not final_results:
        print("\n" + "=" * 60)
        print("No results found.")
        print("=" * 60)
        print("\nTip: If you haven't ingested any documents yet, run:")
        print("  python scripts/ingest.py --path <your documents>")
        return 0

    # Print results
    if args.verbose:
        print_intermediate_results(
            dense_results=dense_results,
            sparse_results=sparse_results,
            fusion_results=fusion_results,
            rerank_results=final_results if used_rerank else None,
            used_rerank=used_rerank,
        )

    print_results(final_results, "FINAL RESULTS")

    return 0


if __name__ == "__main__":
    sys.exit(main())
