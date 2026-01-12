"""Configuration classes for RAG systems.

Each RAG architecture has its own config class. The main RAGConfig class
holds the common settings and architecture-specific configs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class NaiveRAGConfig:
    """Configuration for Naive RAG system.

    Attributes:
        chunk_documents: Whether to chunk documents before indexing.
        chunk_size: Size of chunks in tokens.
        chunk_overlap: Overlap between chunks in tokens.
        embedding_model: Name of the embedding model to use.
        vector_store: Type of vector store ("faiss" or "chroma").
        k_retrieve: Number of documents to retrieve initially.
        similarity_threshold: Maximum distance threshold for retrieval.
        use_reranker: Whether to use a reranker after retrieval.
        reranker_model: Model name for reranking.
        rerank_threshold: Minimum score threshold after reranking.
        min_docs: Minimum number of documents to return.
        max_docs: Maximum number of documents to return.
        cache_dir: Directory for storing the index cache.
        inline_metadata: Whether to include metadata inline in content.
    """

    chunk_documents: bool = False
    chunk_size: int = 400
    chunk_overlap: int = 50
    embedding_model: str = "text-embedding-3-small"
    vector_store: Literal["faiss", "chroma"] = "faiss"
    k_retrieve: int = 10
    similarity_threshold: float = 1.0
    use_reranker: bool = False
    reranker_model: str = "BAAI/bge-reranker-base"
    rerank_threshold: float = 0.5
    min_docs: int = 0
    max_docs: int = 4
    cache_dir: Path = field(default_factory=lambda: Path("./rag_cache"))
    inline_metadata: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir).expanduser()


@dataclass
class GraphRAGIndexingConfig:
    """Configuration for GraphRAG indexing.

    Attributes:
        cache_dir: Directory for storing the graph cache.
        max_parallel_insert: Maximum parallel insertions.
        batch_size: Batch size for indexing.
        force_reindex: Whether to force reindexing.
        inline_metadata: Whether to include metadata inline.
    """

    cache_dir: Path = field(default_factory=lambda: Path("./graphrag_cache"))
    max_parallel_insert: int = 4
    batch_size: int = 128
    force_reindex: bool = False
    inline_metadata: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir).expanduser()


@dataclass
class GraphRAGQueryConfig:
    """Configuration for GraphRAG queries.

    Attributes:
        mode: Query mode - "semantic", "graph", or "hybrid".
        top_k: Number of results to consider.
    """

    mode: Literal["semantic", "graph", "hybrid"] = "hybrid"
    top_k: int = 50


@dataclass
class GraphRAGConfig:
    """Configuration for GraphRAG system (LightRAG).

    Attributes:
        indexing: Indexing configuration.
        query: Query configuration.
    """

    indexing: GraphRAGIndexingConfig = field(default_factory=GraphRAGIndexingConfig)
    query: GraphRAGQueryConfig = field(default_factory=GraphRAGQueryConfig)

    def __post_init__(self) -> None:
        if isinstance(self.indexing, dict):
            self.indexing = GraphRAGIndexingConfig(**self.indexing)
        if isinstance(self.query, dict):
            self.query = GraphRAGQueryConfig(**self.query)


@dataclass
class RAGConfig:
    """Main configuration for RAG systems.

    This is the top-level config that holds common settings (LLM model,
    temperature, etc.) and architecture-specific configs.

    Attributes:
        rag_type: Which RAG architecture to use.
        llm_provider: LLM provider - "anthropic" or "openai".
        llm_model: Name of the LLM model for generation.
        temperature: Sampling temperature for generation.
        max_tokens: Maximum tokens in generated response.
        naive: Configuration for Naive RAG (if rag_type="naive").
        graphrag: Configuration for GraphRAG (if rag_type="graphrag").

    Example:
        ```python
        # Using Anthropic Claude (default)
        config = RAGConfig(
            rag_type="naive",
            llm_provider="anthropic",
            llm_model="claude-sonnet-4-20250514",
            naive=NaiveRAGConfig(
                chunk_documents=True,
                use_reranker=True,
            ),
        )

        # Using OpenAI
        config = RAGConfig(
            rag_type="naive",
            llm_provider="openai",
            llm_model="gpt-4o-mini",
        )
        ```
    """

    rag_type: Literal["naive", "graphrag", "custom"] = "naive"
    llm_provider: Literal["anthropic", "openai"] = "anthropic"
    llm_model: str = "claude-sonnet-4-20250514"
    temperature: float = 1.0
    max_tokens: int = 512
    naive: NaiveRAGConfig = field(default_factory=NaiveRAGConfig)
    graphrag: GraphRAGConfig = field(default_factory=GraphRAGConfig)

    def __post_init__(self) -> None:
        # Handle dict initialization for nested configs
        if isinstance(self.naive, dict):
            self.naive = NaiveRAGConfig(**self.naive)
        if isinstance(self.graphrag, dict):
            self.graphrag = GraphRAGConfig(**self.graphrag)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "rag_type": self.rag_type,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "naive": {
                "chunk_documents": self.naive.chunk_documents,
                "chunk_size": self.naive.chunk_size,
                "chunk_overlap": self.naive.chunk_overlap,
                "embedding_model": self.naive.embedding_model,
                "vector_store": self.naive.vector_store,
                "k_retrieve": self.naive.k_retrieve,
                "similarity_threshold": self.naive.similarity_threshold,
                "use_reranker": self.naive.use_reranker,
                "reranker_model": self.naive.reranker_model,
                "rerank_threshold": self.naive.rerank_threshold,
                "min_docs": self.naive.min_docs,
                "max_docs": self.naive.max_docs,
                "cache_dir": str(self.naive.cache_dir),
                "inline_metadata": self.naive.inline_metadata,
            },
            "graphrag": {
                "indexing": {
                    "cache_dir": str(self.graphrag.indexing.cache_dir),
                    "max_parallel_insert": self.graphrag.indexing.max_parallel_insert,
                    "batch_size": self.graphrag.indexing.batch_size,
                    "force_reindex": self.graphrag.indexing.force_reindex,
                    "inline_metadata": self.graphrag.indexing.inline_metadata,
                },
                "query": {
                    "mode": self.graphrag.query.mode,
                    "top_k": self.graphrag.query.top_k,
                },
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> RAGConfig:
        """Create a RAGConfig from a dictionary."""
        return cls(**data)

    def get_config_signature(self) -> str:
        """Generate a unique signature for this config.

        Used for caching and identifying experiment runs.
        """
        import hashlib
        import json

        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha1(config_str.encode()).hexdigest()[:10]
