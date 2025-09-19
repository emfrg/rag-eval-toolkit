# rag_system/config.py
from dataclasses import dataclass
from typing import Optional, Literal
from pathlib import Path


@dataclass
class RAGConfig:
    # Chunking
    chunking: bool = False
    chunk_size: int = 400
    chunk_overlap: int = 50

    # Embeddings
    embedding_model: Literal["text-embedding-3-small", "text-embedding-3-large"] = (
        "text-embedding-3-small"
    )

    # Retrieval
    vector_store_type: Literal["faiss", "chroma"] = "faiss"
    k_retrieve: int = 10
    similarity_threshold: float = 1.0  # FAISS distance threshold (lower = more similar)

    # Reranking
    use_reranker: bool = False
    reranker_model: Optional[str] = "BAAI/bge-reranker-base"
    rerank_threshold: float = 0.5
    min_docs: int = 0
    max_docs: int = 4

    # Generation
    llm_model: str = "gpt-4o-mini"
    temperature: float = 1
    max_tokens: int = 512

    # Storage
    cache_dir: Optional[Path] = Path("./rag_cache")
