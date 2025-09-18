# rag_system/config.py
from dataclasses import dataclass
from typing import Optional, Literal
from pathlib import Path


@dataclass
class RAGConfig:
    # Chunking
    chunk_size: int = 400
    chunk_overlap: int = 50

    # Embeddings
    embedding_model: Literal["text-embedding-3-small", "text-embedding-3-large"] = (
        "text-embedding-3-small"
    )

    # Retrieval
    vector_store_type: Literal["faiss", "chroma"] = "faiss"
    k_retrieve: int = 5

    # Reranking
    use_reranker: bool = False
    reranker_model: Optional[str] = "BAAI/bge-reranker-base"
    k_rerank: int = 3

    # Generation
    llm_model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 512

    # Storage
    cache_dir: Optional[Path] = Path("./rag_cache")
