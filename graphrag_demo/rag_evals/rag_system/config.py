# # rag_system/config.py
# from dataclasses import dataclass
# from typing import Optional, Literal
# from pathlib import Path


# @dataclass
# class RAGConfig:
#     # Chunking
#     chunking: bool = False
#     chunk_size: int = 400
#     chunk_overlap: int = 50

#     # Embeddings
#     embedding_model: Literal["text-embedding-3-small", "text-embedding-3-large"] = (
#         "text-embedding-3-small"
#     )

#     # Retrieval
#     vector_store_type: Literal["faiss", "chroma"] = "faiss"
#     k_retrieve: int = 10
#     similarity_threshold: float = 1.0  # FAISS distance threshold (lower = more similar)

#     # Reranking
#     use_reranker: bool = False
#     reranker_model: Optional[str] = "BAAI/bge-reranker-base"
#     rerank_threshold: float = 0.5
#     min_docs: int = 0
#     max_docs: int = 4

#     # Generation
#     llm_model: str = "gpt-4o-mini"
#     temperature: float = 1
#     max_tokens: int = 512

#     # Storage
#     cache_dir: Optional[Path] = Path("./rag_cache")


from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union


def _as_path(value: Union[str, Path]) -> Path:
    if isinstance(value, Path):
        return value
    return Path(value).expanduser()


@dataclass
class NaiveRetrievalConfig:
    chunk_documents: bool = False
    chunk_size: int = 400
    chunk_overlap: int = 50
    embedding_model: str = "text-embedding-3-small"
    vector_store: Literal["faiss", "chroma"] = "faiss"
    k_retrieve: int = 10
    similarity_threshold: float = 1.0
    use_reranker: bool = False
    reranker_model: Optional[str] = "BAAI/bge-reranker-base"
    rerank_threshold: float = 0.5
    min_docs: int = 0
    max_docs: int = 4
    cache_dir: Path = Path("./rag_cache")

    def __post_init__(self) -> None:
        self.cache_dir = _as_path(self.cache_dir)


@dataclass
class LightRAGIndexingConfig:
    graph_cache_dir: Path = Path("./graphrag_cache")
    max_parallel_insert: int = 4
    batch_size: int = 128
    force_reindex: bool = False

    def __post_init__(self) -> None:
        self.graph_cache_dir = _as_path(self.graph_cache_dir)


@dataclass
class LightRAGQueryConfig:
    mode: Literal["semantic", "graph", "hybrid"] = "hybrid"
    top_k: int = 50
    summary_top_k: Optional[int] = None


@dataclass
class LightRAGConfig:
    indexing: LightRAGIndexingConfig = field(default_factory=LightRAGIndexingConfig)
    query: LightRAGQueryConfig = field(default_factory=LightRAGQueryConfig)


@dataclass
class RAGConfig:
    rag_model: Literal["naive", "graphrag"] = "naive"
    llm_model: str = "gpt-4o-mini"
    temperature: float = 1.0
    max_tokens: int = 512
    naive: NaiveRetrievalConfig = field(default_factory=NaiveRetrievalConfig)
    graphrag: LightRAGConfig = field(default_factory=LightRAGConfig)

    def __post_init__(self) -> None:
        if isinstance(self.naive, dict):
            self.naive = NaiveRetrievalConfig(**self.naive)
        if isinstance(self.graphrag, dict):
            self.graphrag = self._build_graphrag_config(self.graphrag)

    def _build_graphrag_config(self, data: Dict[str, Any]) -> LightRAGConfig:
        indexing = data.get("indexing", {})
        query = data.get("query", {})
        if not isinstance(indexing, LightRAGIndexingConfig):
            indexing = LightRAGIndexingConfig(**indexing)
        if not isinstance(query, LightRAGQueryConfig):
            query = LightRAGQueryConfig(**query)
        return LightRAGConfig(indexing=indexing, query=query)
