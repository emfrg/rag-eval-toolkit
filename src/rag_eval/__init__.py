"""RAG Eval Toolkit - A complete RAG experimentation framework."""

__version__ = "0.1.0"

from rag_eval.dataset.corpus import Corpus, Document
from rag_eval.dataset.eval_dataset import EvalDataset, EvalQuestion
from rag_eval.systems.base import RAGSystemBase
from rag_eval.systems.config import EvalConfig, RAGConfig
from rag_eval.systems.response import RAGResponse, RetrievedDocument

__all__ = [
    # Dataset
    "Corpus",
    "Document",
    "EvalDataset",
    "EvalQuestion",
    # Systems
    "RAGSystemBase",
    "RAGConfig",
    "EvalConfig",
    "RAGResponse",
    "RetrievedDocument",
]
