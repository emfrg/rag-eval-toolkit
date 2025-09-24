from .base import BaseRAGSystem, IndexReport
from .config import RAGConfig
from .dataset import RAGDataset
from .naive import NaiveRAGSystem
from .rag import RAGSystem

try:
    from .graphrag import LightRAGSystem  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    LightRAGSystem = None  # type: ignore

__all__ = [
    "BaseRAGSystem",
    "IndexReport",
    "RAGConfig",
    "RAGDataset",
    "RAGSystem",
    "NaiveRAGSystem",
]

if LightRAGSystem is not None:
    __all__.append("LightRAGSystem")
