from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

from langchain.schema import Document

from .config import RAGConfig
from .dataset import RAGDataset


@dataclass
class IndexReport:
    total_documents: int
    ingested_documents: int
    reused_existing: bool


class BaseRAGSystem(ABC):
    def __init__(self, config: RAGConfig, dataset: Optional[RAGDataset] = None) -> None:
        self.config = config
        self.dataset = dataset

    @abstractmethod
    def build_index(self, dataset: RAGDataset) -> IndexReport:
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, question: str) -> List[Document]:
        raise NotImplementedError

    @abstractmethod
    def query(self, question: str) -> Tuple[str, List[Document]]:
        raise NotImplementedError
