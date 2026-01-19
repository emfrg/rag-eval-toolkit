"""Base class for RAG systems.

All RAG implementations must inherit from RAGSystemBase and implement
the required methods: create_index, load_index, retrieve, generate, and query.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_eval.dataset.corpus import Corpus
    from rag_eval.systems.config import RAGConfig
    from rag_eval.systems.response import RAGResponse, RetrievedDocument

logger = logging.getLogger(__name__)


@dataclass
class IndexReport:
    """Report from index creation/loading.

    Attributes:
        total_documents: Total number of documents in the corpus.
        indexed_documents: Number of documents that were indexed.
        reused_existing: Whether an existing index was reused.
        index_path: Path where the index is stored.
    """

    total_documents: int
    indexed_documents: int
    reused_existing: bool
    index_path: str | None = None


class RAGSystemBase(ABC):
    """Abstract base class for RAG system implementations.

    All RAG architectures (Naive, GraphRAG, Agentic, Custom) must implement
    this interface to be compatible with the evaluation framework.

    Example:
        ```python
        class MyCustomRAG(RAGSystemBase):
            def __init__(self, config: RAGConfig):
                super().__init__(config)
                # Custom initialization

            def create_index(self, corpus: Corpus) -> IndexReport:
                # Build your index
                ...

            def load_index(self) -> None:
                # Load existing index
                ...

            def retrieve(self, query: str) -> list[RetrievedDocument]:
                # Retrieve relevant documents
                ...

            def generate(self, query: str, contexts: list[str]) -> str:
                # Generate answer from contexts
                ...

            def query(self, question: str) -> RAGResponse:
                # Full RAG pipeline: retrieve + generate
                ...
        ```
    """

    def __init__(self, config: RAGConfig) -> None:
        """Initialize the RAG system.

        Args:
            config: Configuration for this RAG system.
        """
        self.config = config
        self._index_loaded = False

    @property
    def name(self) -> str:
        """Return the name of this RAG system."""
        return self.__class__.__name__

    @abstractmethod
    def create_index(self, corpus: Corpus) -> IndexReport:
        """Build the index from a corpus.

        Args:
            corpus: The document corpus to index.

        Returns:
            IndexReport with indexing statistics.
        """
        raise NotImplementedError

    @abstractmethod
    def load_index(self) -> None:
        """Load an existing index.

        Raises:
            FileNotFoundError: If no index exists at the configured path.
        """
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, query: str) -> list[RetrievedDocument]:
        """Retrieve relevant documents for a query.

        Args:
            query: The search query.

        Returns:
            List of retrieved documents with scores.
        """
        raise NotImplementedError

    @abstractmethod
    def generate(self, query: str, contexts: list[str]) -> str:
        """Generate an answer given contexts.

        Args:
            query: The question to answer.
            contexts: List of context strings from retrieval.

        Returns:
            Generated answer string.
        """
        raise NotImplementedError

    @abstractmethod
    def query(self, question: str) -> RAGResponse:
        """Execute the full RAG pipeline: retrieve + generate.

        This is the main entry point for querying the RAG system.

        Args:
            question: The question to answer.

        Returns:
            RAGResponse containing the answer and retrieved evidence.
        """
        raise NotImplementedError

    def ensure_index_loaded(self) -> None:
        """Ensure the index is loaded before querying.

        Raises:
            RuntimeError: If index is not loaded.
        """
        if not self._index_loaded:
            raise RuntimeError(
                f"{self.name}: Index not loaded. "
                "Call create_index() or load_index() first."
            )
