"""Response types for RAG systems."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RetrievedDocument:
    """A document retrieved by the RAG system.

    Attributes:
        doc_id: Unique identifier for the document.
        content: The text content of the document.
        score: Relevance score (higher is more relevant).
        metadata: Optional metadata dict (source, author, etc.).
    """

    doc_id: str
    content: str
    score: float = 0.0
    metadata: dict[str, str] | None = None

    def __repr__(self) -> str:
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"RetrievedDocument(doc_id={self.doc_id!r}, score={self.score:.3f}, content={content_preview!r})"


@dataclass
class RAGResponse:
    """Response from a RAG system query.

    Attributes:
        response: The generated answer.
        retrieved_evidence: List of document IDs that were retrieved.
        retrieved_evidence_texts: List of document contents that were retrieved.
        retrieved_documents: Full RetrievedDocument objects (optional).
    """

    response: str
    retrieved_evidence: list[str] = field(default_factory=list)
    retrieved_evidence_texts: list[str] = field(default_factory=list)
    retrieved_documents: list[RetrievedDocument] = field(default_factory=list)

    @classmethod
    def from_documents(cls, response: str, documents: list[RetrievedDocument]) -> RAGResponse:
        """Create a RAGResponse from a list of RetrievedDocuments.

        Args:
            response: The generated answer.
            documents: List of retrieved documents.

        Returns:
            RAGResponse with all fields populated.
        """
        return cls(
            response=response,
            retrieved_evidence=[doc.doc_id for doc in documents],
            retrieved_evidence_texts=[doc.content for doc in documents],
            retrieved_documents=documents,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary format for JSON serialization."""
        return {
            "response": self.response,
            "retrieved_evidence": self.retrieved_evidence,
            "retrieved_evidence_texts": self.retrieved_evidence_texts,
        }
