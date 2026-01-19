"""Corpus class for managing document collections."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A single document in the corpus.

    Attributes:
        doc_id: Unique identifier for the document.
        content: The text content of the document.
        metadata: Optional metadata dict (source, author, category, etc.).

    Example:
        ```python
        doc = Document(
            doc_id="doc_123",
            content="The quick brown fox jumps over the lazy dog.",
            metadata={
                "source": "Wikipedia",
                "author": "Unknown",
                "category": "animals",
            }
        )
        ```
    """

    doc_id: str
    content: str
    metadata: dict[str, str] | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "doc_id": self.doc_id,
            "content": self.content,
        }
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: dict) -> Document:
        """Create a Document from a dictionary."""
        return cls(
            doc_id=data["doc_id"],
            content=data["content"],
            metadata=data.get("metadata"),
        )

    def get_content_with_metadata(self, separator: str = "\n---METADATA---\n") -> str:
        """Get content with metadata appended.

        Useful for inline metadata embedding.

        Args:
            separator: String to separate content from metadata.

        Returns:
            Content string with metadata appended.
        """
        if not self.metadata:
            return self.content

        metadata_str = "\n".join(f"{k}: {v}" for k, v in self.metadata.items())
        return f"{self.content}{separator}{metadata_str}"


@dataclass
class Corpus:
    """A collection of documents for RAG evaluation.

    The corpus can be loaded from a JSONL file where each line is a document,
    or created programmatically by adding documents.

    Attributes:
        documents: List of Document objects.
        name: Optional name for the corpus.

    Example:
        ```python
        # Load from file
        corpus = Corpus.from_jsonl("./data/corpus.jsonl")

        # Or create programmatically
        corpus = Corpus(name="my_corpus")
        corpus.add(Document(doc_id="doc_1", content="Hello world"))

        # Access documents
        for doc in corpus:
            print(doc.content)

        # Get by ID
        doc = corpus.get("doc_1")
        ```
    """

    documents: list[Document] = field(default_factory=list)
    name: str = "corpus"

    def __post_init__(self) -> None:
        self._id_to_doc: dict[str, Document] = {}
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuild the document ID index."""
        self._id_to_doc = {doc.doc_id: doc for doc in self.documents}

    def __len__(self) -> int:
        return len(self.documents)

    def __iter__(self) -> Iterator[Document]:
        return iter(self.documents)

    def __getitem__(self, index: int) -> Document:
        return self.documents[index]

    def add(self, document: Document) -> None:
        """Add a document to the corpus.

        Args:
            document: The document to add.

        Raises:
            ValueError: If a document with the same ID already exists.
        """
        if document.doc_id in self._id_to_doc:
            raise ValueError(f"Document with ID {document.doc_id!r} already exists")
        self.documents.append(document)
        self._id_to_doc[document.doc_id] = document

    def get(self, doc_id: str) -> Document | None:
        """Get a document by ID.

        Args:
            doc_id: The document ID to look up.

        Returns:
            The document if found, None otherwise.
        """
        return self._id_to_doc.get(doc_id)

    def get_content(self, doc_id: str) -> str | None:
        """Get document content by ID.

        Args:
            doc_id: The document ID to look up.

        Returns:
            The document content if found, None otherwise.
        """
        doc = self.get(doc_id)
        return doc.content if doc else None

    def get_contents_by_ids(self, doc_ids: list[str]) -> list[str]:
        """Get contents for a list of document IDs.

        Args:
            doc_ids: List of document IDs.

        Returns:
            List of document contents (empty string for missing docs).
        """
        return [self.get_content(doc_id) or "" for doc_id in doc_ids]

    @classmethod
    def from_jsonl(cls, path: str | Path, name: str | None = None) -> Corpus:
        """Load a corpus from a JSONL file.

        Each line should be a JSON object with at least 'doc_id' and 'content'.

        Args:
            path: Path to the JSONL file.
            name: Optional name for the corpus.

        Returns:
            Corpus loaded from the file.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            json.JSONDecodeError: If a line is invalid JSON.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Corpus file not found: {path}")

        if name is None:
            name = path.stem

        documents = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    documents.append(Document.from_dict(data))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                except KeyError as e:
                    logger.warning(f"Skipping line {line_num} missing required field: {e}")

        logger.info(f"Loaded {len(documents)} documents from {path}")
        corpus = cls(documents=documents, name=name)
        return corpus

    def to_jsonl(self, path: str | Path) -> None:
        """Save the corpus to a JSONL file.

        Args:
            path: Path to save the file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for doc in self.documents:
                f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(self.documents)} documents to {path}")

    def sample(self, n: int, seed: int | None = None) -> Corpus:
        """Get a random sample of documents.

        Args:
            n: Number of documents to sample.
            seed: Random seed for reproducibility.

        Returns:
            New Corpus with sampled documents.
        """
        import random

        if seed is not None:
            random.seed(seed)

        sampled = random.sample(self.documents, min(n, len(self.documents)))
        return Corpus(documents=sampled, name=f"{self.name}_sample_{n}")
