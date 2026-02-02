"""Naive RAG system implementation.

This is a traditional RAG system that uses:
- FAISS for vector storage and retrieval
- OpenAI embeddings (for now - embeddings still use OpenAI)
- LLM generation with Anthropic (default) or OpenAI
- Optional reranking with cross-encoder models
- Optional document chunking
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document as LangChainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from rag_eval.systems.base import IndexReport, RAGSystemBase
from rag_eval.systems.prompts import STRICT_RAG_PROMPT
from rag_eval.systems.response import RAGResponse, RetrievedDocument

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from rag_eval.dataset.corpus import Corpus
    from rag_eval.systems.config import RAGConfig

logger = logging.getLogger(__name__)


def _create_llm(config: RAGConfig) -> BaseChatModel:
    """Create LLM based on config provider.

    Args:
        config: RAG configuration.

    Returns:
        LangChain chat model (Anthropic or OpenAI).
    """
    if config.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=config.llm_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
    elif config.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.llm_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {config.llm_provider}")


def _build_document_text(
    content: str,
    metadata: dict[str, str] | None,
    inline_metadata: bool,
) -> str:
    """Build document text, optionally including metadata inline.

    Args:
        content: The document content.
        metadata: Optional metadata dictionary.
        inline_metadata: Whether to include metadata in the text.

    Returns:
        The document text, possibly with metadata appended.
    """
    if not inline_metadata or not metadata:
        return content

    meta_lines = []
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            value_str = ", ".join(str(item) for item in value)
        else:
            value_str = str(value)
        meta_lines.append(f"{key}: {value_str}")

    if not meta_lines:
        return content

    meta_block = "\n".join(meta_lines)
    return f"---CONTENT---\n{content.strip()}\n---METADATA---\n{meta_block}"


def _chunk_documents(
    documents: list[LangChainDocument],
    chunk_size: int,
    chunk_overlap: int,
) -> list[LangChainDocument]:
    """Split documents into smaller chunks.

    Args:
        documents: List of documents to chunk.
        chunk_size: Target size of each chunk.
        chunk_overlap: Overlap between adjacent chunks.

    Returns:
        List of chunked documents with preserved metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = []
    for doc in documents:
        doc_chunks = text_splitter.split_documents([doc])
        # Preserve original doc_id as source_doc_id
        for chunk in doc_chunks:
            if "doc_id" in doc.metadata:
                chunk.metadata["source_doc_id"] = doc.metadata["doc_id"]
        chunks.extend(doc_chunks)

    return chunks


class NaiveRAGSystem(RAGSystemBase):
    """Naive RAG system using FAISS vector store.

    This implementation provides:
    - Vector similarity search with FAISS
    - OpenAI embeddings (configurable model)
    - Optional document chunking
    - Optional reranking with cross-encoder
    - Index caching for reuse

    Example:
        ```python
        from rag_eval import RAGConfig, Corpus
        from rag_eval.systems.implementations import NaiveRAGSystem

        config = RAGConfig(
            rag_type="naive",
            llm_model="gpt-4o-mini",
            naive=NaiveRAGConfig(
                use_reranker=True,
                k_retrieve=10,
            ),
        )

        rag = NaiveRAGSystem(config)
        rag.create_index(corpus)

        response = rag.query("What is the capital of France?")
        print(response.response)
        ```
    """

    def __init__(self, config: RAGConfig) -> None:
        """Initialize the Naive RAG system.

        Args:
            config: RAG configuration with naive-specific settings.
        """
        super().__init__(config)

        # LLM for generation (Anthropic by default, OpenAI as option)
        self._llm = _create_llm(config)

        # Embeddings
        self._embeddings = OpenAIEmbeddings(model=config.naive.embedding_model)

        # Vector store (initialized on index creation)
        self._vector_store: FAISS | None = None

        # Reranker (lazy loaded)
        self._reranker = None
        self._reranker_loaded = False

        # Prompt template
        self._prompt = ChatPromptTemplate.from_template(STRICT_RAG_PROMPT)

        # Index metadata
        self._index_path: Path | None = None
        self._corpus_name: str | None = None

    @property
    def _cfg(self):
        """Shorthand for naive config."""
        return self.config.naive

    def _get_reranker(self):
        """Lazy load the reranker model."""
        if not self._reranker_loaded and self._cfg.use_reranker:
            from sentence_transformers import CrossEncoder

            logger.info(f"Loading reranker model: {self._cfg.reranker_model}")
            self._reranker = CrossEncoder(self._cfg.reranker_model)
            self._reranker_loaded = True
        return self._reranker

    def _get_index_path(self, corpus_name: str) -> Path:
        """Get the index path for a corpus.

        Args:
            corpus_name: Name of the corpus.

        Returns:
            Path where the index should be stored.
        """
        suffix = (
            f"{self._cfg.chunk_size}_{self._cfg.embedding_model}"
            if self._cfg.chunk_documents
            else f"no_chunk_{self._cfg.embedding_model}"
        )
        if self._cfg.inline_metadata:
            suffix = f"{suffix}_meta"

        return self._cfg.cache_dir / f"{corpus_name}_{suffix}"

    def create_index(self, corpus: Corpus) -> IndexReport:
        """Build the FAISS index from a corpus.

        If an index already exists at the cache path, it will be loaded
        instead of rebuilding.

        Args:
            corpus: The document corpus to index.

        Returns:
            IndexReport with indexing statistics.
        """
        self._corpus_name = corpus.name
        self._index_path = self._get_index_path(corpus.name)
        self._cfg.cache_dir.mkdir(parents=True, exist_ok=True)

        # Try to load existing index
        if self._index_path.exists():
            logger.info(f"Loading existing index from {self._index_path}")
            self._vector_store = FAISS.load_local(
                str(self._index_path),
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
            self._index_loaded = True

            # Pre-load reranker to avoid race condition in parallel queries
            if self._cfg.use_reranker:
                self._get_reranker()

            return IndexReport(
                total_documents=len(corpus),
                indexed_documents=0,
                reused_existing=True,
                index_path=str(self._index_path),
            )

        # Build new index
        logger.info(f"Building new index for {len(corpus)} documents")

        # Convert corpus to LangChain documents
        documents: list[LangChainDocument] = []
        for doc in corpus:
            content = _build_document_text(
                content=doc.content,
                metadata=doc.metadata,
                inline_metadata=self._cfg.inline_metadata,
            )
            lc_doc = LangChainDocument(
                page_content=content,
                metadata={"doc_id": doc.doc_id, **(doc.metadata or {})},
            )
            documents.append(lc_doc)

        # Optionally chunk documents
        if self._cfg.chunk_documents:
            logger.info(
                f"Chunking documents (size={self._cfg.chunk_size}, "
                f"overlap={self._cfg.chunk_overlap})"
            )
            documents = _chunk_documents(
                documents,
                chunk_size=self._cfg.chunk_size,
                chunk_overlap=self._cfg.chunk_overlap,
            )

        # Build FAISS index
        if self._cfg.vector_store != "faiss":
            raise NotImplementedError(
                f"Vector store {self._cfg.vector_store!r} not implemented. "
                "Only 'faiss' is supported."
            )

        logger.info(f"Creating FAISS index with {len(documents)} documents/chunks")
        self._vector_store = FAISS.from_documents(documents, self._embeddings)

        # Save index
        self._vector_store.save_local(str(self._index_path))
        logger.info(f"Saved index to {self._index_path}")

        self._index_loaded = True

        # Pre-load reranker to avoid race condition in parallel queries
        if self._cfg.use_reranker:
            self._get_reranker()

        return IndexReport(
            total_documents=len(corpus),
            indexed_documents=len(documents),
            reused_existing=False,
            index_path=str(self._index_path),
        )

    def load_index(self) -> None:
        """Load an existing index.

        Raises:
            FileNotFoundError: If no index exists.
            RuntimeError: If corpus name is not set.
        """
        if self._corpus_name is None:
            raise RuntimeError(
                "Cannot load index: corpus name not set. "
                "Use create_index() first or set _corpus_name manually."
            )

        self._index_path = self._get_index_path(self._corpus_name)

        if not self._index_path.exists():
            raise FileNotFoundError(f"No index found at {self._index_path}")

        logger.info(f"Loading index from {self._index_path}")
        self._vector_store = FAISS.load_local(
            str(self._index_path),
            self._embeddings,
            allow_dangerous_deserialization=True,
        )
        self._index_loaded = True

        # Pre-load reranker to avoid race condition in parallel queries
        if self._cfg.use_reranker:
            self._get_reranker()

    def retrieve(self, query: str) -> list[RetrievedDocument]:
        """Retrieve relevant documents for a query.

        Args:
            query: The search query.

        Returns:
            List of retrieved documents with scores.

        Raises:
            RuntimeError: If index is not loaded.
        """
        self.ensure_index_loaded()
        assert self._vector_store is not None

        # Initial retrieval
        docs_and_scores = self._vector_store.similarity_search_with_score(
            query, k=self._cfg.k_retrieve
        )

        # Filter by similarity threshold
        filtered: list[tuple[LangChainDocument, float]] = []
        for doc, distance in docs_and_scores:
            if distance <= self._cfg.similarity_threshold:
                filtered.append((doc, distance))
            if len(filtered) >= self._cfg.max_docs:
                break

        # Ensure minimum docs
        if len(filtered) < self._cfg.min_docs:
            filtered = list(docs_and_scores[: self._cfg.max_docs])

        # Optional reranking
        reranker = self._get_reranker()
        if reranker and filtered:
            pairs = [[query, doc.page_content] for doc, _ in filtered]
            scores = reranker.predict(pairs)

            # Filter by rerank threshold and sort
            reranked = [
                (doc, score)
                for (doc, _), score in zip(filtered, scores)
                if score >= self._cfg.rerank_threshold
            ]

            if reranked:
                reranked.sort(key=lambda x: x[1], reverse=True)
                filtered = [(doc, score) for doc, score in reranked[: self._cfg.max_docs]]

        # Convert to RetrievedDocument
        results: list[RetrievedDocument] = []
        for doc, score in filtered:
            # Get doc_id from metadata, falling back to source_doc_id for chunks
            doc_id = doc.metadata.get("doc_id") or doc.metadata.get("source_doc_id", "unknown")
            results.append(
                RetrievedDocument(
                    doc_id=doc_id,
                    content=doc.page_content,
                    score=float(score),
                    metadata=dict(doc.metadata),
                )
            )

        return results

    def generate(self, query: str, contexts: list[str]) -> str:
        """Generate an answer from retrieved contexts.

        Args:
            query: The question to answer.
            contexts: List of context strings.

        Returns:
            Generated answer string.
        """
        context_text = "\n\n".join(contexts)
        messages = self._prompt.format_messages(context=context_text, question=query)
        response = self._llm.invoke(messages)
        return str(response.content)

    def query(self, question: str) -> RAGResponse:
        """Execute the full RAG pipeline.

        Args:
            question: The question to answer.

        Returns:
            RAGResponse with answer and retrieved evidence.
        """
        # Retrieve
        retrieved_docs = self.retrieve(question)

        # Generate
        contexts = [doc.content for doc in retrieved_docs]
        answer = self.generate(question, contexts)

        return RAGResponse.from_documents(answer, retrieved_docs)
