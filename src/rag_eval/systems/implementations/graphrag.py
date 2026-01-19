"""GraphRAG system implementation using LightRAG.

This implementation uses LightRAG for knowledge graph-based retrieval,
combining semantic search with graph traversal for better context retrieval.

Requires the optional 'graphrag' dependency:
    uv add rag-eval-toolkit[graphrag]
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.prompts import ChatPromptTemplate

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


# LightRAG imports - lazy loaded
_lightrag_available = False
try:
    from lightrag import LightRAG, QueryParam
    from lightrag.kg.shared_storage import initialize_pipeline_status
    from lightrag.llm.openai import (
        gpt_4o_complete,
        gpt_4o_mini_complete,
        openai_embed,
    )
    from lightrag.utils import Tokenizer, setup_logger

    _lightrag_available = True

    # Patch Tokenizer to allow special tokens
    if not getattr(Tokenizer, "_allow_special_tokens", False):
        _original_encode = Tokenizer.encode

        def _encode_allow_special(self, content: str):
            return self.tokenizer.encode(content, disallowed_special=())

        Tokenizer.encode = _encode_allow_special
        Tokenizer._allow_special_tokens = True

except ImportError:
    LightRAG = None  # type: ignore
    QueryParam = None  # type: ignore


def _check_lightrag_available() -> None:
    """Check if LightRAG is available."""
    if not _lightrag_available:
        raise ImportError(
            "LightRAG is required for GraphRAGSystem. "
            "Install with: uv add rag-eval-toolkit[graphrag]"
        )


def _select_llm_func(model_name: str):
    """Select the appropriate LightRAG LLM function.

    Args:
        model_name: Name of the model.

    Returns:
        LightRAG LLM function.

    Raises:
        ValueError: If model is not supported.
    """
    if model_name == "gpt-4o":
        return gpt_4o_complete
    if model_name == "gpt-4o-mini":
        return gpt_4o_mini_complete
    raise ValueError(
        f"Unsupported LightRAG model: {model_name!r}. "
        "Supported models: gpt-4o, gpt-4o-mini"
    )


def _build_document_text(
    content: str,
    metadata: dict[str, Any] | None,
    inline_metadata: bool,
) -> str:
    """Build document text for indexing.

    Args:
        content: Document content.
        metadata: Optional metadata.
        inline_metadata: Whether to include metadata inline.

    Returns:
        Document text.
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


class GraphRAGSystem(RAGSystemBase):
    """GraphRAG system using LightRAG for knowledge graph-based retrieval.

    LightRAG builds a knowledge graph from documents and uses a combination
    of semantic search and graph traversal for retrieval.

    Query modes:
    - "semantic": Pure vector similarity search
    - "graph": Knowledge graph traversal only
    - "hybrid": Combined semantic + graph (default, recommended)

    Example:
        ```python
        from rag_eval import RAGConfig, Corpus
        from rag_eval.systems.config import GraphRAGConfig
        from rag_eval.systems.implementations import GraphRAGSystem

        config = RAGConfig(
            rag_type="graphrag",
            llm_model="gpt-4o-mini",
            graphrag=GraphRAGConfig(
                query=GraphRAGQueryConfig(mode="hybrid", top_k=50),
            ),
        )

        rag = GraphRAGSystem(config)
        rag.create_index(corpus)

        response = rag.query("What is the capital of France?")
        print(response.response)
        ```
    """

    def __init__(self, config: RAGConfig) -> None:
        """Initialize the GraphRAG system.

        Args:
            config: RAG configuration with graphrag-specific settings.

        Raises:
            ImportError: If LightRAG is not installed.
        """
        _check_lightrag_available()
        super().__init__(config)

        # Set up LightRAG logging
        setup_logger("lightrag", level="INFO")

        # LightRAG instance (initialized on index creation)
        self._rag: LightRAG | None = None
        self._workdir: Path | None = None

        # Event loop for async operations
        self._loop: asyncio.AbstractEventLoop | None = None

        # LLM for generation (Anthropic by default, separate from LightRAG's internal LLM)
        # Note: LightRAG's internal LLM for KG construction still uses OpenAI
        self._llm = _create_llm(config)
        self._prompt = ChatPromptTemplate.from_template(STRICT_RAG_PROMPT)

        # Corpus name for workspace naming
        self._corpus_name: str | None = None

    @property
    def _cfg(self):
        """Shorthand for graphrag config."""
        return self.config.graphrag

    def _get_workspace_name(self, corpus_name: str) -> str:
        """Get workspace name for a corpus.

        Args:
            corpus_name: Name of the corpus.

        Returns:
            Workspace name.
        """
        suffix = "meta" if self._cfg.indexing.inline_metadata else "base"
        return f"{corpus_name}_{suffix}"

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create an event loop for async operations."""
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        return self._loop

    def _run_async(self, coro):
        """Run an async coroutine synchronously."""
        loop = self._ensure_loop()
        return loop.run_until_complete(coro)

    async def _initialize_lightrag(self, workdir: Path, workspace: str) -> LightRAG:
        """Initialize a LightRAG instance.

        Args:
            workdir: Working directory for LightRAG.
            workspace: Workspace name.

        Returns:
            Initialized LightRAG instance.
        """
        rag = LightRAG(
            working_dir=str(workdir),
            workspace=workspace,
            embedding_func=openai_embed,
            llm_model_func=_select_llm_func(self.config.llm_model),
            max_parallel_insert=self._cfg.indexing.max_parallel_insert,
        )
        await rag.initialize_storages()
        await initialize_pipeline_status()
        return rag

    async def _insert_documents(
        self,
        rag: LightRAG,
        records: list[tuple[str, str | None]],
        batch_size: int,
    ) -> int:
        """Insert documents into LightRAG.

        Args:
            rag: LightRAG instance.
            records: List of (text, doc_id) tuples.
            batch_size: Batch size for insertion.

        Returns:
            Number of documents inserted.
        """
        from tqdm.auto import tqdm

        if not records:
            return 0

        total = 0
        for start in tqdm(range(0, len(records), batch_size), desc="Indexing"):
            batch = records[start : start + batch_size]
            texts = [text for text, _ in batch]
            ids = [doc_id for _, doc_id in batch]

            # Build file_paths for document tracking
            file_paths = []
            for idx, (_, doc_id) in enumerate(batch):
                if isinstance(doc_id, str) and doc_id.strip():
                    file_paths.append(doc_id.strip())
                else:
                    file_paths.append(f"doc-{start + idx}")

            # Only pass ids if all are valid strings
            ids_arg = ids if all(isinstance(d, str) and d for d in ids) else None

            await rag.ainsert(texts, ids=ids_arg, file_paths=file_paths)
            total += len(texts)

        return total

    def create_index(self, corpus: Corpus) -> IndexReport:
        """Build the LightRAG index from a corpus.

        If an index already exists and force_reindex is False, the existing
        index will be reused.

        Args:
            corpus: The document corpus to index.

        Returns:
            IndexReport with indexing statistics.
        """
        self._corpus_name = corpus.name
        cfg = self._cfg.indexing

        workspace = self._get_workspace_name(corpus.name)
        workspace_dir = cfg.cache_dir / workspace
        self._workdir = workspace_dir

        # Check if we need to build a new index
        workspace_has_files = workspace_dir.exists() and any(workspace_dir.iterdir())
        needs_index = cfg.force_reindex or not workspace_has_files

        # Force reindex: remove existing workspace
        if cfg.force_reindex and workspace_dir.exists():
            logger.info(f"Force reindex: removing {workspace_dir}")
            shutil.rmtree(workspace_dir)
            workspace_has_files = False
            needs_index = True

        # Ensure directories exist
        cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        workspace_dir.mkdir(parents=True, exist_ok=True)

        # Load corpus records
        records: list[tuple[str, str | None]] = []
        for doc in corpus:
            text = _build_document_text(
                content=doc.content,
                metadata=doc.metadata,
                inline_metadata=cfg.inline_metadata,
            )
            if text.strip():
                records.append((text, doc.doc_id))

        total_docs = len(records)

        # Initialize LightRAG
        logger.info(f"Initializing LightRAG workspace: {workspace}")
        self._rag = self._run_async(
            self._initialize_lightrag(cfg.cache_dir, workspace)
        )

        # Insert documents if needed
        ingested = 0
        if needs_index:
            logger.info(f"Indexing {total_docs} documents...")
            ingested = self._run_async(
                self._insert_documents(self._rag, records, cfg.batch_size)
            )
        else:
            logger.info("Reusing existing index")

        self._index_loaded = True
        return IndexReport(
            total_documents=total_docs,
            indexed_documents=ingested,
            reused_existing=not needs_index,
            index_path=str(workspace_dir),
        )

    def load_index(self) -> None:
        """Load an existing LightRAG index.

        Raises:
            FileNotFoundError: If no index exists.
            RuntimeError: If corpus name is not set.
        """
        if self._corpus_name is None:
            raise RuntimeError(
                "Cannot load index: corpus name not set. "
                "Use create_index() first or set _corpus_name manually."
            )

        cfg = self._cfg.indexing
        workspace = self._get_workspace_name(self._corpus_name)
        workspace_dir = cfg.cache_dir / workspace

        if not workspace_dir.exists():
            raise FileNotFoundError(f"No index found at {workspace_dir}")

        logger.info(f"Loading LightRAG index from {workspace_dir}")
        self._workdir = workspace_dir
        self._rag = self._run_async(
            self._initialize_lightrag(cfg.cache_dir, workspace)
        )
        self._index_loaded = True

    async def _query_context(self, question: str) -> str:
        """Query LightRAG for context only.

        Args:
            question: The query.

        Returns:
            Context text from LightRAG.
        """
        assert self._rag is not None

        param = QueryParam(
            mode=self._cfg.query.mode,
            top_k=self._cfg.query.top_k,
            only_need_context=True,
        )

        result = await self._rag.aquery(question, param=param)

        if isinstance(result, str):
            return result
        raise TypeError(f"LightRAG context response must be str, got {type(result)}")

    def _parse_context_documents(self, context_text: str) -> list[RetrievedDocument]:
        """Parse LightRAG context text into documents.

        LightRAG returns context in a specific format with document chunks.
        This method extracts the individual documents.

        Args:
            context_text: Raw context text from LightRAG.

        Returns:
            List of parsed documents.
        """
        if not context_text:
            return []

        documents: list[RetrievedDocument] = []
        in_chunk_section = False

        for line in context_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            # Detect start of document chunks section
            if stripped.startswith("-----Document Chunks"):
                in_chunk_section = True
                continue

            # Detect end of chunks section
            if in_chunk_section and stripped.startswith("-----"):
                break

            if in_chunk_section:
                # Skip code fence markers
                if stripped.startswith("```"):
                    continue

                # Try to parse as JSON
                try:
                    data = json.loads(stripped)
                except json.JSONDecodeError:
                    continue

                content = data.get("content")
                if not isinstance(content, str) or not content.strip():
                    continue

                doc_id = data.get("file_path") or data.get("doc_id")
                if not isinstance(doc_id, str) or not doc_id.strip():
                    doc_id = f"context_{len(documents)}"

                documents.append(
                    RetrievedDocument(
                        doc_id=doc_id,
                        content=content,
                        score=1.0,  # LightRAG doesn't provide scores
                        metadata={"source": "lightrag"},
                    )
                )

        # Fallback: if no structured documents found, use entire context
        if not documents and context_text.strip():
            documents.append(
                RetrievedDocument(
                    doc_id="lightrag_context",
                    content=context_text,
                    score=1.0,
                    metadata={"source": "lightrag_raw"},
                )
            )

        return documents

    def retrieve(self, query: str) -> list[RetrievedDocument]:
        """Retrieve relevant documents using LightRAG.

        Args:
            query: The search query.

        Returns:
            List of retrieved documents.

        Raises:
            RuntimeError: If index is not loaded.
        """
        self.ensure_index_loaded()
        context_text = self._run_async(self._query_context(query))
        return self._parse_context_documents(context_text)

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
