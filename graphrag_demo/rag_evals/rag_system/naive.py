from __future__ import annotations

from typing import List, Optional, Tuple

from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from sentence_transformers import CrossEncoder

from .base import BaseRAGSystem, IndexReport
from .config import RAGConfig
from .dataset import RAGDataset
from .document_processor import build_document_text, chunk_documents
from .prompts import STRICT_RAG_PROMPT


class NaiveRAGSystem(BaseRAGSystem):
    def __init__(self, config: RAGConfig, dataset: Optional[RAGDataset] = None) -> None:
        super().__init__(config, dataset)
        self.llm = ChatOpenAI(
            model=config.llm_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        self.embeddings = OpenAIEmbeddings(model=config.naive.embedding_model)
        self.vector_store = None
        self.reranker = (
            CrossEncoder(config.naive.reranker_model)
            if config.naive.use_reranker
            else None
        )
        self.prompt = ChatPromptTemplate.from_template(STRICT_RAG_PROMPT)
        self._index_report: Optional[IndexReport] = None
        if dataset:
            self._index_report = self.build_index(dataset)

    @property
    def index_report(self) -> Optional[IndexReport]:
        return self._index_report

    def build_index(self, dataset: RAGDataset) -> IndexReport:
        cfg = self.config.naive
        corpus_docs = dataset.load_corpus()
        total_docs = len(corpus_docs)

        suffix = (
            f"{cfg.chunk_size}_{cfg.embedding_model}"
            if cfg.chunk_documents
            else f"no_chunk_{cfg.embedding_model}"
        )

        if cfg.inline_metadata:
            suffix = f"{suffix}_meta"

        index_path = cfg.cache_dir / f"{dataset.name}_{suffix}"
        cfg.cache_dir.mkdir(parents=True, exist_ok=True)

        if index_path.exists():
            self.vector_store = FAISS.load_local(
                str(index_path),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            report = IndexReport(
                total_documents=total_docs,
                ingested_documents=0,
                reused_existing=True,
            )
            self._index_report = report
            return report

        documents: List[Document] = [
            Document(
                page_content=build_document_text(
                    content=doc.get("content", ""),
                    metadata=doc.get("metadata", {}),
                    inline_metadata=cfg.inline_metadata,
                ),
                metadata={"doc_id": doc["doc_id"], **doc.get("metadata", {})},
            )
            for doc in corpus_docs
        ]

        if cfg.chunk_documents:
            documents = chunk_documents(
                documents,
                chunk_size=cfg.chunk_size,
                chunk_overlap=cfg.chunk_overlap,
            )

        if cfg.vector_store != "faiss":
            raise NotImplementedError("Only FAISS vector stores are implemented")

        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        self.vector_store.save_local(str(index_path))

        report = IndexReport(
            total_documents=total_docs,
            ingested_documents=len(documents),
            reused_existing=False,
        )
        self._index_report = report
        return report

    def retrieve(self, question: str) -> List[Document]:
        if self.vector_store is None:
            raise ValueError("Index not built. Call build_index() first.")

        cfg = self.config.naive
        docs_and_scores = self.vector_store.similarity_search_with_score(
            question, k=cfg.k_retrieve
        )

        filtered: List[Document] = []
        for doc, distance in docs_and_scores:
            if distance <= cfg.similarity_threshold:
                filtered.append(doc)
            if len(filtered) >= cfg.max_docs:
                break

        if filtered and len(filtered) >= cfg.min_docs:
            docs = filtered
        else:
            docs = [doc for doc, _ in docs_and_scores[: cfg.max_docs]]

        if self.reranker and docs:
            pairs = [[question, doc.page_content] for doc in docs]
            scores = self.reranker.predict(pairs)
            reranked = [
                (doc, score)
                for doc, score in zip(docs, scores)
                if score >= self.config.naive.rerank_threshold
            ]
            if reranked:
                reranked.sort(key=lambda item: item[1], reverse=True)
                docs = [doc for doc, _ in reranked[: cfg.max_docs]]

        return docs

    def query(self, question: str) -> Tuple[str, List[Document]]:
        contexts = self.retrieve(question)
        context_text = "\n\n".join(doc.page_content for doc in contexts)
        messages = self.prompt.format_messages(context=context_text, question=question)
        response = self.llm.invoke(messages)
        return response.content, contexts
