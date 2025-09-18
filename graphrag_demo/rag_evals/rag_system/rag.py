# rag_system/rag.py
import os
from typing import List, Tuple, Optional
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS, Chroma
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder
import numpy as np

from .config import RAGConfig
from .dataset import RAGDataset
from .document_processor import chunk_documents


class RAGSystem:
    def __init__(self, config: RAGConfig, dataset: RAGDataset = None):
        self.config = config
        self.dataset = dataset
        self.llm = ChatOpenAI(
            model=config.llm_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        self.embeddings = OpenAIEmbeddings(model=config.embedding_model)
        self.vector_store = None
        self.reranker = None

        if config.use_reranker:
            self.reranker = CrossEncoder(config.reranker_model)

        self._setup_prompt()

        # Build index if dataset provided
        if dataset:
            self.build_index_from_dataset(dataset)

    def _setup_prompt(self):
        """Setup the prompt template for generation."""
        self.prompt = ChatPromptTemplate.from_template(
            """
You are a helpful assistant answering questions based on the provided context.
Use only the information from the context to answer. If the context doesn't contain 
enough information, say so clearly.

Context:
{context}

Question: {question}

Answer: """
        )

    def build_index_from_dataset(self, dataset: RAGDataset):
        """Build index from a RAGDataset."""
        # Create unique index path based on dataset and config
        index_name = (
            f"{dataset.name}_{self.config.chunk_size}_{self.config.embedding_model}"
        )
        index_path = self.config.cache_dir / index_name

        if index_path.exists():
            print(f"Loading existing index from {index_path}")
            if self.config.vector_store_type == "faiss":
                self.vector_store = FAISS.load_local(
                    str(index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
            return

        # Build new index
        print(f"Building new index for {dataset.name}")
        corpus_docs = dataset.load_corpus()

        # Convert to LangChain documents
        documents = []
        for doc in corpus_docs:
            documents.append(
                Document(
                    page_content=doc["content"],
                    metadata={"doc_id": doc["doc_id"], **doc.get("metadata", {})},
                )
            )

        # Chunk documents
        documents = chunk_documents(
            documents, self.config.chunk_size, self.config.chunk_overlap
        )

        # Create vector store
        if self.config.vector_store_type == "faiss":
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            index_path.parent.mkdir(parents=True, exist_ok=True)
            self.vector_store.save_local(str(index_path))
            print(f"Index saved to {index_path}")

    def retrieve(self, query: str) -> List[Document]:
        """Retrieve relevant documents for a query."""
        if self.vector_store is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Initial retrieval
        docs = self.vector_store.similarity_search(query, k=self.config.k_retrieve)

        # Optional reranking
        if self.reranker and len(docs) > 0:
            # Prepare pairs for reranking
            pairs = [[query, doc.page_content] for doc in docs]
            scores = self.reranker.predict(pairs)

            # Sort by score and take top k
            sorted_indices = np.argsort(scores)[::-1][: self.config.k_rerank]
            docs = [docs[i] for i in sorted_indices]

        return docs

    def generate(self, query: str, contexts: List[Document]) -> str:
        """Generate answer based on query and retrieved contexts."""
        context_text = "\n\n".join([doc.page_content for doc in contexts])

        messages = self.prompt.format_messages(context=context_text, question=query)

        response = self.llm.invoke(messages)
        return response.content

    def query(self, question: str) -> Tuple[str, List[Document]]:
        """End-to-end RAG query."""
        # Retrieve
        retrieved_docs = self.retrieve(question)

        # Generate
        answer = self.generate(question, retrieved_docs)

        return answer, retrieved_docs
