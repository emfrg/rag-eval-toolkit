# rag_system/document_processor.py
import json
from typing import List, Dict, Any, Mapping
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


def load_corpus(corpus_path: str) -> List[Document]:
    """Load corpus from JSONL file."""
    documents = []
    with open(corpus_path, "r") as f:
        for line in f:
            doc = json.loads(line)
            documents.append(
                Document(
                    page_content=doc["content"],
                    metadata={"doc_id": doc["doc_id"], **doc.get("metadata", {})},
                )
            )
    return documents


def build_document_text(
    *,
    content: str,
    metadata: Mapping[str, Any] | None,
    inline_metadata: bool,
) -> str:
    if not inline_metadata:
        return content
    meta = metadata or {}
    if not meta:
        return content
    lines = []
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            value_str = ", ".join(str(item) for item in value)
        else:
            value_str = str(value)
        lines.append(f"{key}: {value_str}")
    if not lines:
        return content
    meta_block = "\n".join(lines)
    return f"---CONTENT---\n{content.strip()}\n---METADATA---\n{meta_block}"


def chunk_documents(
    documents: List[Document], chunk_size: int = 400, chunk_overlap: int = 50
) -> List[Document]:
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = []
    for doc in documents:
        doc_chunks = text_splitter.split_documents([doc])
        # Preserve original doc_id in metadata
        for chunk in doc_chunks:
            chunk.metadata["source_doc_id"] = doc.metadata["doc_id"]
        chunks.extend(doc_chunks)

    return chunks
