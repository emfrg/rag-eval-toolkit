"""Dataset management utilities for loading and processing datasets."""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import datasets
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument


class DatasetManager:
    """Manages dataset loading, caching, and processing."""

    def __init__(self, cache_dir: Path, dataset_type: str = "multi_hop"):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_type = dataset_type
        self.logger = logging.getLogger(__name__)

    def load_or_download(
        self, dataset_name: str, force_download: bool = False
    ) -> datasets.Dataset:
        """Load dataset from cache or download from HuggingFace."""
        cache_path = self.cache_dir / dataset_name.replace("/", "_")

        if cache_path.exists() and not force_download:
            self.logger.info(f"Loading dataset from cache: {cache_path}")
            dataset = datasets.load_from_disk(str(cache_path))
        else:
            self.logger.info(f"Downloading dataset: {dataset_name}")
            try:
                dataset = datasets.load_dataset(dataset_name, split="train")
            except ValueError as e:
                if "MultiHopRAG" in dataset_name:
                    dataset = datasets.load_dataset(
                        dataset_name, "MultiHopRAG", split="train"
                    )
                else:
                    raise e
            dataset.save_to_disk(str(cache_path))
            self.logger.info(f"Dataset cached to: {cache_path}")

        return dataset

    def process_multi_hop_dataset(
        self, dataset: datasets.Dataset
    ) -> Tuple[List[Dict], List[Dict]]:
        """Process multi-hop dataset into corpus and evaluation questions."""
        corpus_docs = []
        eval_questions = []
        seen_evidence = set()

        for idx, item in enumerate(dataset):
            evidence_ids = []

            for evidence in item.get("evidence_list", []):
                evidence_text = evidence.get("fact", "")
                if not evidence_text:
                    continue

                evidence_hash = hash(evidence_text)

                if evidence_hash not in seen_evidence:
                    seen_evidence.add(evidence_hash)
                    doc_id = f"doc_{len(corpus_docs)}"

                    corpus_docs.append(
                        {
                            "doc_id": doc_id,
                            "content": evidence_text,
                            "metadata": {
                                "source": evidence.get("source", "unknown"),
                                "author": evidence.get("author", ""),
                                "category": evidence.get("category", ""),
                                "published_at": evidence.get("published_at", ""),
                                "title": evidence.get("title", ""),
                                "url": evidence.get("url", ""),
                            },
                        }
                    )
                    evidence_ids.append(doc_id)
                else:
                    for doc in corpus_docs:
                        if hash(doc["content"]) == evidence_hash:
                            evidence_ids.append(doc["doc_id"])
                            break

            eval_questions.append(
                {
                    "question_id": f"q_{idx}",
                    "question": item.get("query", ""),
                    "answer": item.get("answer", ""),
                    "question_type": item.get("question_type", "inference_query"),
                    "required_evidence": evidence_ids,
                    "evidence_count": len(evidence_ids),
                }
            )

        self.logger.info(
            f"Processed {len(dataset)} items into {len(corpus_docs)} corpus documents "
            f"and {len(eval_questions)} evaluation questions"
        )

        return corpus_docs, eval_questions

    def process_documents(
        self,
        dataset: datasets.Dataset,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
    ) -> List[Dict[str, Any]]:
        """Process dataset documents into chunks for QA generation."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", ".", " ", ""],
        )

        langchain_docs = []
        for item in dataset:
            text_field = "text" if "text" in item else "content"
            source_field = "source" if "source" in item else "id"

            doc = LangchainDocument(
                page_content=item.get(text_field, ""),
                metadata={"source": item.get(source_field, "unknown")},
            )
            langchain_docs.append(doc)

        processed_docs = []
        for doc in langchain_docs:
            chunks = text_splitter.split_documents([doc])
            for chunk in chunks:
                processed_docs.append(
                    {
                        "content": chunk.page_content,
                        "source": chunk.metadata.get("source"),
                        "start_index": chunk.metadata.get("start_index", 0),
                    }
                )

        self.logger.info(
            f"Processed {len(langchain_docs)} documents into {len(processed_docs)} chunks"
        )
        return processed_docs
