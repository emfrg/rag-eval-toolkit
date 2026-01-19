"""Dataset adapter for converting various formats to Corpus + EvalDataset.

This module provides functionality to adapt existing datasets from different
sources (HuggingFace, JSON files, custom formats) into our standardized
Corpus and EvalDataset format.

Flow:
    External Data / Internal Data → adapt_structure() → Corpus + EvalDataset

Supported sources:
    - HuggingFace datasets (e.g., MultiHopRAG, SQUAD, etc.)
    - Raw text files (directory of .txt files)
    - JSON/JSONL files with custom schema
    - Lists of dictionaries

Example:
    ```python
    from rag_eval.dataset.adapter import adapt_structure, adapt_huggingface_dataset

    # From HuggingFace
    corpus, eval_dataset = adapt_huggingface_dataset(
        "yixuantt/MultiHopRAG",
        dataset_type="multi_hop"
    )

    # From custom JSON
    corpus, eval_dataset = adapt_structure(
        documents=[{"id": "1", "text": "..."}],
        questions=[{"question": "...", "answer": "...", "sources": ["1"]}]
    )
    ```
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from rag_eval.dataset.corpus import Corpus, Document
from rag_eval.dataset.eval_dataset import EvalDataset, EvalQuestion

logger = logging.getLogger(__name__)


def adapt_structure(
    documents: list[dict[str, Any]] | None = None,
    questions: list[dict[str, Any]] | None = None,
    document_id_field: str = "id",
    document_content_field: str = "text",
    question_id_field: str = "id",
    question_text_field: str = "question",
    answer_field: str = "answer",
    evidence_field: str = "sources",
    corpus_name: str = "corpus",
    dataset_name: str = "eval_dataset",
) -> tuple[Corpus, EvalDataset | None]:
    """Adapt custom data structures to Corpus and EvalDataset.

    This is the main entry point for converting arbitrary data into our
    standardized format. It handles flexible field mapping.

    Args:
        documents: List of document dicts (required).
        questions: List of question dicts (optional).
        document_id_field: Field name for document ID.
        document_content_field: Field name for document content.
        question_id_field: Field name for question ID.
        question_text_field: Field name for question text.
        answer_field: Field name for answer.
        evidence_field: Field name for required evidence (doc IDs).
        corpus_name: Name for the corpus.
        dataset_name: Name for the eval dataset.

    Returns:
        Tuple of (Corpus, EvalDataset or None if no questions provided).

    Example:
        ```python
        # With custom field names
        corpus, dataset = adapt_structure(
            documents=[{"doc_id": "1", "content": "Hello world"}],
            questions=[{"q_id": "1", "q": "What?", "a": "Hello", "refs": ["1"]}],
            document_id_field="doc_id",
            document_content_field="content",
            question_id_field="q_id",
            question_text_field="q",
            answer_field="a",
            evidence_field="refs",
        )
        ```
    """
    if documents is None:
        raise ValueError("documents parameter is required")

    # Build corpus
    corpus_docs: list[Document] = []
    for idx, doc in enumerate(documents):
        doc_id = str(doc.get(document_id_field, f"doc_{idx}"))
        content = doc.get(document_content_field, "")

        if not content:
            # Try common alternative field names
            for alt in ["text", "content", "body", "page_content"]:
                if alt in doc and doc[alt]:
                    content = doc[alt]
                    break

        # Extract metadata (everything except id and content fields)
        metadata = {
            k: v
            for k, v in doc.items()
            if k not in [document_id_field, document_content_field, "text", "content"]
            and v is not None
        }

        corpus_docs.append(
            Document(doc_id=doc_id, content=content, metadata=metadata or None)
        )

    corpus = Corpus(documents=corpus_docs, name=corpus_name)
    logger.info(f"Created corpus with {len(corpus)} documents")

    # Build eval dataset if questions provided
    eval_dataset = None
    if questions:
        eval_questions: list[EvalQuestion] = []

        for idx, q in enumerate(questions):
            q_id = str(q.get(question_id_field, f"q_{idx}"))
            question_text = q.get(question_text_field, "")
            answer = q.get(answer_field, "")

            # Get evidence - could be list of IDs or list of dicts
            evidence = q.get(evidence_field, [])
            if evidence and isinstance(evidence[0], dict):
                # Extract IDs from dict format
                evidence_ids = [str(e.get("doc_id", e.get("id", ""))) for e in evidence]
            else:
                evidence_ids = [str(e) for e in evidence]

            # Get question type
            q_type = q.get("question_type", q.get("type", "factoid"))

            eval_questions.append(
                EvalQuestion(
                    question_id=q_id,
                    question=question_text,
                    answer=answer,
                    question_type=q_type,
                    required_evidence=evidence_ids,
                    evidence_count=len(evidence_ids),
                )
            )

        eval_dataset = EvalDataset(questions=eval_questions, name=dataset_name)
        logger.info(f"Created eval dataset with {len(eval_dataset)} questions")

    return corpus, eval_dataset


def adapt_huggingface_dataset(
    dataset_name: str,
    dataset_type: str = "multi_hop",
    split: str = "train",
    cache_dir: str | Path | None = None,
) -> tuple[Corpus, EvalDataset]:
    """Adapt a HuggingFace dataset to Corpus and EvalDataset.

    Supports known dataset formats like MultiHopRAG, SQUAD, etc.

    Args:
        dataset_name: HuggingFace dataset name (e.g., "yixuantt/MultiHopRAG").
        dataset_type: Type of dataset processing to apply:
            - "multi_hop": For datasets with evidence lists
            - "qa": For standard QA datasets (SQUAD-like)
        split: Dataset split to load.
        cache_dir: Local cache directory for the dataset.

    Returns:
        Tuple of (Corpus, EvalDataset).

    Example:
        ```python
        corpus, dataset = adapt_huggingface_dataset(
            "yixuantt/MultiHopRAG",
            dataset_type="multi_hop"
        )
        ```
    """
    try:
        import datasets
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    logger.info(f"Loading HuggingFace dataset: {dataset_name}")

    # Load dataset
    try:
        hf_dataset = datasets.load_dataset(dataset_name, split=split)
    except ValueError:
        # Try with config name for MultiHopRAG-style datasets
        if "MultiHopRAG" in dataset_name:
            hf_dataset = datasets.load_dataset(
                dataset_name, "MultiHopRAG", split=split
            )
        else:
            raise

    # Cache if requested
    if cache_dir:
        cache_path = Path(cache_dir) / dataset_name.replace("/", "_")
        cache_path.mkdir(parents=True, exist_ok=True)
        hf_dataset.save_to_disk(str(cache_path))
        logger.info(f"Cached dataset to {cache_path}")

    # Process based on type
    if dataset_type == "multi_hop":
        return _adapt_multi_hop_dataset(hf_dataset)
    elif dataset_type == "qa":
        return _adapt_qa_dataset(hf_dataset)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")


def _adapt_multi_hop_dataset(hf_dataset) -> tuple[Corpus, EvalDataset]:
    """Adapt a multi-hop QA dataset (like MultiHopRAG)."""
    corpus_docs: list[Document] = []
    eval_questions: list[EvalQuestion] = []
    seen_evidence: dict[int, str] = {}  # hash -> doc_id

    for idx, item in enumerate(hf_dataset):
        evidence_ids: list[str] = []

        # Process evidence list
        evidence_list = item.get("evidence_list", [])
        for evidence in evidence_list:
            evidence_text = evidence.get("fact", "")
            if not evidence_text:
                continue

            evidence_hash = hash(evidence_text)

            if evidence_hash not in seen_evidence:
                doc_id = f"doc_{len(corpus_docs)}"
                seen_evidence[evidence_hash] = doc_id

                # Build metadata from evidence fields
                metadata = {}
                for field in ["source", "author", "category", "published_at", "title", "url"]:
                    if field in evidence and evidence[field]:
                        metadata[field] = evidence[field]

                corpus_docs.append(
                    Document(
                        doc_id=doc_id,
                        content=evidence_text,
                        metadata=metadata or None,
                    )
                )
                evidence_ids.append(doc_id)
            else:
                evidence_ids.append(seen_evidence[evidence_hash])

        # Create question
        eval_questions.append(
            EvalQuestion(
                question_id=f"q_{idx}",
                question=item.get("query", ""),
                answer=item.get("answer", ""),
                question_type=item.get("question_type", "inference_query"),
                required_evidence=evidence_ids,
                evidence_count=len(evidence_ids),
            )
        )

    corpus = Corpus(documents=corpus_docs, name="multi_hop_corpus")
    eval_dataset = EvalDataset(questions=eval_questions, name="multi_hop_eval")

    logger.info(
        f"Adapted multi-hop dataset: {len(corpus)} docs, {len(eval_dataset)} questions"
    )
    return corpus, eval_dataset


def _adapt_qa_dataset(hf_dataset) -> tuple[Corpus, EvalDataset]:
    """Adapt a standard QA dataset (SQUAD-like format)."""
    corpus_docs: list[Document] = []
    eval_questions: list[EvalQuestion] = []
    seen_contexts: dict[int, str] = {}  # hash -> doc_id

    for idx, item in enumerate(hf_dataset):
        # Get context
        context = item.get("context", item.get("passage", ""))
        context_hash = hash(context)

        if context_hash not in seen_contexts:
            doc_id = f"doc_{len(corpus_docs)}"
            seen_contexts[context_hash] = doc_id
            corpus_docs.append(Document(doc_id=doc_id, content=context))
        else:
            doc_id = seen_contexts[context_hash]

        # Get answer (handle SQUAD format where answers is a dict)
        answer = item.get("answer", "")
        if not answer and "answers" in item:
            answers = item["answers"]
            if isinstance(answers, dict) and "text" in answers:
                answer = answers["text"][0] if answers["text"] else ""
            elif isinstance(answers, list) and answers:
                answer = answers[0].get("text", "") if isinstance(answers[0], dict) else str(answers[0])

        eval_questions.append(
            EvalQuestion(
                question_id=f"q_{idx}",
                question=item.get("question", ""),
                answer=answer,
                question_type="factoid",
                required_evidence=[doc_id],
                evidence_count=1,
            )
        )

    corpus = Corpus(documents=corpus_docs, name="qa_corpus")
    eval_dataset = EvalDataset(questions=eval_questions, name="qa_eval")

    logger.info(
        f"Adapted QA dataset: {len(corpus)} docs, {len(eval_dataset)} questions"
    )
    return corpus, eval_dataset


def adapt_from_directory(
    directory: str | Path,
    file_pattern: str = "*.txt",
    questions_file: str | Path | None = None,
) -> tuple[Corpus, EvalDataset | None]:
    """Create a corpus from a directory of text files.

    Args:
        directory: Path to directory containing text files.
        file_pattern: Glob pattern for files to include.
        questions_file: Optional path to JSONL file with questions.

    Returns:
        Tuple of (Corpus, EvalDataset or None).

    Example:
        ```python
        corpus, _ = adapt_from_directory("./documents/", "*.txt")
        corpus.to_jsonl("./corpus.jsonl")
        ```
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    documents: list[dict[str, Any]] = []
    for idx, file_path in enumerate(sorted(directory.glob(file_pattern))):
        content = file_path.read_text(encoding="utf-8")
        documents.append(
            {
                "id": f"doc_{idx}",
                "text": content,
                "source": file_path.name,
                "path": str(file_path),
            }
        )

    logger.info(f"Loaded {len(documents)} documents from {directory}")

    # Load questions if provided
    questions = None
    if questions_file:
        questions_path = Path(questions_file)
        if questions_path.exists():
            questions = []
            with open(questions_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        questions.append(json.loads(line))
            logger.info(f"Loaded {len(questions)} questions from {questions_file}")

    return adapt_structure(documents=documents, questions=questions)


def adapt_from_jsonl(
    corpus_file: str | Path,
    questions_file: str | Path | None = None,
    **field_mappings,
) -> tuple[Corpus, EvalDataset | None]:
    """Load corpus and questions from JSONL files.

    Args:
        corpus_file: Path to corpus JSONL file.
        questions_file: Optional path to questions JSONL file.
        **field_mappings: Field name mappings passed to adapt_structure.

    Returns:
        Tuple of (Corpus, EvalDataset or None).
    """
    corpus_path = Path(corpus_file)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    documents = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                documents.append(json.loads(line))

    questions = None
    if questions_file:
        questions_path = Path(questions_file)
        if questions_path.exists():
            questions = []
            with open(questions_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        questions.append(json.loads(line))

    return adapt_structure(documents=documents, questions=questions, **field_mappings)
