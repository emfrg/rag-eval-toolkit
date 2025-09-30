#!/usr/bin/env python3
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import List, Tuple

import click
from dotenv import load_dotenv
from langchain.schema import Document

from rag_system import IndexReport, RAGConfig, RAGDataset, RAGSystem


def _index_directory(dataset: RAGDataset, config: RAGConfig) -> Path:
    cfg = config.naive
    suffix = (
        f"{cfg.chunk_size}_{cfg.embedding_model}"
        if cfg.chunk_documents
        else f"no_chunk_{cfg.embedding_model}"
    )
    if cfg.inline_metadata:
        suffix = f"{suffix}_meta"
    return cfg.cache_dir / f"{dataset.name}_{suffix}"


def _wipe_index(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _print_report(report: IndexReport, index_dir: Path) -> None:
    if report.reused_existing:
        click.echo(
            f"Reused existing FAISS index ({report.total_documents} docs) at {index_dir}"
        )
        return
    click.echo(
        f"Indexed {report.ingested_documents} / {report.total_documents} documents into {index_dir}"
    )


def _print_contexts(contexts: List[Document], limit: int) -> None:
    if not contexts:
        click.echo("No supporting contexts retrieved.")
        return

    click.echo(f"Retrieved {len(contexts)} context documents:")
    for doc in contexts[:limit]:
        doc_id = doc.metadata["doc_id"] if "doc_id" in doc.metadata else "unknown"
        snippet = doc.page_content.replace("\n", " ")[:200]
        click.echo(f"  - {doc_id}: {snippet}...")
        # click.echo(f" - {doc_id}: {doc.page_content}")


def _resolve_question(dataset: RAGDataset, override: str | None) -> Tuple[str, str]:
    if override is not None:
        return "manual", override

    questions = dataset.load_questions()
    if not questions:
        raise ValueError("No questions available in questions file.")

    first = questions[0]
    if "question_id" in first:
        question_id = first["question_id"]
    else:
        question_id = "q_0"

    if "question" not in first:
        raise KeyError("Expected 'question' field in questions file")

    question_text = first["question"]
    return question_id, question_text


@click.command()
@click.option(
    "--dataset-dir",
    required=True,
    type=click.Path(path_type=str),
    help="Directory produced by eval_dataset_builder (contains corpus.jsonl).",
)
@click.option(
    "--cache-dir",
    default="rag_cache",
    show_default=True,
    type=click.Path(path_type=str),
    help="Directory where FAISS indices are stored.",
)
@click.option(
    "--chunk-docs/--no-chunk-docs",
    default=False,
    show_default=True,
    help="Whether to chunk documents before embedding.",
)
@click.option(
    "--chunk-size",
    default=400,
    show_default=True,
    help="Chunk size used when chunking is enabled.",
)
@click.option(
    "--chunk-overlap",
    default=50,
    show_default=True,
    help="Token overlap between chunks when chunking is enabled.",
)
@click.option(
    "--embedding-model",
    default="text-embedding-3-small",
    show_default=True,
    help="Embedding model for document indexing.",
)
@click.option(
    "--k",
    default=10,
    show_default=True,
    help="Number of nearest neighbours to retrieve before filtering.",
)
@click.option(
    "--similarity-threshold",
    default=1.0,
    show_default=True,
    help="Maximum FAISS distance to keep a document.",
)
@click.option(
    "--min-docs",
    default=0,
    show_default=True,
    help="Minimum number of documents to return after filtering.",
)
@click.option(
    "--max-docs",
    default=4,
    show_default=True,
    help="Maximum number of documents to keep after optional reranking.",
)
@click.option(
    "--use-reranker",
    is_flag=True,
    help="Enable sentence-transformer reranker after vector retrieval.",
)
@click.option(
    "--reranker-model",
    default="BAAI/bge-reranker-base",
    show_default=True,
    help="Cross-encoder reranker to use when --use-reranker is enabled.",
)
@click.option(
    "--rerank-threshold",
    default=0.5,
    show_default=True,
    help="Minimum reranker score to keep a document when reranking is enabled.",
)
@click.option(
    "--llm-model",
    default="gpt-4o-mini",
    show_default=True,
    help="Chat model used to generate the final answer.",
)
@click.option(
    "--temperature",
    default=1.0,
    show_default=True,
    help="Sampling temperature for the answer model.",
)
@click.option(
    "--max-tokens",
    default=512,
    show_default=True,
    help="Maximum number of tokens for the answer model.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Force rebuild of the FAISS index even if it exists.",
)
@click.option(
    "--inline-metadata/--no-inline-metadata",
    default=False,
    show_default=True,
    help="Embed document metadata within the indexed content.",
)
@click.option(
    "--sanity-question",
    default=None,
    help="Run this question after indexing; defaults to the first question in the dataset.",
)
@click.option(
    "--skip-sanity-query",
    is_flag=True,
    help="Skip running the sanity query after indexing completes.",
)
@click.option(
    "--context-limit",
    default=5,
    show_default=True,
    help="Maximum number of contexts to display.",
)
def main(
    dataset_dir: str,
    cache_dir: str,
    chunk_docs: bool,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
    k: int,
    similarity_threshold: float,
    min_docs: int,
    max_docs: int,
    use_reranker: bool,
    reranker_model: str,
    rerank_threshold: float,
    llm_model: str,
    temperature: float,
    max_tokens: int,
    force: bool,
    sanity_question: str | None,
    skip_sanity_query: bool,
    context_limit: int,
    inline_metadata: bool,
) -> None:
    load_dotenv()

    try:
        dataset = RAGDataset.from_dataset_dir(dataset_dir)
    except Exception as exc:
        click.echo(f"Failed to load dataset: {exc}", err=True)
        sys.exit(1)

    config = RAGConfig(
        rag_model="naive",
        llm_model=llm_model,
        temperature=temperature,
        max_tokens=max_tokens,
        naive={
            "chunk_documents": chunk_docs,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding_model": embedding_model,
            "vector_store": "faiss",
            "k_retrieve": k,
            "similarity_threshold": similarity_threshold,
            "use_reranker": use_reranker,
            "reranker_model": reranker_model,
            "rerank_threshold": rerank_threshold,
            "min_docs": min_docs,
            "max_docs": max_docs,
            "cache_dir": cache_dir,
            "inline_metadata": inline_metadata,
        },
    )

    index_dir = _index_directory(dataset, config)
    if force:
        _wipe_index(index_dir)
    elif not index_dir.exists():
        click.echo("No cached index found; a new one will be created.")

    rag = RAGSystem(config, dataset)

    report = rag.index_report
    if report is None:
        click.echo("Index report unavailable; indexing may have failed.", err=True)
        sys.exit(1)

    _print_report(report, index_dir)

    if skip_sanity_query:
        return

    try:
        question_id, query_text = _resolve_question(dataset, sanity_question)
    except Exception as exc:
        click.echo(f"Failed to pick a sanity question: {exc}", err=True)
        sys.exit(1)

    try:
        answer, contexts = rag.query(query_text)
    except Exception as exc:
        click.echo(f"Sanity query failed: {exc}", err=True)
        sys.exit(1)

    click.echo("\nSanity query:")
    click.echo(f"Question ID: {question_id}")
    click.echo(f"Q: {query_text}")
    click.echo(f"A: {answer}\n")
    _print_contexts(contexts, context_limit)


if __name__ == "__main__":
    main()
