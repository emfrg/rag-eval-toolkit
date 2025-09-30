#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import click
from dotenv import load_dotenv
from langchain.schema import Document

from rag_system import IndexReport, RAGConfig, RAGDataset, RAGSystem


def answer_and_context(rag: RAGSystem, question: str) -> Tuple[str, List[Document]]:
    return rag.query(question)


def _print_report(report: IndexReport, working_dir: Path) -> None:
    if report.reused_existing:
        click.echo(
            f"Reused existing LightRAG index ({report.total_documents} docs) at {working_dir}"
        )
    else:
        click.echo(
            f"Indexed {report.ingested_documents} / {report.total_documents} documents into {working_dir}"
        )


def _print_contexts(contexts: List[Document], limit: int = 5) -> None:
    if not contexts:
        click.echo("No supporting contexts retrieved.")
        return

    # TODO: check context length to remove duplicates
    click.echo(f"Retrieved {len(contexts)} context chunks:")
    for doc in contexts[:limit]:
        doc_id = doc.metadata.get("doc_id", "unknown")
        snippet = doc.page_content.replace("\n", " ")[:200]
        click.echo(f"  - {doc_id}: {snippet}...")


@click.command()
@click.option(
    "--dataset-dir",
    required=True,
    type=click.Path(path_type=str),
    help="Directory produced by eval_dataset_builder (contains corpus.jsonl).",
)
@click.option(
    "--graph-cache",
    default="graphrag_cache",
    type=click.Path(path_type=str),
    show_default=True,
    help="Where LightRAG will store its artifacts.",
)
@click.option(
    "--max-parallel-insert",
    default=4,
    show_default=True,
    help="Maximum concurrent insert tasks during indexing.",
)
@click.option(
    "--batch-size",
    default=128,
    show_default=True,
    help="Number of records per insert batch.",
)
@click.option(
    "--llm-model",
    default="gpt-4o-mini",
    show_default=True,
    help="OpenAI model used for LightRAG reasoning.",
)
@click.option(
    "--query-mode",
    type=click.Choice(["semantic", "graph", "hybrid"]),
    default="hybrid",
    show_default=True,
    help="Retrieval mode for sanity query.",
)
@click.option(
    "--top-k",
    default=50,
    show_default=True,
    help="Number of candidates LightRAG should consider.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Force rebuild of the LightRAG index.",
)
@click.option(
    "--inline-metadata/--no-inline-metadata",
    default=False,
    show_default=True,
    help="Embed document metadata within the indexed content.",
)
@click.option(
    "--skip-sanity-query",
    is_flag=True,
    help="Skip running the sanity query after indexing.",
)
@click.option(
    "--sanity-query",
    default="Who was the CEO that ended up in jail? Just name. \nName:",  # "What is this dataset about?",
    show_default=True,
    help="Question to run after indexing completes.",
)
def main(
    dataset_dir: str,
    graph_cache: str,
    max_parallel_insert: int,
    batch_size: int,
    llm_model: str,
    query_mode: str,
    top_k: int,
    force: bool,
    skip_sanity_query: bool,
    sanity_query: str,
    inline_metadata: bool,
) -> None:
    load_dotenv()

    try:
        dataset = RAGDataset.from_dataset_dir(dataset_dir)
    except Exception as exc:  # pragma: no cover
        click.echo(f"Failed to load dataset: {exc}", err=True)
        sys.exit(1)

    config = RAGConfig(
        rag_model="graphrag",
        llm_model=llm_model,
        graphrag={
            "indexing": {
                "graph_cache_dir": graph_cache,
                "max_parallel_insert": max_parallel_insert,
                "batch_size": batch_size,
                "force_reindex": force,
                "inline_metadata": inline_metadata,
            },
            "query": {
                "mode": query_mode,
                "top_k": top_k,
            },
        },
    )

    try:
        rag = RAGSystem(config, dataset)
    except RuntimeError as exc:  # LightRAG missing or misconfigured
        click.echo(str(exc), err=True)
        sys.exit(1)

    report = rag.index_report
    if report is None:
        click.echo("Index report unavailable; build may have failed.", err=True)
        sys.exit(1)

    working_dir = getattr(rag.backend, "working_directory", None)
    if isinstance(working_dir, Path):
        _print_report(report, working_dir)
    else:
        click.echo("Indexing completed.")

    if skip_sanity_query:
        return

    try:
        answer, contexts = answer_and_context(rag, sanity_query)
    except Exception as exc:
        click.echo(f"Sanity query failed: {exc}", err=True)
        sys.exit(1)

    click.echo("\nSanity query:")
    click.echo(f"Q: {sanity_query}")
    click.echo(f"A: {answer}\n")
    _print_contexts(contexts)


if __name__ == "__main__":
    main()
