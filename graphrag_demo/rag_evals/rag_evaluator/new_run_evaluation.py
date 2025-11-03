#!/usr/bin/env python3
# rag_evaluator/new_run_evaluation.py
from __future__ import annotations

import click
from dotenv import load_dotenv

from .new_runner import run


@click.command()
@click.option(
    "--dataset-dir",
    required=True,
    help="Path to the dataset directory produced by eval_dataset_builder.",
)
@click.option(
    "--questions-file",
    default="sampled_eval_questions.jsonl",
    show_default=True,
    help="Evaluation questions file relative to the dataset directory.",
)
@click.option(
    "--output-dir",
    default="./results",
    show_default=True,
    help="Directory where evaluation outputs will be stored.",
)
@click.option(
    "--experiment-name",
    default=None,
    help="Optional experiment name; defaults to a timestamp when omitted.",
)
@click.option(
    "--reuse-latest/--no-reuse-latest",
    default=True,
    show_default=True,
    help="Reuse the latest cached answers when available.",
)
@click.option(
    "--reuse-cached-scores/--no-reuse-cached-scores",
    default=True,
    show_default=True,
    help="Reuse cached RAGAS scores when answers are reused.",
)
@click.option(
    "--rag-cache",
    default="rag_cache",
    show_default=True,
    help="Location for naive FAISS indices.",
)
@click.option(
    "--graph-cache",
    default="graphrag_cache",
    show_default=True,
    help="Location for LightRAG graph indices.",
)
@click.option(
    "--force-graphrag",
    is_flag=True,
    help="Force rebuild of the LightRAG index.",
)
def main(
    dataset_dir: str,
    questions_file: str,
    output_dir: str,
    experiment_name: str | None,
    reuse_latest: bool,
    reuse_cached_scores: bool,
    rag_cache: str,
    graph_cache: str,
    force_graphrag: bool,
) -> None:
    """
    Run one naive and one GraphRAG configuration and print their evaluation summary.
    """
    load_dotenv()

    results = run(
        dataset_dir=dataset_dir,
        questions_file=questions_file,
        output_dir=output_dir,
        experiment_name=experiment_name,
        reuse_latest=reuse_latest,
        reuse_cached_scores=reuse_cached_scores,
        rag_cache=rag_cache,
        graph_cache=graph_cache,
        force_graphrag=force_graphrag,
    )

    click.echo("\n" + "=" * 50)
    click.echo("EXPERIMENT SUMMARY")
    click.echo("=" * 50)

    for result in results:
        config_id = result["config_id"]
        config = result["config"]
        scores = result.get("scores", {})

        click.echo(f"\nConfig {config_id}:")
        rag_model = config.get("rag_model", "unknown")
        click.echo(f"  RAG backend: {rag_model}")

        if rag_model == "naive":
            naive_cfg = config.get("naive", {})
            click.echo(f"  Chunking: {naive_cfg.get('chunk_documents', False)}")
            click.echo(f"  k_retrieve: {naive_cfg.get('k_retrieve')}")
            click.echo(f"  Use reranker: {naive_cfg.get('use_reranker', False)}")
            click.echo(
                f"  Inline metadata: {naive_cfg.get('inline_metadata', False)}"
            )
        elif rag_model == "graphrag":
            graph_cfg = config.get("graphrag", {})
            query_cfg = graph_cfg.get("query", {})
            click.echo(f"  Query mode: {query_cfg.get('mode', 'hybrid')}")
            click.echo(f"  top_k: {query_cfg.get('top_k')}")
            indexing_cfg = graph_cfg.get("indexing", {})
            click.echo(
                f"  Inline metadata: {indexing_cfg.get('inline_metadata', False)}"
            )

        if scores:
            click.echo("  Scores:")
            for metric, score in scores.items():
                click.echo(f"    {metric}: {score:.4f}")
        else:
            click.echo("  Scores: (none)")


if __name__ == "__main__":
    main()
