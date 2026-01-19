"""CLI for RAG Eval Toolkit."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def setup_logging(verbose: bool) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@click.group()
@click.version_option(version="0.1.0")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
def main(verbose: bool) -> None:
    """RAG Eval Toolkit - Evaluate and compare RAG systems."""
    setup_logging(verbose)


@main.command()
@click.option(
    "--corpus",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to corpus JSONL file",
)
@click.option(
    "--dataset",
    "-d",
    required=True,
    type=click.Path(exists=True),
    help="Path to evaluation dataset JSONL file",
)
@click.option(
    "--config",
    "-C",
    type=click.Path(exists=True),
    help="Path to config JSON/YAML file",
)
@click.option(
    "--system",
    "-s",
    type=click.Choice(["naive", "graphrag"]),
    default="naive",
    help="RAG system type",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="./results",
    help="Output directory for results",
)
@click.option(
    "--sample",
    "-n",
    type=int,
    default=None,
    help="Run on a sample of N questions (for quick testing)",
)
def run(
    corpus: str,
    dataset: str,
    config: str | None,
    system: str,
    output: str,
    sample: int | None,
) -> None:
    """Run RAG evaluation experiment.

    Examples:

        # Run with default naive RAG
        rag-eval run -c corpus.jsonl -d questions.jsonl

        # Run with GraphRAG
        rag-eval run -c corpus.jsonl -d questions.jsonl -s graphrag

        # Run on a sample of 10 questions
        rag-eval run -c corpus.jsonl -d questions.jsonl -n 10

        # Run with custom config
        rag-eval run -c corpus.jsonl -d questions.jsonl -C config.json
    """
    from rich.console import Console

    from rag_eval.dataset.corpus import Corpus
    from rag_eval.dataset.eval_dataset import EvalDataset
    from rag_eval.evaluator.runner import ExperimentRunner
    from rag_eval.systems.config import RAGConfig

    console = Console()

    # Load corpus and dataset
    console.print(f"[blue]Loading corpus from {corpus}...[/blue]")
    corpus_data = Corpus.from_jsonl(corpus)
    console.print(f"  Loaded {len(corpus_data)} documents")

    console.print(f"[blue]Loading dataset from {dataset}...[/blue]")
    eval_dataset = EvalDataset.from_jsonl(dataset)
    console.print(f"  Loaded {len(eval_dataset)} questions")

    # Sample if requested
    if sample:
        console.print(f"[yellow]Using sample of {sample} questions[/yellow]")
        eval_dataset = eval_dataset.sample(n=sample, seed=42)

    # Load or create config
    if config:
        console.print(f"[blue]Loading config from {config}...[/blue]")
        with open(config, "r") as f:
            config_data = json.load(f)
        rag_config = RAGConfig.from_dict(config_data)
    else:
        console.print(f"[blue]Using default {system} config[/blue]")
        rag_config = RAGConfig(rag_type=system)

    configs = [rag_config]

    # Run experiment
    console.print("\n[green]Starting evaluation...[/green]\n")
    runner = ExperimentRunner(output_dir=output)
    summary = runner.run_experiments(configs, corpus_data, eval_dataset)

    # Print results
    console.print("\n[green]Results:[/green]")
    for result in summary.results:
        console.print(f"\n  Config: {result.config_sig}")
        console.print("  Scores:")
        for metric, score in result.scores.items():
            console.print(f"    {metric}: {score:.4f}")

    console.print(f"\n[blue]Results saved to {output}[/blue]")


@main.command()
@click.option(
    "--corpus",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to corpus JSONL file",
)
@click.option(
    "--dataset",
    "-d",
    required=True,
    type=click.Path(exists=True),
    help="Path to evaluation dataset JSONL file",
)
@click.option(
    "--configs",
    "-C",
    required=True,
    type=click.Path(exists=True),
    help="Path to configs JSON file (list of configs)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="./results",
    help="Output directory for results",
)
@click.option(
    "--sample",
    "-n",
    type=int,
    default=None,
    help="Run on a sample of N questions",
)
def compare(
    corpus: str,
    dataset: str,
    configs: str,
    output: str,
    sample: int | None,
) -> None:
    """Compare multiple RAG configurations.

    The configs file should be a JSON array of RAG configurations.

    Example:

        rag-eval compare -c corpus.jsonl -d questions.jsonl -C configs.json
    """
    from rich.console import Console
    from rich.table import Table

    from rag_eval.dataset.corpus import Corpus
    from rag_eval.dataset.eval_dataset import EvalDataset
    from rag_eval.evaluator.runner import ExperimentRunner
    from rag_eval.systems.config import RAGConfig

    console = Console()

    # Load data
    console.print(f"[blue]Loading corpus from {corpus}...[/blue]")
    corpus_data = Corpus.from_jsonl(corpus)

    console.print(f"[blue]Loading dataset from {dataset}...[/blue]")
    eval_dataset = EvalDataset.from_jsonl(dataset)

    if sample:
        console.print(f"[yellow]Using sample of {sample} questions[/yellow]")
        eval_dataset = eval_dataset.sample(n=sample, seed=42)

    # Load configs
    console.print(f"[blue]Loading configs from {configs}...[/blue]")
    with open(configs, "r") as f:
        configs_data = json.load(f)

    rag_configs = [RAGConfig.from_dict(c) for c in configs_data]
    console.print(f"  Loaded {len(rag_configs)} configurations")

    # Run experiments
    console.print("\n[green]Starting comparison...[/green]\n")
    runner = ExperimentRunner(output_dir=output)
    summary = runner.run_experiments(rag_configs, corpus_data, eval_dataset)

    # Print results table
    console.print("\n[green]Results Comparison:[/green]\n")

    # Get all metrics
    all_metrics = set()
    for result in summary.results:
        all_metrics.update(result.scores.keys())

    table = Table(title="RAG Configuration Comparison")
    table.add_column("Config", style="cyan")
    table.add_column("Type", style="blue")
    for metric in sorted(all_metrics):
        table.add_column(metric, justify="right")

    for result in summary.results:
        row = [
            result.config_sig[:8],
            result.config.get("rag_type", "unknown"),
        ]
        for metric in sorted(all_metrics):
            score = result.scores.get(metric, 0)
            row.append(f"{score:.3f}")
        table.add_row(*row)

    console.print(table)

    # Find and print best
    best = summary.find_best("faithfulness")
    if best:
        console.print(f"\n[green]Best config (by faithfulness): {best.config_sig}[/green]")

    console.print(f"\n[blue]Results saved to {output}[/blue]")


@main.group()
def dataset() -> None:
    """Dataset management commands."""
    pass


@dataset.command(name="info")
@click.argument("path", type=click.Path(exists=True))
def dataset_info(path: str) -> None:
    """Show information about a dataset.

    Example:

        rag-eval dataset info questions.jsonl
    """
    from rich.console import Console
    from rich.table import Table

    from rag_eval.dataset.eval_dataset import EvalDataset

    console = Console()

    eval_dataset = EvalDataset.from_jsonl(path)

    console.print(f"\n[blue]Dataset: {path}[/blue]")
    console.print(f"  Total questions: {len(eval_dataset)}")

    # Question type distribution
    type_dist = eval_dataset.get_question_types_distribution()
    console.print("\n  Question Types:")
    for qtype, count in sorted(type_dist.items()):
        console.print(f"    {qtype}: {count}")

    # Evidence count distribution
    evidence_dist = eval_dataset.get_evidence_count_distribution()
    console.print("\n  Evidence Counts:")
    for count, num in sorted(evidence_dist.items()):
        console.print(f"    {count} docs: {num} questions")


@dataset.command(name="sample")
@click.argument("path", type=click.Path(exists=True))
@click.option("-n", "--num", type=int, default=10, help="Number of samples")
@click.option("-o", "--output", required=True, help="Output file path")
@click.option("--seed", type=int, default=42, help="Random seed")
def dataset_sample(path: str, num: int, output: str, seed: int) -> None:
    """Create a sample from a dataset.

    Example:

        rag-eval dataset sample questions.jsonl -n 50 -o sample.jsonl
    """
    from rich.console import Console

    from rag_eval.dataset.eval_dataset import EvalDataset

    console = Console()

    eval_dataset = EvalDataset.from_jsonl(path)
    sampled = eval_dataset.sample(n=num, seed=seed)
    sampled.to_jsonl(output)

    console.print(f"[green]Saved {len(sampled)} samples to {output}[/green]")


@dataset.command(name="build")
@click.option(
    "--corpus",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to corpus JSONL file",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(),
    help="Output path for generated dataset",
)
@click.option("-n", "--num", type=int, default=50, help="Number of questions to generate")
@click.option(
    "--types",
    "-t",
    multiple=True,
    default=["factoid"],
    help="Question types to generate (factoid, multi_hop)",
)
@click.option("--model", default="gpt-4o-mini", help="LLM model to use")
@click.option("--workers", type=int, default=5, help="Number of parallel workers")
@click.option("--seed", type=int, default=None, help="Random seed")
def dataset_build(
    corpus: str,
    output: str,
    num: int,
    types: tuple[str, ...],
    model: str,
    workers: int,
    seed: int | None,
) -> None:
    """Generate an evaluation dataset from a corpus.

    Uses LLM to generate QA pairs from corpus documents.

    Examples:

        # Generate 50 factoid questions
        rag-eval dataset build -c corpus.jsonl -o questions.jsonl -n 50

        # Generate mix of factoid and multi-hop questions
        rag-eval dataset build -c corpus.jsonl -o questions.jsonl -t factoid -t multi_hop
    """
    from rich.console import Console

    from rag_eval.dataset.corpus import Corpus
    from rag_eval.dataset.builder import generate_eval_dataset

    console = Console()

    console.print(f"[blue]Loading corpus from {corpus}...[/blue]")
    corpus_data = Corpus.from_jsonl(corpus)
    console.print(f"  Loaded {len(corpus_data)} documents")

    console.print(f"\n[green]Generating {num} questions...[/green]")
    console.print(f"  Types: {', '.join(types)}")
    console.print(f"  Model: {model}")

    eval_dataset = generate_eval_dataset(
        corpus_data,
        num_questions=num,
        question_types=list(types),
        model=model,
        max_workers=workers,
        seed=seed,
    )

    eval_dataset.to_jsonl(output)
    console.print(f"\n[green]Saved {len(eval_dataset)} questions to {output}[/green]")


@dataset.command(name="adapt")
@click.option(
    "--source",
    "-s",
    required=True,
    help="HuggingFace dataset name (e.g., 'yixuantt/MultiHopRAG')",
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    type=click.Path(),
    help="Output directory for corpus.jsonl and questions.jsonl",
)
@click.option(
    "--type",
    "-t",
    "dataset_type",
    type=click.Choice(["multi_hop", "qa"]),
    default="multi_hop",
    help="Dataset type for processing",
)
@click.option("--split", default="train", help="Dataset split to load")
def dataset_adapt(
    source: str,
    output_dir: str,
    dataset_type: str,
    split: str,
) -> None:
    """Adapt a HuggingFace dataset to our format.

    Downloads and converts HuggingFace datasets to Corpus + EvalDataset.

    Examples:

        # Adapt MultiHopRAG dataset
        rag-eval dataset adapt -s yixuantt/MultiHopRAG -o ./data/

        # Adapt a SQUAD-like dataset
        rag-eval dataset adapt -s squad -t qa -o ./data/
    """
    from pathlib import Path

    from rich.console import Console

    from rag_eval.dataset.adapter import adapt_huggingface_dataset

    console = Console()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    console.print(f"[blue]Adapting dataset: {source}[/blue]")
    console.print(f"  Type: {dataset_type}")
    console.print(f"  Split: {split}")

    corpus, eval_dataset = adapt_huggingface_dataset(
        source, dataset_type=dataset_type, split=split
    )

    corpus_path = output_path / "corpus.jsonl"
    questions_path = output_path / "questions.jsonl"

    corpus.to_jsonl(corpus_path)
    eval_dataset.to_jsonl(questions_path)

    console.print(f"\n[green]Saved {len(corpus)} documents to {corpus_path}[/green]")
    console.print(f"[green]Saved {len(eval_dataset)} questions to {questions_path}[/green]")


@dataset.command(name="score")
@click.option(
    "--dataset",
    "-d",
    required=True,
    type=click.Path(exists=True),
    help="Path to evaluation dataset JSONL file",
)
@click.option(
    "--corpus",
    "-c",
    type=click.Path(exists=True),
    help="Path to corpus JSONL (for groundedness scoring)",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(),
    help="Output path for filtered dataset",
)
@click.option("--model", default="gpt-4o-mini", help="LLM model for scoring")
@click.option("--workers", type=int, default=5, help="Parallel workers")
@click.option("--groundedness-min", type=int, default=3, help="Min groundedness score (1-5)")
@click.option("--relevance-min", type=int, default=3, help="Min relevance score (1-5)")
@click.option("--standalone-min", type=int, default=3, help="Min standalone score (1-5)")
@click.option("--complexity-min", type=int, default=2, help="Min complexity score (1-5)")
def dataset_score(
    dataset: str,
    corpus: str | None,
    output: str,
    model: str,
    workers: int,
    groundedness_min: int,
    relevance_min: int,
    standalone_min: int,
    complexity_min: int,
) -> None:
    """Score and filter dataset by quality.

    Uses LLM-based critique to evaluate question quality across multiple
    dimensions and filters out low-quality questions.

    Examples:

        # Score with default thresholds
        rag-eval dataset score -d questions.jsonl -o filtered.jsonl

        # Score with corpus (enables groundedness)
        rag-eval dataset score -d questions.jsonl -c corpus.jsonl -o filtered.jsonl

        # Custom thresholds
        rag-eval dataset score -d questions.jsonl -o filtered.jsonl --complexity-min 4
    """
    from rich.console import Console

    from rag_eval.dataset.corpus import Corpus
    from rag_eval.dataset.eval_dataset import EvalDataset
    from rag_eval.dataset.critique import score_eval_dataset

    console = Console()

    console.print(f"[blue]Loading dataset from {dataset}...[/blue]")
    eval_dataset = EvalDataset.from_jsonl(dataset)
    console.print(f"  Loaded {len(eval_dataset)} questions")

    corpus_data = None
    if corpus:
        console.print(f"[blue]Loading corpus from {corpus}...[/blue]")
        corpus_data = Corpus.from_jsonl(corpus)
        console.print(f"  Loaded {len(corpus_data)} documents")

    console.print(f"\n[green]Scoring questions...[/green]")
    console.print(f"  Model: {model}")
    console.print(f"  Thresholds: groundedness>={groundedness_min}, relevance>={relevance_min}, "
                  f"standalone>={standalone_min}, complexity>={complexity_min}")

    filtered_dataset, scores = score_eval_dataset(
        eval_dataset,
        corpus=corpus_data,
        model=model,
        max_workers=workers,
        groundedness_min=groundedness_min,
        relevance_min=relevance_min,
        standalone_min=standalone_min,
        complexity_min=complexity_min,
    )

    filtered_dataset.to_jsonl(output)
    console.print(f"\n[green]Saved {len(filtered_dataset)}/{len(eval_dataset)} "
                  f"questions to {output}[/green]")


if __name__ == "__main__":
    main()
