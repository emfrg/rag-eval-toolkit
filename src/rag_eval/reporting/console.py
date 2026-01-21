"""Console output using rich library."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from rag_eval.evaluator.results import ExperimentResult, ExperimentSummary


def get_console() -> Console:
    """Get a rich Console instance."""
    return Console()


def print_results_summary(
    result: ExperimentResult,
    console: Console | None = None,
) -> None:
    """Print a summary of a single experiment result.

    Args:
        result: The experiment result to display.
        console: Rich console instance. If None, creates a new one.
    """
    if console is None:
        console = get_console()

    # Header
    console.print(
        Panel(
            f"[bold blue]Config: {result.config_sig}[/bold blue]\n"
            f"Type: {result.config.get('rag_type', 'unknown')}",
            title="Experiment Result",
        )
    )

    # Scores table
    table = Table(title="Evaluation Scores", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Score", justify="right", style="green")

    for metric, score in sorted(result.scores.items()):
        # Color code scores
        if score >= 0.8:
            style = "green"
        elif score >= 0.5:
            style = "yellow"
        else:
            style = "red"
        table.add_row(metric, f"[{style}]{score:.4f}[/{style}]")

    console.print(table)

    # Metadata
    if result.answers_file:
        console.print(f"\n[dim]Answers saved to: {result.answers_file}[/dim]")


def print_comparison_table(
    summary: ExperimentSummary,
    console: Console | None = None,
    highlight_best: bool = True,
) -> None:
    """Print a comparison table of multiple experiments.

    Args:
        summary: The experiment summary with multiple results.
        console: Rich console instance. If None, creates a new one.
        highlight_best: Whether to highlight the best score for each metric.
    """
    if console is None:
        console = get_console()

    if not summary.results:
        console.print("[yellow]No results to display[/yellow]")
        return

    # Collect all metrics
    all_metrics = set()
    for result in summary.results:
        all_metrics.update(result.scores.keys())
    metrics = sorted(all_metrics)

    # Find best scores for each metric
    best_scores: dict[str, float] = {}
    for metric in metrics:
        best_score = -1.0
        for result in summary.results:
            score = result.scores.get(metric, 0)
            if score > best_score:
                best_score = score
        best_scores[metric] = best_score

    # Build table
    table = Table(title="RAG Configuration Comparison", show_lines=True)
    table.add_column("Config", style="cyan", no_wrap=True)
    table.add_column("Type", style="blue")

    for metric in metrics:
        table.add_column(metric, justify="right")

    # Add rows
    for result in summary.results:
        row = [
            result.config_sig[:10],
            result.config.get("rag_type", "?"),
        ]

        for metric in metrics:
            score = result.scores.get(metric, 0)
            # Mark as best if score equals the best score (handles ties)
            is_best = highlight_best and abs(score - best_scores[metric]) < 0.0001

            # Format score with color
            if score >= 0.8:
                color = "green"
            elif score >= 0.5:
                color = "yellow"
            else:
                color = "red"

            if is_best:
                formatted = f"[bold {color}]{score:.3f}*[/bold {color}]"
            else:
                formatted = f"[{color}]{score:.3f}[/{color}]"

            row.append(formatted)

        table.add_row(*row)

    console.print(table)

    # Print legend
    if highlight_best:
        console.print("\n[dim]* = best score for metric[/dim]")

    # Print summary stats
    if summary.corpus_name:
        console.print(f"\n[dim]Corpus: {summary.corpus_name}[/dim]")
    if summary.dataset_name:
        console.print(f"[dim]Dataset: {summary.dataset_name}[/dim]")
    console.print(f"[dim]Total experiments: {len(summary.results)}[/dim]")


def print_progress_header(
    current: int,
    total: int,
    config_sig: str,
    rag_type: str,
    console: Console | None = None,
) -> None:
    """Print a progress header for an experiment.

    Args:
        current: Current experiment number (1-indexed).
        total: Total number of experiments.
        config_sig: Configuration signature.
        rag_type: Type of RAG system.
        console: Rich console instance.
    """
    if console is None:
        console = get_console()

    console.print()
    console.rule(f"[bold]Experiment {current}/{total}[/bold]")
    console.print(f"  [cyan]Config:[/cyan] {config_sig}")
    console.print(f"  [cyan]Type:[/cyan] {rag_type}")
    console.print()


def print_metric_progress(
    metric: str,
    score: float,
    console: Console | None = None,
) -> None:
    """Print a single metric score inline.

    Args:
        metric: Name of the metric.
        score: The score value.
        console: Rich console instance.
    """
    if console is None:
        console = get_console()

    if score >= 0.8:
        color = "green"
    elif score >= 0.5:
        color = "yellow"
    else:
        color = "red"

    console.print(f"  [{color}]{metric}: {score:.4f}[/{color}]")
