"""JSON export functionality."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_eval.evaluator.results import ExperimentResult, ExperimentSummary


def export_results(
    summary: ExperimentSummary,
    output_dir: str | Path,
    include_configs: bool = True,
) -> dict[str, Path]:
    """Export experiment results to JSON files.

    Creates the following files:
    - summary.json: Overview of all experiments with scores
    - results/{config_sig}.json: Individual result for each config

    Args:
        summary: The experiment summary to export.
        output_dir: Directory to save files.
        include_configs: Whether to include full configs in output.

    Returns:
        Dictionary mapping file type to path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True)

    exported: dict[str, Path] = {}

    # Export summary
    summary_path = output_dir / "summary.json"
    summary.save(summary_path)
    exported["summary"] = summary_path

    # Export individual results
    for result in summary.results:
        result_data = result.to_dict()
        if not include_configs:
            # Remove large config data
            result_data.pop("config", None)

        result_path = results_dir / f"{result.config_sig}.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        exported[f"result_{result.config_sig}"] = result_path

    return exported


def export_single_result(
    result: ExperimentResult,
    output_path: str | Path,
) -> Path:
    """Export a single experiment result to JSON.

    Args:
        result: The experiment result to export.
        output_path: Path for the output file.

    Returns:
        Path to the created file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    return output_path


def load_summary(path: str | Path) -> ExperimentSummary:
    """Load an experiment summary from JSON.

    Args:
        path: Path to the summary JSON file.

    Returns:
        Loaded ExperimentSummary.
    """
    from rag_eval.evaluator.results import ExperimentSummary

    return ExperimentSummary.load(path)
