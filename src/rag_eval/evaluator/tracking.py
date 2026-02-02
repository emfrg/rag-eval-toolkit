"""MLflow experiment tracking integration."""

from __future__ import annotations

import json
import logging
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_eval.evaluator.results import ExperimentResult
    from rag_eval.systems.config import RAGConfig

logger = logging.getLogger(__name__)

# Check if mlflow is available
try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None  # type: ignore[assignment]
    MLFLOW_AVAILABLE = False


def is_tracking_available() -> bool:
    """Check if MLflow is installed and available."""
    return MLFLOW_AVAILABLE


def _sanitize_metric_name(name: str) -> str:
    """Sanitize metric name for MLflow compatibility.

    MLflow accepts: alphanumerics, underscores, dashes, periods, spaces, colons, slashes.
    RAGAS produces names like 'factual_correctness(mode=f1)' which need sanitization.
    """
    # Replace ( ) = with underscores
    sanitized = re.sub(r"[()=]", "_", name)
    # Replace any remaining invalid chars
    sanitized = re.sub(r"[^a-zA-Z0-9_\-.\s:/]", "_", sanitized)
    # Clean up multiple underscores
    sanitized = re.sub(r"_+", "_", sanitized)
    # Remove trailing underscores
    sanitized = sanitized.strip("_")
    return sanitized


def log_experiment(
    config: RAGConfig,
    result: ExperimentResult,
    answers_path: Path | None = None,
    experiment_name: str = "rag-eval",
) -> str | None:
    """Log an experiment run to MLflow.

    Args:
        config: The RAG configuration used.
        result: The experiment result with scores.
        answers_path: Path to the answers JSONL file (logged as artifact).
        experiment_name: Name of the MLflow experiment.

    Returns:
        The MLflow run ID if successful, None otherwise.
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not installed. Run: uv sync --extra tracking")
        return None

    mlflow.set_experiment(experiment_name)

    # Generate run name from config.name or fallback to rag_type_config_sig
    run_name = config.name if config.name else f"{config.rag_type}_{result.config_sig}"

    with mlflow.start_run(run_name=run_name) as run:
        # Log common parameters
        params = {
            "rag_type": config.rag_type,
            "llm_provider": config.llm_provider,
            "llm_model": config.llm_model,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "config_sig": result.config_sig,
        }

        # Add type-specific parameters
        if config.rag_type == "naive":
            params.update({
                "naive.k_retrieve": config.naive.k_retrieve,
                "naive.chunk_documents": config.naive.chunk_documents,
                "naive.chunk_size": config.naive.chunk_size,
                "naive.chunk_overlap": config.naive.chunk_overlap,
                "naive.embedding_model": config.naive.embedding_model,
                "naive.use_reranker": config.naive.use_reranker,
                "naive.max_docs": config.naive.max_docs,
            })
        elif config.rag_type == "graphrag":
            params.update({
                "graphrag.query_mode": config.graphrag.query.mode,
                "graphrag.top_k": config.graphrag.query.top_k,
            })

        # Add eval config
        params.update({
            "eval.batch_size": config.eval.batch_size,
            "eval.max_workers": config.eval.max_workers,
        })

        mlflow.log_params(params)

        # Log metrics (sanitize names for MLflow compatibility)
        for metric_name, score in result.scores.items():
            safe_name = _sanitize_metric_name(metric_name)
            mlflow.log_metric(safe_name, score)

        # Log num_samples as metric
        if hasattr(result, "num_samples"):
            mlflow.log_metric("num_samples", result.num_samples)

        # Log answers file as artifact
        if answers_path and answers_path.exists():
            mlflow.log_artifact(str(answers_path))

        # Log full config as JSON artifact
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(config.to_dict(), f, indent=2)
            config_temp_path = f.name

        mlflow.log_artifact(config_temp_path, "config")

        # Clean up temp file
        Path(config_temp_path).unlink(missing_ok=True)

        logger.info(f"Logged to MLflow: run_id={run.info.run_id}")
        return run.info.run_id

    return None
