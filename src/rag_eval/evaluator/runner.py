"""Experiment runner for comparing RAG configurations."""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from rag_eval.dataset.corpus import Corpus
    from rag_eval.dataset.eval_dataset import EvalDataset
    from rag_eval.systems.base import RAGSystemBase
    from rag_eval.systems.config import RAGConfig

from rag_eval.evaluator.metrics import RAGEvaluator
from rag_eval.evaluator.results import ExperimentResult, ExperimentSummary
from rag_eval.evaluator.tracking import is_tracking_available, log_experiment

logger = logging.getLogger(__name__)


# Type alias for RAG system factory
RAGSystemFactory = Callable[["RAGConfig"], "RAGSystemBase"]


def get_default_rag_factory(rag_type: str) -> RAGSystemFactory:
    """Get the default factory for a RAG type.

    Args:
        rag_type: Type of RAG system ("naive" or "graphrag").

    Returns:
        Factory function that creates RAG systems.

    Raises:
        ValueError: If rag_type is not supported.
    """
    if rag_type == "naive":
        from rag_eval.systems.implementations.naive import NaiveRAGSystem

        return NaiveRAGSystem
    elif rag_type == "graphrag":
        from rag_eval.systems.implementations.graphrag import GraphRAGSystem

        return GraphRAGSystem
    else:
        raise ValueError(f"Unknown RAG type: {rag_type!r}")


class ExperimentRunner:
    """Run experiments comparing RAG configurations.

    The runner evaluates multiple RAG configurations and collects results
    for comparison. It supports:
    - Multiple RAG system types (Naive, GraphRAG, custom)
    - Multiple configurations per system
    - Answer and score caching
    - Result persistence

    Example:
        ```python
        from rag_eval.evaluator import ExperimentRunner
        from rag_eval import RAGConfig, Corpus, EvalDataset

        corpus = Corpus.from_jsonl("corpus.jsonl")
        eval_dataset = EvalDataset.from_jsonl("questions.jsonl")

        configs = [
            RAGConfig(rag_type="naive", naive=NaiveRAGConfig(k_retrieve=5)),
            RAGConfig(rag_type="naive", naive=NaiveRAGConfig(k_retrieve=10)),
            RAGConfig(rag_type="graphrag"),
        ]

        runner = ExperimentRunner(output_dir="./results")
        summary = runner.run_experiments(configs, corpus, eval_dataset)

        best = summary.find_best("faithfulness")
        print(f"Best config: {best.config_sig}")
        ```
    """

    def __init__(
        self,
        output_dir: str | Path = "./experiment_results",
        evaluator: RAGEvaluator | None = None,
    ) -> None:
        """Initialize the experiment runner.

        Args:
            output_dir: Directory to save results.
            evaluator: RAGEvaluator instance. If None, uses default metrics.
        """
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.evaluator = evaluator or RAGEvaluator()

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def run_experiments(
        self,
        configs: list[RAGConfig],
        corpus: Corpus,
        eval_dataset: EvalDataset,
        rag_factories: dict[str, RAGSystemFactory] | None = None,
        recreate: bool = False,
        tracking: bool = True,
        experiment_name: str = "rag-eval",
    ) -> ExperimentSummary:
        """Run experiments for multiple configurations.

        Args:
            configs: List of RAG configurations to test.
            corpus: Document corpus.
            eval_dataset: Evaluation dataset.
            rag_factories: Optional dict mapping rag_type to factory function.
                          If not provided, uses default factories.
            recreate: If True, ignore existing checkpoints and start fresh.
            tracking: If True, log experiments to MLflow (if available).
            experiment_name: MLflow experiment name.

        Returns:
            ExperimentSummary with all results.
        """
        # Load existing summary if present (to append, not overwrite)
        summary_path = self.output_dir / "summary.json"
        if summary_path.exists():
            logger.info(f"Loading existing summary from {summary_path}")
            summary = ExperimentSummary.load(summary_path)
            # Update corpus/dataset name if different
            summary.corpus_name = corpus.name
            summary.dataset_name = eval_dataset.name
        else:
            summary = ExperimentSummary(
                corpus_name=corpus.name,
                dataset_name=eval_dataset.name,
            )

        # Track existing config signatures to avoid duplicates
        existing_sigs = {r.config_sig for r in summary.results}

        for idx, config in enumerate(configs):
            config_sig = config.get_config_signature()

            logger.info(f"\n{'='*60}")
            logger.info(f"Experiment {idx + 1}/{len(configs)}")
            logger.info(f"Config: {config.rag_type}, sig={config_sig}")
            logger.info(f"{'='*60}")

            # Skip if this config was already run with enough samples (unless recreate=True)
            if config_sig in existing_sigs and not recreate:
                existing_result = next(r for r in summary.results if r.config_sig == config_sig)
                requested_samples = len(eval_dataset)

                # Re-run if we need more samples than previously evaluated
                if existing_result.num_samples < requested_samples:
                    logger.info(
                        f"Expanding {config_sig}: {existing_result.num_samples} → {requested_samples} samples "
                        f"(index cached, only new questions evaluated)"
                    )
                    # Don't skip - fall through to run_single_experiment which uses checkpoints
                else:
                    logger.info(f"Skipping {config_sig} - already in summary (use --recreate to re-run)")
                    # Log to MLflow so the current experiment view is complete (but skip if already logged)
                    if tracking and is_tracking_available():
                        answers_path = None
                        if existing_result.answers_file:
                            answers_path = self.output_dir / "answers" / existing_result.answers_file
                        log_experiment(config, existing_result, answers_path, experiment_name, allow_duplicate=False)
                    continue

            try:
                result, answers_path = self.run_single_experiment(
                    config=config,
                    corpus=corpus,
                    eval_dataset=eval_dataset,
                    config_id=config_sig,  # Use config_sig as ID for consistency
                    rag_factories=rag_factories,
                    recreate=recreate,
                )

                # Remove old result if re-running
                if config_sig in existing_sigs:
                    summary.results = [r for r in summary.results if r.config_sig != config_sig]

                summary.add_result(result)
                existing_sigs.add(config_sig)
                logger.info(f"Scores: {result.scores}")

                # Save summary after each experiment (checkpoint for interruption recovery)
                summary.save(summary_path)

                # Log to MLflow if tracking is enabled
                if tracking and is_tracking_available():
                    log_experiment(config, result, answers_path, experiment_name)

            except Exception as e:
                logger.error(f"Experiment {idx} failed: {e}")
                continue

        # Save summary (appends to existing)
        summary.save(summary_path)
        logger.info(f"Saved summary to {summary_path} ({len(summary.results)} total experiments)")

        return summary

    def run_single_experiment(
        self,
        config: RAGConfig,
        corpus: Corpus,
        eval_dataset: EvalDataset,
        config_id: int | str = 0,
        rag_factories: dict[str, RAGSystemFactory] | None = None,
        recreate: bool = False,
    ) -> tuple[ExperimentResult, Path]:
        """Run a single experiment with one configuration.

        Args:
            config: RAG configuration.
            corpus: Document corpus.
            eval_dataset: Evaluation dataset.
            config_id: Identifier for this configuration.
            rag_factories: Optional dict mapping rag_type to factory function.
            recreate: If True, ignore existing checkpoints and start fresh.

        Returns:
            Tuple of (ExperimentResult, answers_path).
        """
        # Get or create RAG system factory
        if rag_factories and config.rag_type in rag_factories:
            factory = rag_factories[config.rag_type]
        else:
            factory = get_default_rag_factory(config.rag_type)

        # Create RAG system
        logger.info(f"Creating {config.rag_type} RAG system...")
        rag_system = factory(config)

        # Build index
        logger.info("Building index...")
        index_report = rag_system.create_index(corpus)
        logger.info(
            f"Index: {index_report.indexed_documents} docs indexed, "
            f"reused_existing={index_report.reused_existing}"
        )

        # Run evaluation with checkpointing and parallel processing
        config_sig = config.get_config_signature()

        # GraphRAG doesn't support parallel queries - force sequential
        eval_config = config.eval
        if config.rag_type == "graphrag":
            logger.info("GraphRAG: using sequential queries (max_workers=1)")
            eval_config = replace(config.eval, max_workers=1)

        logger.info("Running evaluation...")
        eval_result = self.evaluator.evaluate(
            rag_system=rag_system,
            corpus=corpus,
            eval_dataset=eval_dataset,
            checkpoint_dir=self.checkpoint_dir,
            config_sig=config_sig,
            eval_config=eval_config,
            recreate=recreate,
        )

        # Save answers
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        answers_file = f"{config_sig}__{timestamp}.jsonl"
        answers_path = self.output_dir / "answers" / answers_file
        eval_result.save_answers(answers_path)
        logger.info(f"Saved answers to {answers_path}")

        # Clean up RAG system resources
        if hasattr(rag_system, "close"):
            rag_system.close()

        result = ExperimentResult(
            config_id=config_id,
            config_sig=config_sig,
            config=config.to_dict(),
            scores=eval_result.scores,
            num_samples=eval_result.num_samples,
            answers_file=answers_file,
        )
        return result, answers_path


def run_experiment(
    configs: list[RAGConfig],
    corpus: Corpus,
    eval_dataset: EvalDataset,
    output_dir: str | Path = "./experiment_results",
) -> ExperimentSummary:
    """Convenience function to run experiments.

    Args:
        configs: List of RAG configurations.
        corpus: Document corpus.
        eval_dataset: Evaluation dataset.
        output_dir: Directory for results.

    Returns:
        ExperimentSummary with results.
    """
    runner = ExperimentRunner(output_dir=output_dir)
    return runner.run_experiments(configs, corpus, eval_dataset)


def get_sample(
    eval_dataset: EvalDataset,
    n: int = 10,
    seed: int | None = 42,
) -> EvalDataset:
    """Get a sample of the evaluation dataset for quick testing.

    Args:
        eval_dataset: Full evaluation dataset.
        n: Number of samples.
        seed: Random seed for reproducibility.

    Returns:
        Sampled EvalDataset.
    """
    return eval_dataset.sample(n=n, seed=seed)
