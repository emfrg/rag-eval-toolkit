"""Experiment runner for comparing RAG configurations."""

from __future__ import annotations

import logging
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
        self.evaluator = evaluator or RAGEvaluator()

    def run_experiments(
        self,
        configs: list[RAGConfig],
        corpus: Corpus,
        eval_dataset: EvalDataset,
        rag_factories: dict[str, RAGSystemFactory] | None = None,
    ) -> ExperimentSummary:
        """Run experiments for multiple configurations.

        Args:
            configs: List of RAG configurations to test.
            corpus: Document corpus.
            eval_dataset: Evaluation dataset.
            rag_factories: Optional dict mapping rag_type to factory function.
                          If not provided, uses default factories.

        Returns:
            ExperimentSummary with all results.
        """
        summary = ExperimentSummary(
            corpus_name=corpus.name,
            dataset_name=eval_dataset.name,
        )

        for idx, config in enumerate(configs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Experiment {idx + 1}/{len(configs)}")
            logger.info(f"Config: {config.rag_type}, sig={config.get_config_signature()}")
            logger.info(f"{'='*60}")

            try:
                result = self.run_single_experiment(
                    config=config,
                    corpus=corpus,
                    eval_dataset=eval_dataset,
                    config_id=idx,
                    rag_factories=rag_factories,
                )
                summary.add_result(result)
                logger.info(f"Scores: {result.scores}")

            except Exception as e:
                logger.error(f"Experiment {idx} failed: {e}")
                continue

        # Save summary
        summary_path = self.output_dir / "summary.json"
        summary.save(summary_path)
        logger.info(f"Saved summary to {summary_path}")

        return summary

    def run_single_experiment(
        self,
        config: RAGConfig,
        corpus: Corpus,
        eval_dataset: EvalDataset,
        config_id: int | str = 0,
        rag_factories: dict[str, RAGSystemFactory] | None = None,
    ) -> ExperimentResult:
        """Run a single experiment with one configuration.

        Args:
            config: RAG configuration.
            corpus: Document corpus.
            eval_dataset: Evaluation dataset.
            config_id: Identifier for this configuration.
            rag_factories: Optional dict mapping rag_type to factory function.

        Returns:
            ExperimentResult with scores and metadata.
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

        # Run evaluation
        logger.info("Running evaluation...")
        eval_result = self.evaluator.evaluate(
            rag_system=rag_system,
            corpus=corpus,
            eval_dataset=eval_dataset,
        )

        # Save answers
        config_sig = config.get_config_signature()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        answers_file = f"{config_sig}__{timestamp}.jsonl"
        answers_path = self.output_dir / "answers" / answers_file
        eval_result.save_answers(answers_path)
        logger.info(f"Saved answers to {answers_path}")

        return ExperimentResult(
            config_id=config_id,
            config_sig=config_sig,
            config=config.to_dict(),
            scores=eval_result.scores,
            answers_file=answers_file,
        )


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
