"""RAGAS metrics integration for RAG evaluation."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    FactualCorrectness,
    Faithfulness,
    LLMContextRecall,
    SemanticSimilarity,
)
from tqdm import tqdm

if TYPE_CHECKING:
    from ragas.metrics.base import Metric

    from rag_eval.dataset.corpus import Corpus
    from rag_eval.dataset.eval_dataset import EvalDataset, EvalQuestion
    from rag_eval.evaluator.results import AnswerRecord, EvaluationResult
    from rag_eval.systems.base import RAGSystemBase
    from rag_eval.systems.config import EvalConfig

logger = logging.getLogger(__name__)


def get_default_metrics() -> list[Metric]:
    """Get the default set of RAGAS metrics.

    Returns:
        List of RAGAS metric instances.
    """
    return [
        Faithfulness(),
        LLMContextRecall(),
        FactualCorrectness(),
        SemanticSimilarity(),
    ]


def _create_ragas_llm() -> LangchainLLMWrapper:
    """Create a wrapped LLM for RAGAS evaluation."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return LangchainLLMWrapper(llm)


def _create_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    """Create wrapped embeddings for RAGAS evaluation."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return LangchainEmbeddingsWrapper(embeddings)


class RAGEvaluator:
    """Evaluate RAG systems using RAGAS metrics.

    The evaluator runs queries through a RAG system and evaluates the
    responses using RAGAS metrics like Faithfulness, Context Recall,
    Factual Correctness, and Semantic Similarity.

    Supports:
    - Parallel batch processing for faster evaluation
    - Checkpointing to resume interrupted runs

    Example:
        ```python
        from rag_eval.evaluator import RAGEvaluator
        from rag_eval.systems.config import EvalConfig

        evaluator = RAGEvaluator()
        results = evaluator.evaluate(
            rag_system, corpus, eval_dataset,
            eval_config=EvalConfig(batch_size=40, max_workers=15),
        )

        print(f"Faithfulness: {results.scores['faithfulness']:.2f}")
        ```
    """

    def __init__(self, metrics: list[Metric] | None = None) -> None:
        """Initialize the evaluator with RAGAS metrics.

        Args:
            metrics: List of RAGAS metrics to use. If None, uses default metrics.
        """
        self.metrics = metrics or get_default_metrics()

    def evaluate(
        self,
        rag_system: RAGSystemBase,
        corpus: Corpus,
        eval_dataset: EvalDataset,
        show_progress: bool = True,
        checkpoint_dir: Path | None = None,
        config_sig: str | None = None,
        eval_config: EvalConfig | None = None,
        recreate: bool = False,
    ) -> EvaluationResult:
        """Evaluate a RAG system on an evaluation dataset.

        Args:
            rag_system: The RAG system to evaluate.
            corpus: The document corpus (for reference evidence texts).
            eval_dataset: The evaluation dataset with questions and answers.
            show_progress: Whether to show a progress bar.
            checkpoint_dir: Directory for checkpoints. If None, no checkpointing.
            config_sig: Config signature for checkpoint identification.
            eval_config: Evaluation config with batch_size, max_workers, etc.
            recreate: If True, ignore existing checkpoint and start fresh.

        Returns:
            EvaluationResult with scores and detailed results.

        Raises:
            ValueError: If no valid samples are generated.
        """
        from rag_eval.evaluator.results import AnswerRecord, EvaluationResult
        from rag_eval.systems.config import EvalConfig

        # Use defaults from EvalConfig if not provided
        if eval_config is None:
            eval_config = EvalConfig()

        batch_size = eval_config.batch_size
        max_workers = eval_config.max_workers
        checkpoint_interval = eval_config.checkpoint_interval

        # Delete existing checkpoint if recreate=True
        if recreate and checkpoint_dir and config_sig:
            self._delete_checkpoint(checkpoint_dir, config_sig)
            logger.info("Recreate mode: deleted existing checkpoint (if any)")

        # Load checkpoint if exists (will be empty if recreate deleted it)
        checkpoint = self._load_checkpoint(checkpoint_dir, config_sig)
        processed_ids: set[str] = set(checkpoint.get("processed_ids", []))
        samples: list[SingleTurnSample] = []
        answer_records: list[AnswerRecord] = []

        # Restore from checkpoint if available
        if checkpoint.get("samples"):
            for s in checkpoint["samples"]:
                samples.append(SingleTurnSample(**s))
        if checkpoint.get("answer_records"):
            for r in checkpoint["answer_records"]:
                answer_records.append(AnswerRecord.from_dict(r))

        # Filter out already processed questions
        all_questions = list(eval_dataset)
        remaining = [q for q in all_questions if q.question_id not in processed_ids]

        if processed_ids:
            if remaining:
                logger.info(
                    f"Resuming from checkpoint: {len(processed_ids)} already processed, "
                    f"{len(remaining)} remaining"
                )
            else:
                logger.info(
                    f"All {len(processed_ids)} questions already queried (from checkpoint). "
                    f"Skipping to RAGAS evaluation..."
                )
        else:
            logger.info(f"Generating RAG responses for {len(remaining)} questions...")

        # Process in batches with parallel execution
        total_batches = (len(remaining) + batch_size - 1) // batch_size

        with tqdm(total=len(remaining), desc="Querying RAG", disable=not show_progress) as pbar:
            for batch_idx in range(0, len(remaining), batch_size):
                batch = remaining[batch_idx : batch_idx + batch_size]

                # Process batch in parallel
                batch_results = self._process_batch(
                    batch, rag_system, corpus, max_workers
                )

                # Collect results
                for question_id, sample, record in batch_results:
                    if sample is not None:
                        samples.append(sample)
                        answer_records.append(record)
                        processed_ids.add(question_id)

                pbar.update(len(batch))

                # Save checkpoint after each batch
                if checkpoint_dir and config_sig:
                    self._save_checkpoint(
                        checkpoint_dir,
                        config_sig,
                        processed_ids,
                        samples,
                        answer_records,
                    )

        if not samples:
            raise ValueError("No valid samples generated for evaluation")

        # Run RAGAS evaluation in batches with checkpointing
        ragas_batch_size = eval_config.ragas_batch_size

        # Restore evaluation progress from checkpoint
        evaluated_indices: set[int] = set(checkpoint.get("evaluated_indices", []))
        per_sample_scores: list[dict[str, float]] = checkpoint.get("per_sample_scores", [])

        # Determine which samples still need evaluation
        unevaluated_indices = [i for i in range(len(samples)) if i not in evaluated_indices]

        if evaluated_indices:
            if unevaluated_indices:
                logger.info(
                    f"Resuming RAGAS evaluation: {len(evaluated_indices)} already evaluated, "
                    f"{len(unevaluated_indices)} remaining"
                )
            else:
                logger.info(
                    f"All {len(evaluated_indices)} samples already evaluated (from checkpoint). "
                    f"Computing final scores..."
                )
        else:
            logger.info(f"Evaluating {len(samples)} samples with RAGAS...")

        # Create RAGAS LLM and embeddings wrappers (reuse for all batches)
        ragas_llm = _create_ragas_llm()
        ragas_embeddings = _create_ragas_embeddings()

        # Configure RAGAS parallelization
        run_config = RunConfig(
            max_workers=eval_config.ragas_max_workers,
            timeout=eval_config.ragas_timeout,
            max_retries=eval_config.ragas_max_retries,
        )
        logger.info(
            f"RAGAS config: batch_size={ragas_batch_size}, "
            f"max_workers={eval_config.ragas_max_workers}, "
            f"timeout={eval_config.ragas_timeout}s"
        )

        # Process unevaluated samples in batches
        total_batches = (len(unevaluated_indices) + ragas_batch_size - 1) // ragas_batch_size if unevaluated_indices else 0

        with tqdm(total=len(unevaluated_indices), desc="Evaluating", disable=not show_progress) as pbar:
            for batch_start in range(0, len(unevaluated_indices), ragas_batch_size):
                batch_indices = unevaluated_indices[batch_start : batch_start + ragas_batch_size]
                batch_samples = [samples[i] for i in batch_indices]

                # Evaluate this batch
                ragas_dataset = EvaluationDataset(samples=batch_samples)
                batch_result = evaluate(
                    dataset=ragas_dataset,
                    metrics=self.metrics,
                    llm=ragas_llm,
                    embeddings=ragas_embeddings,
                    run_config=run_config,
                )

                # Extract per-sample scores from batch result
                batch_scores = self._extract_per_sample_scores(batch_result, len(batch_samples))

                # Ensure per_sample_scores list is large enough
                while len(per_sample_scores) < len(samples):
                    per_sample_scores.append({})

                # Store scores at correct indices
                for idx, sample_idx in enumerate(batch_indices):
                    per_sample_scores[sample_idx] = batch_scores[idx]
                    evaluated_indices.add(sample_idx)

                pbar.update(len(batch_indices))

                # Save checkpoint after each batch
                if checkpoint_dir and config_sig:
                    self._save_checkpoint(
                        checkpoint_dir,
                        config_sig,
                        processed_ids,
                        samples,
                        answer_records,
                        evaluated_indices,
                        per_sample_scores,
                    )

        # Aggregate per-sample scores into final scores
        scores = self._aggregate_scores(per_sample_scores)

        # Delete checkpoint on success
        if checkpoint_dir and config_sig:
            self._delete_checkpoint(checkpoint_dir, config_sig)

        return EvaluationResult(
            scores=scores,
            num_samples=len(samples),
            answer_records=answer_records,
            ragas_result=None,  # No single result when batched
        )

    def _process_question(
        self,
        question: EvalQuestion,
        rag_system: RAGSystemBase,
        corpus: Corpus,
    ) -> tuple[str, SingleTurnSample | None, AnswerRecord | None]:
        """Process a single question through the RAG system.

        Args:
            question: The question to process.
            rag_system: The RAG system to query.
            corpus: The document corpus.

        Returns:
            Tuple of (question_id, sample, answer_record) or (question_id, None, None) on error.
        """
        from rag_eval.evaluator.results import AnswerRecord

        try:
            # Query the RAG system
            response = rag_system.query(question.question)

            # Build RAGAS sample
            sample = SingleTurnSample(
                user_input=question.question,
                response=response.response,
                retrieved_contexts=response.retrieved_evidence_texts,
                reference=question.answer,
            )

            # Build answer record for detailed results
            record = AnswerRecord(
                question_id=question.question_id,
                question=question.question,
                question_type=question.question_type,
                evidence_count=question.evidence_count,
                model_response=response.response,
                retrieved_evidence=response.retrieved_evidence,
                retrieved_evidence_texts=response.retrieved_evidence_texts,
                ground_truth_answer=question.answer,
                required_evidence=question.required_evidence,
                required_evidence_texts=corpus.get_contents_by_ids(
                    question.required_evidence
                ),
            )

            return (question.question_id, sample, record)

        except Exception as e:
            logger.warning(f"Error processing question {question.question_id}: {e}")
            return (question.question_id, None, None)

    def _process_batch(
        self,
        batch: list[EvalQuestion],
        rag_system: RAGSystemBase,
        corpus: Corpus,
        max_workers: int,
    ) -> list[tuple[str, SingleTurnSample | None, AnswerRecord | None]]:
        """Process a batch of questions in parallel.

        Args:
            batch: List of questions to process.
            rag_system: The RAG system to query.
            corpus: The document corpus.
            max_workers: Maximum number of concurrent workers.

        Returns:
            List of (question_id, sample, answer_record) tuples.
        """
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all questions
            futures = {
                executor.submit(self._process_question, q, rag_system, corpus): q
                for q in batch
            }

            # Collect results as they complete
            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        return results

    def _save_checkpoint(
        self,
        checkpoint_dir: Path,
        config_sig: str,
        processed_ids: set[str],
        samples: list[SingleTurnSample],
        answer_records: list[AnswerRecord],
        evaluated_indices: set[int] | None = None,
        per_sample_scores: list[dict[str, float]] | None = None,
    ) -> None:
        """Save evaluation progress to a checkpoint file.

        Args:
            checkpoint_dir: Directory for checkpoints.
            config_sig: Config signature for identification.
            processed_ids: Set of processed question IDs (RAG querying).
            samples: List of RAGAS samples.
            answer_records: List of answer records.
            evaluated_indices: Set of sample indices that have been RAGAS evaluated.
            per_sample_scores: Per-sample RAGAS scores.
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_dir / f"{config_sig}.json"

        # Convert samples to serializable format
        samples_data = []
        for s in samples:
            samples_data.append({
                "user_input": s.user_input,
                "response": s.response,
                "retrieved_contexts": s.retrieved_contexts,
                "reference": s.reference,
            })

        # Convert answer records to serializable format
        records_data = [r.to_dict() for r in answer_records]

        checkpoint_data = {
            "config_sig": config_sig,
            "processed_ids": list(processed_ids),
            "samples": samples_data,
            "answer_records": records_data,
            "evaluated_indices": list(evaluated_indices) if evaluated_indices else [],
            "per_sample_scores": per_sample_scores or [],
            "timestamp": datetime.now().isoformat(),
        }

        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f)

        eval_status = f", {len(evaluated_indices or [])} evaluated" if evaluated_indices else ""
        logger.debug(f"Saved checkpoint: {len(processed_ids)} questions queried{eval_status}")

    def _load_checkpoint(
        self, checkpoint_dir: Path | None, config_sig: str | None
    ) -> dict:
        """Load checkpoint if it exists.

        Args:
            checkpoint_dir: Directory for checkpoints.
            config_sig: Config signature for identification.

        Returns:
            Checkpoint data dict, or empty dict if no checkpoint.
        """
        if not checkpoint_dir or not config_sig:
            return {}

        checkpoint_file = Path(checkpoint_dir) / f"{config_sig}.json"

        if not checkpoint_file.exists():
            return {}

        try:
            with open(checkpoint_file) as f:
                data = json.load(f)
            logger.info(f"Loaded checkpoint from {checkpoint_file}")
            return data
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return {}

    def _delete_checkpoint(self, checkpoint_dir: Path, config_sig: str) -> None:
        """Delete checkpoint file after successful completion.

        Args:
            checkpoint_dir: Directory for checkpoints.
            config_sig: Config signature for identification.
        """
        checkpoint_file = Path(checkpoint_dir) / f"{config_sig}.json"

        if checkpoint_file.exists():
            checkpoint_file.unlink()
            logger.debug(f"Deleted checkpoint: {checkpoint_file}")

    def _extract_scores(self, ragas_result) -> dict[str, float]:
        """Extract aggregate scores from RAGAS result.

        Args:
            ragas_result: RAGAS evaluation result.

        Returns:
            Dictionary mapping metric names to scores.
        """
        scores: dict[str, float] = {}

        # RAGAS EvaluationResult can be accessed like a dict
        # Try direct dict-like access first
        try:
            for key in ragas_result:
                value = ragas_result[key]
                if isinstance(value, (int, float)):
                    scores[key] = float(value)
        except (TypeError, KeyError):
            pass

        # Also check the result dataframe for per-sample scores
        # We average them to get aggregate scores
        if hasattr(ragas_result, "to_pandas"):
            df = ragas_result.to_pandas()
            for col in df.columns:
                if col not in ["user_input", "response", "retrieved_contexts", "reference"]:
                    try:
                        scores[col] = float(df[col].mean())
                    except (ValueError, TypeError):
                        pass

        return scores

    def _extract_per_sample_scores(
        self, ragas_result, num_samples: int
    ) -> list[dict[str, float]]:
        """Extract per-sample scores from RAGAS batch result.

        Args:
            ragas_result: RAGAS evaluation result for a batch.
            num_samples: Number of samples in the batch.

        Returns:
            List of dicts, each mapping metric names to scores for one sample.
        """
        per_sample: list[dict[str, float]] = [{} for _ in range(num_samples)]

        if hasattr(ragas_result, "to_pandas"):
            df = ragas_result.to_pandas()
            metric_cols = [
                col for col in df.columns
                if col not in ["user_input", "response", "retrieved_contexts", "reference"]
            ]

            for idx in range(min(len(df), num_samples)):
                for col in metric_cols:
                    try:
                        value = df.iloc[idx][col]
                        if value is not None and not (isinstance(value, float) and value != value):  # Check for NaN
                            per_sample[idx][col] = float(value)
                    except (ValueError, TypeError, KeyError):
                        pass

        return per_sample

    def _aggregate_scores(
        self, per_sample_scores: list[dict[str, float]]
    ) -> dict[str, float]:
        """Aggregate per-sample scores into final average scores.

        Args:
            per_sample_scores: List of per-sample score dicts.

        Returns:
            Dictionary mapping metric names to averaged scores.
        """
        if not per_sample_scores:
            return {}

        # Collect all metric names
        all_metrics: set[str] = set()
        for scores in per_sample_scores:
            all_metrics.update(scores.keys())

        # Average each metric
        aggregated: dict[str, float] = {}
        for metric in all_metrics:
            values = [s[metric] for s in per_sample_scores if metric in s]
            if values:
                aggregated[metric] = sum(values) / len(values)

        return aggregated
