"""RAGAS metrics integration for RAG evaluation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, SingleTurnSample, evaluate
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
    from rag_eval.dataset.eval_dataset import EvalDataset
    from rag_eval.evaluator.results import EvaluationResult
    from rag_eval.systems.base import RAGSystemBase

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

    Example:
        ```python
        from rag_eval.evaluator import RAGEvaluator

        evaluator = RAGEvaluator()
        results = evaluator.evaluate(rag_system, corpus, eval_dataset)

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
    ) -> EvaluationResult:
        """Evaluate a RAG system on an evaluation dataset.

        Args:
            rag_system: The RAG system to evaluate.
            corpus: The document corpus (for reference evidence texts).
            eval_dataset: The evaluation dataset with questions and answers.
            show_progress: Whether to show a progress bar.

        Returns:
            EvaluationResult with scores and detailed results.

        Raises:
            ValueError: If no valid samples are generated.
        """
        from rag_eval.evaluator.results import AnswerRecord, EvaluationResult

        samples: list[SingleTurnSample] = []
        answer_records: list[AnswerRecord] = []

        logger.info(f"Generating RAG responses for {len(eval_dataset)} questions...")

        questions = list(eval_dataset)
        if show_progress:
            questions = tqdm(questions, desc="Querying RAG")

        for question in questions:
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
                samples.append(sample)

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
                answer_records.append(record)

            except Exception as e:
                logger.warning(
                    f"Error processing question {question.question_id}: {e}"
                )
                continue

        if not samples:
            raise ValueError("No valid samples generated for evaluation")

        # Run RAGAS evaluation with configured LLM and embeddings
        logger.info(f"Evaluating {len(samples)} samples with RAGAS...")
        ragas_dataset = EvaluationDataset(samples=samples)

        # Create RAGAS LLM and embeddings wrappers
        ragas_llm = _create_ragas_llm()
        ragas_embeddings = _create_ragas_embeddings()

        ragas_result = evaluate(
            dataset=ragas_dataset,
            metrics=self.metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )

        # Extract scores from the result
        scores = self._extract_scores(ragas_result)

        return EvaluationResult(
            scores=scores,
            num_samples=len(samples),
            answer_records=answer_records,
            ragas_result=ragas_result,
        )

    def _extract_scores(self, ragas_result) -> dict[str, float]:
        """Extract scores from RAGAS result.

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
