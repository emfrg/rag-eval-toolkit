# rag_evaluator/evaluator.py
import json
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from ragas import evaluate, SingleTurnSample, EvaluationDataset
from ragas.evaluation import EvaluationResult
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    FactualCorrectness,
    SemanticSimilarity,
    ContextEntityRecall,
)

try:
    # Preferred when running as part of the rag_evals package
    from ..rag_system import RAGSystem, RAGDataset
except ImportError:  # pragma: no cover - fallback for flat execution
    from rag_system import RAGSystem, RAGDataset


class RAGEvaluator:
    """Evaluate RAG systems using RAGAS."""

    def __init__(self, metrics=None):
        """Initialize with RAGAS metrics."""
        if metrics is None:
            self.metrics = [
                Faithfulness(),
                # ResponseRelevancy(),  # BUG: this creates indexerror
                # LLMContextPrecisionWithReference(),
                LLMContextRecall(),
                FactualCorrectness(),
                FactualCorrectness(mode="precision"),
                SemanticSimilarity(),
                # ContextEntityRecall(),
            ]
        else:
            self.metrics = metrics

    def evaluate(self, rag_system: RAGSystem, dataset: RAGDataset) -> EvaluationResult:
        """Run RAGAS evaluation on a RAG system."""

        questions = dataset.load_questions()
        samples = []

        print(f"Generating RAG responses for {len(questions)} questions...")
        for q in tqdm(questions):
            try:
                answer, retrieved_docs = rag_system.query(q["question"])

                sample = SingleTurnSample(
                    user_input=q["question"],
                    response=answer,
                    retrieved_contexts=[doc.page_content for doc in retrieved_docs],
                    reference=q.get("answer", ""),
                )
                samples.append(sample)
            except Exception as e:
                print(
                    f"\nError processing question {q.get('question_id', 'unknown')}: {e}"
                )
                continue

        if not samples:
            raise ValueError("No valid samples generated for evaluation")

        print(f"Evaluating {len(samples)} samples with RAGAS...")

        eval_dataset = EvaluationDataset(samples=samples)
        result = evaluate(dataset=eval_dataset, metrics=self.metrics)

        return result
