# rag_evaluator/evaluator.py
import json
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from ragas import evaluate, SingleTurnSample, EvaluationDataset
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
    answer_similarity,
)

from rag_system import RAGSystem, RAGDataset


class RAGEvaluator:
    """Evaluate RAG systems using RAGAS."""

    def __init__(self, metrics=None):
        """Initialize with RAGAS metrics."""
        if metrics is None:
            self.metrics = [
                faithfulness,
                # answer_relevancy, # TODO: Add back in and fix
                # context_precision, # TODO: Add back in and fix
                # context_recall, # TODO: Add back in and fix
                answer_correctness,
                answer_similarity,
            ]
        else:
            self.metrics = metrics

    def evaluate(self, rag_system: RAGSystem, dataset: RAGDataset) -> Dict[str, Any]:
        """Run RAGAS evaluation on a RAG system."""

        questions = dataset.load_questions()
        samples = []

        print(f"Generating RAG responses for {len(questions)} questions...")
        for q in tqdm(questions):
            try:
                answer, retrieved_docs = rag_system.query(q["question"])

                # Create RAGAS sample for evaluation
                sample = SingleTurnSample(
                    user_input=q["question"],
                    response=answer,
                    retrieved_contexts=[doc.page_content for doc in retrieved_docs],
                    reference=q.get("answer", ""),  # Safe access with default
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

        # Run RAGAS evaluation
        eval_dataset = EvaluationDataset(samples=samples)
        result = evaluate(dataset=eval_dataset, metrics=self.metrics)

        return result
