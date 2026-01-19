"""Dataset module - Corpus, EvalDataset, and dataset building utilities."""

from rag_eval.dataset.corpus import Corpus, Document
from rag_eval.dataset.eval_dataset import EvalDataset, EvalQuestion
from rag_eval.dataset.builder import generate_eval_dataset, QAGenerator
from rag_eval.dataset.adapter import (
    adapt_structure,
    adapt_huggingface_dataset,
    adapt_from_directory,
    adapt_from_jsonl,
)
from rag_eval.dataset.critique import (
    QACritique,
    QAScores,
    score_eval_dataset,
    score_multi_hop_dataset,
)

__all__ = [
    # Core classes
    "Corpus",
    "Document",
    "EvalDataset",
    "EvalQuestion",
    # Builder
    "generate_eval_dataset",
    "QAGenerator",
    # Adapter
    "adapt_structure",
    "adapt_huggingface_dataset",
    "adapt_from_directory",
    "adapt_from_jsonl",
    # Critique
    "QACritique",
    "QAScores",
    "score_eval_dataset",
    "score_multi_hop_dataset",
]
