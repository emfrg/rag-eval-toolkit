"""Dataset module - Corpus and EvalDataset classes."""

from rag_eval.dataset.corpus import Corpus, Document
from rag_eval.dataset.eval_dataset import EvalDataset, EvalQuestion

__all__ = [
    "Corpus",
    "Document",
    "EvalDataset",
    "EvalQuestion",
]
