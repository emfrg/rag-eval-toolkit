# rag_system/dataset.py
from pathlib import Path
from typing import List, Dict, Optional
import json
from dataclasses import dataclass


@dataclass
class RAGDataset:
    """Container for RAG dataset paths."""

    name: str
    corpus_path: Path
    questions_path: Optional[Path] = None

    @classmethod
    def from_dataset_dir(
        cls, dataset_dir: str, dataset_name: str = None, questions_file: str = None
    ):
        """Load dataset from eval_dataset_builder output directory."""
        base_path = Path(dataset_dir)

        if dataset_name is None:
            dataset_name = base_path.name

        # Allow specifying which questions file to use
        if questions_file:
            questions_path = base_path / questions_file
        else:
            # Default
            questions_path = (
                base_path / "filtered_eval_questions.jsonl"
                if (base_path / "filtered_eval_questions.jsonl").exists()
                else base_path / "sampled_eval_questions.jsonl"
            )

        return cls(
            name=dataset_name,
            corpus_path=base_path / "corpus.jsonl",
            questions_path=questions_path,
        )

    def load_corpus(self) -> List[Dict]:
        """Load corpus documents."""
        docs = []
        with open(self.corpus_path, "r") as f:
            for line in f:
                docs.append(json.loads(line))
        return docs

    def load_questions(self) -> List[Dict]:
        """Load evaluation questions."""
        if not self.questions_path or not self.questions_path.exists():
            raise ValueError(f"Questions file not found: {self.questions_path}")

        questions = []
        with open(self.questions_path, "r") as f:
            for line in f:
                questions.append(json.loads(line))
        return questions
