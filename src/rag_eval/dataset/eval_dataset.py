"""Evaluation dataset class for RAG evaluation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal

logger = logging.getLogger(__name__)


QuestionType = Literal[
    "factoid",
    "comparison_query",
    "inference_query",
    "temporal_query",
    "multi_hop",
    "single_hop",
]


@dataclass
class EvalQuestion:
    """A single evaluation question.

    Attributes:
        question_id: Unique identifier for the question.
        question: The question text.
        answer: The ground truth answer.
        question_type: Type of question (factoid, comparison, inference, etc.).
        required_evidence: List of doc_ids needed to answer the question.
        evidence_count: Number of documents required.

    Example:
        ```python
        question = EvalQuestion(
            question_id="q_001",
            question="What is the capital of France?",
            answer="Paris",
            question_type="factoid",
            required_evidence=["doc_123"],
            evidence_count=1,
        )
        ```
    """

    question_id: str
    question: str
    answer: str
    question_type: QuestionType = "factoid"
    required_evidence: list[str] = field(default_factory=list)
    evidence_count: int = 0

    def __post_init__(self) -> None:
        if self.evidence_count == 0:
            self.evidence_count = len(self.required_evidence)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "question_id": self.question_id,
            "question": self.question,
            "answer": self.answer,
            "question_type": self.question_type,
            "required_evidence": self.required_evidence,
            "evidence_count": self.evidence_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> EvalQuestion:
        """Create an EvalQuestion from a dictionary."""
        return cls(
            question_id=data["question_id"],
            question=data["question"],
            answer=data["answer"],
            question_type=data.get("question_type", "factoid"),
            required_evidence=data.get("required_evidence", []),
            evidence_count=data.get("evidence_count", 0),
        )


@dataclass
class EvalDataset:
    """A dataset of evaluation questions.

    The dataset can be loaded from a JSONL file or created programmatically.

    Attributes:
        questions: List of EvalQuestion objects.
        name: Optional name for the dataset.

    Example:
        ```python
        # Load from file
        dataset = EvalDataset.from_jsonl("./data/eval_questions.jsonl")

        # Get a sample for quick testing
        sample = dataset.sample(10)

        # Filter by question type
        multi_hop = dataset.filter_by_type("multi_hop")

        # Iterate
        for q in dataset:
            print(q.question)
        ```
    """

    questions: list[EvalQuestion] = field(default_factory=list)
    name: str = "eval_dataset"

    def __post_init__(self) -> None:
        self._id_to_question: dict[str, EvalQuestion] = {}
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuild the question ID index."""
        self._id_to_question = {q.question_id: q for q in self.questions}

    def __len__(self) -> int:
        return len(self.questions)

    def __iter__(self) -> Iterator[EvalQuestion]:
        return iter(self.questions)

    def __getitem__(self, index: int) -> EvalQuestion:
        return self.questions[index]

    def add(self, question: EvalQuestion) -> None:
        """Add a question to the dataset.

        Args:
            question: The question to add.

        Raises:
            ValueError: If a question with the same ID already exists.
        """
        if question.question_id in self._id_to_question:
            raise ValueError(f"Question with ID {question.question_id!r} already exists")
        self.questions.append(question)
        self._id_to_question[question.question_id] = question

    def get(self, question_id: str) -> EvalQuestion | None:
        """Get a question by ID.

        Args:
            question_id: The question ID to look up.

        Returns:
            The question if found, None otherwise.
        """
        return self._id_to_question.get(question_id)

    def filter_by_type(self, question_type: QuestionType) -> EvalDataset:
        """Get questions of a specific type.

        Args:
            question_type: The type to filter by.

        Returns:
            New EvalDataset with filtered questions.
        """
        filtered = [q for q in self.questions if q.question_type == question_type]
        return EvalDataset(questions=filtered, name=f"{self.name}_{question_type}")

    def filter_by_evidence_count(self, min_count: int = 1, max_count: int | None = None) -> EvalDataset:
        """Filter questions by evidence count.

        Args:
            min_count: Minimum number of required evidence documents.
            max_count: Maximum number of required evidence documents (None for no limit).

        Returns:
            New EvalDataset with filtered questions.
        """
        filtered = []
        for q in self.questions:
            if q.evidence_count >= min_count:
                if max_count is None or q.evidence_count <= max_count:
                    filtered.append(q)
        return EvalDataset(questions=filtered, name=f"{self.name}_evidence_{min_count}_{max_count}")

    @classmethod
    def from_jsonl(cls, path: str | Path, name: str | None = None) -> EvalDataset:
        """Load an evaluation dataset from a JSONL file.

        Each line should be a JSON object with question data.

        Args:
            path: Path to the JSONL file.
            name: Optional name for the dataset.

        Returns:
            EvalDataset loaded from the file.

        Raises:
            FileNotFoundError: If the file doesn't exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        if name is None:
            name = path.stem

        questions = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    questions.append(EvalQuestion.from_dict(data))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                except KeyError as e:
                    logger.warning(f"Skipping line {line_num} missing required field: {e}")

        logger.info(f"Loaded {len(questions)} questions from {path}")
        return cls(questions=questions, name=name)

    def to_jsonl(self, path: str | Path) -> None:
        """Save the dataset to a JSONL file.

        Args:
            path: Path to save the file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for q in self.questions:
                f.write(json.dumps(q.to_dict(), ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(self.questions)} questions to {path}")

    def sample(self, n: int, seed: int | None = None) -> EvalDataset:
        """Get a random sample of questions.

        Args:
            n: Number of questions to sample.
            seed: Random seed for reproducibility.

        Returns:
            New EvalDataset with sampled questions.
        """
        import random

        if seed is not None:
            random.seed(seed)

        sampled = random.sample(self.questions, min(n, len(self.questions)))
        return EvalDataset(questions=sampled, name=f"{self.name}_sample_{n}")

    def get_question_types_distribution(self) -> dict[str, int]:
        """Get distribution of question types.

        Returns:
            Dict mapping question type to count.
        """
        distribution: dict[str, int] = {}
        for q in self.questions:
            distribution[q.question_type] = distribution.get(q.question_type, 0) + 1
        return distribution

    def get_evidence_count_distribution(self) -> dict[int, int]:
        """Get distribution of evidence counts.

        Returns:
            Dict mapping evidence count to number of questions.
        """
        distribution: dict[int, int] = {}
        for q in self.questions:
            distribution[q.evidence_count] = distribution.get(q.evidence_count, 0) + 1
        return distribution
