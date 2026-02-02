"""Result classes for RAG evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class AnswerRecord:
    """Record of a single RAG query and its results.

    Attributes:
        question_id: Unique identifier for the question.
        question: The question text.
        question_type: Type of question (factoid, multi_hop, etc.).
        evidence_count: Number of evidence documents required.
        model_response: The RAG system's response.
        retrieved_evidence: List of retrieved document IDs.
        retrieved_evidence_texts: List of retrieved document contents.
        ground_truth_answer: The expected answer.
        required_evidence: List of required document IDs.
        required_evidence_texts: List of required document contents.
    """

    question_id: str
    question: str
    question_type: str
    evidence_count: int
    model_response: str
    retrieved_evidence: list[str]
    retrieved_evidence_texts: list[str]
    ground_truth_answer: str
    required_evidence: list[str]
    required_evidence_texts: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "question_id": self.question_id,
            "question": self.question,
            "question_type": self.question_type,
            "evidence_count": self.evidence_count,
            "model": {
                "response": self.model_response,
                "retrieved_evidence": self.retrieved_evidence,
                "retrieved_evidence_texts": self.retrieved_evidence_texts,
            },
            "ground_truth": {
                "answer": self.ground_truth_answer,
                "required_evidence": self.required_evidence,
                "required_evidence_texts": self.required_evidence_texts,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnswerRecord":
        """Create from dictionary (reverse of to_dict)."""
        return cls(
            question_id=data["question_id"],
            question=data["question"],
            question_type=data["question_type"],
            evidence_count=data["evidence_count"],
            model_response=data["model"]["response"],
            retrieved_evidence=data["model"]["retrieved_evidence"],
            retrieved_evidence_texts=data["model"]["retrieved_evidence_texts"],
            ground_truth_answer=data["ground_truth"]["answer"],
            required_evidence=data["ground_truth"]["required_evidence"],
            required_evidence_texts=data["ground_truth"]["required_evidence_texts"],
        )


@dataclass
class EvaluationResult:
    """Result of evaluating a RAG system.

    Attributes:
        scores: Dictionary mapping metric names to scores.
        num_samples: Number of samples evaluated.
        answer_records: Detailed records of each answer.
        ragas_result: Raw RAGAS evaluation result.
    """

    scores: dict[str, float]
    num_samples: int
    answer_records: list[AnswerRecord] = field(default_factory=list)
    ragas_result: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "scores": self.scores,
            "num_samples": self.num_samples,
        }

    def save_answers(self, path: str | Path) -> None:
        """Save answer records to a JSONL file.

        Args:
            path: Path to save the file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for record in self.answer_records:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


@dataclass
class ExperimentResult:
    """Result of a single experiment run (RAG system + config).

    Attributes:
        config_id: Identifier for this configuration.
        config_sig: Unique signature/hash of the config.
        config: The RAG configuration used.
        scores: Evaluation scores.
        num_samples: Number of samples evaluated.
        answers_file: Path to the answers JSONL file.
        timestamp: When the experiment was run.
    """

    config_id: int | str
    config_sig: str
    config: dict[str, Any]
    scores: dict[str, float]
    num_samples: int = 0
    answers_file: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "config_id": self.config_id,
            "config_sig": self.config_sig,
            "config": self.config,
            "scores": self.scores,
            "num_samples": self.num_samples,
            "answers_file": self.answers_file,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentResult:
        """Create from dictionary."""
        return cls(
            config_id=data["config_id"],
            config_sig=data["config_sig"],
            config=data["config"],
            scores=data["scores"],
            num_samples=data.get("num_samples", 0),
            answers_file=data.get("answers_file"),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
        )


@dataclass
class ExperimentSummary:
    """Summary of multiple experiment runs.

    Attributes:
        results: List of individual experiment results.
        best_config_id: ID of the best performing config.
        best_metric: Metric used to determine best config.
        corpus_name: Name of the corpus used.
        dataset_name: Name of the evaluation dataset.
    """

    results: list[ExperimentResult] = field(default_factory=list)
    best_config_id: int | str | None = None
    best_metric: str | None = None
    corpus_name: str | None = None
    dataset_name: str | None = None

    def add_result(self, result: ExperimentResult) -> None:
        """Add an experiment result."""
        self.results.append(result)

    def find_best(self, metric: str, higher_is_better: bool = True) -> ExperimentResult | None:
        """Find the best performing configuration.

        Args:
            metric: The metric to optimize.
            higher_is_better: Whether higher values are better.

        Returns:
            The best ExperimentResult, or None if no results.
        """
        if not self.results:
            return None

        valid_results = [r for r in self.results if metric in r.scores]
        if not valid_results:
            return None

        if higher_is_better:
            best = max(valid_results, key=lambda r: r.scores[metric])
        else:
            best = min(valid_results, key=lambda r: r.scores[metric])

        self.best_config_id = best.config_id
        self.best_metric = metric
        return best

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "results": [r.to_dict() for r in self.results],
            "best_config_id": self.best_config_id,
            "best_metric": self.best_metric,
            "corpus_name": self.corpus_name,
            "dataset_name": self.dataset_name,
            "num_experiments": len(self.results),
        }

    def save(self, path: str | Path) -> None:
        """Save summary to JSON file.

        Args:
            path: Path to save the file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> ExperimentSummary:
        """Load summary from JSON file.

        Args:
            path: Path to the file.

        Returns:
            Loaded ExperimentSummary.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        summary = cls(
            results=[ExperimentResult.from_dict(r) for r in data.get("results", [])],
            best_config_id=data.get("best_config_id"),
            best_metric=data.get("best_metric"),
            corpus_name=data.get("corpus_name"),
            dataset_name=data.get("dataset_name"),
        )
        return summary
