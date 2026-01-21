"""QA pair critique and quality scoring.

This module provides functionality to evaluate the quality of generated
QA pairs using LLM-based critique. Quality dimensions include:

- Groundedness: Can the question be answered from the context?
- Relevance: Is the question practically relevant?
- Standalone: Is the question understandable without additional context?
- Complexity: How much reasoning is required?

Example:
    ```python
    from rag_eval.dataset.critique import QACritique, score_eval_dataset

    # Score individual QA pairs
    critique = QACritique()
    scores = critique.evaluate(question="...", answer="...", context="...")

    # Score an entire dataset
    scored_dataset = score_eval_dataset(eval_dataset, corpus, threshold=3)
    ```
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain_openai import ChatOpenAI
from tqdm import tqdm

if TYPE_CHECKING:
    from rag_eval.dataset.corpus import Corpus
    from rag_eval.dataset.eval_dataset import EvalDataset, EvalQuestion

logger = logging.getLogger(__name__)


GROUNDEDNESS_PROMPT = """You will be given a context and a question.
Your task is to rate how well the question can be answered from the given context.

Scale (1-5):
1 = Not answerable at all from context
2 = Barely answerable, context is very insufficient
3 = Partially answerable, some information missing
4 = Mostly answerable, minor details might be missing
5 = Completely answerable from context alone

Context:
{context}

Question: {question}

Provide your rating as a single number (1-5):
Rating: """

RELEVANCE_PROMPT = """You will be given a question.
Your task is to rate how relevant and useful this question would be for practitioners.

Scale (1-5):
1 = Not relevant at all, trivial or nonsensical
2 = Barely relevant, very narrow interest
3 = Somewhat relevant, moderate practical value
4 = Quite relevant, good practical value
5 = Highly relevant, excellent practical value

Question: {question}

Provide your rating as a single number (1-5):
Rating: """

STANDALONE_PROMPT = """You will be given a question.
Your task is to rate how understandable the question is without any additional context.

Scale (1-5):
1 = Completely unclear, requires external context
2 = Mostly unclear, references unknown entities
3 = Somewhat clear, but some references are vague
4 = Mostly clear, self-contained with minor ambiguity
5 = Perfectly clear and self-contained

Question: {question}

Provide your rating as a single number (1-5):
Rating: """

COMPLEXITY_PROMPT = """You will be given a question and the number of evidence pieces required to answer it.
Your task is to rate the reasoning complexity required to answer this question.

Scale (1-5):
1 = Simple lookup (single fact retrieval)
2 = Basic connection (two directly related facts)
3 = Multi-step reasoning (connecting 2-3 facts with simple inference)
4 = Complex synthesis (multiple sources requiring inference)
5 = Advanced reasoning (temporal, causal, or comparative analysis across sources)

Question: {question}
Evidence pieces required: {evidence_count}

Provide your rating as a single number (1-5):
Rating: """


@dataclass
class QAScores:
    """Quality scores for a QA pair.

    Attributes:
        groundedness: How well answerable from context (1-5).
        relevance: Practical relevance (1-5).
        standalone: Clarity without context (1-5).
        complexity: Reasoning complexity (1-5).
    """

    groundedness: int = 0
    relevance: int = 0
    standalone: int = 0
    complexity: int = 0

    def passes_threshold(
        self,
        groundedness_min: int = 3,
        relevance_min: int = 3,
        standalone_min: int = 3,
        complexity_min: int = 2,
    ) -> bool:
        """Check if all scores meet minimum thresholds.

        Args:
            groundedness_min: Minimum groundedness score.
            relevance_min: Minimum relevance score.
            standalone_min: Minimum standalone score.
            complexity_min: Minimum complexity score.

        Returns:
            True if all scores meet thresholds.
        """
        return (
            self.groundedness >= groundedness_min
            and self.relevance >= relevance_min
            and self.standalone >= standalone_min
            and self.complexity >= complexity_min
        )

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary."""
        return {
            "groundedness_score": self.groundedness,
            "relevance_score": self.relevance,
            "standalone_score": self.standalone,
            "complexity_score": self.complexity,
        }


class QACritique:
    """Evaluate QA pair quality using LLM-based critique.

    Attributes:
        model: The LLM model name.
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        """Initialize the critique evaluator.

        Args:
            model: OpenAI model name to use.
        """
        self.model = model
        self.llm = ChatOpenAI(model=model, temperature=0, max_tokens=50)
        logger.info(f"Initialized QACritique with model={model}")

    def evaluate(
        self,
        question: str,
        answer: str,
        context: str | None = None,
        evidence_count: int = 1,
    ) -> QAScores:
        """Evaluate a single QA pair on all quality dimensions.

        Args:
            question: The question text.
            answer: The answer text.
            context: The source context (for groundedness).
            evidence_count: Number of evidence pieces required.

        Returns:
            QAScores with all dimensions rated.
        """
        scores = QAScores()

        # Groundedness (requires context)
        if context:
            try:
                response = self.llm.invoke(
                    GROUNDEDNESS_PROMPT.format(context=context, question=question)
                ).content
                scores.groundedness = self._extract_score(response)
            except Exception as e:
                logger.warning(f"Groundedness evaluation failed: {e}")

        # Relevance
        try:
            response = self.llm.invoke(
                RELEVANCE_PROMPT.format(question=question)
            ).content
            scores.relevance = self._extract_score(response)
        except Exception as e:
            logger.warning(f"Relevance evaluation failed: {e}")

        # Standalone
        try:
            response = self.llm.invoke(
                STANDALONE_PROMPT.format(question=question)
            ).content
            scores.standalone = self._extract_score(response)
        except Exception as e:
            logger.warning(f"Standalone evaluation failed: {e}")

        # Complexity
        try:
            response = self.llm.invoke(
                COMPLEXITY_PROMPT.format(
                    question=question, evidence_count=evidence_count
                )
            ).content
            scores.complexity = self._extract_score(response)
        except Exception as e:
            logger.warning(f"Complexity evaluation failed: {e}")

        return scores

    def evaluate_question(
        self,
        question: EvalQuestion,
        corpus: Corpus | None = None,
    ) -> QAScores:
        """Evaluate an EvalQuestion object.

        Args:
            question: The EvalQuestion to evaluate.
            corpus: Optional corpus to look up evidence for groundedness.

        Returns:
            QAScores for the question.
        """
        # Get context from corpus if available
        context = None
        if corpus and question.required_evidence:
            contexts = corpus.get_contents_by_ids(question.required_evidence)
            context = "\n\n".join(c for c in contexts if c)

        return self.evaluate(
            question=question.question,
            answer=question.answer,
            context=context,
            evidence_count=question.evidence_count,
        )

    def _extract_score(self, response: str) -> int:
        """Extract numeric score from LLM response.

        Args:
            response: Raw LLM response text.

        Returns:
            Score (1-5), 0 if extraction failed.
        """
        try:
            # Try to find a number in the response
            # Look for "Rating: X" or just a digit
            match = re.search(r"Rating:\s*(\d)", response)
            if match:
                score = int(match.group(1))
            else:
                # Find any single digit
                match = re.search(r"\b([1-5])\b", response)
                if match:
                    score = int(match.group(1))
                else:
                    return 0

            # Clamp to valid range
            return max(1, min(5, score))

        except (ValueError, AttributeError):
            return 0


def score_eval_dataset(
    eval_dataset: EvalDataset,
    corpus: Corpus | None = None,
    model: str = "gpt-4o-mini",
    max_workers: int = 5,
    groundedness_min: int = 3,
    relevance_min: int = 3,
    standalone_min: int = 3,
    complexity_min: int = 2,
) -> tuple[EvalDataset, dict[str, QAScores]]:
    """Score and filter an evaluation dataset by quality.

    Args:
        eval_dataset: The dataset to score.
        corpus: Optional corpus for groundedness evaluation.
        model: LLM model to use for critique.
        max_workers: Number of parallel workers.
        groundedness_min: Minimum groundedness score to keep.
        relevance_min: Minimum relevance score to keep.
        standalone_min: Minimum standalone score to keep.
        complexity_min: Minimum complexity score to keep.

    Returns:
        Tuple of (filtered EvalDataset, dict of question_id -> scores).

    Example:
        ```python
        filtered_dataset, all_scores = score_eval_dataset(
            dataset,
            corpus,
            groundedness_min=4,
            standalone_min=4,
        )
        print(f"Kept {len(filtered_dataset)}/{len(dataset)} questions")
        ```
    """
    from rag_eval.dataset.eval_dataset import EvalDataset

    critique = QACritique(model=model)
    all_scores: dict[str, QAScores] = {}
    filtered_questions: list[EvalQuestion] = []

    def evaluate_single(question: EvalQuestion) -> tuple[str, QAScores]:
        scores = critique.evaluate_question(question, corpus)
        return question.question_id, scores

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(evaluate_single, q) for q in eval_dataset.questions
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Scoring questions",
        ):
            q_id, scores = future.result()
            all_scores[q_id] = scores

            # Check if passes threshold
            if scores.passes_threshold(
                groundedness_min=groundedness_min,
                relevance_min=relevance_min,
                standalone_min=standalone_min,
                complexity_min=complexity_min,
            ):
                # Find and add the question
                q = eval_dataset.get(q_id)
                if q:
                    filtered_questions.append(q)

    filtered_dataset = EvalDataset(
        questions=filtered_questions, name=f"{eval_dataset.name}_filtered"
    )

    logger.info(
        f"Filtered dataset: {len(filtered_dataset)}/{len(eval_dataset)} "
        f"questions passed thresholds"
    )

    return filtered_dataset, all_scores


def score_multi_hop_dataset(
    eval_dataset: EvalDataset,
    model: str = "gpt-4o-mini",
    max_workers: int = 5,
    complexity_min: int = 3,
    standalone_min: int = 4,
) -> tuple[EvalDataset, dict[str, QAScores]]:
    """Score a multi-hop dataset focusing on complexity and standalone quality.

    Optimized for multi-hop QA where we care about:
    - Complexity: Questions should require multi-step reasoning
    - Standalone: Questions should be clear without context

    Args:
        eval_dataset: The dataset to score.
        model: LLM model to use.
        max_workers: Number of parallel workers.
        complexity_min: Minimum complexity score.
        standalone_min: Minimum standalone score.

    Returns:
        Tuple of (filtered EvalDataset, dict of question_id -> scores).
    """
    from rag_eval.dataset.eval_dataset import EvalDataset

    critique = QACritique(model=model)
    all_scores: dict[str, QAScores] = {}
    filtered_questions: list[EvalQuestion] = []

    def evaluate_single(question: EvalQuestion) -> tuple[str, QAScores]:
        scores = QAScores()

        # Only evaluate complexity and standalone for multi-hop
        try:
            response = critique.llm.invoke(
                COMPLEXITY_PROMPT.format(
                    question=question.question,
                    evidence_count=question.evidence_count,
                )
            ).content
            scores.complexity = critique._extract_score(response)
        except Exception as e:
            logger.warning(f"Complexity evaluation failed: {e}")

        try:
            response = critique.llm.invoke(
                STANDALONE_PROMPT.format(question=question.question)
            ).content
            scores.standalone = critique._extract_score(response)
        except Exception as e:
            logger.warning(f"Standalone evaluation failed: {e}")

        return question.question_id, scores

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(evaluate_single, q) for q in eval_dataset.questions
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Scoring multi-hop questions",
        ):
            q_id, scores = future.result()
            all_scores[q_id] = scores

            if (
                scores.complexity >= complexity_min
                and scores.standalone >= standalone_min
            ):
                q = eval_dataset.get(q_id)
                if q:
                    filtered_questions.append(q)

    filtered_dataset = EvalDataset(
        questions=filtered_questions, name=f"{eval_dataset.name}_filtered"
    )

    logger.info(
        f"Filtered multi-hop dataset: {len(filtered_dataset)}/{len(eval_dataset)} "
        f"questions passed thresholds"
    )

    return filtered_dataset, all_scores
