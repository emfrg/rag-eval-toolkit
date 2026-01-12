"""Dataset builder for generating evaluation datasets from documents.

This module provides functionality to generate QA pairs from a corpus
of documents using LLMs, then format them into an EvalDataset.

Flow:
    Corpus (documents) → generate_eval_dataset() → EvalDataset

Example:
    ```python
    from rag_eval.dataset import Corpus
    from rag_eval.dataset.builder import generate_eval_dataset

    corpus = Corpus.from_jsonl("./corpus.jsonl")
    eval_dataset = generate_eval_dataset(corpus, num_questions=50)
    eval_dataset.to_jsonl("./eval_questions.jsonl")
    ```
"""

from __future__ import annotations

import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from langchain_openai import ChatOpenAI
from tqdm import tqdm

from rag_eval.dataset.corpus import Document
from rag_eval.dataset.eval_dataset import EvalDataset, EvalQuestion

if TYPE_CHECKING:
    from rag_eval.dataset.corpus import Corpus

logger = logging.getLogger(__name__)


FACTOID_QA_PROMPT = """Given the following context, generate a factoid question and answer pair.
The question should:
- Be answerable from the context alone
- Be specific and unambiguous
- Not be too obvious or trivial

Context:
{context}

Generate a question-answer pair in the following format:
Question: [Your question here]
Answer: [Brief answer based on the context, max 2-3 sentences]
"""

MULTI_HOP_QA_PROMPT = """Given the following contexts from multiple sources, generate a question that requires reasoning across both contexts to answer.

Context 1:
{context1}

Context 2:
{context2}

Generate a multi-hop question that requires information from BOTH contexts to answer.
Format:
Question: [Your multi-hop question here]
Answer: [Answer that synthesizes information from both contexts]
"""


class QAGenerator:
    """Generate QA pairs from documents using an LLM.

    Attributes:
        model: The LLM model name to use.
        temperature: Sampling temperature for generation.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
    ) -> None:
        """Initialize the QA generator.

        Args:
            model: OpenAI model name to use.
            temperature: Sampling temperature (0-1).
        """
        self.model = model
        self.temperature = temperature
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=500,
        )
        logger.info(f"Initialized QAGenerator with model={model}")

    def generate_factoid(self, document: Document) -> EvalQuestion | None:
        """Generate a factoid QA pair from a single document.

        Args:
            document: The source document.

        Returns:
            Generated EvalQuestion or None if generation failed.
        """
        try:
            prompt = FACTOID_QA_PROMPT.format(context=document.content)
            response = self.llm.invoke(prompt).content

            # Parse response
            question, answer = self._parse_qa_response(response)

            if not question or not answer:
                return None

            # Skip if answer is too long (likely extracted too much context)
            if len(answer) > 500:
                logger.debug(f"Skipping: answer too long ({len(answer)} chars)")
                return None

            return EvalQuestion(
                question_id=f"gen_{document.doc_id}",
                question=question,
                answer=answer,
                question_type="factoid",
                required_evidence=[document.doc_id],
                evidence_count=1,
            )

        except Exception as e:
            logger.warning(f"Failed to generate QA for doc {document.doc_id}: {e}")
            return None

    def generate_multi_hop(
        self, doc1: Document, doc2: Document, question_id: str
    ) -> EvalQuestion | None:
        """Generate a multi-hop QA pair from two documents.

        Args:
            doc1: First source document.
            doc2: Second source document.
            question_id: ID to assign to the question.

        Returns:
            Generated EvalQuestion or None if generation failed.
        """
        try:
            prompt = MULTI_HOP_QA_PROMPT.format(
                context1=doc1.content,
                context2=doc2.content,
            )
            response = self.llm.invoke(prompt).content

            question, answer = self._parse_qa_response(response)

            if not question or not answer:
                return None

            return EvalQuestion(
                question_id=question_id,
                question=question,
                answer=answer,
                question_type="multi_hop",
                required_evidence=[doc1.doc_id, doc2.doc_id],
                evidence_count=2,
            )

        except Exception as e:
            logger.warning(f"Failed to generate multi-hop QA: {e}")
            return None

    def _parse_qa_response(self, response: str) -> tuple[str, str]:
        """Parse LLM response to extract question and answer.

        Args:
            response: Raw LLM response text.

        Returns:
            Tuple of (question, answer), empty strings if parsing failed.
        """
        question = ""
        answer = ""

        try:
            # Try to extract question
            if "Question:" in response:
                parts = response.split("Question:", 1)[1]
                if "Answer:" in parts:
                    question = parts.split("Answer:", 1)[0].strip()
                    answer = parts.split("Answer:", 1)[1].strip()
                else:
                    question = parts.strip()

            # Clean up
            question = question.strip().strip('"').strip("'")
            answer = answer.strip().strip('"').strip("'")

        except Exception as e:
            logger.debug(f"Failed to parse QA response: {e}")

        return question, answer


def generate_eval_dataset(
    corpus: Corpus,
    num_questions: int = 50,
    question_types: list[str] | None = None,
    model: str = "gpt-4o-mini",
    max_workers: int = 5,
    seed: int | None = None,
) -> EvalDataset:
    """Generate an evaluation dataset from a corpus of documents.

    This function samples documents from the corpus and uses an LLM to
    generate QA pairs, returning them as an EvalDataset.

    Args:
        corpus: Source corpus of documents.
        num_questions: Number of questions to generate.
        question_types: Types to generate (default: ["factoid"]).
            Options: "factoid", "multi_hop"
        model: LLM model to use for generation.
        max_workers: Number of parallel workers for generation.
        seed: Random seed for reproducibility.

    Returns:
        EvalDataset with generated questions.

    Example:
        ```python
        corpus = Corpus.from_jsonl("./corpus.jsonl")
        dataset = generate_eval_dataset(
            corpus,
            num_questions=100,
            question_types=["factoid", "multi_hop"],
        )
        dataset.to_jsonl("./eval_questions.jsonl")
        ```
    """
    if seed is not None:
        random.seed(seed)

    if question_types is None:
        question_types = ["factoid"]

    generator = QAGenerator(model=model)
    questions: list[EvalQuestion] = []

    # Calculate how many of each type to generate
    per_type = num_questions // len(question_types)
    remainder = num_questions % len(question_types)

    for i, q_type in enumerate(question_types):
        count = per_type + (1 if i < remainder else 0)

        if q_type == "factoid":
            new_questions = _generate_factoid_questions(
                corpus, generator, count, max_workers
            )
        elif q_type == "multi_hop":
            new_questions = _generate_multi_hop_questions(
                corpus, generator, count, max_workers
            )
        else:
            logger.warning(f"Unknown question type: {q_type}, skipping")
            continue

        questions.extend(new_questions)

    logger.info(f"Generated {len(questions)} questions total")
    return EvalDataset(questions=questions, name="generated")


def _generate_factoid_questions(
    corpus: Corpus,
    generator: QAGenerator,
    count: int,
    max_workers: int,
) -> list[EvalQuestion]:
    """Generate factoid questions from sampled documents."""
    # Sample more documents than needed (some will fail)
    sample_size = min(count * 2, len(corpus))
    sampled_docs = random.sample(list(corpus.documents), sample_size)

    questions: list[EvalQuestion] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(generator.generate_factoid, doc): doc
            for doc in sampled_docs
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Generating factoid questions",
        ):
            if len(questions) >= count:
                break

            result = future.result()
            if result is not None:
                # Assign unique ID
                result.question_id = f"q_factoid_{len(questions)}"
                questions.append(result)

    return questions[:count]


def _generate_multi_hop_questions(
    corpus: Corpus,
    generator: QAGenerator,
    count: int,
    max_workers: int,
) -> list[EvalQuestion]:
    """Generate multi-hop questions from document pairs."""
    # Create document pairs
    docs = list(corpus.documents)
    pairs: list[tuple[Document, Document]] = []

    # Sample pairs more than needed
    sample_size = min(count * 2, len(docs) * (len(docs) - 1) // 2)

    for _ in range(sample_size):
        if len(docs) < 2:
            break
        pair = random.sample(docs, 2)
        pairs.append((pair[0], pair[1]))

    questions: list[EvalQuestion] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                generator.generate_multi_hop,
                pair[0],
                pair[1],
                f"q_multihop_{idx}",
            ): idx
            for idx, pair in enumerate(pairs)
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Generating multi-hop questions",
        ):
            if len(questions) >= count:
                break

            result = future.result()
            if result is not None:
                result.question_id = f"q_multihop_{len(questions)}"
                questions.append(result)

    return questions[:count]
