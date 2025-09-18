"""QA pair critique and quality evaluation."""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List
from langchain_openai import ChatOpenAI
from tqdm.auto import tqdm


GROUNDEDNESS_PROMPT = """
You will be given a context and a question.
Your task is to provide a 'total rating' scoring how well the question can be answered from the given context.

Give your answer on a scale of 1 to 5, where 1 means "not answerable" and 5 means "completely answerable".

Context: {context}
Question: {question}

Evaluation:
Total rating: """

RELEVANCE_PROMPT = """
You will be given a question.
Your task is to provide a 'total rating' scoring how relevant this question is for practitioners.

Give your answer on a scale of 1 to 5, where 1 means "not relevant" and 5 means "highly relevant".

Question: {question}

Evaluation:
Total rating: """

STANDALONE_PROMPT = """
You will be given a question.
Your task is to provide a 'total rating' scoring how understandable the question is without any context.

Give your answer on a scale of 1 to 5, where 1 means "not understandable" and 5 means "fully understandable".

Question: {question}

Evaluation:
Total rating: """

COMPLEXITY_PROMPT = """
You will be given a question and the number of evidence pieces required to answer it.
Your task is to provide a 'total rating' scoring the reasoning complexity required.

Give your answer on a scale of 1 to 5, where:
1 = Simple lookup (single fact)
2 = Basic connection (two related facts)
3 = Multi-step reasoning (connecting 2-3 facts)
4 = Complex synthesis (multiple sources with inference)
5 = Advanced reasoning (temporal, causal, or comparative analysis across sources)

Question: {question}
Evidence pieces required: {evidence_count}

Evaluation:
Total rating: """


class QACritique:
    """Evaluates QA pair quality using multiple criteria."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.logger = logging.getLogger(__name__)
        self.llm = ChatOpenAI(model=model, temperature=0, max_tokens=200)

    def evaluate_single(self, qa_pair: Dict[str, str]) -> Dict[str, any]:
        """Evaluate a single QA pair on multiple criteria."""
        qa_copy = qa_pair.copy()

        try:
            groundedness_response = self.llm.invoke(
                GROUNDEDNESS_PROMPT.format(
                    context=qa_pair["context"], question=qa_pair["question"]
                )
            ).content
            qa_copy["groundedness_score"] = self._extract_score(groundedness_response)

            relevance_response = self.llm.invoke(
                RELEVANCE_PROMPT.format(question=qa_pair["question"])
            ).content
            qa_copy["relevance_score"] = self._extract_score(relevance_response)

            standalone_response = self.llm.invoke(
                STANDALONE_PROMPT.format(question=qa_pair["question"])
            ).content
            qa_copy["standalone_score"] = self._extract_score(standalone_response)

        except Exception as e:
            self.logger.warning(f"Critique failed for question: {e}")
            qa_copy.update(
                {"groundedness_score": 0, "relevance_score": 0, "standalone_score": 0}
            )

        return qa_copy

    def evaluate_batch(
        self, qa_pairs: List[Dict[str, str]], max_workers: int = 5
    ) -> List[Dict[str, any]]:
        """Evaluate multiple QA pairs in parallel."""
        evaluated_pairs = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.evaluate_single, qa) for qa in qa_pairs]

            for future in tqdm(futures, desc="Critiquing QA pairs"):
                result = future.result()
                if result:
                    evaluated_pairs.append(result)

        return evaluated_pairs

    def _extract_score(self, response: str) -> int:
        """Extract numeric score from critique response."""
        try:
            score = int(response.split("Total rating:")[-1].strip())
            return min(max(score, 1), 5)
        except:
            return 0


class MultiHopCritique:
    """Evaluates multi-hop question complexity and quality."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.logger = logging.getLogger(__name__)
        self.llm = ChatOpenAI(model=model, temperature=0, max_tokens=200)

    def evaluate_single(self, question: Dict[str, any]) -> Dict[str, any]:
        """Evaluate a single multi-hop question."""
        question_copy = question.copy()

        try:
            complexity_response = self.llm.invoke(
                COMPLEXITY_PROMPT.format(
                    question=question["question"],
                    evidence_count=question.get("evidence_count", 1),
                )
            ).content
            question_copy["complexity_score"] = self._extract_score(complexity_response)

            standalone_response = self.llm.invoke(
                STANDALONE_PROMPT.format(question=question["question"])
            ).content
            question_copy["standalone_score"] = self._extract_score(standalone_response)

        except Exception as e:
            self.logger.warning(f"Evaluation failed for question: {e}")
            question_copy.update({"complexity_score": 0, "standalone_score": 0})

        return question_copy

    def evaluate_batch(self, questions: List[Dict], max_workers: int = 5) -> List[Dict]:
        """Evaluate multiple questions in parallel."""
        evaluated_questions = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.evaluate_single, q) for q in questions]

            for future in tqdm(futures, desc="Evaluating question complexity"):
                result = future.result()
                if result:
                    evaluated_questions.append(result)

        return evaluated_questions

    def _extract_score(self, response: str) -> int:
        """Extract numeric score from evaluation response."""
        try:
            score = int(response.split("Total rating:")[-1].strip())
            return min(max(score, 1), 5)
        except:
            return 0
