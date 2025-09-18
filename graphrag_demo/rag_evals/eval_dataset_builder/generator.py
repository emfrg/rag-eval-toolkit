"""QA pair generation from documents using LLMs."""

import logging
from typing import Dict, Optional
from langchain_openai import ChatOpenAI


QA_GENERATION_PROMPT = """
Given the following context, generate a factoid question and answer pair.
The question should be answerable from the context alone.

Context: {context}

Generate a question-answer pair in the following format:
Factoid question: [Your question here]
Answer: [Brief answer based on the context]
"""


class QAGenerator:
    """Generates QA pairs from document chunks using LLMs."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.7):
        self.logger = logging.getLogger(__name__)
        self.llm = ChatOpenAI(model=model, temperature=temperature, max_tokens=500)

    def generate_single(self, document: Dict[str, str]) -> Optional[Dict[str, str]]:
        """Generate a single QA pair from a document chunk."""
        try:
            prompt = QA_GENERATION_PROMPT.format(context=document["content"])
            response = self.llm.invoke(prompt).content

            # Parse response
            question = (
                response.split("Factoid question: ")[-1].split("Answer: ")[0].strip()
            )
            answer = response.split("Answer: ")[-1].strip()

            # Validate answer length
            if len(answer) > 300:
                self.logger.debug(f"Answer too long ({len(answer)} chars), skipping")
                return None

            return {
                "context": document["content"],
                "question": question,
                "answer": answer,
                "source_doc": document.get("source", "unknown"),
            }

        except Exception as e:
            self.logger.warning(f"Failed to generate QA pair: {e}")
            return None
