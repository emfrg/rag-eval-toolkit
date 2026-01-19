"""Prompt templates for RAG systems."""

# Default RAG prompt for question answering
DEFAULT_RAG_PROMPT = """You are a question-answering system. Answer the question based on the provided context.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

# Strict RAG prompt that enforces using only the context
STRICT_RAG_PROMPT = """You are a question-answering system that MUST operate under strict constraints.

CRITICAL RULES:
1. You may ONLY use information explicitly stated in or easily inferred from the provided context
2. You are FORBIDDEN from using any knowledge from your training data to accommodate for missing information
3. Every claim in your answer must be directly traceable to the context
4. If the context is empty, you MUST respond with exactly: "Insufficient information."

ANSWER STYLE:
Answer with Yes/No OR a short factoid answer. Do NOT include additional text or explanations.

CONTEXT:
{context}

QUESTION: {question}

Based SOLELY on the above context, provide your answer. If there is no context given, respond with "Insufficient information."

ANSWER:"""
