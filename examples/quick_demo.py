#!/usr/bin/env python3
"""Quick demo of RAG Eval Toolkit.

This runs the FULL pipeline in ~1-2 minutes using:
- Small synthetic corpus (10 documents)
- Small eval dataset (5 questions)
- Naive RAG only (no GraphRAG - that takes hours)
- No reranker (faster)

Usage:
    uv run python examples/quick_demo.py
"""

import json
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Create synthetic data
CORPUS_DATA = [
    {
        "doc_id": "doc_001",
        "content": "Paris is the capital city of France. It is located in northern France on the river Seine. Paris is known for the Eiffel Tower, which was built in 1889.",
        "metadata": {"topic": "geography", "country": "France"}
    },
    {
        "doc_id": "doc_002",
        "content": "The Eiffel Tower is a wrought-iron lattice tower in Paris. It was designed by Gustave Eiffel and stands 330 meters tall. It is the most visited paid monument in the world.",
        "metadata": {"topic": "landmarks", "country": "France"}
    },
    {
        "doc_id": "doc_003",
        "content": "Tokyo is the capital of Japan. It is the most populous metropolitan area in the world with over 37 million people. Tokyo hosted the 2020 Summer Olympics.",
        "metadata": {"topic": "geography", "country": "Japan"}
    },
    {
        "doc_id": "doc_004",
        "content": "Mount Fuji is the highest mountain in Japan at 3,776 meters. It is an active volcano located about 100 kilometers southwest of Tokyo. It last erupted in 1707.",
        "metadata": {"topic": "geography", "country": "Japan"}
    },
    {
        "doc_id": "doc_005",
        "content": "Python is a high-level programming language created by Guido van Rossum in 1991. It emphasizes code readability and uses significant indentation.",
        "metadata": {"topic": "programming", "language": "Python"}
    },
    {
        "doc_id": "doc_006",
        "content": "Machine learning is a subset of artificial intelligence. It enables computers to learn from data without being explicitly programmed. Common algorithms include neural networks and decision trees.",
        "metadata": {"topic": "programming", "field": "AI"}
    },
    {
        "doc_id": "doc_007",
        "content": "The Great Wall of China is a series of fortifications made of stone, brick, and other materials. It was built over centuries to protect against invasions. Its total length is over 20,000 kilometers.",
        "metadata": {"topic": "landmarks", "country": "China"}
    },
    {
        "doc_id": "doc_008",
        "content": "Berlin is the capital of Germany. It has a population of about 3.6 million people. The Berlin Wall divided the city from 1961 to 1989.",
        "metadata": {"topic": "geography", "country": "Germany"}
    },
    {
        "doc_id": "doc_009",
        "content": "RAG stands for Retrieval-Augmented Generation. It combines information retrieval with language models to generate more accurate and grounded responses.",
        "metadata": {"topic": "programming", "field": "AI"}
    },
    {
        "doc_id": "doc_010",
        "content": "The Louvre Museum in Paris is the world's largest art museum. It houses the Mona Lisa painting by Leonardo da Vinci. The museum receives about 10 million visitors annually.",
        "metadata": {"topic": "landmarks", "country": "France"}
    },
]

EVAL_DATA = [
    {
        "question_id": "q_001",
        "question": "What is the capital of France?",
        "answer": "Paris",
        "question_type": "factual",
        "required_evidence": ["doc_001"],
    },
    {
        "question_id": "q_002",
        "question": "How tall is the Eiffel Tower?",
        "answer": "The Eiffel Tower is 330 meters tall.",
        "question_type": "factual",
        "required_evidence": ["doc_002"],
    },
    {
        "question_id": "q_003",
        "question": "When was Python created and by whom?",
        "answer": "Python was created by Guido van Rossum in 1991.",
        "question_type": "factual",
        "required_evidence": ["doc_005"],
    },
    {
        "question_id": "q_004",
        "question": "What is the highest mountain in Japan and how tall is it?",
        "answer": "Mount Fuji is the highest mountain in Japan at 3,776 meters.",
        "question_type": "factual",
        "required_evidence": ["doc_004"],
    },
    {
        "question_id": "q_005",
        "question": "What famous painting is housed in the Louvre Museum?",
        "answer": "The Mona Lisa by Leonardo da Vinci is housed in the Louvre Museum.",
        "question_type": "factual",
        "required_evidence": ["doc_010"],
    },
]


def write_jsonl(data: list[dict], path: Path) -> None:
    """Write data to JSONL file."""
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def main():
    print("=" * 60)
    print("RAG Eval Toolkit - Quick Demo")
    print("=" * 60)

    # Create temp directory for data
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Write synthetic data
        corpus_path = tmpdir / "corpus.jsonl"
        eval_path = tmpdir / "questions.jsonl"
        output_dir = tmpdir / "results"

        print("\n1. Creating synthetic dataset...")
        write_jsonl(CORPUS_DATA, corpus_path)
        write_jsonl(EVAL_DATA, eval_path)
        print(f"   - Corpus: {len(CORPUS_DATA)} documents")
        print(f"   - Questions: {len(EVAL_DATA)} questions")

        # Import after dotenv
        from rag_eval import Corpus, EvalDataset, RAGConfig
        from rag_eval.systems.config import NaiveRAGConfig
        from rag_eval.evaluator import ExperimentRunner
        from rag_eval.reporting import print_comparison_table

        # Load data
        print("\n2. Loading data...")
        corpus = Corpus.from_jsonl(corpus_path)
        eval_dataset = EvalDataset.from_jsonl(eval_path)

        # Configure RAG - using Naive RAG only (fast!)
        print("\n3. Configuring Naive RAG system...")
        configs = [
            # Basic Naive RAG
            RAGConfig(
                rag_type="naive",
                llm_provider="anthropic",
                llm_model="claude-sonnet-4-20250514",
                naive=NaiveRAGConfig(
                    k_retrieve=3,
                    max_docs=3,
                    use_reranker=False,  # No reranker = faster
                ),
            ),
            # Naive RAG with more retrieval
            RAGConfig(
                rag_type="naive",
                llm_provider="anthropic",
                llm_model="claude-sonnet-4-20250514",
                naive=NaiveRAGConfig(
                    k_retrieve=5,
                    max_docs=5,
                    use_reranker=False,
                ),
            ),
        ]

        print(f"   - Comparing {len(configs)} configurations")

        # Run experiment
        print("\n4. Running evaluation...")
        print("   (This should take ~1-2 minutes)")
        print("-" * 40)

        runner = ExperimentRunner(output_dir=output_dir)
        summary = runner.run_experiments(configs, corpus, eval_dataset)

        # Print results
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)

        print_comparison_table(summary)

        # Show best config
        best = summary.find_best("faithfulness")
        if best:
            print(f"\nBest config (by faithfulness): {best.config_sig[:8]}")
            print(f"  Scores: {best.scores}")

        print("\n" + "=" * 60)
        print("Demo complete!")
        print("=" * 60)


if __name__ == "__main__":
    main()
