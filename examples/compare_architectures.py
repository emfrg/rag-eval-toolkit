"""Example: Compare Naive RAG vs GraphRAG.

This example shows how to compare different RAG architectures
using the RAG Eval Toolkit.

Usage:
    python examples/compare_architectures.py

Prerequisites:
    - Set ANTHROPIC_API_KEY environment variable (for generation)
    - Set OPENAI_API_KEY environment variable (for embeddings)
    - Have corpus.jsonl and questions.jsonl in data/ directory
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for API keys
if not os.getenv("ANTHROPIC_API_KEY"):
    print("Error: ANTHROPIC_API_KEY environment variable not set")
    exit(1)

if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY not set - required for embeddings")
    print("Set OPENAI_API_KEY to use embeddings and GraphRAG indexing")

from rag_eval import Corpus, EvalDataset, RAGConfig
from rag_eval.systems.config import GraphRAGConfig, GraphRAGQueryConfig, NaiveRAGConfig
from rag_eval.evaluator import ExperimentRunner
from rag_eval.reporting import print_comparison_table


def main():
    # Paths to data files
    data_dir = Path("data")
    corpus_path = data_dir / "corpus.jsonl"
    questions_path = data_dir / "questions.jsonl"

    # Check if data files exist
    if not corpus_path.exists():
        print(f"Error: Corpus file not found at {corpus_path}")
        print("Please create a corpus.jsonl file with your documents.")
        return

    if not questions_path.exists():
        print(f"Error: Questions file not found at {questions_path}")
        print("Please create a questions.jsonl file with evaluation questions.")
        return

    # Load data
    print("Loading corpus...")
    corpus = Corpus.from_jsonl(corpus_path)
    print(f"  Loaded {len(corpus)} documents")

    print("Loading evaluation dataset...")
    eval_dataset = EvalDataset.from_jsonl(questions_path)
    print(f"  Loaded {len(eval_dataset)} questions")

    # Use a sample for quick testing
    print("\nUsing a sample of 10 questions for quick testing...")
    eval_dataset = eval_dataset.sample(n=10, seed=42)

    # Define configurations to compare
    configs = [
        # Naive RAG - basic configuration (using Anthropic Claude)
        RAGConfig(
            rag_type="naive",
            llm_provider="anthropic",
            llm_model="claude-sonnet-4-20250514",
            naive=NaiveRAGConfig(
                k_retrieve=5,
                use_reranker=False,
            ),
        ),
        # Naive RAG - with reranker (using Anthropic Claude)
        RAGConfig(
            rag_type="naive",
            llm_provider="anthropic",
            llm_model="claude-sonnet-4-20250514",
            naive=NaiveRAGConfig(
                k_retrieve=10,
                use_reranker=True,
            ),
        ),
        # GraphRAG - hybrid mode (using Anthropic Claude for generation)
        # Note: LightRAG indexing still uses OpenAI internally
        RAGConfig(
            rag_type="graphrag",
            llm_provider="anthropic",
            llm_model="claude-sonnet-4-20250514",
            graphrag=GraphRAGConfig(
                query=GraphRAGQueryConfig(mode="hybrid"),
            ),
        ),
    ]

    print(f"\nComparing {len(configs)} configurations...")

    # Run experiments
    runner = ExperimentRunner(output_dir="./results/comparison")
    summary = runner.run_experiments(configs, corpus, eval_dataset)

    # Print comparison table
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print_comparison_table(summary)

    # Find and print best configuration
    best = summary.find_best("faithfulness")
    if best:
        print(f"\nBest configuration (by faithfulness):")
        print(f"  Config: {best.config_sig}")
        print(f"  Type: {best.config.get('rag_type')}")
        print(f"  Faithfulness: {best.scores.get('faithfulness', 0):.3f}")

    print(f"\nResults saved to ./results/comparison/")


if __name__ == "__main__":
    main()
