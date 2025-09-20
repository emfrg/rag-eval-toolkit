# rag_evaluator/run_evaluation.py
import click
import json
from pathlib import Path
from rag_evaluator import ExperimentRunner
from dotenv import load_dotenv

load_dotenv()

# Default RAG configurations
DEFAULT_CONFIGS = [
    {
        "chunking": False,
        "k_retrieve": 10,
        "use_reranker": False,
        "min_docs": 0,
        "max_docs": 4,
        "llm_model": "gpt-4o-mini",
    },
    {
        "chunking": False,
        "k_retrieve": 10,
        "use_reranker": True,
        "min_docs": 0,
        "max_docs": 4,
        "llm_model": "gpt-4o-mini",
    },
    # {
    #     "chunking": False,
    #     "k_retrieve": 10,
    #     "use_reranker": True,
    #     "rerank_threshold": 0.5,
    #     "min_docs": 0,
    #     "max_docs": 4,
    #     "llm_model": "gpt-4o-mini",
    # },
    # {
    #     "chunking": False,
    #     "k_retrieve": 15,
    #     "use_reranker": True,
    #     "rerank_threshold": 0.3,
    #     "min_docs": 0,
    #     "max_docs": 4,
    #     "llm_model": "gpt-4o-mini",
    # },
]


@click.command()
@click.option(
    "--dataset-dir",
    default="data/eval_datasets/yixuantt_MultiHopRAG_89ba9d15",
    help="Path to dataset directory",
)
@click.option(
    "--questions-file",
    default="sampled_eval_questions.jsonl",
    help="Questions file to use (default: sampled_eval_questions.jsonl)",
)
@click.option(
    "--experiment-name",
    default=None,
    help="Experiment name (default: timestamp)",
)
@click.option(
    "--output-dir",
    default="./results",
    help="Output directory for results",
)
@click.option(
    "--configs-file",
    default=None,
    type=click.Path(exists=True),
    help="JSON file with custom RAG configs (overrides defaults)",
)
@click.option(
    "--quick",
    is_flag=True,
    help="Run only the first config for quick testing",
)
def main(dataset_dir, questions_file, experiment_name, output_dir, configs_file, quick):
    """Run RAG evaluation experiments with RAGAS metrics."""

    # Load configurations
    if configs_file:
        click.echo(f"Loading configs from {configs_file}")
        with open(configs_file, "r") as f:
            rag_configs = json.load(f)
    else:
        rag_configs = DEFAULT_CONFIGS
        if quick:
            rag_configs = [rag_configs[0]]  # Use only first config for quick test

    click.echo(f"Dataset: {dataset_dir}")
    click.echo(f"Questions: {questions_file}")
    click.echo(f"Configs to test: {len(rag_configs)}")

    # Run experiments
    runner = ExperimentRunner(output_dir=output_dir)
    results = runner.run_experiment(
        dataset_dir=dataset_dir,
        rag_configs=rag_configs,
        experiment_name=experiment_name,
        questions_file=questions_file,
    )

    # Print summary
    click.echo("\n" + "=" * 50)
    click.echo("EXPERIMENT SUMMARY")
    click.echo("=" * 50)

    for result in results:
        config_id = result["config_id"]
        scores = result["scores"]
        config = result["config"]

        click.echo(f"\nConfig {config_id}:")

        # Display actual config parameters safely
        if "chunking" in config:
            if config.get("chunking", False):
                click.echo(f"  Chunking: Enabled")
                click.echo(f"    - Chunk size: {config.get('chunk_size', 'default')}")
                click.echo(
                    f"    - Chunk overlap: {config.get('chunk_overlap', 'default')}"
                )
            else:
                click.echo(f"  Chunking: Disabled")

        click.echo(f"  Reranker: {config.get('use_reranker', False)}")
        if config.get("use_reranker"):
            click.echo(
                f"    - Rerank threshold: {config.get('rerank_threshold', 'default')}"
            )
            click.echo(f"    - Min docs: {config.get('min_docs', 'default')}")
            click.echo(f"    - Max docs: {config.get('max_docs', 'default')}")

        click.echo(f"  k_retrieve: {config.get('k_retrieve', 'default')}")
        click.echo(f"  LLM Model: {config.get('llm_model', 'default')}")

        # Print individual scores
        if scores:
            click.echo("  Scores:")
            for metric, score in scores.items():
                click.echo(f"    {metric}: {score:.4f}")

            avg_score = sum(scores.values()) / len(scores)
            click.echo(f"  Average Score: {avg_score:.4f}")
        else:
            click.echo("  No scores available (check errors above)")

    click.echo(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()


# # Default: Run all 3 configs on 50-question sample
# python run_evaluation.py

# # Quick test with just first config
# python run_evaluation.py --quick

# # Use different questions file (441 questions)
# python run_evaluation.py --questions-file filtered_eval_questions.jsonl

# # Custom experiment name
# python run_evaluation.py --experiment-name baseline_test

# # Use custom configurations from file
# python run_evaluation.py --configs-file example_configs.json

# # Different dataset
# python run_evaluation.py --dataset-dir data/eval_datasets/custom_dataset

# # Full custom run
# python run_evaluation.py \
#     --dataset-dir data/eval_datasets/yixuantt_MultiHopRAG_89ba9d15 \
#     --questions-file sampled_eval_questions.jsonl \
#     --experiment-name multihop_baseline \
#     --output-dir ./results/baseline \
#     --quick
