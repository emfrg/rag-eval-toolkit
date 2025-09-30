#!/usr/bin/env python3
"""Example usage of eval_dataset_builder."""

import subprocess
import json
from pathlib import Path


# Method 1: Direct CLI usage
def run_basic_download():
    """Download dataset and sample questions without scoring."""
    cmd = [
        "python",
        "-m",
        "eval_dataset_builder.create_eval_dataset",
        "--source-dataset",
        "yixuantt/MultiHopRAG",
        "--output-dir",
        "./data",
        "--dataset-type",
        "multi_hop",
        "--num-samples",
        "100",  # 100
    ]
    subprocess.run(cmd)


# Method 2: Using config file
def run_with_config():
    """Create config and run."""
    config = {
        "source_dataset": "yixuantt/MultiHopRAG",
        "output_dir": "./data",
        "dataset_type": "multi_hop",
        "num_samples": 50,
        "score_questions": False,
        "model": "gpt-4o-mini",
        "verbose": True,
    }

    # Save config
    with open("my_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Run with config
    cmd = [
        "python",
        "-m",
        "eval_dataset_builder.create_eval_dataset",
        "--config",
        "my_config.json",
    ]
    subprocess.run(cmd)


# Method 3: Direct module import
def run_programmatically():
    """Use the module directly."""
    from eval_dataset_builder.create_eval_dataset import Config, EvalDatasetBuilder

    config = Config(
        source_dataset="yixuantt/MultiHopRAG",
        output_dir="./data",
        dataset_type="multi_hop",
        num_samples=50,
        score_questions=False,
        verbose=True,
    )

    builder = EvalDatasetBuilder(config)
    results = builder.build()
    print(f"Created dataset with {results['corpus_size']} documents")
    print(
        f"Sampled {results.get('questions_sampled', results['num_questions'])} questions"
    )


if __name__ == "__main__":
    # For testing - just download 50 questions without scoring
    run_basic_download()

    # Check output
    output_dir = Path("data/eval_datasets/yixuantt_MultiHopRAG_89ba9d15")
    if output_dir.exists():
        print(f"\nDataset created at: {output_dir}")
        print(f"Files created:")
        for file in output_dir.glob("*.jsonl"):
            print(f"  - {file.name}")
