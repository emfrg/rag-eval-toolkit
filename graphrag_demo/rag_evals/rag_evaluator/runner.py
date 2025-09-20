# rag_evaluator/runner.py
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from .evaluator import RAGEvaluator
from rag_system import RAGConfig, RAGSystem, RAGDataset


class ExperimentRunner:
    """Run RAG experiments with different configurations."""

    def __init__(self, output_dir: str = "./results"):
        self.evaluator = RAGEvaluator()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def run_experiment(
        self,
        dataset_dir: str,
        rag_configs: List[Dict[str, Any]],
        experiment_name: str = None,
        questions_file: str = None,
    ) -> List[Dict]:
        """Run experiments with multiple RAG configurations."""

        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Load dataset
        dataset = RAGDataset.from_dataset_dir(
            dataset_dir, questions_file=questions_file
        )

        results = []
        for i, config_dict in enumerate(rag_configs):
            print(f"\n{'='*50}")
            print(f"Running Config {i+1}/{len(rag_configs)}")
            print(f"Config: {config_dict}")
            print("=" * 50)

            # Create RAG system
            rag_config = RAGConfig(**config_dict)
            rag_system = RAGSystem(rag_config, dataset)

            # Evaluate with RAGAS
            ragas_result = self.evaluator.evaluate(rag_system, dataset)

            # Convert to pandas DataFrame to access scores
            df = ragas_result.to_pandas()

            # Get metric columns (exclude metadata columns like question, answer, etc.)
            metric_columns = [
                col
                for col in df.columns
                if col
                in [
                    "faithfulness",
                    "answer_correctness",
                    "answer_similarity",
                    "context_precision",
                    "context_recall",
                    "answer_relevancy",
                ]
            ]

            # Calculate mean scores for each metric
            scores_dict = {}
            for metric in metric_columns:
                mean_score = df[metric].dropna().mean()
                if not pd.isna(mean_score):
                    scores_dict[metric] = float(mean_score)

            result = {
                "config_id": i,
                "config": config_dict,
                "scores": scores_dict,
            }
            results.append(result)

            # Save individual result
            output_file = self.output_dir / f"{experiment_name}_config_{i}.json"
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2, default=str)

            print(f"\nResults:")
            for metric, score in scores_dict.items():
                print(f"  {metric}: {score:.4f}")

        # Save summary
        summary_file = self.output_dir / f"{experiment_name}_summary.json"
        with open(summary_file, "w") as f:
            json.dump(
                {
                    "experiment": experiment_name,
                    "dataset": dataset_dir,
                    "results": results,
                },
                f,
                indent=2,
            )

        return results
