# rag_evaluator/runner.py
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

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

            # Store results - handle lists by averaging them
            scores_dict = {}
            for metric in ["faithfulness", "answer_correctness", "answer_similarity"]:
                try:
                    score = ragas_result[metric]
                    # If it's a list, average it
                    if isinstance(score, list):
                        scores_dict[metric] = sum(score) / len(score)
                    else:
                        scores_dict[metric] = score
                except:
                    pass

            result = {
                "config_id": i,
                "config": config_dict,
                "scores": scores_dict,
            }
            results.append(result)

            # Save individual result
            output_file = self.output_dir / f"{experiment_name}_config_{i}.json"
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)

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
