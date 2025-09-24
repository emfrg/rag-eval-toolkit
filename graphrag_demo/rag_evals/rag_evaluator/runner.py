# rag_evaluator/runner.py
import json
import traceback
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from numpy import True_
import pandas as pd
from tqdm import tqdm

from .evaluator import RAGEvaluator
from rag_system import RAGConfig, RAGSystem, RAGDataset

from ragas import SingleTurnSample, EvaluationDataset, evaluate


class ExperimentRunner:
    """Run RAG experiments with different configurations."""

    def __init__(self, output_dir: str = "./results"):
        self.evaluator = RAGEvaluator()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "answers").mkdir(exist_ok=True)

    def _config_signature(self, config_dict: Dict[str, Any]) -> str:
        data = json.dumps(config_dict, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(data.encode("utf-8")).hexdigest()[:10]

    def _answers_dir(self) -> Path:
        return self.output_dir / "answers"

    def _answers_filename(self, config_sig: str) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self._answers_dir() / f"{config_sig}__{ts}.jsonl"

    def _find_latest_answers_file(self, config_sig: str) -> Optional[Path]:
        files = sorted(self._answers_dir().glob(f"{config_sig}__*.jsonl"))
        return files[-1] if files else None

    def _save_records(self, path: Path, records: List[Dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def _load_records(self, path: Path) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out

    def _records_to_samples(
        self, records: List[Dict[str, Any]]
    ) -> List[SingleTurnSample]:
        samples: List[SingleTurnSample] = []
        for r in records:
            samples.append(
                SingleTurnSample(
                    user_input=r["question"],
                    response=r["model"]["response"],
                    retrieved_contexts=r["model"]["retrieved_evidence_texts"],
                    reference=r["ground_truth"]["answer"],
                )
            )
        return samples

    def _required_texts(
        self, dataset: RAGDataset, required_ids: List[str]
    ) -> List[str]:
        if not hasattr(dataset, "get_doc_text"):
            raise AttributeError(
                "RAGDataset must implement get_doc_text(doc_id: str) -> str"
            )
        return [dataset.get_doc_text(doc_id) for doc_id in required_ids]

    def _find_latest_result_file(self, config_sig: str) -> Optional[Path]:
        """Find the most recent per-config result JSON produced previously."""
        candidates = list(self.output_dir.glob("*_config_*.json"))
        latest_path: Optional[Path] = None
        latest_mtime: float = -1.0
        for p in candidates:
            try:
                with p.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("config_sig") == config_sig:
                    mtime = p.stat().st_mtime
                    if mtime > latest_mtime:
                        latest_mtime = mtime
                        latest_path = p
            except Exception:
                continue
        return latest_path

    def _load_scores_from_result(self, path: Path) -> Dict[str, float]:
        """Load scores dict from a previous result file."""
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        scores = data.get("scores", {}) or {}  # TODO: check this syntax
        out: Dict[str, float] = {}
        for k, v in scores.items():
            try:
                out[k] = float(v)
            except Exception:
                continue
        return out

    def run_experiment(
        self,
        dataset_dir: str,
        rag_configs: List[Dict[str, Any]],
        experiment_name: str = None,
        questions_file: str = None,
        reuse_latest: bool = True,
        reuse_cached_scores: bool = True,
    ) -> List[Dict]:
        """Run experiments with multiple RAG configurations."""

        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        dataset = RAGDataset.from_dataset_dir(
            dataset_dir, questions_file=questions_file
        )

        results = []
        for i, config_dict in enumerate(rag_configs):
            print(f"\n{'='*50}")
            print(f"Running Config {i+1}/{len(rag_configs)}")
            print(f"Config: {config_dict}")
            print("=" * 50)

            config_sig = self._config_signature(config_dict)
            answers_file: Optional[Path] = None
            records: List[Dict[str, Any]] = []
            samples: List[SingleTurnSample] = []
            used_cached_answers = False

            if reuse_latest:
                latest = self._find_latest_answers_file(config_sig)
                if latest:
                    print(f"Reusing answers: {latest.name}")
                    answers_file = latest
                    records = self._load_records(latest)
                    samples = self._records_to_samples(records)
                    used_cached_answers = True

            if not samples:
                print("Generating answers...")
                rag_config = RAGConfig(**config_dict)
                rag_system = RAGSystem(rag_config, dataset)

                questions = dataset.load_questions()
                for idx, q in enumerate(
                    tqdm(questions, desc="Generating", leave=False)
                ):
                    try:
                        q_id = q.get("question_id", f"q_{idx:05d}")
                        question_text = q["question"]
                        answer, retrieved_docs = rag_system.query(question_text)

                        retrieved_ids = [
                            doc.metadata["doc_id"] for doc in retrieved_docs
                        ]
                        retrieved_texts = [doc.page_content for doc in retrieved_docs]

                        required_ids = (
                            q.get("required_evidence", []) or []
                        )  # TODO: check this syntax
                        required_texts = self._required_texts(dataset, required_ids)

                        record = {
                            "question_id": q_id,
                            "question": question_text,
                            "question_type": q.get("question_type"),
                            "evidence_count": q.get("evidence_count"),
                            "model": {
                                "response": answer,
                                "retrieved_evidence": retrieved_ids,
                                "retrieved_evidence_texts": retrieved_texts,
                            },
                            "ground_truth": {
                                "answer": q.get("answer", ""),
                                "required_evidence": required_ids,
                                "required_evidence_texts": required_texts,
                            },
                        }
                        records.append(record)

                        samples.append(
                            SingleTurnSample(
                                user_input=question_text,
                                response=answer,
                                retrieved_contexts=retrieved_texts,
                                reference=q.get("answer", ""),
                            )
                        )
                    except Exception as e:
                        print(
                            f"\nError processing question {q.get('question_id', 'unknown')}: {e}"
                        )
                        continue

                if not samples:
                    raise ValueError("No valid samples generated for evaluation")

                answers_file = self._answers_filename(config_sig)
                self._save_records(answers_file, records)
                print(f"Saved answers: {answers_file.name}")

            # Decide whether to run evaluation or reuse cached scores
            scores_dict: Dict[str, float] = {}
            if used_cached_answers and reuse_cached_scores:
                prev_result_file = self._find_latest_result_file(config_sig)
                if prev_result_file:
                    print(f"Loading cached scores from: {prev_result_file.name}")
                    scores_dict = self._load_scores_from_result(prev_result_file)
                else:
                    print("No cached scores found; running evaluation...")

            if not scores_dict:
                print(f"Evaluating {len(samples)} samples with RAGAS...")
                eval_dataset = EvaluationDataset(samples=samples)
                ragas_result = evaluate(
                    dataset=eval_dataset, metrics=self.evaluator.metrics
                )
                df = ragas_result.to_pandas()

                metric_columns = [
                    col
                    for col in df.columns
                    if col
                    in [
                        "faithfulness",
                        "factual_correctness(mode=f1)",
                        "factual_correctness(mode=precision)",
                        "semantic_similarity",
                        "llm_context_precision_with_reference",
                        "context_recall",
                        "context_entity_recall",
                    ]
                ]

                for metric in metric_columns:
                    mean_score = df[metric].dropna().mean()
                    if not pd.isna(mean_score):
                        scores_dict[metric] = float(mean_score)

            result = {
                "config_id": i,
                "config_sig": config_sig,
                "config": config_dict,
                "answers_file": answers_file.name if answers_file else None,
                "scores": scores_dict,
            }
            results.append(result)

            output_file = self.output_dir / f"{experiment_name}_config_{i}.json"
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2, default=str)

            print(f"\nResults:")
            for metric, score in scores_dict.items():
                print(f"  {metric}: {score:.4f}")

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

    # def run_experiment(
    #     self,
    #     dataset_dir: str,
    #     rag_configs: List[Dict[str, Any]],
    #     experiment_name: str = None,
    #     questions_file: str = None,
    #     reuse_latest: bool = True,
    # ) -> List[Dict]:
    #     """Run experiments with multiple RAG configurations."""

    #     if experiment_name is None:
    #         experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    #     dataset = RAGDataset.from_dataset_dir(
    #         dataset_dir, questions_file=questions_file
    #     )

    #     results = []
    #     for i, config_dict in enumerate(rag_configs):
    #         print(f"\n{'='*50}")
    #         print(f"Running Config {i+1}/{len(rag_configs)}")
    #         print(f"Config: {config_dict}")
    #         print("=" * 50)

    #         config_sig = self._config_signature(config_dict)
    #         answers_file: Optional[Path] = None
    #         records: List[Dict[str, Any]] = []
    #         samples: List[SingleTurnSample] = []

    #         if reuse_latest:
    #             latest = self._find_latest_answers_file(config_sig)
    #             if latest:
    #                 print(f"Reusing answers: {latest.name}")
    #                 answers_file = latest
    #                 records = self._load_records(latest)
    #                 samples = self._records_to_samples(records)

    #         if not samples:
    #             print("Generating answers...")
    #             rag_config = RAGConfig(**config_dict)
    #             rag_system = RAGSystem(rag_config, dataset)

    #             questions = dataset.load_questions()
    #             for idx, q in enumerate(
    #                 tqdm(questions, desc="Generating", leave=False)
    #             ):
    #                 try:
    #                     q_id = q.get("question_id", f"q_{idx:05d}")
    #                     question_text = q["question"]
    #                     answer, retrieved_docs = rag_system.query(question_text)

    #                     retrieved_ids = [
    #                         doc.metadata["doc_id"] for doc in retrieved_docs
    #                     ]

    #                     retrieved_texts = [doc.page_content for doc in retrieved_docs]

    #                     required_ids = q.get("required_evidence", []) or []
    #                     required_texts = self._required_texts(dataset, required_ids)

    #                     record = {
    #                         "question_id": q_id,
    #                         "question": question_text,
    #                         "question_type": q.get("question_type"),
    #                         "evidence_count": q.get("evidence_count"),
    #                         "model": {
    #                             "response": answer,
    #                             "retrieved_evidence": retrieved_ids,
    #                             "retrieved_evidence_texts": retrieved_texts,
    #                         },
    #                         "ground_truth": {
    #                             "answer": q.get("answer", ""),
    #                             "required_evidence": required_ids,
    #                             "required_evidence_texts": required_texts,
    #                         },
    #                     }
    #                     records.append(record)

    #                     samples.append(
    #                         SingleTurnSample(
    #                             user_input=question_text,
    #                             response=answer,
    #                             retrieved_contexts=retrieved_texts,
    #                             reference=q.get("answer", ""),
    #                         )
    #                     )
    #                 except Exception as e:
    #                     print(
    #                         f"\nError processing question {q.get('question_id', 'unknown')}: {e}"
    #                     )
    #                     continue

    #             if not samples:
    #                 raise ValueError("No valid samples generated for evaluation")

    #             answers_file = self._answers_filename(config_sig)
    #             self._save_records(answers_file, records)
    #             print(f"Saved answers: {answers_file.name}")

    #         print(f"Evaluating {len(samples)} samples with RAGAS...")
    #         eval_dataset = EvaluationDataset(samples=samples)
    #         ragas_result = evaluate(
    #             dataset=eval_dataset, metrics=self.evaluator.metrics
    #         )

    #         df = ragas_result.to_pandas()

    #         metric_columns = [
    #             col
    #             for col in df.columns
    #             if col
    #             in [
    #                 "faithfulness",
    #                 "factual_correctness(mode=f1)",
    #                 "factual_correctness(mode=precision)",
    #                 "semantic_similarity",
    #                 "llm_context_precision_with_reference",
    #                 "context_recall",
    #                 "context_entity_recall",
    #             ]
    #         ]

    #         scores_dict = {}
    #         for metric in metric_columns:
    #             mean_score = df[metric].dropna().mean()
    #             if not pd.isna(mean_score):
    #                 scores_dict[metric] = float(mean_score)

    #         result = {
    #             "config_id": i,
    #             "config_sig": config_sig,
    #             "config": config_dict,
    #             "answers_file": answers_file.name if answers_file else None,
    #             "scores": scores_dict,
    #         }
    #         results.append(result)

    #         output_file = self.output_dir / f"{experiment_name}_config_{i}.json"
    #         with open(output_file, "w") as f:
    #             json.dump(result, f, indent=2, default=str)

    #         print(f"\nResults:")
    #         for metric, score in scores_dict.items():
    #             print(f"  {metric}: {score:.4f}")

    #     summary_file = self.output_dir / f"{experiment_name}_summary.json"
    #     with open(summary_file, "w") as f:
    #         json.dump(
    #             {
    #                 "experiment": experiment_name,
    #                 "dataset": dataset_dir,
    #                 "results": results,
    #             },
    #             f,
    #             indent=2,
    #         )

    #     return results
