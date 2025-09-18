#!/usr/bin/env python3
"""
Create evaluation datasets for RAG systems with QA pair generation and critique.
"""

import hashlib
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import click
import datasets
import pandas as pd
from tqdm.auto import tqdm
from dotenv import load_dotenv
from huggingface_hub import login as hf_login

from eval_dataset_builder.generator import QAGenerator
from eval_dataset_builder.critique import QACritique, MultiHopCritique
from eval_dataset_builder.dataset_manager import DatasetManager
from eval_dataset_builder.utils import write_jsonl, read_jsonl, setup_logging

load_dotenv()


@dataclass
class Config:
    """Configuration for evaluation dataset creation."""

    source_dataset: str
    output_dir: str
    dataset_type: str = "multi_hop"
    force_download: bool = False
    score_questions: bool = False
    filter_only: bool = False
    num_samples: Optional[int] = None
    eval_dataset: Optional[str] = None
    chunk_size: int = 2000
    chunk_overlap: int = 200
    model: str = "gpt-4o-mini"
    groundedness_threshold: int = 4
    relevance_threshold: int = 4
    standalone_threshold: int = 4
    complexity_threshold: int = 3
    max_workers: int = 5
    verbose: bool = False
    hf_token: Optional[str] = None


class EvalDatasetBuilder:
    """Main orchestrator for building evaluation datasets."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self._setup_authentication()
        self._setup_directories()

        self.dataset_manager = DatasetManager(
            self.base_dir / "source_datasets", dataset_type=config.dataset_type
        )

        if config.dataset_type == "multi_hop":
            self.qa_critique = MultiHopCritique(model=config.model)
        else:
            self.qa_generator = QAGenerator(model=config.model)
            self.qa_critique = QACritique(model=config.model)

    def _setup_authentication(self) -> None:
        """Setup HuggingFace authentication if token is provided."""
        token = self.config.hf_token or os.getenv("HF_TOKEN")
        if token:
            hf_login(token=token, add_to_git_credential=False)
            self.logger.info("HuggingFace authentication successful")

    def _setup_directories(self) -> None:
        """Create directory structure for outputs."""
        dataset_hash = hashlib.md5(self.config.source_dataset.encode()).hexdigest()[:8]
        dataset_name = self.config.source_dataset.replace("/", "_")

        self.base_dir = Path(self.config.output_dir)
        self.output_dir = (
            self.base_dir / "eval_datasets" / f"{dataset_name}_{dataset_hash}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.logger.info(f"Output directory: {self.output_dir}")

    def build(self) -> Dict[str, Any]:
        """Execute the main pipeline for building evaluation dataset."""
        results = {}

        if self.config.filter_only:
            return self._filter_from_cache(results)

        self.logger.info(f"Loading source dataset: {self.config.source_dataset}")
        source_dataset = self.dataset_manager.load_or_download(
            self.config.source_dataset, force_download=self.config.force_download
        )
        results["source_dataset_size"] = len(source_dataset)

        if self.config.dataset_type == "multi_hop":
            return self._build_multi_hop(source_dataset, results)
        else:
            return self._build_single_hop(source_dataset, results)

    def _filter_from_cache(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Filter questions from cached scores without API calls."""
        scored_path = self.output_dir / "eval_questions_with_scores.jsonl"

        if not scored_path.exists():
            raise FileNotFoundError(
                f"Scored questions not found at {scored_path}. "
                "Run with --score-questions first to generate scores."
            )

        self.logger.info("Loading scored questions from cache...")
        scored_questions = read_jsonl(scored_path)
        results["total_scored_questions"] = len(scored_questions)

        filtered_questions = [
            q
            for q in scored_questions
            if q.get("complexity_score", 0) >= self.config.complexity_threshold
            and q.get("standalone_score", 0) >= self.config.standalone_threshold
        ]

        if self.config.num_samples and self.config.num_samples < len(
            filtered_questions
        ):
            filtered_questions = random.sample(
                filtered_questions, self.config.num_samples
            )
            self.logger.info(
                f"Sampled {self.config.num_samples} from {len(filtered_questions)} filtered questions"
            )

        results["questions_filtered"] = len(filtered_questions)

        filtered_path = self.output_dir / "filtered_eval_questions.jsonl"
        write_jsonl(filtered_path, filtered_questions)
        self.logger.info(f"Saved {len(filtered_questions)} filtered questions")

        self._save_metadata(results)
        return results

    def _build_multi_hop(
        self, source_dataset: datasets.Dataset, results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build multi-hop evaluation dataset."""
        self.logger.info("Processing multi-hop dataset...")

        corpus_docs, eval_questions = self.dataset_manager.process_multi_hop_dataset(
            source_dataset
        )
        results["corpus_size"] = len(corpus_docs)
        results["num_questions"] = len(eval_questions)

        corpus_path = self.output_dir / "corpus.jsonl"
        write_jsonl(corpus_path, corpus_docs)
        self.logger.info(f"Saved {len(corpus_docs)} corpus documents")

        complete_questions_path = self.output_dir / "complete_eval_questions.jsonl"
        write_jsonl(complete_questions_path, eval_questions)
        self.logger.info(f"Saved {len(eval_questions)} complete evaluation questions")

        # Apply sampling if requested (even without scoring)
        output_questions = eval_questions
        if self.config.num_samples and self.config.num_samples < len(eval_questions):
            output_questions = random.sample(eval_questions, self.config.num_samples)
            self.logger.info(f"Sampled {self.config.num_samples} questions")
            results["questions_sampled"] = self.config.num_samples

        if not self.config.score_questions:
            # Just save the (possibly sampled) questions without scoring
            sampled_path = self.output_dir / "sampled_eval_questions.jsonl"
            write_jsonl(sampled_path, output_questions)
            self.logger.info(
                f"Saved {len(output_questions)} questions to sampled_eval_questions.jsonl"
            )
        else:
            # Score questions
            self.logger.info("Scoring questions for complexity and quality...")

            scored_path = self.output_dir / "eval_questions_with_scores.jsonl"

            if scored_path.exists():
                self.logger.info("Loading existing scored questions...")
                existing_scored = read_jsonl(scored_path)
                scored_ids = {q["question_id"] for q in existing_scored}

                questions_to_score = [
                    q for q in output_questions if q["question_id"] not in scored_ids
                ]

                if questions_to_score:
                    self.logger.info(
                        f"Scoring {len(questions_to_score)} new questions..."
                    )
                    newly_scored = self.qa_critique.evaluate_batch(
                        questions_to_score, max_workers=self.config.max_workers
                    )
                    all_scored = existing_scored + newly_scored
                else:
                    self.logger.info("All questions already scored")
                    all_scored = [
                        q
                        for q in existing_scored
                        if q["question_id"]
                        in {eq["question_id"] for eq in output_questions}
                    ]
            else:
                self.logger.info(f"Scoring {len(output_questions)} questions...")
                all_scored = self.qa_critique.evaluate_batch(
                    output_questions, max_workers=self.config.max_workers
                )

            write_jsonl(scored_path, all_scored)
            results["questions_scored"] = len(all_scored)
            self.logger.info(f"Saved {len(all_scored)} scored questions")

            # Filter by complexity and standalone scores
            filtered_questions = [
                q
                for q in all_scored
                if q.get("complexity_score", 0) >= self.config.complexity_threshold
                and q.get("standalone_score", 0) >= self.config.standalone_threshold
            ]

            results["questions_filtered"] = len(filtered_questions)

            filtered_path = self.output_dir / "filtered_eval_questions.jsonl"
            write_jsonl(filtered_path, filtered_questions)
            self.logger.info(f"Saved {len(filtered_questions)} filtered questions")

        self._save_metadata(results)
        return results

    def _build_single_hop(
        self, source_dataset: datasets.Dataset, results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build single-hop evaluation dataset."""
        self.logger.info("Processing documents into chunks...")
        processed_docs = self.dataset_manager.process_documents(
            source_dataset,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        results["num_chunks"] = len(processed_docs)

        docs_path = self.output_dir / "processed_docs.jsonl"
        write_jsonl(docs_path, processed_docs)

        if self.config.score_questions:
            self.logger.info(f"Generating {self.config.num_samples or 100} QA pairs...")
            qa_pairs = self._generate_qa_pairs(processed_docs)
            results["qa_pairs_generated"] = len(qa_pairs)

            raw_path = self.output_dir / "qa_pairs_raw.jsonl"
            write_jsonl(raw_path, qa_pairs)

            self.logger.info("Running critique agents on QA pairs...")
            filtered_pairs = self._critique_and_filter(qa_pairs)
            results["qa_pairs_filtered"] = len(filtered_pairs)

            filtered_path = self.output_dir / "qa_pairs_filtered.jsonl"
            write_jsonl(filtered_path, filtered_pairs)

        self._save_metadata(results)
        return results

    def _generate_qa_pairs(self, documents: List[Dict]) -> List[Dict]:
        """Generate QA pairs with checkpoint support."""
        checkpoint_file = self.checkpoint_dir / "qa_generation.jsonl"

        existing_pairs = []
        if checkpoint_file.exists():
            existing_pairs = read_jsonl(checkpoint_file)
            self.logger.info(
                f"Loaded {len(existing_pairs)} existing QA pairs from checkpoint"
            )

        num_samples = self.config.num_samples or 100
        remaining = num_samples - len(existing_pairs)
        if remaining <= 0:
            return existing_pairs[:num_samples]

        sampled_docs = random.sample(documents, min(remaining, len(documents)))

        new_pairs = []
        for doc in tqdm(sampled_docs, desc="Generating QA pairs"):
            qa_pair = self.qa_generator.generate_single(doc)
            if qa_pair:
                new_pairs.append(qa_pair)
                write_jsonl(checkpoint_file, existing_pairs + new_pairs)

        return existing_pairs + new_pairs

    def _critique_and_filter(self, qa_pairs: List[Dict]) -> List[Dict]:
        """Critique QA pairs and filter based on thresholds."""
        critiqued_pairs = self.qa_critique.evaluate_batch(
            qa_pairs, max_workers=self.config.max_workers
        )

        filtered_pairs = [
            qa
            for qa in critiqued_pairs
            if qa.get("groundedness_score", 0) >= self.config.groundedness_threshold
            and qa.get("relevance_score", 0) >= self.config.relevance_threshold
            and qa.get("standalone_score", 0) >= self.config.standalone_threshold
        ]

        self.logger.info(f"Filtered {len(filtered_pairs)}/{len(qa_pairs)} QA pairs")
        return filtered_pairs

    def _save_metadata(self, results: Dict[str, Any]) -> None:
        """Save configuration and results metadata."""
        config_path = self.output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(asdict(self.config), f, indent=2)

        results_path = self.output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"Pipeline completed. Results saved to {self.output_dir}")


@click.command()
@click.option(
    "--source-dataset", required=True, help="HuggingFace dataset name or local path"
)
@click.option("--output-dir", default="./data", help="Base output directory")
@click.option(
    "--dataset-type",
    type=click.Choice(["multi_hop", "single_hop"]),
    default="multi_hop",
    help="Type of dataset to process",
)
@click.option("--force-download", is_flag=True, help="Force re-download of datasets")
@click.option(
    "--score-questions",
    is_flag=True,
    help="Score questions for complexity and quality (requires API)",
)
@click.option(
    "--filter-only",
    is_flag=True,
    help="Filter from cached scores without downloading or scoring (offline)",
)
@click.option(
    "--num-samples", type=int, help="Number of samples to include in filtered output"
)
@click.option(
    "--eval-dataset", help="Existing evaluation dataset to use (single-hop mode)"
)
@click.option(
    "--complexity-threshold",
    default=3,
    type=int,
    help="Minimum complexity score for filtering (1-5)",
)
@click.option(
    "--config", type=click.Path(exists=True), help="Path to configuration file"
)
@click.option("--model", default="gpt-4o-mini", help="LLM model to use")
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
def main(
    source_dataset,
    output_dir,
    dataset_type,
    force_download,
    score_questions,
    filter_only,
    num_samples,
    eval_dataset,
    complexity_threshold,
    config,
    model,
    verbose,
):
    """Create evaluation dataset for RAG systems."""

    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    if config:
        with open(config, "r") as f:
            config_dict = json.load(f)
        config_obj = Config(**config_dict)
    else:
        config_obj = Config(
            source_dataset=source_dataset,
            output_dir=output_dir,
            dataset_type=dataset_type,
            force_download=force_download,
            score_questions=score_questions,
            filter_only=filter_only,
            num_samples=num_samples,
            eval_dataset=eval_dataset,
            complexity_threshold=complexity_threshold,
            model=model,
            verbose=verbose,
        )

    try:
        builder = EvalDatasetBuilder(config_obj)
        results = builder.build()

        logger.info("Pipeline completed successfully!")
        logger.info(f"Results: {results}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
