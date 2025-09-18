# rag_system/batch_processor.py
from typing import List, Dict
import json
from tqdm import tqdm
from .rag import RAGSystem


def process_questions(rag: RAGSystem, questions_path: str) -> List[Dict]:
    """Process evaluation questions in batch."""
    with open(questions_path, "r") as f:
        questions = [json.loads(line) for line in f]

    results = []
    for q in tqdm(questions, desc="Processing questions"):
        answer, retrieved_docs = rag.query(q["question"])

        # Extract doc IDs for evaluation
        retrieved_ids = [
            doc.metadata.get("source_doc_id", "unknown") for doc in retrieved_docs
        ]

        results.append(
            {
                "question_id": q["question_id"],
                "question": q["question"],
                "generated_answer": answer,
                "retrieved_doc_ids": retrieved_ids,
                "ground_truth_answer": q["answer"],
                "required_evidence": q["required_evidence"],
            }
        )

    return results
