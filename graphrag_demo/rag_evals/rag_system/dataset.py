# rag_system/dataset.py
from pathlib import Path
from typing import List, Dict, Optional
import json
from dataclasses import dataclass


@dataclass
class RAGDataset:
    """Container for RAG dataset paths."""

    name: str
    corpus_path: Path
    questions_path: Optional[Path] = None

    def __post_init__(self):
        self._doc_index: Optional[Dict[str, str]] = None

    @classmethod
    def from_dataset_dir(
        cls, dataset_dir: str, dataset_name: str = None, questions_file: str = None
    ):
        """Load dataset from eval_dataset_builder output directory."""
        base_path = Path(dataset_dir)

        if dataset_name is None:
            dataset_name = base_path.name

        if questions_file:
            questions_path = base_path / questions_file
        else:
            questions_path = base_path / "sampled_eval_questions.jsonl"
            if not questions_path.exists():
                raise FileNotFoundError(
                    f"Default questions file not found: {questions_path}. "
                    f"Please ensure sampled_eval_questions.jsonl exists in {base_path}"
                )

        return cls(
            name=dataset_name,
            corpus_path=base_path / "corpus.jsonl",
            questions_path=questions_path,
        )

    def load_corpus(self) -> List[Dict]:
        """Load corpus documents."""
        docs: List[Dict] = []
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                docs.append(json.loads(line))
        return docs

    def load_questions(self) -> List[Dict]:
        """Load evaluation questions."""
        if not self.questions_path or not self.questions_path.exists():
            raise ValueError(f"Questions file not found: {self.questions_path}")

        questions: List[Dict] = []
        with open(self.questions_path, "r", encoding="utf-8") as f:
            for line in f:
                questions.append(json.loads(line))
        return questions

    def _ensure_index(self) -> None:
        """Build an in-memory index from doc_id to content."""
        if self._doc_index is not None:
            return
        index: Dict[str, str] = {}
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                doc_id = rec.get("doc_id")
                if not isinstance(doc_id, str):
                    continue
                content = rec.get("content")
                if not isinstance(content, str):
                    content = ""
                index[doc_id] = content
        self._doc_index = index

    def get_doc_text(self, doc_id: str) -> str:
        """Return the full text content for the given doc_id."""
        self._ensure_index()
        assert self._doc_index is not None
        try:
            return self._doc_index[doc_id]
        except KeyError:
            raise KeyError(f"Document not found: {doc_id}")

    def get_doc_content(self, doc_id: str) -> str:
        """Alias to maintain naming aligned with 'content' field."""
        return self.get_doc_text(doc_id)
