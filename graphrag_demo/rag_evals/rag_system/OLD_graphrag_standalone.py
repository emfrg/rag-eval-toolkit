# init_graphrag_standalone.py
import asyncio
import json
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import click
from dotenv import load_dotenv
from tqdm.auto import tqdm

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_complete, gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import Tokenizer, setup_logger

from rag_system import RAGConfig


# Allow LightRAG tokenizer to accept special tokens without raising errors.
_original_encode = Tokenizer.encode


def _encode_allow_special(self, content: str):
    return self.tokenizer.encode(content, disallowed_special=())


Tokenizer.encode = _encode_allow_special


# --- helpers ---------------------------------------------------------------
async def answer_and_context(
    rag: LightRAG, query: str, mode: str = "hybrid", top_k: int = 50
) -> tuple[str, str]:
    """Return model answer and retrieved context using two explicit calls."""
    context = await rag.aquery(
        query, param=QueryParam(mode=mode, top_k=top_k, only_need_context=True)
    )
    answer = await rag.aquery(
        query, param=QueryParam(mode=mode, top_k=top_k, only_need_context=False)
    )
    return answer, context


# ---------------------------------------------------------------------------


def select_llm_func(model_name: str):
    if model_name == "gpt-4o":
        return gpt_4o_complete
    if model_name == "gpt-4o-mini":
        return gpt_4o_mini_complete
    raise ValueError(f"Unsupported LLM model for LightRAG: {model_name}")


def build_workdir(cache_root: Path, dataset_dir: Path) -> Path:
    dataset_name = dataset_dir.name
    return cache_root / dataset_name


def prepare_workdir(workdir: Path, force: bool) -> bool:
    if workdir.exists():
        if force:
            shutil.rmtree(workdir)
            workdir.mkdir(parents=True, exist_ok=True)
            return True
        if any(workdir.iterdir()):
            return False
        return True
    workdir.mkdir(parents=True, exist_ok=True)
    return True


def load_corpus_records(corpus_path: Path) -> List[Tuple[str, Optional[str]]]:
    records: List[Tuple[str, Optional[str]]] = []
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            text = data.get("content") or ""
            if not text.strip():
                continue
            doc_id = data.get("doc_id")
            records.append((text, doc_id))
    return records


async def init_graphrag(
    workdir: Path, config: RAGConfig, max_parallel_insert: int
) -> LightRAG:
    rag = LightRAG(
        working_dir=str(workdir),
        embedding_func=openai_embed,
        llm_model_func=select_llm_func(config.llm_model),
        max_parallel_insert=max_parallel_insert,
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


async def insert_records(
    rag: LightRAG, records: List[Tuple[str, Optional[str]]], batch_size: int
) -> int:
    if not records:
        return 0

    total_inserted = 0
    for start in tqdm(
        range(0, len(records), batch_size), desc="Indexing", unit="batch"
    ):
        batch = records[start : start + batch_size]
        texts = [text for text, _ in batch]
        ids = [doc_id for _, doc_id in batch]
        # Construct file paths from provided document identifiers.
        file_paths = [
            (
                doc_id.strip()
                if isinstance(doc_id, str) and doc_id.strip()
                else f"doc-{start + i}"
            )
            for i, (_, doc_id) in enumerate(batch)
        ]
        ids_arg = ids if all(doc_id is not None for doc_id in ids) else None
        await rag.ainsert(texts, ids=ids_arg, file_paths=file_paths)
        total_inserted += len(texts)

    return total_inserted


async def run_indexing(
    dataset_dir: Path,
    cache_root: Path,
    max_parallel_insert: int,
    batch_size: int,
    force: bool,
    skip_sanity_query: bool,
    sanity_query: str,
) -> None:
    load_dotenv()
    setup_logger("lightrag", level="INFO")

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    corpus_path = dataset_dir / "corpus.jsonl"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    workdir = build_workdir(cache_root, dataset_dir)
    is_fresh_index = prepare_workdir(workdir, force)

    config = RAGConfig()
    rag = await init_graphrag(workdir, config, max_parallel_insert)

    inserted = 0
    if is_fresh_index:
        records = load_corpus_records(corpus_path)
        total_docs = len(records)
        print(f"Loaded {total_docs} documents from {corpus_path}")

        inserted = await insert_records(rag, records, batch_size)
        print(f"Indexed {inserted} documents into {workdir}")
    else:
        print(f"Reusing existing LightRAG index at {workdir}")

    if not skip_sanity_query:

        answer, context = await answer_and_context(
            rag, sanity_query, mode="hybrid", top_k=50
        )
        print("\nSanity query:")
        print(f"Q: {sanity_query}")
        print(f"A: {answer}")
        print("\nRetrieved context:")
        print(context)


@click.command()
@click.option(
    "--dataset-dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Path to the dataset directory produced by eval_dataset_builder.",
)
@click.option(
    "--graph-cache",
    default="graphrag_cache",
    type=click.Path(path_type=Path),
    help="Root directory where LightRAG artifacts will be stored.",
)
@click.option(
    "--max-parallel-insert",
    default=4,
    show_default=True,
    help="Maximum number of concurrent insert tasks when building the graph.",
)
@click.option(
    "--batch-size",
    default=128,
    show_default=True,
    help="Number of documents to insert per batch.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Recreate the working directory if it already contains data.",
)
@click.option(
    "--skip-sanity-query",
    is_flag=True,
    help="Skip running a sanity query after indexing completes.",
)
@click.option(
    "--sanity-query",
    default="What is this dataset about?",
    show_default=True,
    help="Query to issue once indexing completes.",
)
def main(
    dataset_dir: Path,
    graph_cache: Path,
    max_parallel_insert: int,
    batch_size: int,
    force: bool,
    skip_sanity_query: bool,
    sanity_query: str,
) -> None:
    """Index a MultiHop RAG dataset into a LightRAG store for offline testing."""
    asyncio.run(
        run_indexing(
            dataset_dir=dataset_dir.resolve(),
            cache_root=graph_cache.expanduser().resolve(),
            max_parallel_insert=max_parallel_insert,
            batch_size=batch_size,
            force=force,
            skip_sanity_query=skip_sanity_query,
            sanity_query=sanity_query,
        )
    )


if __name__ == "__main__":
    main()
