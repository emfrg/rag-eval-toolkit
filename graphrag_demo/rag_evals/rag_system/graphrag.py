from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv
from langchain.schema import Document
from tqdm.auto import tqdm

from .base import BaseRAGSystem, IndexReport
from .config import RAGConfig
from .dataset import RAGDataset

try:
    from lightrag import LightRAG, QueryParam
    from lightrag.llm.openai import (
        gpt_4o_complete,
        gpt_4o_mini_complete,
        openai_embed,
    )
    from lightrag.kg.shared_storage import initialize_pipeline_status
    from lightrag.utils import Tokenizer, setup_logger
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "LightRAG is required when rag_model='graphrag'. "
        "Install the lightrag package to enable this backend."
    ) from exc


if not getattr(Tokenizer, "_allow_special_tokens", False):  # pragma: no cover
    _original_encode = Tokenizer.encode

    def _encode_allow_special(self, content: str):
        return self.tokenizer.encode(content, disallowed_special=())

    Tokenizer.encode = _encode_allow_special
    Tokenizer._allow_special_tokens = True


def _select_llm_func(model_name: str):
    if model_name == "gpt-4o":
        return gpt_4o_complete
    if model_name == "gpt-4o-mini":
        return gpt_4o_mini_complete
    raise ValueError(f"Unsupported LightRAG model: {model_name}")


class LightRAGSystem(BaseRAGSystem):
    def __init__(self, config: RAGConfig, dataset: Optional[RAGDataset] = None) -> None:
        super().__init__(config, dataset)
        load_dotenv()
        setup_logger("lightrag", level="INFO")
        self._rag: Optional[LightRAG] = None
        self._workdir: Optional[Path] = None
        self._index_ready: bool = False
        self._index_report: Optional[IndexReport] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        if dataset:
            self._index_report = self.build_index(dataset)

    @property
    def index_report(self) -> Optional[IndexReport]:
        return self._index_report

    @property
    def working_directory(self) -> Optional[Path]:
        return self._workdir

    def build_index(self, dataset: RAGDataset) -> IndexReport:
        cfg = self.config.graphrag.indexing
        workdir = cfg.graph_cache_dir / dataset.name
        self._workdir = workdir

        # Decide whether a fresh index is needed before LightRAG touches the directory
        needs_index = (
            cfg.force_reindex or not workdir.exists() or not any(workdir.iterdir())
        )
        if cfg.force_reindex and workdir.exists():
            shutil.rmtree(workdir)

        workdir.mkdir(parents=True, exist_ok=True)

        records = self._load_corpus_records(dataset.corpus_path)
        total_docs = len(records)

        self._rag = self._run_async(self._initialize_light_rag(workdir))

        ingested = 0
        if needs_index:
            ingested = self._run_async(
                self._insert_records(self._rag, records, cfg.batch_size)
            )

        report = IndexReport(
            total_documents=total_docs,
            ingested_documents=ingested,
            reused_existing=not needs_index,
        )
        self._index_report = report
        return report

    def retrieve(self, question: str) -> List[Document]:
        context_text = self._fetch_context_text(question)
        return self._parse_context_documents(context_text)

    def query(self, question: str) -> Tuple[str, List[Document]]:
        context_text = self._fetch_context_text(question)
        contexts = self._parse_context_documents(context_text)
        answer_raw = self._run_async(self._async_query_answer(question))
        answer = self._extract_answer_text(answer_raw)
        return answer, contexts

    def _fetch_context_text(self, question: str) -> str:
        rag = self._ensure_rag()
        return self._run_async(self._async_query_context(question))

    def _ensure_rag(self) -> LightRAG:
        if self._rag is None:
            if not self.dataset:
                raise ValueError(
                    "No dataset supplied; build_index must be called first."
                )
            self.build_index(self.dataset)
        assert self._rag is not None
        return self._rag

    async def _initialize_light_rag(self, workdir: Path) -> LightRAG:
        rag = LightRAG(
            working_dir=str(workdir),
            embedding_func=openai_embed,
            llm_model_func=_select_llm_func(self.config.llm_model),
            max_parallel_insert=self.config.graphrag.indexing.max_parallel_insert,
        )
        await rag.initialize_storages()
        await initialize_pipeline_status()
        self._index_ready = True
        return rag

    async def _insert_records(
        self,
        rag: LightRAG,
        records: List[Tuple[str, Optional[str]]],
        batch_size: int,
    ) -> int:
        if not records:
            return 0
        total = 0
        for start in tqdm(range(0, len(records), batch_size), desc="Indexing"):
            batch = records[start : start + batch_size]
            texts = [text for text, _ in batch]
            ids: List[Optional[str]] = [doc_id for _, doc_id in batch]
            file_paths = [
                (
                    doc_id.strip()
                    if isinstance(doc_id, str) and doc_id.strip()
                    else f"doc-{start + idx}"
                )
                for idx, (_, doc_id) in enumerate(batch)
            ]
            ids_arg: Optional[List[str]] = (
                ids
                if all(isinstance(doc_id, str) and doc_id for doc_id in ids)
                else None
            )
            await rag.ainsert(texts, ids=ids_arg, file_paths=file_paths)
            total += len(texts)
        return total

    async def _async_query_context(self, question: str) -> str:
        param = self._query_param(only_context=True)
        result = await self._rag.aquery(question, param=param)  # type: ignore[union-attr]
        if isinstance(result, str):
            return result
        raise TypeError(
            f"LightRAG context response must be a string, got {type(result)}"
        )

    async def _async_query_answer(self, question: str) -> Any:
        param = self._query_param(only_context=False)
        return await self._rag.aquery(question, param=param)  # type: ignore[union-attr]

    def _query_param(self, *, only_context: bool = False) -> QueryParam:
        cfg = self.config.graphrag.query
        params: Dict[str, Any] = {"mode": cfg.mode, "top_k": cfg.top_k}
        params["only_need_context"] = only_context
        return QueryParam(**params)

    def _extract_answer_text(self, raw: Any) -> str:
        if isinstance(raw, str):
            return raw
        if isinstance(raw, dict) and "answer" in raw:
            value = raw["answer"]
            if isinstance(value, str):
                return value
        raise TypeError(
            f"Unexpected answer payload from LightRAG: {type(raw)}. "
            "Expected a string or dict with an 'answer' field."
        )

    def _parse_context_documents(self, context_text: str) -> List[Document]:
        if not context_text:
            return []

        documents: List[Document] = []
        in_chunk_section = False
        for line in context_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("-----Document Chunks"):
                in_chunk_section = True
                continue
            if in_chunk_section and stripped.startswith("-----"):
                break
            if in_chunk_section:
                if stripped.startswith("```"):
                    continue
                try:
                    data = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                content = data.get("content")
                if not isinstance(content, str) or not content.strip():
                    continue
                doc_id = data.get("file_path") or data.get("doc_id")
                if not isinstance(doc_id, str) or not doc_id.strip():
                    doc_id = f"context_{len(documents)}"
                metadata: Dict[str, Any] = {"doc_id": doc_id}
                documents.append(Document(page_content=content, metadata=metadata))
        if not documents and context_text:
            metadata: Dict[str, Any] = {"doc_id": "lightrag_context"}
            documents.append(Document(page_content=context_text, metadata=metadata))
        return documents

    def _load_corpus_records(
        self, corpus_path: Path
    ) -> List[Tuple[str, Optional[str]]]:
        records: List[Tuple[str, Optional[str]]] = []
        with corpus_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                data = json.loads(line)
                text = data.get("content") or ""
                if not text.strip():
                    continue
                records.append((text, data.get("doc_id")))
        return records

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        return self._loop

    def _run_async(self, coro):
        loop = self._ensure_loop()
        return loop.run_until_complete(coro)
