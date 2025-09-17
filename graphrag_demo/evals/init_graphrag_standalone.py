import os, json, asyncio
from pathlib import Path
from tqdm import tqdm

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger, Tokenizer  # Tokenizer is here

from dotenv import load_dotenv
import tiktoken  # needed for encode patch

# --- Monkey-patch Tokenizer.encode to allow special tokens ---
_orig_encode = Tokenizer.encode


def _encode_allow_special(self, content: str):
    return self.tokenizer.encode(
        content,
        disallowed_special=(),  # treat all specials as normal text
        # or: allowed_special={'<|endoftext|>', '<|im_start|>', '<|im_end|>'}
    )


Tokenizer.encode = _encode_allow_special
# ----------------------------------------------------------------

load_dotenv()


# ---------- config ----------
SAVE_DIR = Path("datasets_local/20250914_145157")  # or auto-pick latest
CORPUS = SAVE_DIR / "initial_corpus.jsonl"  # your saved file
WORKING_DIR = "lightrag_store"  # LightRAG storage folder
MAX_PARALLEL_INSERT = 4  # tune 2–10 per docs
# ----------------------------

setup_logger("lightrag", level="INFO")
os.makedirs(WORKING_DIR, exist_ok=True)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            yield rec


async def init_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,  # keep SAME for query time
        llm_model_func=gpt_4o_mini_complete,  # indexing LLM
        max_parallel_insert=MAX_PARALLEL_INSERT,
    )
    await rag.initialize_storages()  # REQUIRED
    await initialize_pipeline_status()  # REQUIRED
    return rag


async def insert_all(rag: LightRAG, records):
    texts = []
    ids = []
    for rec in records:
        text = rec.get("text", "")
        src = rec.get("source") or ""
        if not text.strip():
            continue
        texts.append(text)
        # use your source as a stable document id (or hash it)
        ids.append(src if src else None)

    # LightRAG supports list insert + custom IDs
    # If you pass IDs, lengths MUST match.
    await rag.ainsert(texts, ids=ids)


async def main():
    # rag = await init_rag()
    # print("Reading corpus…")
    # records = list(iter_jsonl(CORPUS))
    # print(f"Inserting {len(records)} docs…")
    # await insert_all(rag, records)

    # # (Optional) quick sanity query
    # ans = await rag.aquery(
    #     "What are the main topics covered in this corpus?",
    #     param=QueryParam(mode="hybrid"),  # "local" | "global" | "hybrid" | "mix"
    # )
    # print(ans)

    rag = await init_rag()

    ans = await rag.aquery(
        "What is this corpus about?",  # "What are the main topics covered in this corpus?",
        param=QueryParam(mode="hybrid"),
    )
    print(ans)


if __name__ == "__main__":
    asyncio.run(main())
