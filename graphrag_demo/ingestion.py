import os
from dotenv import load_dotenv

from lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_share_data, initialize_pipeline_status

load_dotenv()


async def initialize_rag(working_dir: str = "./rag_cache") -> LightRAG:
    """
    Initialize LightRAG with vector and Neo4j graph storage,
    and prepare shared pipeline status to avoid KeyError.
    """
    rag = LightRAG(
        working_dir=working_dir,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
        graph_storage="Neo4JStorage",
        vector_storage="FaissVectorDBStorage",
        chunk_token_size=600,
        chunk_overlap_token_size=200,
    )
    await rag.initialize_storages()

    initialize_share_data()
    await initialize_pipeline_status()

    return rag


async def index_data(rag: LightRAG, file_path: str) -> None:
    """
    Index a text file into LightRAG, tagging chunks with its filename.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    await rag.ainsert(input=text, file_paths=[file_path])


async def index_file(rag: LightRAG, path: str) -> None:
    """
    Alias for index_data to mirror sync naming.
    """
    await index_data(rag, path)
