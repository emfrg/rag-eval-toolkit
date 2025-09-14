from lightrag import LightRAG, QueryParam


async def run_async_query(
    rag: LightRAG, question: str, mode: str, top_k: int = 5
) -> str:
    """
    Execute an async RAG query using .aquery
    """
    return await rag.aquery(question, param=QueryParam(mode=mode, top_k=top_k))
