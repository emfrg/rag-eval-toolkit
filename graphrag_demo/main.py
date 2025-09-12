from ingestion import initialize_rag, index_file
from retrieve import run_async_query
from dotenv import load_dotenv
import asyncio

load_dotenv()


async def get_rag_response(
    question: str, mode: str, data_path: str = "data/data.txt"
) -> str:
    """
    1. Initialize RAG.
    2. Index file (chunk and insert into vector store and knowledge graph).
    3. Run async query and return the response.
    """
    rag = await initialize_rag()
    await index_file(rag, data_path)
    resp = await run_async_query(rag, question, mode)
    return resp


def main() -> None:
    mode = "mix"
    while True:
        question = input("\nAsk a question (or type 'exit' to quit): ").strip()
        if question.lower() in {"exit", "quit"}:
            break

        resp = asyncio.run(get_rag_response(question=question, mode=mode))
        print("\n===== Query Result =====\n")
        print(resp)


if __name__ == "__main__":
    main()
