# quick_test.py
from rag_helpers import answer_with_graphrag


from dotenv import load_dotenv

load_dotenv()

question = "What are the main topics in this corpus?"
answer, contexts = answer_with_graphrag(
    question, llm=None, knowledge_index={"mode": "hybrid"}
)
# print(f"Answer: {answer[:200]}...")
print(f"Answer: {answer}...")
print(f"Got {len(contexts)} contexts")
