# example_usage.py
from rag_system import RAGConfig, RAGSystem, RAGDataset

from dotenv import load_dotenv

load_dotenv()

# Load your dataset
dataset = RAGDataset.from_dataset_dir(
    "data/eval_datasets/yixuantt_MultiHopRAG_89ba9d15"
)

# Configure RAG
config = RAGConfig(
    chunk_size=400,
    embedding_model="text-embedding-3-small",
    use_reranker=True,
    k_retrieve=10,
    k_rerank=5,
    llm_model="gpt-4o-mini",
)

# Initialize RAG with dataset
rag = RAGSystem(config, dataset)

# Single query
answer, docs = rag.query("Who is the CEO ousted from OpenAI?")
print(f"Answer: {answer}")

# Batch evaluation
questions = dataset.load_questions()
for q in questions[:3]:
    answer, docs = rag.query(q["question"])
    print(f"\nQ: {q['question']}")
    print(f"Generated: {answer}")
    print(f"Ground Truth: {q['answer']}")
