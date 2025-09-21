# example_usage.py
from rag_system import RAGConfig, RAGSystem, RAGDataset
from dotenv import load_dotenv

load_dotenv()

raise NotImplementedError("This file is not implemented yet")

# Load your dataset
dataset = RAGDataset.from_dataset_dir(
    "data/eval_datasets/yixuantt_MultiHopRAG_89ba9d15"
)

# Configure RAG
config = RAGConfig(
    chunking=False,  # MultiHopRAG doesn't need chunking
    embedding_model="text-embedding-3-small",
    use_reranker=False,
    similarity_threshold=0.8,  # (FAISS distance) lower better 0 - 2
    rerank_threshold=0.5,  # (BGE cross-encoder) higher better 0 - 1
    llm_model="gpt-4o-mini",
)

# Initialize RAG with dataset
rag = RAGSystem(config, dataset)

# Single query example
answer, docs = rag.query("Who is the CEO ousted from OpenAI?")
print(f"Answer: {answer}")
print(f"Retrieved {len(docs)} documents")
for i, doc in enumerate(docs[:3], 1):
    print(f"  Doc {i}: {doc.page_content[:100]}...")

# Process multiple questions
questions = dataset.load_questions()
results = []

for q in questions[:3]:
    answer, retrieved_docs = rag.query(q["question"])

    # Collect all information
    result = {
        "question_id": q["question_id"],
        "question": q["question"],
        "generated_answer": answer,
        "retrieved_docs": [
            {
                "content": doc.page_content,
                "doc_id": doc.metadata.get("doc_id", "unknown"),
                "metadata": doc.metadata,
            }
            for doc in retrieved_docs
        ],
    }
    results.append(result)

    # Display
    print(f"\n{'='*50}")
    print(f"Question ID: {result['question_id']}")
    print(f"Question: {result['question'][:100]}...")
    print(f"Answer: {result['generated_answer']}")
    print(f"Retrieved {len(result['retrieved_docs'])} documents")

# Save results if needed
import json

with open("rag_results.json", "w") as f:
    json.dump(results, f, indent=2)
