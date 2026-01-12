# RAG Eval Toolkit

A complete RAG (Retrieval-Augmented Generation) experimentation framework for evaluating and comparing RAG architectures.

## Features

- **Multiple RAG Architectures**: Built-in support for Naive RAG and GraphRAG (LightRAG)
- **Pluggable System**: Easy to add custom RAG implementations
- **RAGAS Metrics**: Evaluate with Faithfulness, Context Recall, Factual Correctness, and more
- **Configuration-based Experiments**: Compare different configs across architectures
- **CLI Tool**: Run experiments from the command line
- **Rich Reporting**: Beautiful console output and JSON export

## Installation

```bash
# Basic installation
uv add rag-eval-toolkit

# With GraphRAG support
uv add rag-eval-toolkit[graphrag]

# Development installation (from source)
uv sync --all-extras
```

## Quick Start

### Command Line

```bash
# Run a simple evaluation
rag-eval run -c corpus.jsonl -d questions.jsonl -s naive

# Run with GraphRAG
rag-eval run -c corpus.jsonl -d questions.jsonl -s graphrag

# Run on a sample for quick testing
rag-eval run -c corpus.jsonl -d questions.jsonl -n 10

# Compare multiple configurations
rag-eval compare -c corpus.jsonl -d questions.jsonl -C configs.json
```

### Python API

```python
from rag_eval import Corpus, EvalDataset, RAGConfig
from rag_eval.systems.config import NaiveRAGConfig
from rag_eval.systems.implementations import NaiveRAGSystem
from rag_eval.evaluator import ExperimentRunner

# Load data
corpus = Corpus.from_jsonl("corpus.jsonl")
eval_dataset = EvalDataset.from_jsonl("questions.jsonl")

# Configure RAG system (uses Anthropic Claude by default)
config = RAGConfig(
    rag_type="naive",
    llm_provider="anthropic",  # or "openai"
    llm_model="claude-sonnet-4-20250514",  # or "gpt-4o-mini" for OpenAI
    naive=NaiveRAGConfig(
        k_retrieve=10,
        use_reranker=True,
    ),
)

# Run evaluation
runner = ExperimentRunner(output_dir="./results")
summary = runner.run_experiments([config], corpus, eval_dataset)

# Find best config
best = summary.find_best("faithfulness")
print(f"Best faithfulness: {best.scores['faithfulness']:.3f}")
```

## Data Formats

### Corpus (JSONL)

```json
{
  "doc_id": "doc_001",
  "content": "The capital of France is Paris.",
  "metadata": {
    "source": "Wikipedia",
    "category": "geography"
  }
}
```

### Evaluation Dataset (JSONL)

```json
{
  "question_id": "q_001",
  "question": "What is the capital of France?",
  "answer": "Paris",
  "question_type": "factoid",
  "required_evidence": ["doc_001"],
  "evidence_count": 1
}
```

### Config File (JSON)

```json
[
  {
    "rag_type": "naive",
    "llm_model": "gpt-4o-mini",
    "naive": {
      "k_retrieve": 5,
      "use_reranker": false
    }
  },
  {
    "rag_type": "naive",
    "llm_model": "gpt-4o-mini",
    "naive": {
      "k_retrieve": 10,
      "use_reranker": true
    }
  },
  {
    "rag_type": "graphrag"
  }
]
```

## Custom RAG Systems

Implement the `RAGSystemBase` protocol to create your own RAG system:

```python
from rag_eval.systems.base import RAGSystemBase, IndexReport
from rag_eval.systems.response import RAGResponse, RetrievedDocument

class MyCustomRAG(RAGSystemBase):
    def __init__(self, config):
        super().__init__(config)
        # Your initialization

    def create_index(self, corpus):
        # Build your index
        return IndexReport(
            total_documents=len(corpus),
            indexed_documents=len(corpus),
            reused_existing=False,
        )

    def load_index(self):
        # Load existing index
        self._index_loaded = True

    def retrieve(self, query):
        # Return list of RetrievedDocument
        return [
            RetrievedDocument(
                doc_id="doc_001",
                content="...",
                score=0.95,
            )
        ]

    def generate(self, query, contexts):
        # Generate answer from contexts
        return "Generated answer"

    def query(self, question):
        docs = self.retrieve(question)
        answer = self.generate(question, [d.content for d in docs])
        return RAGResponse.from_documents(answer, docs)
```

Use your custom system:

```python
from rag_eval.evaluator.runner import ExperimentRunner

runner = ExperimentRunner()
summary = runner.run_experiments(
    configs=[config],
    corpus=corpus,
    eval_dataset=eval_dataset,
    rag_factories={"custom": MyCustomRAG},
)
```

## Metrics

The toolkit uses RAGAS metrics by default:

- **Faithfulness**: Does the answer faithfully use the retrieved context?
- **Context Recall**: Are all relevant documents retrieved?
- **Factual Correctness**: Is the answer factually correct?
- **Semantic Similarity**: How similar is the answer to the reference?

## Project Structure

```
rag_eval_toolkit/
├── src/rag_eval/
│   ├── dataset/          # Corpus and EvalDataset classes
│   ├── systems/          # RAG system implementations
│   │   ├── base.py       # RAGSystemBase protocol
│   │   ├── config.py     # Configuration classes
│   │   └── implementations/
│   │       ├── naive.py  # Naive RAG (FAISS)
│   │       └── graphrag.py  # GraphRAG (LightRAG)
│   ├── evaluator/        # Evaluation framework
│   │   ├── metrics.py    # RAGAS integration
│   │   ├── runner.py     # Experiment runner
│   │   └── results.py    # Result classes
│   ├── reporting/        # Console and JSON output
│   └── cli.py            # Command-line interface
├── examples/
└── tests/
```

## Environment Variables

Create a `.env` file in your project root:

```
ANTHROPIC_API_KEY=your-anthropic-key
OPENAI_API_KEY=your-openai-key
```

The toolkit automatically loads this file using `python-dotenv`.

- **ANTHROPIC_API_KEY**: Used for LLM generation (Claude is the default)
- **OPENAI_API_KEY**: Required for embeddings and GraphRAG indexing

## License

MIT
