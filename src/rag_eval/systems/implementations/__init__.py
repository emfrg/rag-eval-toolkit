"""Built-in RAG system implementations."""

# Implementations are imported lazily to avoid dependency issues
# Users who don't need GraphRAG shouldn't need to install lightrag

__all__ = ["NaiveRAGSystem", "GraphRAGSystem"]


def __getattr__(name: str):
    if name == "NaiveRAGSystem":
        from rag_eval.systems.implementations.naive import NaiveRAGSystem
        return NaiveRAGSystem
    elif name == "GraphRAGSystem":
        from rag_eval.systems.implementations.graphrag import GraphRAGSystem
        return GraphRAGSystem
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
