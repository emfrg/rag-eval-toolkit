# #!/usr/bin/env python3
# # rag_evaluator/new_runner.py
# from __future__ import annotations

# from pathlib import Path
# from typing import Any, Dict, List, Optional

# from rag_evaluator import ExperimentRunner


# def build_configs(
#     *, rag_cache: str, graph_cache: str, force_graphrag: bool
# ) -> List[Dict[str, Any]]:
#     """
#     Return default configurations for both backends (two per architecture).
#     """
#     naive_config: Dict[str, Any] = {
#         "rag_model": "naive",
#         "llm_model": "gpt-4o-mini",
#         "temperature": 1.0,
#         "max_tokens": 80,
#         "naive": {
#             "chunk_documents": False,
#             "chunk_size": 400,
#             "chunk_overlap": 50,
#             "embedding_model": "text-embedding-3-small",
#             "vector_store": "faiss",
#             "k_retrieve": 20,
#             "similarity_threshold": 1.0,
#             "use_reranker": False,  # True,
#             "reranker_model": "BAAI/bge-reranker-base",
#             "rerank_threshold": 0.5,
#             "min_docs": 0,  # 0
#             "max_docs": 20,  # 4
#             "cache_dir": rag_cache,
#             "inline_metadata": False,
#         },
#     }

#     naive_chunked_config: Dict[str, Any] = {
#         "rag_model": "naive",
#         "llm_model": "gpt-4o-mini",
#         "temperature": 1.0,
#         "max_tokens": 80,
#         "naive": {
#             "chunk_documents": False,
#             "chunk_size": 600,
#             "chunk_overlap": 100,
#             "embedding_model": "text-embedding-3-small",
#             "vector_store": "faiss",
#             "k_retrieve": 50,
#             "similarity_threshold": 1.0,
#             "use_reranker": True,
#             "reranker_model": "BAAI/bge-reranker-base",
#             "rerank_threshold": 0.5,
#             "min_docs": 0,
#             "max_docs": 20,
#             "cache_dir": rag_cache,
#             "inline_metadata": False,
#         },
#     }

#     naive_meta_config: Dict[str, Any] = {
#         "rag_model": "naive",
#         "llm_model": "gpt-4o-mini",
#         "temperature": 1.0,
#         "max_tokens": 80,
#         "naive": {
#             "chunk_documents": False,
#             "chunk_size": 400,
#             "chunk_overlap": 50,
#             "embedding_model": "text-embedding-3-small",
#             "vector_store": "faiss",
#             "k_retrieve": 20,
#             "similarity_threshold": 1.0,
#             "use_reranker": False,  # True,
#             "reranker_model": "BAAI/bge-reranker-base",
#             "rerank_threshold": 0.5,
#             "min_docs": 0,  # 0
#             "max_docs": 20,  # 4
#             "cache_dir": rag_cache,
#             "inline_metadata": True,
#         },
#     }

#     graphrag_config: Dict[str, Any] = {
#         "rag_model": "graphrag",
#         "llm_model": "gpt-4o-mini",
#         "temperature": 1.0,
#         "max_tokens": 80,
#         "graphrag": {
#             "indexing": {
#                 "graph_cache_dir": graph_cache,
#                 "max_parallel_insert": 4,
#                 "batch_size": 128,
#                 "force_reindex": force_graphrag,
#                 "inline_metadata": False,
#             },
#             "query": {
#                 "mode": "hybrid",
#                 "top_k": 20,
#                 "summary_top_k": None,
#             },
#         },
#     }

#     graphrag_semantic_config: Dict[str, Any] = {
#         "rag_model": "graphrag",
#         "llm_model": "gpt-4o-mini",
#         "temperature": 1.0,
#         "max_tokens": 80,
#         "graphrag": {
#             "indexing": {
#                 "graph_cache_dir": graph_cache,
#                 "max_parallel_insert": 4,
#                 "batch_size": 128,
#                 "force_reindex": force_graphrag,
#                 "inline_metadata": False,
#             },
#             "query": {
#                 "mode": "mix",
#                 "top_k": 50,
#                 "summary_top_k": 10,
#             },
#         },
#     }

#     graphrag_meta_config: Dict[str, Any] = {
#         "rag_model": "graphrag",
#         "llm_model": "gpt-4o-mini",
#         "temperature": 1.0,
#         "max_tokens": 80,
#         "graphrag": {
#             "indexing": {
#                 "graph_cache_dir": graph_cache,
#                 "max_parallel_insert": 4,
#                 "batch_size": 128,
#                 "force_reindex": force_graphrag,
#                 "inline_metadata": True,
#             },
#             "query": {
#                 "mode": "mix",
#                 "top_k": 50,
#                 "summary_top_k": 10,
#             },
#         },
#     }

#     return [
#         naive_config,
#         naive_chunked_config,
#         naive_meta_config,
#         graphrag_config,
#         graphrag_semantic_config,
#         graphrag_meta_config,
#     ]


# def run(
#     *,
#     dataset_dir: str,
#     questions_file: Optional[str],
#     output_dir: str,
#     experiment_name: Optional[str],
#     reuse_latest: bool,
#     reuse_cached_scores: bool,
#     rag_cache: str,
#     graph_cache: str,
#     force_graphrag: bool,
# ) -> List[Dict[str, Any]]:
#     """
#     Execute ExperimentRunner with the two default configs and return its results payload.
#     """
#     runner = ExperimentRunner(output_dir=output_dir)
#     configs = build_configs(
#         rag_cache=rag_cache, graph_cache=graph_cache, force_graphrag=force_graphrag
#     )
#     results = runner.run_experiment(
#         dataset_dir=dataset_dir,
#         rag_configs=configs,
#         experiment_name=experiment_name,
#         questions_file=questions_file,
#         reuse_latest=reuse_latest,
#         reuse_cached_scores=reuse_cached_scores,
#     )
#     return results


#!/usr/bin/env python3

# rag_evaluator/new_runner.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    # Preferred when running within the rag_evaluator package
    from .runner import ExperimentRunner
except ImportError:  # pragma: no cover - fallback when executed flat
    from rag_evaluator.runner import ExperimentRunner


def build_configs(
    *, rag_cache: str, graph_cache: str, force_graphrag: bool
) -> List[Dict[str, Any]]:
    """
    Return default configurations for both backends (two per architecture).
    """
    naive_config: Dict[str, Any] = {
        "rag_model": "naive",
        "llm_model": "gpt-4o-mini",
        "temperature": 1.0,
        "max_tokens": 80,
        "naive": {
            "chunk_documents": False,
            "chunk_size": 400,
            "chunk_overlap": 50,
            "embedding_model": "text-embedding-3-small",
            "vector_store": "faiss",
            "k_retrieve": 20,
            "similarity_threshold": 1.0,
            "use_reranker": False,  # True,
            "reranker_model": "BAAI/bge-reranker-base",
            "rerank_threshold": 0.5,
            "min_docs": 0,  # 0
            "max_docs": 20,  # 4
            "cache_dir": rag_cache,
            "inline_metadata": False,
        },
    }

    naive_meta_config: Dict[str, Any] = {
        "rag_model": "naive",
        "llm_model": "gpt-4o-mini",
        "temperature": 1.0,
        "max_tokens": 80,
        "naive": {
            "chunk_documents": False,
            "chunk_size": 400,
            "chunk_overlap": 50,
            "embedding_model": "text-embedding-3-small",
            "vector_store": "faiss",
            "k_retrieve": 20,
            "similarity_threshold": 1.0,
            "use_reranker": False,  # True,
            "reranker_model": "BAAI/bge-reranker-base",
            "rerank_threshold": 0.5,
            "min_docs": 0,  # 0
            "max_docs": 20,  # 4
            "cache_dir": rag_cache,
            "inline_metadata": True,
        },
    }

    # naive_meta_rerank_config: Dict[str, Any] = {
    #     "rag_model": "naive",
    #     "llm_model": "gpt-4o-mini",
    #     "temperature": 1.0,
    #     "max_tokens": 80,
    #     "naive": {
    #         "chunk_documents": False,
    #         "chunk_size": 400,
    #         "chunk_overlap": 50,
    #         "embedding_model": "text-embedding-3-small",
    #         "vector_store": "faiss",
    #         "k_retrieve": 20,
    #         "similarity_threshold": 1.0,
    #         "use_reranker": True,  # True,
    #         "reranker_model": "BAAI/bge-reranker-base",
    #         "rerank_threshold": 0.5,
    #         "min_docs": 0,  # 0
    #         "max_docs": 20,  # 4
    #         "cache_dir": rag_cache,
    #         "inline_metadata": True,
    #     },
    # }

    graphrag_config: Dict[str, Any] = {
        "rag_model": "graphrag",
        "llm_model": "gpt-4o-mini",
        "temperature": 1.0,
        "max_tokens": 80,
        "graphrag": {
            "indexing": {
                "graph_cache_dir": graph_cache,
                "max_parallel_insert": 4,
                "batch_size": 128,
                "force_reindex": force_graphrag,
                "inline_metadata": False,
            },
            "query": {
                "mode": "hybrid",
                "top_k": 20,
                "summary_top_k": None,
            },
        },
    }

    graphrag_meta_config: Dict[str, Any] = {
        "rag_model": "graphrag",
        "llm_model": "gpt-4o-mini",
        "temperature": 1.0,
        "max_tokens": 80,
        "graphrag": {
            "indexing": {
                "graph_cache_dir": graph_cache,
                "max_parallel_insert": 4,
                "batch_size": 128,
                "force_reindex": force_graphrag,
                "inline_metadata": True,
            },
            "query": {
                "mode": "hybrid",
                "top_k": 20,
                "summary_top_k": None,
            },
        },
    }

    graphrag_top50_config: Dict[str, Any] = {
        "rag_model": "graphrag",
        "llm_model": "gpt-4o-mini",
        "temperature": 1.0,
        "max_tokens": 80,
        "graphrag": {
            "indexing": {
                "graph_cache_dir": graph_cache,
                "max_parallel_insert": 4,
                "batch_size": 128,
                "force_reindex": force_graphrag,
                "inline_metadata": False,
            },
            "query": {
                "mode": "hybrid",
                "top_k": 50,
                "summary_top_k": None,
            },
        },
    }

    graphrag_meta_top50_config: Dict[str, Any] = {
        "rag_model": "graphrag",
        "llm_model": "gpt-4o-mini",
        "temperature": 1.0,
        "max_tokens": 80,
        "graphrag": {
            "indexing": {
                "graph_cache_dir": graph_cache,
                "max_parallel_insert": 4,
                "batch_size": 128,
                "force_reindex": force_graphrag,
                "inline_metadata": True,
            },
            "query": {
                "mode": "hybrid",
                "top_k": 50,
                "summary_top_k": None,
            },
        },
    }

    # graphrag_mix_config: Dict[str, Any] = {
    #     "rag_model": "graphrag",
    #     "llm_model": "gpt-4o-mini",
    #     "temperature": 1.0,
    #     "max_tokens": 80,
    #     "graphrag": {
    #         "indexing": {
    #             "graph_cache_dir": graph_cache,
    #             "max_parallel_insert": 4,
    #             "batch_size": 128,
    #             "force_reindex": force_graphrag,
    #             "inline_metadata": False,
    #         },
    #         "query": {
    #             "mode": "mix",
    #             "top_k": 50,
    #             "summary_top_k": 10,
    #         },
    #     },
    # }

    return [
        # naive_config,
        naive_meta_config,
        # ---
        # naive_meta_rerank_config,
        graphrag_config,
        graphrag_meta_config,
        # graphrag_top50_config,
        # graphrag_meta_top50_config,
        # graphrag_semantic_config,
    ]


def run(
    *,
    dataset_dir: str,
    questions_file: Optional[str],
    output_dir: str,
    experiment_name: Optional[str],
    reuse_latest: bool,
    reuse_cached_scores: bool,
    rag_cache: str,
    graph_cache: str,
    force_graphrag: bool,
) -> List[Dict[str, Any]]:
    """
    Execute ExperimentRunner with the two default configs and return its results payload.
    """
    runner = ExperimentRunner(output_dir=output_dir)
    configs = build_configs(
        rag_cache=rag_cache, graph_cache=graph_cache, force_graphrag=force_graphrag
    )
    results = runner.run_experiment(
        dataset_dir=dataset_dir,
        rag_configs=configs,
        experiment_name=experiment_name,
        questions_file=questions_file,
        reuse_latest=reuse_latest,
        reuse_cached_scores=reuse_cached_scores,
    )
    return results
