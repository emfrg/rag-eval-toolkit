# run_rag_evals
# Auto-save final figure to results/<script_stem>.img (PNG)
# Supports Matplotlib (guaranteed PNG export).
import os, atexit
from pathlib import Path

_script_stem = Path(__file__).stem
_results_dir = Path("results")
_results_dir.mkdir(parents=True, exist_ok=True)
_output_path = _results_dir / f"{_script_stem}.img"

# Matplotlib support
try:
    import matplotlib.pyplot as plt

    def _save_last_matplotlib():
        figs = [plt.figure(num) for num in plt.get_fignums()]
        if figs:
            figs[-1].savefig(_output_path, format="png", bbox_inches="tight")

    atexit.register(_save_last_matplotlib)
except Exception:
    pass


from dotenv import load_dotenv

load_dotenv()

import os
import json
import datasets
from tqdm import tqdm
from langchain.docstore.document import Document as LangchainDocument
from langchain_openai import ChatOpenAI

import warnings
import torch

# Suppress the specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="colbert")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.amp")


# Import shared functions
from rag_helpers import (
    read_jsonl,
    load_embeddings,
    answer_with_rag,
    run_rag_tests,
    evaluate_answers,
    answer_with_graphrag,
    RAG_PROMPT_TEMPLATE,
    EVALUATION_PROMPT,
)

load_dir = "datasets_local/20250914_145157"

# Load initial corpus
ds = read_jsonl(os.path.join(load_dir, "initial_corpus.jsonl"))
RAW_KNOWLEDGE_BASE = [
    LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]})
    for doc in tqdm(ds)
]

# Load eval dataset
eval_path_hf = os.path.join(load_dir, "eval_dataset_hf.jsonl")
if os.path.exists(eval_path_hf):
    eval_dataset = datasets.Dataset.from_list(read_jsonl(eval_path_hf))
    print(f"Loaded {len(eval_dataset)} eval questions from HuggingFace dataset")
else:
    print("Downloading HuggingFace eval dataset...")
    eval_dataset = datasets.load_dataset("m-ric/huggingface_doc_qa_eval", split="train")
    from rag_helpers import write_jsonl

    write_jsonl(eval_path_hf, [item for item in eval_dataset])

from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage

evaluation_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="You are a fair evaluator language model."),
        HumanMessagePromptTemplate.from_template(EVALUATION_PROMPT),
    ]
)

eval_chat_model = ChatOpenAI(model="gpt-4o", temperature=0)
evaluator_name = "GPT4"


from ragatouille import RAGPretrainedModel

# Create output directory if it doesn't exist
if not os.path.exists("output"):
    os.mkdir("output")

# Define reader models
READER_MODELS = {
    "gpt-4o-mini": ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=512),
    "gpt-4o": ChatOpenAI(model="gpt-4o", temperature=0.1, max_tokens=512),
}

# Minimal configuration - focus on what matters most
CONFIGS = [
    {
        "method": "graphrag",
        "mode": "hybrid",
        "rerank": False,
        "reader": "gpt-4o-mini",
    },
    {
        "chunk": 400,
        "reader": "gpt-4o-mini",
        "rerank": True,
        "embeddings": "text-embedding-3-small",
    },
    # {
    #     "chunk": 400,
    #     "reader": "gpt-4o-mini",
    #     "rerank": False,
    #     "embeddings": "text-embedding-3-small",
    # },
    # {
    #     "chunk": 400,
    #     "reader": "gpt-4o",
    #     "rerank": False,
    #     "embeddings": "text-embedding-3-small",
    # },
    # {
    #     "chunk": 400,
    #     "reader": "gpt-4o",
    #     "rerank": True,
    #     "embeddings": "text-embedding-3-small",
    # },
    # {
    #     "chunk": 400,
    #     "reader": "gpt-4o",
    #     "rerank": False,
    #     "embeddings": "text-embedding-3-small",
    # },
    # {
    #     "chunk": 400,
    #     "reader": "gpt-4o",
    #     "rerank": True,
    #     "embeddings": "text-embedding-3-small",
    # },
    # GraphRAG configs
    # {
    #     "method": "graphrag",
    #     "mode": "mix",
    #     "rerank": False,
    #     "reader": "gpt-4o-mini",
    # },
    # {
    #     "method": "graphrag",
    #     "mode": "mix",
    #     "rerank": True,
    #     "reader": "gpt-4o-mini",
    # },
    # {
    #     "method": "graphrag",
    #     "mode": "hybrid",
    #     "rerank": True,
    #     "reader": "gpt-4o-mini",
    # },
    # {
    #     "method": "graphrag",
    #     "mode": "mix",
    #     "rerank": False,
    #     "reader": "gpt-4o",
    # },
    # {
    #     "method": "graphrag",
    #     "mode": "mix",
    #     "rerank": True,
    #     "reader": "gpt-4o",
    # },
    # {
    #     "method": "graphrag",
    #     "mode": "local",
    #     "rerank": False,
    #     "reader": "gpt-4o-mini",
    # },
    # {
    #     "method": "graphrag",
    #     "mode": "global",
    #     "rerank": False,
    #     "reader": "gpt-4o-mini",
    # },
    # {
    #     "method": "graphrag",
    #     "mode": "hybrid",
    #     "rerank": False,
    #     "reader": "gpt-4o-mini",
    # },
    # {
    #     "method": "graphrag",
    #     "mode": "hybrid",
    #     "rerank": True,
    #     "reader": "gpt-4o-mini",
    # },
    # {
    #     "method": "graphrag",
    #     "mode": "hybrid",
    #     "rerank": True,
    #     "reader": "gpt-4o",
    # },
    # {
    #     "method": "graphrag",
    #     "mode": "mix",
    #     "rerank": True,
    #     "reader": "gpt-4o",
    # },
]

# Run tests for each configuration
for config in CONFIGS:
    method = config.get("method", "classic")

    if method == "graphrag":
        # GraphRAG configuration
        mode = config["mode"]
        reader_name = config["reader"]  # Get the reader name
        rerank = config.get("rerank", False)  # Get rerank setting

        settings_name = f"graphrag_mode:{mode}_rerank:{rerank}_reader:{reader_name}"
        output_file_name = f"output/rag_{settings_name}.json"

        print(f"\n{'='*60}")
        print(f"Running GraphRAG configuration: {settings_name}")
        print(f"{'='*60}")

        print(
            f"Using GraphRAG with mode: {mode}, reader: {reader_name}, rerank: {rerank}"
        )

        # Pass the actual LLM and reranker like classic RAG!
        knowledge_index = {"mode": mode}
        reader_llm = READER_MODELS[reader_name]  # Use the same LLM!
        reranker = (
            RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
            if rerank
            else None
        )

        print("Running GraphRAG tests...")
        try:
            with open(output_file_name, "r") as f:
                outputs = json.load(f)
        except:
            outputs = []

        for example in tqdm(eval_dataset):
            question = example["question"]
            if question in [output["question"] for output in outputs]:
                continue

            # Use GraphRAG with proper LLM and reranker
            answer, relevant_docs = answer_with_graphrag(
                question,
                reader_llm,  # Pass actual LLM instance
                knowledge_index,
                reranker=reranker,  # Pass reranker if enabled!
            )

            result = {
                "question": question,
                "true_answer": example["answer"],
                "source_doc": example["source_doc"],
                "generated_answer": answer,
                "retrieved_docs": relevant_docs,
                "test_settings": settings_name,
            }
            outputs.append(result)

            with open(output_file_name, "w") as f:
                json.dump(outputs, f)
    else:
        # Classic RAG configuration
        chunk_size = config["chunk"]
        reader_name = config["reader"]
        rerank = config["rerank"]
        embeddings = config["embeddings"]

        reader_llm = READER_MODELS[reader_name]
        settings_name = f"chunk:{chunk_size}_embeddings:{embeddings.replace('/', '~')}_rerank:{rerank}_reader:{reader_name}"
        output_file_name = f"output/rag_{settings_name}.json"

        print(f"\n{'='*60}")
        print(f"Running configuration: {settings_name}")
        print(f"{'='*60}")

        print("Loading knowledge base embeddings...")
        knowledge_index = load_embeddings(
            RAW_KNOWLEDGE_BASE, chunk_size=chunk_size, embedding_model_name=embeddings
        )

        print("Setting up reranker..." if rerank else "Skipping reranker...")
        reranker = (
            RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
            if rerank
            else None
        )

        print("Running RAG tests...")
        run_rag_tests(
            eval_dataset=eval_dataset,
            llm=reader_llm,
            knowledge_index=knowledge_index,
            output_file=output_file_name,
            reranker=reranker,
            verbose=False,
            test_settings=settings_name,
        )

    # Evaluate for both methods
    print("Evaluating answers...")
    evaluate_answers(
        output_file_name, eval_chat_model, evaluator_name, evaluation_prompt_template
    )

    print(f"✓ Completed: {settings_name}")

print(f"\n{'='*60}")
print("All configurations completed!")
print(f"{'='*60}")


import glob, json, regex as re
import pandas as pd
import matplotlib.pyplot as plt  # Matplotlib plotting

# Load all results
outputs = []
for file in glob.glob("output/*.json"):
    output = pd.DataFrame(json.load(open(file, "r")))
    output["settings"] = file
    outputs.append(output)

result = pd.concat(outputs)

# Process scores
result["eval_score_GPT4"] = result["eval_score_GPT4"].apply(
    lambda x: int(x) if isinstance(x, str) else 1
)
result["eval_score_GPT4"] = (result["eval_score_GPT4"] - 1) / 4

# Calculate average scores
average_scores = result.groupby("settings")["eval_score_GPT4"].mean()


# Filter configs - handle both classic and GraphRAG
config_files = []
for config in CONFIGS:
    method = config.get("method", "classic")
    if method == "graphrag":
        mode = config["mode"]
        reader_name = config["reader"]
        rerank = config.get("rerank", False)
        settings_name = f"graphrag_mode:{mode}_rerank:{rerank}_reader:{reader_name}"
    else:
        settings_name = f"chunk:{config['chunk']}_embeddings:{config['embeddings'].replace('/', '~')}_rerank:{config['rerank']}_reader:{config['reader']}"
    config_files.append(f"output/rag_{settings_name}.json")

filtered_scores = average_scores[average_scores.index.isin(config_files)]

# Convert to %
scores_percentage = filtered_scores * 100


# Build DataFrame for plotting
def create_label(filepath):
    if "graphrag" in filepath:
        # Extract GraphRAG settings
        match = re.search(r"graphrag_mode:(\w+)_rerank:(\w+)_reader:([^.]+)", filepath)
        if match:
            mode, rerank, reader = match.groups()
            rerank_str = " w/ rerank" if rerank == "True" else ""
            return f"GraphRAG ({mode}) {reader}{rerank_str}"

    else:
        # Original pattern for classic RAG
        pattern = r"chunk:(\d+)_embeddings:([^_]+)_rerank:(\w+)_reader:([^.]+)"
        match = re.search(pattern, filepath)
        if match:
            chunk, embeddings, rerank, reader = match.groups()
            rerank_str = "w/ rerank" if rerank == "True" else ""
            return f"Chunk {chunk}, {reader}{', ' + rerank_str if rerank_str else ''}"
    return filepath


# Create plot data with safe extraction of optional fields
plot_data = []
for idx in scores_percentage.index:
    config_dict = {
        "Configuration": create_label(idx),
        "Accuracy": scores_percentage[idx],
    }

    # Try to extract chunk size (only for classic RAG)
    chunk_match = re.search(r"chunk:(\d+)", idx)
    if chunk_match:
        config_dict["Chunk Size"] = int(chunk_match.group(1))
    else:
        config_dict["Chunk Size"] = 0  # or None for GraphRAG

    # Extract reader model or method
    if "graphrag" in idx:
        config_dict["Reader Model"] = "graphrag"
    else:
        reader_match = re.search(r"reader:([^.]+)", idx)
        config_dict["Reader Model"] = (
            reader_match.group(1) if reader_match else "unknown"
        )

    plot_data.append(config_dict)

df_plot = pd.DataFrame(plot_data).sort_values("Accuracy", ascending=True)

# Plot and export
fig, ax = plt.subplots(figsize=(12, 4 + 0.4 * len(df_plot)))

y = df_plot["Configuration"]
x = df_plot["Accuracy"]
ax.barh(y, x)

# Annotate bars with percentages
for i, v in enumerate(x):
    ax.text(v + 1, i, f"{v:.1f}%", va="center")

ax.set_title("RAG Performance: Classic vs GraphRAG Comparison")
ax.set_xlabel("Accuracy (%)")
ax.set_xlim(0, 100)
ax.invert_yaxis()  # highest at top after ascending sort
ax.grid(axis="x", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.show()

# Summary
print("\n" + "=" * 60)
print("EVALUATION SUMMARY")
print("=" * 60)
summary_df = df_plot[["Configuration", "Accuracy"]].copy()
summary_df["Accuracy"] = summary_df["Accuracy"].apply(lambda x: f"{x:.1f}%")
print(summary_df.sort_values("Configuration").to_string(index=False))

best_config = df_plot.loc[df_plot["Accuracy"].idxmax()]
print("\n" + "-" * 60)
print(f"✨ Best Configuration: {best_config['Configuration']}")
print(f"   Accuracy: {best_config['Accuracy']:.1f}%")
print("-" * 60)
