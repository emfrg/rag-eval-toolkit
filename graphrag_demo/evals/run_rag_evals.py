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
import datasets
from tqdm import tqdm
from langchain.docstore.document import Document as LangchainDocument
from langchain_openai import ChatOpenAI

# Import shared functions
from rag_helpers import (
    read_jsonl,
    load_embeddings,
    answer_with_rag,
    run_rag_tests,
    evaluate_answers,
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
        "chunk": 400,
        "reader": "gpt-4o-mini",
        "rerank": False,
        "embeddings": "text-embedding-3-small",
    },
    {
        "chunk": 400,
        "reader": "gpt-4o",
        "rerank": False,
        "embeddings": "text-embedding-3-small",
    },
]

# Run tests for each configuration
for config in CONFIGS:
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
        RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0") if rerank else None
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

# Filter configs
config_files = []
for config in CONFIGS:
    settings_name = f"chunk:{config['chunk']}_embeddings:{config['embeddings'].replace('/', '~')}_rerank:{config['rerank']}_reader:{config['reader']}"
    config_files.append(f"output/rag_{settings_name}.json")
filtered_scores = average_scores[average_scores.index.isin(config_files)]

# Convert to %
scores_percentage = filtered_scores * 100


# Build DataFrame for plotting
def create_label(filepath):
    pattern = r"chunk:(\d+)_embeddings:([^_]+)_rerank:(\w+)_reader:([^.]+)"
    match = re.search(pattern, filepath)
    if match:
        chunk, embeddings, rerank, reader = match.groups()
        rerank_str = "w/ rerank" if rerank == "True" else ""
        return f"Chunk {chunk}, {reader}{', ' + rerank_str if rerank_str else ''}"
    return filepath


df_plot = pd.DataFrame(
    {
        "Configuration": [create_label(idx) for idx in scores_percentage.index],
        "Accuracy": scores_percentage.values,
        "Chunk Size": [
            int(re.search(r"chunk:(\d+)", idx).group(1))
            for idx in scores_percentage.index
        ],
        "Reader Model": [
            re.search(r"reader:([^.]+)", idx).group(1)
            for idx in scores_percentage.index
        ],
    }
).sort_values("Accuracy", ascending=True)

# Plot and export
fig, ax = plt.subplots(figsize=(12, 4 + 0.4 * len(df_plot)))

y = df_plot["Configuration"]
x = df_plot["Accuracy"]
ax.barh(y, x)

# Annotate bars with percentages
for i, v in enumerate(x):
    ax.text(v + 1, i, f"{v:.1f}%", va="center")

ax.set_title("RAG Performance: Chunk Size and Reader Model Comparison")
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
