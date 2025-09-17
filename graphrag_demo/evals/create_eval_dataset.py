from dotenv import load_dotenv

load_dotenv()

from tqdm.auto import tqdm
import pandas as pd
from typing import Optional, List, Tuple
import json
import datasets
import random
import datetime
import os

from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import from helpers
from rag_helpers import (
    read_jsonl,
    write_jsonl,
    write_csv,
    to_serializable,
    call_llm,
    QA_generation_prompt,
    question_groundedness_critique_prompt,
    question_relevance_critique_prompt,
    question_standalone_critique_prompt,
)

pd.set_option("display.max_colwidth", None)

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=500,
)

# !pip install -q torch transformers langchain sentence-transformers tqdm openpyxl openai pandas datasets langchain-community ragatouille

from tqdm.auto import tqdm
import pandas as pd
from typing import Optional, List, Tuple
import json
import datasets

pd.set_option("display.max_colwidth", None)

# from huggingface_hub import notebook_login

# notebook_login()

# # download the dataset from the hub
# ds = datasets.load_dataset("m-ric/huggingface_doc", split="train")

# -------------------------------------------------------------------------
# Fixed directory for loading and saving datasets
# -------------------------------------------------------------------------
from pathlib import Path

save_dir = Path("datasets_local/20250914_145157")
os.makedirs(save_dir, exist_ok=True)

ds = datasets.load_dataset(
    "json",
    data_files=str(save_dir / "initial_corpus.jsonl"),
    split="train",
)

print(ds)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument

langchain_docs = [
    LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]})
    for doc in tqdm(ds)
]


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    add_start_index=True,
    separators=["\n\n", "\n", ".", " ", ""],
)

docs_processed = []
for doc in langchain_docs:
    docs_processed += text_splitter.split_documents([doc])


print(call_llm(llm, "This is a test context"))

import random
from tqdm import tqdm

N_GENERATIONS = 10  # keep it low for testing

print(f"Generating {N_GENERATIONS} QA couples...")

outputs = []
for sampled_context in tqdm(
    random.sample(docs_processed, min(N_GENERATIONS, len(docs_processed)))
):
    output_QA_couple = call_llm(
        llm, QA_generation_prompt.format(context=sampled_context.page_content)
    )
    try:
        question = (
            output_QA_couple.split("Factoid question: ")[-1]
            .split("Answer: ")[0]
            .strip()
        )
        answer = output_QA_couple.split("Answer: ")[-1].strip()
        assert len(answer) < 300, "Answer is too long"
        outputs.append(
            {
                "context": sampled_context.page_content,
                "question": question,
                "answer": answer,
                "source_doc": sampled_context.metadata.get("source"),
            }
        )
    except Exception:
        continue

# Preview a few rows in console
if outputs:
    print(pd.DataFrame(outputs).head(5).to_string(index=False))

print("Generating critique for each QA couple...")
for output in tqdm(outputs):
    evaluations = {
        "groundedness": call_llm(
            llm,
            question_groundedness_critique_prompt.format(
                context=output["context"], question=output["question"]
            ),
        ),
        "relevance": call_llm(
            llm,
            question_relevance_critique_prompt.format(question=output["question"]),
        ),
        "standalone": call_llm(
            llm,
            question_standalone_critique_prompt.format(question=output["question"]),
        ),
    }
    try:
        for criterion, evaluation in evaluations.items():
            score, eval = (
                int(evaluation.split("Total rating: ")[-1].strip()),
                evaluation.split("Total rating: ")[-2].split("Evaluation: ")[1],
            )
            output.update(
                {
                    f"{criterion}_score": score,
                    f"{criterion}_eval": eval,
                }
            )
    except Exception:
        continue

import pandas as pd

pd.set_option("display.max_colwidth", None)

generated_questions = pd.DataFrame.from_dict(outputs)

print("Evaluation dataset before filtering:")
cols = [
    "question",
    "answer",
    "groundedness_score",
    "relevance_score",
    "standalone_score",
]
print(generated_questions[cols].to_string(index=False))


generated_questions = generated_questions.loc[
    (generated_questions["groundedness_score"] >= 4)
    & (generated_questions["relevance_score"] >= 4)
    & (generated_questions["standalone_score"] >= 4)
]

print("============================================")
print("Final evaluation dataset:")
print(generated_questions[cols].to_string(index=False))

eval_dataset = datasets.Dataset.from_pandas(
    generated_questions, split="train", preserve_index=False
)

# # download the dataset from the hub
# eval_dataset = datasets.load_dataset("m-ric/huggingface_doc_qa_eval", split="train")

# If present, also load eval dataset from json in the same fixed directory
try:
    eval_dataset = datasets.load_dataset(
        "json",
        data_files=str(save_dir / "eval_dataset_hf.jsonl"),
        split="train",
    )
except Exception:
    pass

# Save ALL datasets to the same fixed directory
if "ds" in globals():
    initial_corpus = [{"text": doc["text"], "source": doc["source"]} for doc in ds]
    write_jsonl(os.path.join(save_dir, "initial_corpus.jsonl"), initial_corpus)

if "docs_processed" in globals():
    write_jsonl(os.path.join(save_dir, "processed_docs.jsonl"), docs_processed)

if "outputs" in globals():
    write_jsonl(os.path.join(save_dir, "qa_generated_all.jsonl"), outputs)

if "generated_questions" in globals():
    outputs_filtered = generated_questions.to_dict("records")
    write_jsonl(os.path.join(save_dir, "qa_generated_filtered.jsonl"), outputs_filtered)

if "eval_dataset" in globals():
    eval_data = [item for item in eval_dataset]
    write_jsonl(os.path.join(save_dir, "eval_dataset_hf.jsonl"), eval_data)

print(str(save_dir))
