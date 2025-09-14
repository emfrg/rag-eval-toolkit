from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.docstore.document import Document as LangchainDocument
from langchain.schema import SystemMessage, HumanMessage
from typing import List, Optional, Tuple
import tiktoken
import os
import json
import csv


# File I/O helpers
def read_jsonl(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def to_serializable(x):
    try:
        json.dumps(x)
        return x
    except TypeError:
        if hasattr(x, "page_content"):
            return {
                "page_content": getattr(x, "page_content", None),
                "metadata": getattr(x, "metadata", {}),
            }
        if isinstance(x, dict):
            return {k: to_serializable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple, set)):
            return [to_serializable(i) for i in x]
        if hasattr(x, "__dict__"):
            return {k: to_serializable(v) for k, v in vars(x).items()}
        return repr(x)


def write_jsonl(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(to_serializable(r), ensure_ascii=False) + "\n")


def write_csv(path, records):
    if not records or not isinstance(records[0], dict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            pass
        return
    keys = sorted({k for r in records for k in r.keys()})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in keys})


# LLM helper
def call_llm(llm_client, prompt: str):
    response = llm_client.invoke(
        [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=prompt),
        ]
    )
    return response.content


# RAG functions
def split_documents(
    chunk_size: int,
    knowledge_base: List[LangchainDocument],
    tokenizer_name: str,
) -> List[LangchainDocument]:
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=tokenizer_name,
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n\n", "\n", ".", " ", ""],
        disallowed_special=(),  # to disable the special token check
    )
    docs_processed = []
    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])

    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)
    return docs_processed_unique


def load_embeddings(
    langchain_docs: List[LangchainDocument],
    chunk_size: int,
    embedding_model_name: Optional[str] = "text-embedding-3-small",
) -> FAISS:
    embedding_model = OpenAIEmbeddings(model=embedding_model_name)
    index_name = (
        f"index_chunk:{chunk_size}_embeddings:{embedding_model_name.replace('/', '~')}"
    )
    index_folder_path = f"data/indexes/{index_name}/"
    if os.path.isdir(index_folder_path):
        return FAISS.load_local(
            index_folder_path,
            embedding_model,
            allow_dangerous_deserialization=True,
        )
    else:
        print("Index not found, generating it...")
        docs_processed = split_documents(chunk_size, langchain_docs, "cl100k_base")

        # Batch the embedding creation to avoid token limits
        batch_size = 300  # Adjust based on your chunk size

        # Create initial index with first batch
        knowledge_index = FAISS.from_documents(
            docs_processed[:batch_size],
            embedding_model,
            distance_strategy=DistanceStrategy.COSINE,
        )

        # Add remaining documents in batches
        for i in range(batch_size, len(docs_processed), batch_size):
            batch = docs_processed[i : i + batch_size]
            print(f"Processing embedding batch {i//batch_size + 1}...")
            knowledge_index.add_documents(batch)

        knowledge_index.save_local(index_folder_path)
        return knowledge_index


def answer_with_rag(
    question: str,
    llm,
    knowledge_index,
    reranker: Optional[any] = None,
    num_retrieved_docs: int = 30,
    num_docs_final: int = 7,
) -> Tuple[str, List[str]]:
    relevant_docs = knowledge_index.similarity_search(
        query=question, k=num_retrieved_docs
    )
    relevant_docs_text = [doc.page_content for doc in relevant_docs]

    if reranker:
        reranked = reranker.rerank(question, relevant_docs_text, k=num_docs_final)
        relevant_docs_text = [doc["content"] for doc in reranked]

    relevant_docs_text = relevant_docs_text[:num_docs_final]

    context = "\nExtracted documents:\n"
    context += "".join(
        [f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs_text)]
    )

    final_prompt = RAG_PROMPT_TEMPLATE.format(question=question, context=context)

    if hasattr(llm, "invoke"):
        response = llm.invoke(final_prompt)
        answer = response.content if hasattr(response, "content") else str(response)
    else:
        answer = llm(final_prompt)

    return answer, relevant_docs_text


from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.vectorstores import VectorStore
from typing import Optional, List
import datasets
import json
from tqdm import tqdm


def run_rag_tests(
    eval_dataset: datasets.Dataset,
    llm: BaseLanguageModel,
    knowledge_index: VectorStore,
    output_file: str,
    reranker: Optional[any] = None,
    verbose: Optional[bool] = True,
    test_settings: Optional[str] = None,  # To document the test settings used
):
    """Runs RAG tests on the given dataset and saves the results to the given output file."""
    try:  # load previous generations if they exist
        with open(output_file, "r") as f:
            outputs = json.load(f)
    except:
        outputs = []

    for example in tqdm(eval_dataset):
        question = example["question"]
        if question in [output["question"] for output in outputs]:
            continue

        answer, relevant_docs = answer_with_rag(
            question, llm, knowledge_index, reranker=reranker
        )
        if verbose:
            print("=======================================================")
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            print(f'True answer: {example["answer"]}')
        result = {
            "question": question,
            "true_answer": example["answer"],
            "source_doc": example["source_doc"],
            "generated_answer": answer,
            "retrieved_docs": [doc for doc in relevant_docs],
        }
        if test_settings:
            result["test_settings"] = test_settings
        outputs.append(result)

        with open(output_file, "w") as f:
            json.dump(outputs, f)


# Prompts
RAG_PROMPT_TEMPLATE = """
<|system|>
Using the information contained in the context,
give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
Provide the number of the source document when relevant.
If the answer cannot be deduced from the context, do not give an answer.</s>
<|user|>
Context:
{context}
---
Now here is the question you need to answer.

Question: {question}
</s>
<|assistant|>
"""

QA_generation_prompt = """
Your task is to write a factoid question and an answer given a context.
Your factoid question should be answerable with a specific, concise piece of factual information from the context.
Your factoid question should be formulated in the same style as questions users could ask in a search engine.
This means that your factoid question MUST NOT mention something like "according to the passage" or "context".

Provide your answer as follows:

Output:::
Factoid question: (your factoid question)
Answer: (your answer to the factoid question)

Now here is the context.

Context: {context}\n
Output:::"""

question_groundedness_critique_prompt = """
You will be given a context and a question.
Your task is to provide a 'total rating' scoring how well one can answer the given question unambiguously with the given context.
Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at all given the context, and 5 means that the question is clearly and unambiguously answerable with the context.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here are the question and context.

Question: {question}\n
Context: {context}\n
Answer::: """

question_relevance_critique_prompt = """
You will be given a question.
Your task is to provide a 'total rating' representing how useful this question can be to machine learning developers building NLP applications with the Hugging Face ecosystem.
Give your answer on a scale of 1 to 5, where 1 means that the question is not useful at all, and 5 means that the question is extremely useful.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question.

Question: {question}\n
Answer::: """

question_standalone_critique_prompt = """
You will be given a question.
Your task is to provide a 'total rating' representing how context-independent this question is.
Give your answer on a scale of 1 to 5, where 1 means that the question depends on additional information to be understood, and 5 means that the question makes sense by itself.
For instance, if the question refers to a particular setting, like 'in the context' or 'in the document', the rating must be 1.
The questions can contain obscure technical nouns or acronyms like Gradio, Hub, Hugging Face or Space and still be a 5: it must simply be clear to an operator with access to documentation what the question is about.

For instance, "What is the name of the checkpoint from which the ViT model is imported?" should receive a 1, since there is an implicit mention of a context, thus the question is not independent from the context.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question.

Question: {question}\n
Answer::: """


EVALUATION_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 5}}\"
4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
[Is the response correct, accurate, and factual based on the reference answer?]
Score 1: The response is completely incorrect, inaccurate, and/or not factual.
Score 2: The response is mostly incorrect, inaccurate, and/or not factual.
Score 3: The response is somewhat correct, accurate, and/or factual.
Score 4: The response is mostly correct, accurate, and factual.
Score 5: The response is completely correct, accurate, and factual.

###Feedback:"""
