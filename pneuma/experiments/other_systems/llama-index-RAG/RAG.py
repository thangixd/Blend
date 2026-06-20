import json
import os
import sys
import time
from datetime import datetime

import torch

sys.path.append("../../..")


from huggingface_hub import login
from llama_index.core import Document, PromptTemplate, Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.readers.database import DatabaseReader
from sqlalchemy import create_engine
from tqdm import tqdm
from transformers import set_seed

from benchmark_generator.context.utils.jsonl import read_jsonl

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
set_seed(42, deterministic=True)


def get_documents(path: str, duckdb_filename=""):
    engine = create_engine(f"duckdb:///{duckdb_filename}.duckdb")
    contexts = read_jsonl(
        f"../../../data_src/benchmarks/context/{duckdb_filename}/contexts_{duckdb_filename}.jsonl"
    )
    db_reader = DatabaseReader(engine)
    final_documents = []

    for table in tqdm(
        sorted([table[:-4] for table in os.listdir(path)]), desc="Process tables"
    ):
        # Inserting contents
        documents = db_reader.load_data(f"select * from {table}")
        for i in range(len(documents)):
            documents[i].id_ = f"{table}.csv_part_{i}"
            documents[i].text = documents[i].id_ + f": {documents[i].text}"
            final_documents.append(documents[i])

        # Inserting contexts
        specific_contexts = [
            context for context in contexts if context["table"] == table
        ]
        base_id = len(documents)
        for j in range(len(specific_contexts)):
            document = Document(
                text=f"{table}.csv_part_{base_id+j}: "
                + specific_contexts[j]["context"],
                doc_id=f"{table}.csv_part_{base_id+j}",
            )
            final_documents.append(document)
    return final_documents


# Adjust names and tokens
long_name = "pneuma_adventure_works"
short_name = "adventure"
# login("TODO: HF_Token")

results = {}

start_time = time.time()
documents = get_documents(f"../../../data_src/tables/{long_name}", short_name)
end_time = time.time()
results["ingestion"] = {
    "file_count": len(documents),
    "time": end_time - start_time,
}

ctx_benchmark = read_jsonl(
    f"../../../data_src/benchmarks/context/{short_name}/bx_{short_name}.jsonl"
)
ctn_benchmark = read_jsonl(
    f"../../../data_src/benchmarks/content/{long_name}_questions.jsonl"
)


query_wrapper_prompt = PromptTemplate(
    "[INST] {query_str} [/INST]"
)  # Taken from Mistral-7B-Instruct-v0.3's chat template
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = HuggingFaceLLM(
    context_window=32768,
    max_new_tokens=100,
    query_wrapper_prompt=query_wrapper_prompt,
    generate_kwargs={"do_sample": False, "pad_token_id": 2},
    model_kwargs={
        "torch_dtype": torch.bfloat16,
    },
    tokenizer_name="mistralai/Mistral-7B-Instruct-v0.3",
    model_name="mistralai/Mistral-7B-Instruct-v0.3",
    device_map="auto",
    tokenizer_kwargs={"max_length": 32768},
)


start_time = time.time()
index = VectorStoreIndex.from_documents(documents, show_progress=True)
end_time = time.time()
results["generate_index"] = {
    "time": end_time - start_time,
}
query_engine = index.as_query_engine(similarity_top_k=1)


def get_query(question: str):
    # Instruction for the LLM to return relevant dataset(s) in ranked format
    return f"""{question} Mention the name of the table (csv) that is relevant to the
query first, then explain the reason."""


def evaluate_benchmark(benchmark: list[dict[str, str]], question_key: str):
    hit_rate_sum = 0
    for idx, datum in enumerate(tqdm(benchmark)):
        question = datum[question_key]
        answer_tables = datum["answer_tables"]

        query = get_query(question)
        response = query_engine.query(query)
        for table in answer_tables:
            if f"{table}.csv" in str(response):
                hit_rate_sum += 1
                break

        if idx % 10 == 0:
            print(f"Current hit rate sum in row {idx}: {hit_rate_sum}")
            print(f"Response: {response}")
            print("=" * 200)
    print(f"Final Hit Rate: {hit_rate_sum/len(benchmark)}")


results["query_index"] = []

print("Benchmark results for BX1")
start_time = time.time()
evaluate_benchmark(ctx_benchmark, "question_bx1")
end_time = time.time()
results["query_index"].append(
    {
        "benchmark": "BX1",
        "time": end_time - start_time,
    }
)

print("Benchmark results for BX2")
start_time = time.time()
evaluate_benchmark(ctx_benchmark, "question_bx2")
end_time = time.time()
results["query_index"].append(
    {
        "benchmark": "BX2",
        "time": end_time - start_time,
    }
)


print("Benchmark results for BC1")
start_time = time.time()
evaluate_benchmark(ctn_benchmark, "question_from_sql_1")
end_time = time.time()
results["query_index"].append(
    {
        "benchmark": "BC1",
        "time": end_time - start_time,
    }
)

print("Benchmark results for BC2")
start_time = time.time()
evaluate_benchmark(ctn_benchmark, "question")
end_time = time.time()
results["query_index"].append(
    {
        "benchmark": "BC2",
        "time": end_time - start_time,
    }
)

os.makedirs("benchmark_results", exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
with open(
    f"benchmark_results/benchmark-{long_name}-{timestamp}.json", "w", encoding="utf-8"
) as f:
    json_results = {
        "dataset": long_name,
        "timestamp": timestamp,
        "results": results,
    }
    json.dump(json_results, f, indent=4)
