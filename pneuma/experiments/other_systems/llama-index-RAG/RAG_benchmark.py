import json
import os
import sys
import time
from datetime import datetime

import torch

sys.path.append("../..")

import duckdb
from huggingface_hub import login
from llama_index.core import (
    Document,
    PromptTemplate,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.readers.database import DatabaseReader
from sqlalchemy import create_engine
from tqdm import tqdm
from transformers import set_seed

from benchmark_generator.context.utils.jsonl import read_jsonl

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
set_seed(42, deterministic=True)

# Adjust names and tokens
file_count = 15000
query_count = 50
short_name = f"fetaqa_{str(file_count)}"
long_name = "pneuma_fetaqa"
path = "../../data_src/tables/pneuma_fetaqa"

long_name = "pneuma_adventure_works"
short_name = "adventure"
path = "../../data_src/tables/pneuma_adventure_works"
duckdb_name = f"storage/{short_name}.duckdb"
# login("TODO: HF_Token")


def get_documents(path: str, duckdb_filename=""):
    engine = create_engine(f"duckdb:///storage/{duckdb_filename}.duckdb")
    db_reader = DatabaseReader(engine)
    final_documents = []

    for table in sorted([table[:-4] for table in os.listdir(path)][:file_count]):
        # Inserting contents
        documents = db_reader.load_data(f"select * from {table}")
        for i in range(len(documents)):
            documents[i].id_ = f"{table}.csv_part_{i}"
            documents[i].text = documents[i].id_ + f": {documents[i].text}"
            final_documents.append(documents[i])

    return final_documents


results = {}

start_time = time.time()
con = duckdb.connect(duckdb_name)
print("Creating tables")
file_paths = os.listdir(path)[:file_count]
for table in tqdm([table[:-4] for table in file_paths]):
    con.execute(
        f"create table '{table}' as select * from read_csv('{path}/{table}.csv', ignore_errors=true)"
    )

con.checkpoint()
con.close()

documents = get_documents(f"../../data_src/tables/{long_name}", short_name)
end_time = time.time()

print(f"Time to read {len(file_paths)} tables: {end_time - start_time} seconds")
results["ingestion"] = {
    "file_count": len(file_paths),
    "time": end_time - start_time,
}

ctn_benchmark = read_jsonl(
    f"../../data_src/benchmarks/content/{long_name}_questions_annotated.jsonl"
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

# start_time = time.time()
# index = VectorStoreIndex.from_documents(documents, show_progress=True)
# end_time = time.time()
# index.storage_context.persist(persist_dir=f"./index_{file_count}")
# print(f"Time to generate index: {end_time - start_time} seconds")
# results["generate_index"] = {
#     "time": end_time - start_time,
# }

storage_context = StorageContext.from_defaults(persist_dir=f"./index_{file_count}")
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine(similarity_top_k=1)


def get_query(question: str):
    # Instruction for the LLM to return relevant dataset(s) in ranked format
    return f"""{question} Mention the name of the table (csv) that is relevant to the
query first, then explain the reason."""


def evaluate_benchmark(benchmark: list[dict[str, str]], question_key: str):
    hit_rate_sum = 0
    cnt = 0
    for idx, datum in enumerate(tqdm(benchmark)):
        if cnt > query_count:
            break
        cnt += 1
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


print("Benchmark results for BC1")
start_time = time.time()
evaluate_benchmark(ctn_benchmark, "question_from_sql_1")
end_time = time.time()
print(f"Time to evaluate BC1: {end_time - start_time} seconds")
results["BC1_query_index"] = {
    "query_count": query_count,
    "time": end_time - start_time,
}


print("Benchmark results for BC2")
start_time = time.time()
evaluate_benchmark(ctn_benchmark, "question")
end_time = time.time()
print(f"Time to evaluate BC2: {end_time - start_time} seconds")
results["BC2_query_index"] = {
    "query_count": query_count,
    "time": end_time - start_time,
}


results["query_index"] = {
    "query_count": results["BC1_query_index"]["query_count"]
    + results["BC2_query_index"]["query_count"],
    "time": results["BC1_query_index"]["time"] + results["BC2_query_index"]["time"],
}


os.makedirs("benchmark_results", exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
with open(
    f"benchmark_results/benchmark-{short_name}-{timestamp}.json", "w", encoding="utf-8"
) as f:
    json_results = {
        "dataset": short_name,
        "timestamp": timestamp,
        "results": results,
    }
    json.dump(json_results, f, indent=4)
