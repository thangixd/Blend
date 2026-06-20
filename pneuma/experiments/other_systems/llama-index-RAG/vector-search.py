import setproctitle
setproctitle.setproctitle("python3")
import os
import sys
import time

sys.path.append("../..")

from huggingface_hub import login
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
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
        f"../../data_src/benchmarks/context/{duckdb_filename}/contexts_{duckdb_filename}.jsonl"
    )
    db_reader = DatabaseReader(engine)
    final_documents = []

    for table in sorted([table[:-4] for table in os.listdir(path)]):
        # Inserting contents
        documents = db_reader.load_data(f"select * from {table}")
        for i in range(len(documents)):
            final_documents.append(Document(
                text=documents[i].text,
                doc_id=f"{table}.csv_part_{i}",
            ))

        # Inserting contexts
        specific_contexts = [
            context for context in contexts if context["table"] == table
        ]
        base_id = len(documents)
        for j in range(len(specific_contexts)):
            document = Document(
                text=specific_contexts[j]["context"],
                doc_id=f"{table}.csv_part_{base_id+j}",
            )
            final_documents.append(document)
    return final_documents


def start(long_name: str, short_name: str, k=10):
    print(f"Benchmarking for: {short_name}")
    ctx_benchmark = read_jsonl(
        f"../../data_src/benchmarks/context/{short_name}/bx_{short_name}.jsonl"
    )
    ctn_benchmark = read_jsonl(
        f"../../data_src/benchmarks/content/{long_name}_questions_annotated.jsonl"
    )

    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

    start = time.time()
    documents = get_documents(f"../../data_src/tables/{long_name}", short_name)
    end = time.time()
    print(f"Time to create documents: {end-start} seconds")


    start = time.time()
    index = VectorStoreIndex(documents, show_progress=True)
    retriever = index.as_retriever(similarity_top_k=k)
    end = time.time()
    print(f"Time to indexing: {end-start} seconds")

    def convert_retrieved_data_to_tables_ranks(retrieved_data):
        # Convert all retrieved data to the format (table, rank)
        rank = 1
        prev_score = retrieved_data[0].get_score()
        tables_ranks = []
        for data in retrieved_data:
            if data.get_score() < prev_score:
                rank += 1
            table = data.id_.split(".csv")[0]
            tables_ranks.append((table, rank))
            prev_score = data.get_score()
        return tables_ranks


    def evaluate_benchmark(benchmark: list[dict[str, str]], question_key: str):
        hit_rate_sum = 0
        for idx, datum in enumerate(tqdm(benchmark)):
            question = datum[question_key]
            answer_tables = datum["answer_tables"]
            response = retriever.retrieve(question)
            tables_ranks = convert_retrieved_data_to_tables_ranks(response)
            for table, _ in tables_ranks:
                if table in answer_tables:
                    hit_rate_sum += 1
                    break
            if idx % 10 == 0:
                print(f"Current hit rate sum in row {idx}: {hit_rate_sum}")
                print("=" * 200)
        print(f"Final Hit Rate: {hit_rate_sum/len(benchmark)}")


    print(f"BC1 {short_name}")
    start = time.time()
    evaluate_benchmark(ctn_benchmark, "question_from_sql_1")
    end = time.time()
    print(f"Time for BC1 {short_name}: {end-start} seconds")

    print(f"BC2 {short_name}")
    start = time.time()
    evaluate_benchmark(ctn_benchmark, "question")
    end = time.time()
    print(f"Time for BC2 {short_name}: {end-start} seconds")
    
    print(f"BX1 {short_name}")
    start = time.time()
    evaluate_benchmark(ctx_benchmark, "question_bx1")
    end = time.time()
    print(f"Time for BX1 {short_name}: {end-start} seconds")

    
    print(f"BX2 {short_name}")
    start = time.time()
    evaluate_benchmark(ctx_benchmark, "question_bx2")
    end = time.time()
    print(f"Time for BX2 {short_name}: {end-start} seconds")


long_name = "pneuma_adventure_works"
short_name = "adventure"
start(long_name, short_name)
