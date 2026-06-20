import json
import os
import shutil
from datetime import datetime
from time import time

from pneuma import Pneuma

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

dataset = "fetaqa"
benchmark_type = "content"
out_path = "out_benchmark/storage"


def read_jsonl(file_path: str):
    data: list[dict[str, str]] = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data


def write_jsonl(data: list[dict[str, str]], file_path: str):
    with open(file_path, "w", encoding="utf-8") as file:
        for item in data:
            file.write(json.dumps(item))
            file.write("\n")


def get_question_key(benchmark_type: str, use_rephrased_questions: bool = False):
    if benchmark_type == "content":
        if not use_rephrased_questions:
            question_key = "question_from_sql_1"
        else:
            question_key = "question"
    else:
        if not use_rephrased_questions:
            question_key = "question_bx1"
        else:
            question_key = "question_bx2"
    return question_key


def main():
    if dataset == "chicago":
        content_benchmark = read_jsonl(
            "../data_src/benchmarks/content/pneuma_chicago_10K_questions_annotated.jsonl"
        )
        data_path = "../data_src/tables/pneuma_chicago_10K"
    elif dataset == "public":
        content_benchmark = read_jsonl(
            "../data_src/benchmarks/content/pneuma_public_bi_questions_annotated.jsonl"
        )
        data_path = "../data_src/tables/pneuma_public_bi"
    elif dataset == "chembl":
        content_benchmark = read_jsonl(
            "../data_src/benchmarks/content/pneuma_chembl_10K_questions_annotated.jsonl"
        )
        data_path = "../data_src/tables/pneuma_chembl_10K"
    elif dataset == "adventure":
        content_benchmark = read_jsonl(
            "../data_src/benchmarks/content/pneuma_adventure_works_questions_annotated.jsonl"
        )
        data_path = "../data_src/tables/pneuma_adventure_works"
    elif dataset == "fetaqa":
        content_benchmark = read_jsonl(
            "../data_src/benchmarks/content/pneuma_fetaqa_questions_annotated.jsonl"
        )
        data_path = "../data_src/tables/pneuma_fetaqa"

    question_key = get_question_key(benchmark_type, False)
    questions_bc1 = []
    for data in content_benchmark:
        questions_bc1.append(data[question_key])
    questions_bc1 = questions_bc1[:50]

    question_key = get_question_key(benchmark_type, True)
    questions_bc2 = []
    for data in content_benchmark:
        questions_bc2.append(data[question_key])
    questions_bc2 = questions_bc2[:50]

    metadata_path = ""
    shutil.rmtree(out_path, ignore_errors=True)
    pneuma = Pneuma(out_path=out_path)
    pneuma.setup()

    results = {}
    responses = {}

    # Read Tables
    print("Starting reading tables...")
    start_time = time()
    response = pneuma.add_tables(data_path, "benchmarking", accept_duplicates=True)
    end_time = time()
    response = json.loads(response)
    print(
        f"Time to read {response['data']['file_count']} tables: {end_time - start_time} seconds"
    )
    results["read_tables"] = {
        "file_count": response["data"]["file_count"],
        "time": end_time - start_time,
    }
    responses["read_tables"] = response

    # Add Metadata
    if metadata_path:
        print("Starting adding metadata...")
        start_time = time()
        response = pneuma.add_metadata(metadata_path)
        end_time = time()
        response = json.loads(response)
        print(
            f"Time to add {response['data']['file_count']} contexts: {end_time - start_time} seconds"
        )
        results["add_metadata"] = {
            "file_count": response["data"]["file_count"],
            "time": end_time - start_time,
        }
        responses["add_metadata"] = response

    # Summarize
    print("Starting summarizing tables...")
    start_time = time()
    response = pneuma.summarize()
    end_time = time()
    response = json.loads(response)
    print(
        f"Time to summarize {len(response['data']['table_ids'])} tables: {end_time - start_time} seconds"
    )
    results["summarize"] = {
        "table_count": len(response["data"]["table_ids"]),
        "time": end_time - start_time,
    }
    responses["summarize"] = response

    # Write results to file
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    with open(
        f"{out_path}/../benchmark-{dataset}-{timestamp}.json", "w", encoding="utf-8"
    ) as f:
        json_results = {
            "dataset": dataset,
            "benchmark_type": benchmark_type,
            "timestamp": timestamp,
            "results": results,
            "responses": responses,
        }
        json.dump(json_results, f, indent=4)

    # Generate Index
    print("Starting generating index...")
    start_time = time()
    response = pneuma.generate_index("benchmark_index")
    end_time = time()
    response = json.loads(response)
    print(
        f"Time to generate index with {len(response['data']['table_ids'])} tables: {end_time - start_time} seconds"
    )
    results["generate_index"] = {
        "table_count": len(response["data"]["table_ids"]),
        "vector_index_generation_time": response["data"][
            "vector_index_generation_time"
        ],
        "keyword_index_generation_time": response["data"][
            "keyword_index_generation_time"
        ],
        "time": end_time - start_time,
    }
    responses["generate_index"] = response

    # Write results to file
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    with open(
        f"{out_path}/../benchmark-{dataset}-{timestamp}.json", "w", encoding="utf-8"
    ) as f:
        json_results = {
            "dataset": dataset,
            "benchmark_type": benchmark_type,
            "timestamp": timestamp,
            "results": results,
            "responses": responses,
        }
        json.dump(json_results, f, indent=4)

    # Query Index
    start_time = time()
    for question in questions_bc1:
        response = pneuma.query_index("benchmark_index", question, 3)
    end_time = time()
    print(
        f"Time to query index with {len(questions_bc1)} BC1 questions: {end_time - start_time} seconds"
    )
    results["BC1_query_index"] = {
        "query_count": len(questions_bc1),
        "time": end_time - start_time,
        "query_throughput": len(questions_bc1) / (end_time - start_time),
    }

    start_time = time()
    for question in questions_bc2:
        response = pneuma.query_index("benchmark_index", question, 3)
    end_time = time()
    print(
        f"Time to query index with {len(questions_bc2)} BC2 questions: {end_time - start_time} seconds"
    )
    results["BC2_query_index"] = {
        "query_count": len(questions_bc2),
        "time": end_time - start_time,
        "query_throughput": len(questions_bc2) / (end_time - start_time),
    }

    results["query_index"] = {
        "query_count": results["BC1_query_index"]["query_count"]
        + results["BC2_query_index"]["query_count"],
        "time": results["BC1_query_index"]["time"] + results["BC2_query_index"]["time"],
    }
    results["query_throughput"] = (
        results["query_index"]["query_count"] / results["query_index"]["time"]
    )

    # Write results to file
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    with open(
        f"{out_path}/../benchmark-{dataset}-{timestamp}.json", "w", encoding="utf-8"
    ) as f:
        json_results = {
            "dataset": dataset,
            "benchmark_type": benchmark_type,
            "timestamp": timestamp,
            "results": results,
            "responses": responses,
        }
        json.dump(json_results, f, indent=4)


if __name__ == "__main__":
    main()
