import argparse
import json
import os
import sys
import time

import numpy as np
import Stemmer

sys.path.append("../..")


from sentence_transformers import SentenceTransformer
from sentence_transformers.SentenceTransformer import SentenceTransformer
from transformers import set_seed

from benchmark_generator.context.utils.jsonl import read_jsonl

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
set_seed(42, deterministic=True)


embedding_model = SentenceTransformer("../models/bge-base", local_files_only=True)
stemmer = Stemmer.Stemmer("english")


def get_question_key(benchmark_type: str, use_rephrased_questions: bool):
    if benchmark_type == "content":
        if not use_rephrased_questions:
            print("Processing non-rephrased content questions")
            question_key = "question_from_sql_1"
        else:
            print("Processing rephrased content questions")
            question_key = "question"
    else:
        if not use_rephrased_questions:
            print("Processing non-rephrased context questions")
            question_key = "question_bx1"
        else:
            print("Processing rephrased context questions")
            question_key = "question_bx2"
    return question_key


def produce_embeddings(
    dataset: str, benchmark: list[dict[str, str]], benchmark_type: str,
    use_rephrased_questions=False,
):
    start = time.time()
    print(f"Producing embeddings for benchmark questions of {dataset} dataset")

    questions = []
    question_key = get_question_key(benchmark_type, use_rephrased_questions)
    for data in benchmark:
        questions.append(data[question_key])
    embed_questions = embedding_model.encode(
        questions, batch_size=64, show_progress_bar=True
    )
    np.savetxt(
        f"embeddings/embed-{dataset}-questions-{benchmark_type}-{use_rephrased_questions}.txt",
        embed_questions,
    )

    end = time.time()
    print(f"Embedding production time: {end - start} seconds")


def start_produce_embeddings(
    dataset: str, dataset_long_name: str,
    question_types: list[str], use_rephrased_questions: bool,
):
    if "content" in question_types:
        content_benchmark = read_jsonl(
            f"{QUESTIONS_PATH}/content/{dataset_long_name}_questions_annotated.jsonl"
        )
        produce_embeddings(
            dataset, content_benchmark, "content", use_rephrased_questions
        )

    if "context" in question_types:
        context_benchmark = read_jsonl(
            f"{QUESTIONS_PATH}/context/{dataset}/bx_{dataset}.jsonl"
        )
        produce_embeddings(
            dataset, context_benchmark, "context", use_rephrased_questions
        )


def str_to_bool(value: str) -> bool:
    if value.lower() in ['true', '1', 't', 'y', 'yes']:
        return True
    elif value.lower() in ['false', '0', 'f', 'n', 'no']:
        return False
    else:
        raise ValueError("Invalid boolean value")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This program produces embeddings for questions, reducing \
            the needs to produce them again and again for experiments.",
    )
    parser.add_argument(
        "-d", "--dataset", default="all",
        choices=["chembl", "adventure", "public", "chicago", "fetaqa", "bird"],
    )
    parser.add_argument(
        "-q", "--question-types", nargs="+",
        default=["content", "context"], choices=["content", "context"]
    )
    parser.add_argument(
        "-r", "--use-rephrased-questions", nargs="+", type=str_to_bool,
        default=[True, False], choices=[True, False]
    )
    dataset = parser.parse_args().dataset
    question_types = parser.parse_args().question_types
    use_rephrased_questions = parser.parse_args().use_rephrased_questions

    with open("../constants.json") as file:
        CONSTANTS = json.load(file)
        QUESTIONS_PATH = f"../{CONSTANTS['data_src']}/benchmarks"
        DATASETS: dict[str,str] = CONSTANTS["datasets"]

    if dataset == "all":
        for dataset in DATASETS.keys():
            for i in use_rephrased_questions:
                start_produce_embeddings(
                    dataset, DATASETS[dataset], question_types, i
                )
    else:
        for i in use_rephrased_questions:
            start_produce_embeddings(dataset, DATASETS[dataset], question_types, i)
