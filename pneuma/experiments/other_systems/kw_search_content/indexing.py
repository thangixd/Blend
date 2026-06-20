import os
import sys
import time
import bm25s
import Stemmer

sys.path.append("../../..")

from benchmark_generator.context.utils.jsonl import read_jsonl


def get_table_contents(table_path: str, table_name: str):
    with open(table_path, "r", encoding="utf-8") as file:
        table_content = file.read()
        document = {
            "text": table_content,
            "metadata": {"table": f"{table_name}_SEP_contents"},
        }
        return document


def get_table_contexts(contexts: list[dict[str, str]], table_name: str):
    table_contexts = [context for context in contexts if context["table"] == table_name]
    documents: list[dict] = []
    for idx, context in enumerate(table_contexts):
        documents.append(
            {
                "text": context["context"],
                "metadata": {"table": f"{table_name}_SEP_contexts-{idx}"},
            }
        )
    return documents


def index_dataset(dataset_path: str, dataset_name: str):
    stemmer = Stemmer.Stemmer("english")
    corpus_json: list[dict] = []
    contexts = read_jsonl(
        f"../../../data_src/benchmarks/context/{dataset_name}/contexts_{dataset_name}.jsonl"
    )

    for table in sorted(os.listdir(dataset_path)):
        table_contents = get_table_contents(f"{dataset_path}/{table}", table[:-4])
        table_contexts = get_table_contexts(contexts, table[:-4])

        corpus_json.append(table_contents)
        corpus_json += table_contexts

    corpus_text = [doc["text"] for doc in corpus_json]
    corpus_tokens = bm25s.tokenize(
        corpus_text, stopwords="en", stemmer=stemmer, show_progress=False
    )

    retriever = bm25s.BM25(corpus=corpus_json)
    retriever.index(corpus_tokens, show_progress=True)
    retriever.save(f"indices/keyword-index-{dataset_name}")


if __name__ == "__main__":
    datasets = [
        {
            "path": "../../../data_src/tables/pneuma_chembl_10K",
            "name": "chembl",
        },
        {
            "path": "../../../data_src/tables/pneuma_adventure_works",
            "name": "adventure",
        },
        {
            "path": "../../../data_src/tables/pneuma_public_bi",
            "name": "public",
        },
        {
            "path": "../../../data_src/tables/pneuma_chicago_10K",
            "name": "chicago",
        },
        {
            "path": "../../../data_src/tables/pneuma_fetaqa",
            "name": "fetaqa",
        },
    ]

    for dataset in datasets:
        start = time.time()
        index_dataset(dataset["path"], dataset["name"])
        end = time.time()
        print(f"Indexing time for {dataset['name']}: {end-start} seconds")
