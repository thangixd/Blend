import os
import sys
import json
import argparse

sys.path.append("../..")

from tqdm import tqdm
from sqlalchemy import create_engine
from llama_index.readers.database import DatabaseReader
from benchmark_generator.context.utils.jsonl import write_jsonl


def generate_dbreader_summaries(
    tables_path: str, duckdb_filename: str, summaries_name: str
):
    content_summaries: list[dict[str, str]] = []

    engine = create_engine(
        f"duckdb:///../other_systems/llama-index-RAG/{duckdb_filename}.duckdb"
    )
    db_reader = DatabaseReader(engine)

    for table in tqdm(sorted([table[:-4] for table in os.listdir(tables_path)])):
        documents = db_reader.load_data(f"select * from {table}")
        for i in range(len(documents)):
            content_summaries.append({"table": table, "summary": documents[i].text})

    DBREADER_PATH = "summaries/dbreader"
    try:
        write_jsonl(content_summaries, f"{DBREADER_PATH}/{summaries_name}.jsonl")
    except FileNotFoundError:
        os.mkdir(DBREADER_PATH)
        write_jsonl(content_summaries, f"{DBREADER_PATH}/{summaries_name}.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This program generates DBReader summaries, which is\
                    basically all rows of the tables.",
        epilog="Alternatively, you may download the generated summaries from\
                the `summaries` directory.",
    )
    parser.add_argument("-d", "--dataset", default="all")
    dataset = parser.parse_args().dataset

    with open("constants.json") as file:
        constants: dict[str, any] = json.load(file)

    TABLES_SRC: str = constants["data_src"] + "tables/"
    TABLES: dict[str, str] = constants["tables"]

    if dataset == "all":
        for table_info in TABLES.items():
            summaries_name, table_name = table_info
            tables_path = TABLES_SRC + table_name
            generate_dbreader_summaries(tables_path, summaries_name, summaries_name)
    else:
        try:
            table_name = TABLES[dataset]
            tables_path = TABLES_SRC + table_name
            generate_dbreader_summaries(tables_path, dataset, dataset)
        except KeyError:
            print(
                f"Dataset {dataset} not found! Please define the path in `constants.json`."
            )
