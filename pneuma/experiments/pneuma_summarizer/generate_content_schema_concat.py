import os
import sys
import json
import argparse
import pandas as pd

sys.path.append("../..")

from tqdm import tqdm
from benchmark_generator.context.utils.jsonl import write_jsonl


def generate_schema_concat_summaries(tables_path: str, summaries_name: str):
    content_summaries: list[dict[str, str]] = []
    tables = [
        table[:-4]
        for table in sorted(os.listdir(tables_path))
        if table.endswith(".csv")
    ]

    for table in tqdm(tables, "processing tables"):
        df = pd.read_csv(f"{tables_path}/{table}.csv")
        summary = {
            "id": f"{table}_SEP_contents_SEP_schema",
            "table": table,
            "summary": " | ".join(df.columns),
        }
        content_summaries.append(summary)

    SCHEMA_CONCAT_PATH = "summaries/schema_concat"
    try:
        write_jsonl(content_summaries, f"{SCHEMA_CONCAT_PATH}/{summaries_name}.jsonl")
    except FileNotFoundError:
        os.mkdir(SCHEMA_CONCAT_PATH)
        write_jsonl(content_summaries, f"{SCHEMA_CONCAT_PATH}/{summaries_name}.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This program generates SchemaConcat summaries, which is\
                    basically the concatenation of a table columns.",
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
            generate_schema_concat_summaries(tables_path, summaries_name)
    else:
        try:
            table_name = TABLES[dataset]
            tables_path = TABLES_SRC + table_name
            generate_schema_concat_summaries(tables_path, dataset)
        except KeyError:
            print(
                f"Dataset {dataset} not found! Please define the path in `constants.json`."
            )
