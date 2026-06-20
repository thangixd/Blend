import argparse
import json
import os

import duckdb
from tqdm import tqdm


def ingest_duckdb(duckdb_name: str, dataset_path: str):
    con = duckdb.connect(duckdb_name)
    for table in tqdm([table[:-4] for table in os.listdir(dataset_path)]):
        con.execute(
            f"create table '{table}' as select * from read_csv('{dataset_path}/{table}.csv', ignore_errors=true)"
        )
    con.checkpoint()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This program ingests datasets into duckdb files.",
    )
    parser.add_argument("-d", "--dataset", default="all")
    dataset = parser.parse_args().dataset

    with open("../constants.json") as file:
        constants: dict[str, any] = json.load(file)

    TABLES_SRC: str = constants["data_src"] + "tables/"
    TABLES: dict[str, str] = constants["tables"]

    if dataset == "all":
        for table_info in TABLES.items():
            duckdb_name, dataset_dir = table_info
            dataset_path = TABLES_SRC + dataset_dir
            ingest_duckdb(f"{duckdb_name}.duckdb", dataset_path)
    else:
        try:
            dataset_dir = TABLES[dataset]
            dataset_path = TABLES_SRC + dataset_dir
            ingest_duckdb(f"{dataset}.duckdb", dataset_path)
        except KeyError:
            print(
                f"Dataset {dataset} not found! Please define the path in `constants.json`."
            )
