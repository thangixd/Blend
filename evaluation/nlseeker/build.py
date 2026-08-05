import json
from pathlib import Path

import duckdb
import pandas as pd

from evaluation.nlseeker.datasets import pneuma_id
from evaluation.nlseeker.prepare import PreparedDataset

# Typing imports
from typing import List, Tuple


def write_metadata_csv(contexts_jsonl: Path, assignment: List[Tuple[int, str]], out_csv: Path) -> Path:
    tableid_by_pneuma = {pneuma_id(Path(path).name): table_id for table_id, path in assignment}
    rows = []
    with open(contexts_jsonl) as f:
        for line in f:
            record = json.loads(line)
            rows.append((tableid_by_pneuma[record['table']], record['context']))
    pd.DataFrame(rows, columns=['table_id', 'value']).to_csv(out_csv, index=False)
    return out_csv


def populate_tables_registry(dbcon, dataset: str, assignment: List[Tuple[int, str]]) -> None:
    from scripts.create_index_duckdb import create_table
    create_table(dbcon, dataset)
    dbcon.executemany(f'INSERT INTO "{dataset}_tables" VALUES (?, ?)',
                      [(table_id, Path(path).name) for table_id, path in assignment])


def build_dataset_artifacts(artifacts_dir: Path, prepared: PreparedDataset, dataset: str) -> None:
    """Live build: needs the LLM and embedding endpoints. Not covered by unit tests."""
    from scripts.create_index_duckdb import assign_table_ids
    from scripts.create_index_nl_blend_duckdb import build_nl_index
    from src.NLSeeker.Config import NLSeekerConfig

    assignment = assign_table_ids(str(prepared.lake_dir / '*.csv'))
    metadata_csv = write_metadata_csv(prepared.contexts_jsonl, assignment,
                                      artifacts_dir / 'metadata.csv')

    config = NLSeekerConfig.load(artifacts_dir / 'nlseeker.ini')
    dbcon = duckdb.connect(str(artifacts_dir / 'eval.duckdb'))
    try:
        populate_tables_registry(dbcon, dataset, assignment)
        build_nl_index(dbcon, config, assignment, str(metadata_csv), accept_duplicates=True)
    finally:
        dbcon.close()
