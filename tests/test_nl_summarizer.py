import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import duckdb

from src.NLSeeker.Summarizer import ROW_SAMPLE_SIZE, Summarizer


def _summarizer(connection: duckdb.DuckDBPyConnection) -> Summarizer:
    return Summarizer(connection, 'nl_test', llm=None, embedder=None)


def test_row_samples_are_bounded_and_formatted():
    connection = duckdb.connect(':memory:')
    summarizer = _summarizer(connection)
    connection.execute('CREATE TABLE "7" AS SELECT range AS id, \'v\' || range AS name FROM range(100)')
    rows = summarizer._row_samples(7)
    assert len(rows) == ROW_SAMPLE_SIZE
    assert all('id: ' in row and 'name: ' in row for row in rows)


def test_row_samples_of_small_and_empty_tables():
    connection = duckdb.connect(':memory:')
    summarizer = _summarizer(connection)
    connection.execute('CREATE TABLE "8" AS SELECT 1 AS id')
    connection.execute('CREATE TABLE "9" (id INTEGER)')
    assert len(summarizer._row_samples(8)) == 1
    assert summarizer._row_samples(9) == []


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'{name} ok')
