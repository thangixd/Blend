import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import duckdb

from src.NLSeeker import Schema
from src.NLSeeker.IndexGenerator import IndexGenerator


class _StubEmbedder:
    content_budget = 128

    @staticmethod
    def count_tokens(text: str) -> int:
        return len(text.split())

    @staticmethod
    def encode(texts):
        return [[0.0, 0.0, 0.0] for _ in texts]


def _connection() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect(':memory:')
    Schema.setup(connection, 'nl_test')
    connection.execute(
        "INSERT INTO table_status (id, table_name, status, creator, hash) VALUES "
        f"('1', 'a.csv', '{Schema.SUMMARIZED}', 'test', 'x'), "
        f"('2', 'b.csv', '{Schema.REGISTERED}', 'test', 'y')")
    connection.execute(
        'INSERT INTO table_summaries (table_id, summary, summary_type) VALUES (?, ?, ?)',
        ['1', json.dumps({'payload': 'id: the identifier'}), Schema.COLUMN_NARRATION])
    return connection


def test_table_without_documents_raises():
    with tempfile.TemporaryDirectory() as tmp:
        generator = IndexGenerator(_connection(), 'nl_test', _StubEmbedder(),
                                   Path(tmp) / 'vector', Path(tmp) / 'fulltext')
        try:
            generator.generate_index('test-index')
        except ValueError as error:
            assert '2' in str(error)
        else:
            raise AssertionError('expected ValueError for the unsummarized table')


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'{name} ok')
