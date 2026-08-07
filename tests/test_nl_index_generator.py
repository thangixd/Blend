import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import duckdb

from src.NLSeeker import Schema
from src.NLSeeker.IndexGenerator import IndexGenerator

DOCUMENTS = 'test_nl_documents'
TOKENS = 'test_nl_tokens'


class _StubEmbedder:
    content_budget = 128

    @staticmethod
    def count_tokens(text: str) -> int:
        return len(text.split())

    @staticmethod
    def encode(texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


def _connection(summarize_second: bool = False) -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect(':memory:')
    Schema.setup(connection, 'nl_test')
    status = Schema.SUMMARIZED if summarize_second else Schema.REGISTERED
    connection.execute(
        "INSERT INTO table_status (id, table_name, status, creator, hash) VALUES "
        f"('1', 'a.csv', '{Schema.SUMMARIZED}', 'test', 'x'), "
        f"('2', 'b.csv', '{status}', 'test', 'y')")
    connection.execute(
        'INSERT INTO table_summaries (table_id, summary, summary_type) VALUES (?, ?, ?)',
        ['1', json.dumps({'payload': 'id: the customer identifier'}), Schema.COLUMN_NARRATION])
    if summarize_second:
        connection.execute(
            'INSERT INTO table_summaries (table_id, summary, summary_type) VALUES (?, ?, ?)',
            ['2', json.dumps({'payload': 'city: the city city'}), Schema.COLUMN_NARRATION])
    return connection


def _generator(connection) -> IndexGenerator:
    return IndexGenerator(connection, 'nl_test', _StubEmbedder(), DOCUMENTS, TOKENS)


def test_table_without_documents_raises():
    connection = _connection()
    try:
        _generator(connection).generate_index('test-index')
    except ValueError as error:
        assert '2' in str(error)
    else:
        raise AssertionError('expected ValueError for the unsummarized table')


def test_documents_table_holds_one_row_per_document():
    connection = _connection(summarize_second=True)
    assert _generator(connection).generate_index('test-index') == 2
    rows = connection.execute(
        f'SELECT docid, tableid, length FROM main."{DOCUMENTS}" ORDER BY tableid').fetchall()
    assert [row[1] for row in rows] == [1, 2]
    assert rows[0][0] == '1_SEP_contents_SEP_schema-0'
    # 'id: the customer identifier' -> id, customer, identifier ('the' is a stopword)
    assert rows[0][2] == 3


def test_embedding_column_is_a_fixed_size_float_array():
    connection = _connection(summarize_second=True)
    _generator(connection).generate_index('test-index')
    types = dict(connection.execute(
        f'SELECT column_name, data_type FROM information_schema.columns '
        f"WHERE table_name = '{DOCUMENTS}'").fetchall())
    assert types['embedding'] == 'FLOAT[3]'


def test_tokens_table_holds_term_frequencies():
    connection = _connection(summarize_second=True)
    _generator(connection).generate_index('test-index')
    rows = dict(connection.execute(
        f'SELECT token, tf FROM main."{TOKENS}" '
        f"WHERE docid = '2_SEP_contents_SEP_schema-0'").fetchall())
    assert rows == {'citi': 3}


def test_replace_rebuilds_the_tables():
    connection = _connection(summarize_second=True)
    generator = _generator(connection)
    generator.generate_index('test-index')
    generator.generate_index('test-index', replace=True)
    assert connection.execute(f'SELECT count(*) FROM main."{DOCUMENTS}"').fetchone()[0] == 2
    assert connection.execute("SELECT count(*) FROM indexes WHERE name = 'test-index'").fetchone()[0] == 2


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'{name} ok')
