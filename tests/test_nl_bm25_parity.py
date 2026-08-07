import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import bm25s
import duckdb
import Stemmer
from bm25s.tokenization import convert_tokenized_to_string_list

from src.NLSeeker import Tokenizer
from src.NLSeeker.Retriever import bm25_sql

DOCUMENTS = 'parity_nl_documents'
TOKENS = 'parity_nl_tokens'

CORPUS = [
    'CustomerID: the unique identifier of a customer in the sales system',
    'AddressLine1: the street address of the customer',
    'AddressLine2: the second street address line, often empty',
    'City: the city the address belongs to',
    'PostalCode: the postal code of the city',
    'OrderDate: the date the sales order was placed',
    'TotalDue: the total amount due for the sales order',
    'ProductID: the identifier of the product being sold',
    'Name: the product name as shown in the catalogue',
    'ListPrice: the list price of the product in the catalogue',
]

QUERIES = [
    'customer address',
    'city postal code',
    'sales order total',
    'nonexistent term',
    # A repeated term contributes once per occurrence, and df must still be counted over
    # distinct terms - the two halves of the occurrence-weighting in bm25_sql.
    'address address',
    'addresses address',
    'city city postal',
]


def _connection() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect(':memory:')
    connection.execute(f'CREATE TABLE "{DOCUMENTS}" (docid VARCHAR, tableid INTEGER, '
                       f'text VARCHAR, length INTEGER)')
    connection.execute(f'CREATE TABLE "{TOKENS}" (docid VARCHAR, token VARCHAR, tf INTEGER)')
    for index, text in enumerate(CORPUS):
        frequencies = Tokenizer.term_frequencies(text)
        connection.execute(f'INSERT INTO "{DOCUMENTS}" VALUES (?, ?, ?, ?)',
                           [f'{index}_SEP_contents_SEP_schema-0', index, text,
                            sum(frequencies.values())])
        for token, frequency in frequencies.items():
            connection.execute(f'INSERT INTO "{TOKENS}" VALUES (?, ?, ?)',
                               [f'{index}_SEP_contents_SEP_schema-0', token, frequency])
    return connection


def _bm25s_scores(query: str) -> dict:
    stemmer = Stemmer.Stemmer('english')
    corpus_tokens = bm25s.tokenize(CORPUS, stopwords='en', stemmer=stemmer, show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=False)

    query_tokens = bm25s.tokenize(query, stopwords='en', stemmer=stemmer, show_progress=False)
    scores = retriever.get_scores(convert_tokenized_to_string_list(query_tokens)[0])
    return {f'{index}_SEP_contents_SEP_schema-0': float(score) for index, score in enumerate(scores)}


def test_sql_bm25_matches_bm25s():
    connection = _connection()
    for query in QUERIES:
        expected = _bm25s_scores(query)
        rows = dict(connection.execute(
            bm25_sql(DOCUMENTS, TOKENS, Tokenizer.tokenize(query))).fetchall())
        for docid, expected_score in expected.items():
            actual = rows.get(docid, 0.0)
            assert abs(actual - expected_score) <= 1e-4 * max(1.0, abs(expected_score)), \
                f'{query!r} {docid}: sql={actual} bm25s={expected_score}'


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'{name} ok')
