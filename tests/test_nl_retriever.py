import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import duckdb

from src.NLSeeker.Retriever import Retriever

DOCUMENTS = 'test_nl_documents'
TOKENS = 'test_nl_tokens'

# (docid, tableid, text, embedding) - texts are pre-tokenized by the fixture below.
CORPUS = [
    ('1_SEP_contents_SEP_schema-0', 1, 'customer address city', [1.0, 0.0]),
    ('2_SEP_contents_SEP_schema-0', 2, 'order date total', [0.0, 1.0]),
    ('3_SEP_contents_SEP_schema-0', 3, 'address address postal', [0.7, 0.7]),
    # A second document for table 1, so a table can rank via more than one document -
    # this is what exercises the table-vs-document distinction in retrieve().
    ('1_SEP_contents_SEP_row-0', 1, 'address city customer', [1.0, 0.0]),
]


class _StubConfig:
    documents_table = DOCUMENTS
    tokens_table = TOKENS


class _StubEmbedder:
    @staticmethod
    def encode(texts):
        return [[1.0, 0.0] for _ in texts]


class _StubLLM:
    @staticmethod
    def complete(conversations, max_new_tokens, desc):
        # table 3's document is judged irrelevant so the partition has something to move.
        return ['no' if 'address address postal' in c[0]['content'] else 'yes' for c in conversations]


class _StubDB:
    def __init__(self, connection) -> None:
        self.connection = connection

    def execute_and_fetchall(self, query, parameters=None):
        if parameters is None:
            return self.connection.execute(query).fetchall()
        return self.connection.execute(query, parameters).fetchall()


def _db() -> _StubDB:
    from src.NLSeeker import Tokenizer

    connection = duckdb.connect(':memory:')
    connection.execute(f'CREATE TABLE "{DOCUMENTS}" (docid VARCHAR, tableid INTEGER, '
                       f'text VARCHAR, length INTEGER, embedding FLOAT[2])')
    connection.execute(f'CREATE TABLE "{TOKENS}" (docid VARCHAR, token VARCHAR, tf INTEGER)')
    for docid, tableid, text, embedding in CORPUS:
        frequencies = Tokenizer.term_frequencies(text)
        connection.execute(f'INSERT INTO "{DOCUMENTS}" VALUES (?, ?, ?, ?, ?)',
                           [docid, tableid, text, sum(frequencies.values()), embedding])
        for token, frequency in frequencies.items():
            connection.execute(f'INSERT INTO "{TOKENS}" VALUES (?, ?, ?)', [docid, token, frequency])
    return _StubDB(connection)


def _retriever() -> Retriever:
    return Retriever(_StubConfig(), _StubLLM(), _StubEmbedder())


def test_pure_bm25_ranks_the_repeated_term_first():
    ids = _retriever().retrieve(_db(), 'address', k=3, n=10, alpha=1.0, rerank=False)
    assert ids[0] == 3


def test_pure_vector_ranks_the_aligned_embedding_first():
    ids = _retriever().retrieve(_db(), 'address', k=3, n=10, alpha=0.0, rerank=False)
    assert ids[0] == 1


def test_allow_list_is_applied_inside_retrieval():
    ids = _retriever().retrieve(_db(), 'address', k=3, n=10, alpha=1.0,
                                additionals=' AND TableId IN (1, 2) ', rerank=False)
    assert 3 not in ids
    assert set(ids) <= {1, 2}


def test_allow_list_of_a_low_ranked_table_still_returns_it():
    # The whole reason the predicate is pushed into retrieval: filtering after the
    # cut would drop table 2, which no query term matches.
    ids = _retriever().retrieve(_db(), 'address', k=1, n=1, alpha=1.0,
                                additionals=' AND TableId IN (2) ', rerank=False)
    assert ids == [2]


def test_deny_list_is_applied_inside_retrieval():
    ids = _retriever().retrieve(_db(), 'address', k=3, n=10, alpha=1.0,
                                additionals=' AND TableId NOT IN (3) ', rerank=False)
    assert ids == [1, 2]


def test_k_counts_tables_not_documents():
    # Table 1's two documents rank as the top two by raw document score, so cutting
    # k=2 before deduplicating would return only table 1; cutting after returns both
    # table 1 and table 2.
    assert _retriever().retrieve(_db(), 'address city order', k=2, n=10, alpha=0.5, rerank=False) == [1, 2]


def test_query_of_only_stopwords_still_returns_vector_ranked_tables():
    ids = _retriever().retrieve(_db(), 'the of a', k=3, n=10, alpha=0.0, rerank=False)
    assert ids[0] == 1


def test_repeated_query_token_still_ranks_the_strongest_match_first():
    ids = _retriever().retrieve(_db(), 'address address', k=3, n=10, alpha=1.0, rerank=False)
    assert ids[0] == 3


def test_rerank_partitions_without_dropping():
    ids = _retriever().retrieve(_db(), 'address', k=3, n=10, alpha=1.0, rerank=True)
    assert ids == [1, 2, 3]


def test_same_stem_query_tokens_still_rank_the_strongest_match_first():
    ids = _retriever().retrieve(_db(), 'addresses address', k=3, n=10, alpha=1.0, rerank=False)
    assert ids[0] == 3


def test_a_second_handler_is_used_rather_than_the_captured_first_one():
    # A shared Retriever must not close over whichever DBHandler made the first call -
    # every call passes its own handler, so a later caller works even after the first
    # handler's connection is gone.
    retriever = _retriever()
    first_db = _db()
    retriever.retrieve(first_db, 'address', k=3, n=10, alpha=1.0, rerank=False)
    first_db.connection.close()

    second_db = _db()
    ids = retriever.retrieve(second_db, 'address', k=3, n=10, alpha=1.0, rerank=False)
    assert ids[0] == 3


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'{name} ok')
