import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.DBHandler import DBHandler
from src.Operators.Seekers.NaturalLanguage import NaturalLanguage

DBHandler.USE_ML_OPTIMIZER = False


def _db() -> DBHandler:
    db = DBHandler()
    db.load_config(ROOT / 'tests' / 'tasks' / 'adventure_works.ini')
    return db


def test_empty_retrieval_returns_standard_sentinel():
    nl = NaturalLanguage('anything', k=5)
    nl._retrieve = lambda db, additionals: []
    assert nl.create_sql_query(_db()) == 'SELECT TableId FROM AllTables WHERE 1=0'


def test_sql_preserves_retrieval_rank_order():
    db = _db()
    nl = NaturalLanguage('anything', k=3)
    nl._retrieve = lambda db, additionals: [4, 7, 2]
    rows = db.execute_and_fetchall(nl.create_sql_query(db))
    assert [row[0] for row in rows] == [4, 7, 2]


def test_sql_cuts_to_k():
    db = _db()
    nl = NaturalLanguage('anything', k=2)
    nl._retrieve = lambda db, additionals: [4, 7, 2]
    rows = db.execute_and_fetchall(nl.create_sql_query(db))
    assert [row[0] for row in rows] == [4, 7]


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'{name} ok')
