import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.DBHandler import DBHandler

DBHandler.USE_ML_OPTIMIZER = False


def _db() -> DBHandler:
    db = DBHandler()
    db.load_config(ROOT / 'tests' / 'tasks' / 'adventure_works.ini')
    return db


def test_query_without_parameters_still_works():
    assert _db().execute_and_fetchall('SELECT 1 AS x') == [(1,)]


def test_scalar_parameter_binds():
    assert _db().execute_and_fetchall('SELECT ? AS x', [7]) == [(7,)]


def test_float_list_binds_as_a_fixed_size_array():
    rows = _db().execute_and_fetchall(
        'SELECT array_cosine_similarity(?::FLOAT[3], [1.0, 0.0, 0.0]::FLOAT[3]) AS s',
        [[1.0, 0.0, 0.0]])
    assert abs(rows[0][0] - 1.0) < 1e-6


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'{name} ok')
