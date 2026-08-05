import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.DBHandler import DBHandler
from src.Operators.Seekers.SingleColumnOverlap import SingleColumnOverlap
from src.Operators.Seekers.NaturalLanguage import NaturalLanguage

DBHandler.USE_ML_OPTIMIZER = False


def _db() -> DBHandler:
    db = DBHandler()
    db.load_config(ROOT / 'tests' / 'tasks' / 'adventure_works.ini')
    return db


def test_token_frequencies_counts_from_index():
    db = _db()
    token = db.execute_and_fetchall(
        "SELECT CellValue FROM AllTables WHERE CellValue <> '' AND CellValue NOT LIKE '%''%' LIMIT 1")[0][0]
    expected = db.execute_and_fetchall(
        f"SELECT COUNT(*) FROM AllTables WHERE CellValue = '{token}'")[0][0]
    assert db.get_token_frequencies([token])[token] == expected


def test_token_frequencies_missing_token_defaults_to_one():
    db = _db()
    assert db.get_token_frequencies(['zz_no_such_token_zz'])['zz_no_such_token_zz'] == 1


def test_sc_features_shape():
    db = _db()
    sc = SingleColumnOverlap(['road-350-w', 'll road frame - black- 44'], k=10)
    features = sc._features(db)
    assert len(features) == 3
    assert features[0] == 2      # distinct input rows
    assert features[2] == 1      # one column
    assert features[1] >= 1      # summed corpus frequency, geometric mean over 1 column


def test_ml_cost_flag_off_is_constant():
    db = _db()
    sc = SingleColumnOverlap(['road-350-w'], k=10)
    assert sc.ml_cost(db) == 1


def test_nl_features_explicit_n():
    db = _db()
    nl = NaturalLanguage('which tables contain street addresses', k=7, n=5, rerank=True)
    assert nl._features(db) == [5, 35, 1]


def test_nl_features_n_from_config():
    from src.NLSeeker.Config import NLSeekerConfig
    db = _db()
    config = NLSeekerConfig.load(db.config_path)
    nl = NaturalLanguage('where are the addresses', k=3, rerank=False)
    assert nl._features(db) == [4, 3 * config.n, 0]


def test_nl_ml_cost_flag_off_is_constant():
    db = _db()
    nl = NaturalLanguage('anything at all', k=5, n=2)
    assert nl.ml_cost(db) == 1


def test_ml_cost_flag_on_predicts_from_models():
    db = _db()
    DBHandler.USE_ML_OPTIMIZER = True
    try:
        sc = SingleColumnOverlap(['road-350-w'], k=10)
        nl = NaturalLanguage('which tables contain street addresses', k=5, n=3)
        sc_cost = sc.ml_cost(db)
        nl_cost = nl.ml_cost(db)
    finally:
        DBHandler.USE_ML_OPTIMIZER = False
    assert isinstance(sc_cost, float)
    assert isinstance(nl_cost, float)
    assert sc._cached_predicted_runtime is not None
    assert nl._cached_predicted_runtime is not None


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'{name} ok')
