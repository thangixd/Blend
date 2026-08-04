import sys
from pathlib import Path

_ROOT = Path(__file__).resolve()
for _p in [_ROOT, *_ROOT.parents]:
    if (_p / 'src' / 'Plan.py').exists() and (_p / 'tests' / 'tasks').exists():
        sys.path.insert(0, str(_p))
        import os as _os
        _os.chdir(_p)
        break
else:
    raise RuntimeError('could not locate Blend repo root above ' + str(_ROOT))

import pandas as pd
from pandas.api.types import is_string_dtype
from src.Plan import Plan
from src.Operators import Seekers, Combiners


def main() -> None:
    LAKE_CONFIG = "tests/tasks/adventure_works.ini"
    df = pd.read_csv("tests/tasks/datalake/adventure_works/vEmployee_SEP_table_42.csv")[['Title', 'FirstName', 'MiddleName', 'BusinessEntityID', 'AdditionalContactInfo']].copy()
    for _col in df.columns:
        if is_string_dtype(df[_col]):
            df[_col] = df[_col].astype(str).str.lower()

    k = 50
    plan = Plan()
    plan.add("query_included", Seekers.Correlation(
        df['MiddleName'], df['BusinessEntityID'], k))
    plan.add("query_excluded", Seekers.Correlation(
        df['MiddleName'], df['AdditionalContactInfo'], k))
    plan.add("difference", Combiners.Difference(k=k),
             inputs=["query_included", "query_excluded"])
    plan.add("query_mc", Seekers.MC(df[['Title', 'FirstName']], k*2))
    plan.add("intersection", Combiners.Intersection(k=k),
             inputs=["query_mc", "difference"])

    plan.DB.load_config(Path(LAKE_CONFIG))

    ids = plan.run() or []
    print(f"returned {len(ids)} TableIds: {ids}")


if __name__ == "__main__":
    main()
