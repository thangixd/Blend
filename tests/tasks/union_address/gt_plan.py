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
    df = pd.read_csv("tests/tasks/datalake/adventure_works/Address_SEP_table_45.csv")[['AddressLine1', 'City', 'PostalCode']].copy()
    for _col in df.columns:
        if is_string_dtype(df[_col]):
            df[_col] = df[_col].astype(str).str.lower()

    k = 50
    plan = Plan()
    plan.add('AddressLine1', Seekers.SC(df['AddressLine1'], k*10))
    plan.add('City', Seekers.SC(df['City'], k*10))
    plan.add('PostalCode', Seekers.SC(df['PostalCode'], k*10))
    plan.add("union", Combiners.Union(k=k), inputs=['AddressLine1', 'City', 'PostalCode'])

    plan.DB.load_config(Path(LAKE_CONFIG))

    ids = plan.run() or []
    print(f"returned {len(ids)} TableIds: {ids}")


if __name__ == "__main__":
    main()
