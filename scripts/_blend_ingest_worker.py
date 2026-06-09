from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import df_to_index


@dataclass
class WorkerResult:
    table_id: int
    # None when the source df was empty 
    value_shard: Optional[pa.Table]
    # None when --no-nl-index was set
    raw_table: Optional[pa.Table]


def _read_table(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(file_path)
    return pd.read_csv(file_path, low_memory=False)


def _df_for_value_index(table_id: int, df: pd.DataFrame) -> pd.DataFrame:
    indexed = df.copy()
    indexed.columns.name = str(table_id)
    return df_to_index(indexed)


def build_value_shard(
    table_id: int, file_path: str, want_raw: bool
) -> WorkerResult:
    fail_on = os.environ.get("_BLEND_FAIL_ON_TID")
    if fail_on is not None and fail_on.isdigit() and int(fail_on) == table_id:
        raise RuntimeError("synthetic worker failure")

    df = _read_table(Path(file_path))
    if df.empty:
        return WorkerResult(table_id=table_id, value_shard=None, raw_table=None)

    raw_arrow = pa.Table.from_pandas(df, preserve_index=False) if want_raw else None
    idx_df = _df_for_value_index(table_id, df)
    idx_arrow = pa.Table.from_pandas(idx_df, preserve_index=False)
    return WorkerResult(table_id=table_id, value_shard=idx_arrow, raw_table=raw_arrow)
