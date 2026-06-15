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
    value_shard: Optional[pa.Table]
    raw_table: Optional[pd.DataFrame]


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
    table_id: int,
    file_path: str,
    want_raw: bool,
    want_value: bool = True,
) -> WorkerResult:
    """Compute the value-index Arrow shard (if want_value) plus the raw DataFrame.

    want_value=False skips _df_for_value_index + pa.Table.from_pandas; only
    the raw DataFrame (if want_raw) is returned. Used by benchmark mode where
    NLSeeker is the only consumer.
    """
    fail_on = os.environ.get("_BLEND_FAIL_ON_TID")
    if fail_on is not None and fail_on.isdigit() and int(fail_on) == table_id:
        raise RuntimeError("synthetic worker failure")

    df = _read_table(Path(file_path))
    if df.empty:
        return WorkerResult(table_id=table_id, value_shard=None, raw_table=None)

    raw_df = df if want_raw else None
    value_shard = None
    if want_value:
        idx_df = _df_for_value_index(table_id, df)
        value_shard = pa.Table.from_pandas(idx_df, preserve_index=False)
    return WorkerResult(table_id=table_id, value_shard=value_shard, raw_table=raw_df)
