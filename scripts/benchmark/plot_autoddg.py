"""Aggregate AutoDDG per-cell results into comparison plots."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)


def _bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
) -> tuple[float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed=42)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    samples = values[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha))


def aggregate(results_dir: Path) -> pd.DataFrame:
    """Scan *results_dir* for per-cell per_query.jsonl files and aggregate.

    Directory layout expected (written by the bench-autoddg Make loop):

        results_dir/
            <cell>/
                <ISO-8601-timestamp>/
                    per_query.jsonl
            ...

    For each cell directory the latest timestamp subdirectory is selected
    (ISO-8601 lexicographic sort, descending).  Records without a ``family``
    field are treated as a single ``"all"`` family with a logged warning.

    Returns a DataFrame with one row per ``(cell, family, rerank)`` combination,
    columns: cell, family, rerank, hit@1, hit@1_ci_lo, hit@1_ci_hi,
    hit@5, hit@5_ci_lo, hit@5_ci_hi, mrr, n.
    """
    if not results_dir.is_dir():
        return pd.DataFrame()
    rows = []
    for cell_dir in sorted(results_dir.iterdir()):
        if not cell_dir.is_dir():
            continue

        # Find the latest timestamp subdir (ISO-8601 names are lexicographically ordered).
        ts_dirs = sorted(
            [p for p in cell_dir.iterdir() if p.is_dir() and "T" in p.name],
            reverse=True,
        )
        if not ts_dirs:
            LOG.warning("No timestamp subdir found in %s; skipping.", cell_dir)
            continue
        latest_ts_dir = ts_dirs[0]

        per_q_path = latest_ts_dir / "per_query.jsonl"
        if not per_q_path.exists():
            LOG.warning("Missing per_query.jsonl in %s; skipping.", latest_ts_dir)
            continue

        records = []
        with per_q_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if not records:
            LOG.warning("Empty per_query.jsonl in %s; skipping.", latest_ts_dir)
            continue

        df = pd.DataFrame(records)

        # Normalise: rerank_mode is "off"/"on" in run.py.
        if "rerank_mode" not in df.columns:
            LOG.warning(
                "per_query.jsonl in %s has no 'rerank_mode' column; skipping.",
                latest_ts_dir,
            )
            continue

        # Family-split grouping.  Fall back to a single "all" group for legacy records.
        if "family" not in df.columns:
            LOG.warning(
                "per_query.jsonl in %s has no 'family' column; using single 'all' group.",
                latest_ts_dir,
            )
            families = [("all", df)]
        else:
            families = [(fam, df[df["family"] == fam]) for fam in sorted(df["family"].unique())]

        for fam, fam_df in families:
            for rerank_val in ("off", "on"):
                sub = fam_df[fam_df["rerank_mode"] == rerank_val]
                if sub.empty:
                    continue
                # Hit@1: rows where k==1.
                hit1_mask = sub["k"] == 1
                # Hit@5: rows where k==5 (distinct from k=1 to avoid double-counting).
                hit5_mask = sub["k"] == 5

                hit1 = sub[hit1_mask]["hit"].to_numpy(dtype=float)
                hit5 = sub[hit5_mask]["hit"].to_numpy(dtype=float)
                # MRR: use reciprocal rank from k==1 rows (rr column from run.py).
                mrr_vals = sub[hit1_mask]["rr"].to_numpy(dtype=float)

                h1_lo, h1_hi = _bootstrap_ci(hit1)
                h5_lo, h5_hi = _bootstrap_ci(hit5)
                row: dict = {
                    "cell": cell_dir.name,
                    "family": fam,
                    "rerank": rerank_val == "on",
                    "hit@1": float(hit1.mean()) if hit1.size else float("nan"),
                    "hit@1_ci_lo": h1_lo,
                    "hit@1_ci_hi": h1_hi,
                    "hit@5": float(hit5.mean()) if hit5.size else float("nan"),
                    "hit@5_ci_lo": h5_lo,
                    "hit@5_ci_hi": h5_hi,
                    "mrr": float(mrr_vals.mean()) if mrr_vals.size else float("nan"),
                    "n": int(hit1_mask.sum()),
                }
                rows.append(row)
    return pd.DataFrame(rows)


def main(argv: Optional[list[str]] = None) -> None:
    """Entry point for ``python -m scripts.benchmark.plot_autoddg``."""
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
    )
    p.add_argument(
        "--results-dir",
        required=True,
        type=Path,
        help=(
            "Base results directory for one dataset, e.g. "
            "benchmark-data/results/autoddg/chembl; contains one sub-dir per cell."
        ),
    )
    p.add_argument(
        "--out",
        default=None,
        type=Path,
        help="Write summary CSV here (default: results_dir/summary.csv).",
    )
    args = p.parse_args(argv)
    df = aggregate(args.results_dir)
    if df.empty:
        print("No results aggregated.")
        return
    out = args.out or (args.results_dir / "summary.csv")
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
