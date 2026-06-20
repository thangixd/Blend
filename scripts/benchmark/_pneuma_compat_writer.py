"""Shared writer for ``pneuma_compat.jsonl``.

Used by both the Blend bench (``scripts.benchmark.run``) and the PNEUMA
bench (``scripts.benchmark.pneuma_run``) so the artifact layout is
bit-identical regardless of which side produced it.

The rows are expected to be ``SummaryRow`` instances (or any object with
the same attributes):  ``dataset``, ``family``, ``k``, ``n``, ``alpha``,
``hit_rate``, ``n_questions``, ``rerank_mode``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def write_pneuma_compat_jsonl(path: Path, rows: Iterable[Any]) -> None:
    """One line per (dataset, family, k, rerank_mode) in PNEUMA's hybrid-...jsonl shape."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            n_hits = round(r.hit_rate * r.n_questions / 100)
            obj = {
                "dataset": r.dataset, "benchmark_name": r.family,
                "k": r.k, "n": r.n, "alpha": r.alpha,
                "hitrate": r.hit_rate, "sum": n_hits,
                "rerank_mode": r.rerank_mode,
            }
            f.write(json.dumps(obj) + "\n")


__all__ = ["write_pneuma_compat_jsonl"]
