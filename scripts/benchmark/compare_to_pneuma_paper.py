"""Compare Blend benchmark hit-rates against the PNEUMA paper figures."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PaperRef:
    mixed_k1: float | None    # Fig. 6  (rerank=on, weighted BC2+BX2 at k=1)
    mixed_k5: float | None    # Fig. 6  (rerank=on, weighted BC2+BX2 at k=5)
    content_k1: float | None  # Table 2 (rerank=off, BC2 at k=1)
    context_k1: float | None  # Table 2 (rerank=off, BX2 at k=1)


# Numbers read off the paper figures/tables. Fig. 6 bars are reported
# to 2 decimals in the paper text; Table 2 is exact.
PAPER: dict[str, PaperRef] = {
    "chembl":          PaperRef(mixed_k1=68.07, mixed_k5=83.74,
                                content_k1=83.00, context_k1=50.88),
    "adventure_works": PaperRef(mixed_k1=71.53, mixed_k5=90.45,
                                content_k1=81.40, context_k1=56.08),
    "public_bi":       PaperRef(mixed_k1=65.89, mixed_k5=70.90,
                                content_k1=71.10, context_k1=57.75),
    "chicago_open":    PaperRef(mixed_k1=56.38, mixed_k5=76.76,
                                content_k1=55.09, context_k1=52.25),
    "fetaqa":          PaperRef(mixed_k1=53.71, mixed_k5=59.37,
                                content_k1=56.14, context_k1=46.96),
}

# Aliases tolerated when reading the dataset column / directory.
DATASET_ALIASES: dict[str, str] = {
    "chembl_10k":   "chembl",
    "public":       "public_bi",
    "chicago":      "chicago_open",
    "chicago_10k":  "chicago_open",
    "adventure":    "adventure_works",
}


def _normalise(name: str) -> str:
    return DATASET_ALIASES.get(name, name)


def _latest_run_dir(dataset_dir: Path) -> Path | None:
    """Return the newest timestamped run dir that contains summary.csv."""
    candidates = [
        d for d in dataset_dir.iterdir()
        if d.is_dir() and (d / "summary.csv").exists()
    ]
    if not candidates:
        return None
    # Sort lexicographically; the YYYY-MM-DDTHH-MM-SS layout is collation-safe.
    return sorted(candidates, key=lambda p: p.name)[-1]


def _read_summary(path: Path) -> list[dict]:
    """Parse summary.csv into a list of dicts."""
    with path.open() as f:
        return list(csv.DictReader(f))


def _filter(rows: list[dict], *, rerank_mode: str, k_in: set[int]) -> list[dict]:
    result = []
    for r in rows:
        if "rerank_mode" not in r:
            continue
        if r["rerank_mode"] == rerank_mode and int(r["k"]) in k_in:
            result.append(r)
    return result


def _weighted_mean_hit_rate(rows: list[dict]) -> float | None:
    """BC2+BX2 weighted mean, weighted by n_questions (the paper's own scheme).

    Returns None when no rows are supplied or total n_questions is zero, so
    callers can distinguish 'missing data' from a true 0.0 hit-rate.
    """
    if not rows:
        return None
    total = sum(float(r["hit_rate"]) * int(r["n_questions"]) for r in rows)
    n = sum(int(r["n_questions"]) for r in rows)
    return total / n if n else None


def _fmt_or_dash(value: float | None, width: int = 8, decimals: int = 2) -> str:
    """Format a float to fixed width; emit '---' aligned-width when None."""
    if value is None:
        return "---".rjust(width)
    return f"{value:>{width}.{decimals}f}"


def _delta_or_dash(actual: float | None, ref: float, width: int = 6) -> str:
    """Format a +/- delta vs ref; emit '---' aligned-width when actual is None."""
    if actual is None:
        return "---".rjust(width)
    return f"{actual - ref:+{width}.2f}"


def _print_panel_1(rows_by_ds: dict[str, list[dict]]) -> None:
    print()
    print("=" * 100)
    print("Panel 1: Paper-faithful comparison")
    print("  content_k1 / context_k1 vs Table 2 (§7.3.2 Hybrid, no judge)  -> Blend rerank=off")
    print("  mixed_k1   / mixed_k5   vs Fig. 6  (§7.1.1 full Pneuma)       -> Blend rerank=on")
    print("=" * 100)
    print(f"{'dataset':18} | {'mixed_k1':>8} | {'paper_k1':>8} | {'Δ pp':>6} | "
          f"{'mixed_k5':>8} | {'paper_k5':>8} | {'Δ pp':>6} | "
          f"{'content_k1':>10} | {'paper':>6} | {'Δ pp':>6} | "
          f"{'context_k1':>10} | {'paper':>6} | {'Δ pp':>6}")
    print("-" * 100)
    any_data = False
    for ds, rows in sorted(rows_by_ds.items()):
        if ds not in PAPER:
            continue
        ref = PAPER[ds]

        # Table 2 metrics (no judge) -> rerank=off
        off = _filter(rows, rerank_mode="off", k_in={1})
        bc2_off_k1 = next((r for r in off if r["family"] == "BC2"), None)
        bx2_off_k1 = next((r for r in off if r["family"] == "BX2"), None)

        # Fig. 6 metrics (with judge) -> rerank=on
        on = _filter(rows, rerank_mode="on", k_in={1, 5})
        bc2_on = [r for r in on if r["family"] == "BC2"]
        bx2_on = [r for r in on if r["family"] == "BX2"]

        def mixed_on(k: int) -> float | None:
            return _weighted_mean_hit_rate(
                [r for r in bc2_on if int(r["k"]) == k]
                + [r for r in bx2_on if int(r["k"]) == k]
            )

        if not (bc2_off_k1 or bx2_off_k1 or bc2_on or bx2_on):
            continue

        m1 = mixed_on(1)
        m5 = mixed_on(5)
        c1 = float(bc2_off_k1["hit_rate"]) if bc2_off_k1 else None
        x1 = float(bx2_off_k1["hit_rate"]) if bx2_off_k1 else None
        print(
            f"{ds:18} | {_fmt_or_dash(m1)} | {ref.mixed_k1:8.2f} | {_delta_or_dash(m1, ref.mixed_k1)} | "
            f"{_fmt_or_dash(m5)} | {ref.mixed_k5:8.2f} | {_delta_or_dash(m5, ref.mixed_k5)} | "
            f"{_fmt_or_dash(c1, width=10)} | {ref.content_k1:6.2f} | {_delta_or_dash(c1, ref.content_k1)} | "
            f"{_fmt_or_dash(x1, width=10)} | {ref.context_k1:6.2f} | {_delta_or_dash(x1, ref.context_k1)}"
        )
        any_data = True
    if not any_data:
        print("(no data - run `make benchmark DATASET=<ds>`)")


def _print_panel_2(rows_by_ds: dict[str, list[dict]]) -> None:
    print()
    print("=" * 100)
    print("Panel 2: Judge uplift (rerank=on column already feeds Panel 1's mixed_*)")
    print("  Uplift_k1 = mixed_k1(rerank=on) - mixed_k1(rerank=off)")
    print("  i.e. how much the LLM judge adds over Hybrid Retrieval alone (paper §7.3.3 Fig. 13).")
    print("=" * 100)
    print(f"{'dataset':18} | {'mixed_k1':>8} | {'mixed_k5':>8} | "
          f"{'content_k1':>10} | {'context_k1':>10} | {'uplift_k1':>9}")
    print("-" * 100)
    any_data = False
    for ds, rows in sorted(rows_by_ds.items()):
        on_rows = _filter(rows, rerank_mode="on", k_in={1, 5})
        if not on_rows:
            continue
        off_rows = _filter(rows, rerank_mode="off", k_in={1})
        bc2_on = [r for r in on_rows if r["family"] == "BC2"]
        bx2_on = [r for r in on_rows if r["family"] == "BX2"]

        def mixed_on(k: int) -> float | None:
            return _weighted_mean_hit_rate(
                [r for r in bc2_on if int(r["k"]) == k]
                + [r for r in bx2_on if int(r["k"]) == k]
            )

        m1_on = mixed_on(1)
        m5_on = mixed_on(5)
        c1_on = next((float(r["hit_rate"]) for r in bc2_on if int(r["k"]) == 1), None)
        x1_on = next((float(r["hit_rate"]) for r in bx2_on if int(r["k"]) == 1), None)

        # Compute uplift: on - off at k=1 (mixed). None when either side missing.
        m1_off = _weighted_mean_hit_rate(
            [r for r in off_rows if r["family"] == "BC2"]
            + [r for r in off_rows if r["family"] == "BX2"]
        )
        if m1_on is None or m1_off is None:
            uplift_str = "---".rjust(9)
        else:
            uplift_str = f"{m1_on - m1_off:+9.2f}"

        print(
            f"{ds:18} | {_fmt_or_dash(m1_on)} | {_fmt_or_dash(m5_on)} | "
            f"{_fmt_or_dash(c1_on, width=10)} | {_fmt_or_dash(x1_on, width=10)} | {uplift_str}"
        )
        any_data = True
    if not any_data:
        print("(no data - run `make benchmark DATASET=<ds>`)")


def _print_panel_3(rows_by_ds: dict[str, list[dict]]) -> None:
    print()
    print("=" * 100)
    print("Panel 3: k-sweep ablation (rerank=off; matches PNEUMA §7.4.2 Fig. 15)")
    print("=" * 100)
    print(f"{'dataset':18} | {'k=1':>6} | {'k=5':>6} | {'k=10':>6} | {'k=30':>6} | {'k=50':>6}")
    print("-" * 100)
    any_data = False
    for ds, rows in sorted(rows_by_ds.items()):
        off = _filter(rows, rerank_mode="off", k_in={1, 5, 10, 30, 50})
        bc2 = {int(r["k"]): float(r["hit_rate"]) for r in off if r["family"] == "BC2"}
        if not bc2:
            continue
        print(
            f"{ds:18} | {bc2.get(1, 0.0):6.2f} | {bc2.get(5, 0.0):6.2f} | "
            f"{bc2.get(10, 0.0):6.2f} | {bc2.get(30, 0.0):6.2f} | {bc2.get(50, 0.0):6.2f}"
        )
        any_data = True
    if not any_data:
        print("(no data - run `make benchmark DATASET=<ds>`)")


def _print_panel_4(rows_by_ds: dict[str, list[dict]]) -> None:
    print()
    print("=" * 100)
    print("Panel 4: Online query throughput (paper §7.2.1 / Fig. 9)")
    print("  Paper: 100-question subset, 10x averaged, full Pneuma (=judge on) at k=1.")
    print("  q/s = 1000 / latency_ms_mean of the matching cell. Compare paper Fig. 9 against")
    print("  the 'k=1 rerank=on' rows below; the rerank=off rows show the Hybrid-only baseline.")
    print("=" * 100)
    print(
        f"{'dataset':18} | {'family':6} | {'rerank':6} | "
        f"{'k=1 mean':>9} | {'k=1 p50':>7} | {'k=1 p95':>7} | "
        f"{'k=5 mean':>9} | {'k=5 p50':>7} | {'k=5 p95':>7} | "
        f"{'vector_p50':>10} | {'vector_p95':>10} | {'q/s':>6}"
    )
    print("-" * 100)
    any_data = False
    for ds, rows in sorted(rows_by_ds.items()):
        for fam in ("BC2", "BX2"):
            for mode in ("off", "on"):
                k1 = next(
                    (r for r in rows
                     if r.get("family") == fam
                     and r.get("rerank_mode") == mode
                     and int(r["k"]) == 1),
                    None,
                )
                k5 = next(
                    (r for r in rows
                     if r.get("family") == fam
                     and r.get("rerank_mode") == mode
                     and int(r["k"]) == 5),
                    None,
                )
                if not k1 or not k5:
                    continue
                lat_k1_mean = k1.get("latency_ms_mean", "---")
                lat_k1_p50 = k1.get("latency_ms_p50", "---")
                lat_k1_p95 = k1.get("latency_ms_p95", "---")
                lat_k5_mean = k5.get("latency_ms_mean", "---")
                lat_k5_p50 = k5.get("latency_ms_p50", "---")
                lat_k5_p95 = k5.get("latency_ms_p95", "---")
                vec_p50 = k1.get("vector_ms_p50", "---")
                vec_p95 = k1.get("vector_ms_p95", "---")

                # Per-cell throughput: 1000 / mean k=1 latency. This is the
                # apples-to-apples q/s for paper Fig. 9 when rerank=on (full
                # Pneuma) and the no-judge baseline when rerank=off.
                try:
                    qs_value = 1000.0 / float(lat_k1_mean)
                    qs = f"{qs_value:6.2f}"
                except (ValueError, TypeError, ZeroDivisionError):
                    qs = "  ----"

                def _f1(v: str) -> str:
                    try:
                        return f"{float(v):7.1f}"
                    except (ValueError, TypeError):
                        return f"{'---':>7}"

                def _f1m(v: str) -> str:
                    try:
                        return f"{float(v):9.1f}"
                    except (ValueError, TypeError):
                        return f"{'---':>9}"

                def _f2(v: str) -> str:
                    try:
                        return f"{float(v):10.2f}"
                    except (ValueError, TypeError):
                        return f"{'---':>10}"

                print(
                    f"{ds:18} | {fam:6} | {mode:6} | "
                    f"{_f1m(lat_k1_mean)} | {_f1(lat_k1_p50)} | {_f1(lat_k1_p95)} | "
                    f"{_f1m(lat_k5_mean)} | {_f1(lat_k5_p50)} | {_f1(lat_k5_p95)} | "
                    f"{_f2(vec_p50)} | {_f2(vec_p95)} | "
                    f"{qs}"
                )
                any_data = True
    if not any_data:
        print("(no data - run `make benchmark DATASET=<ds>`)")


@dataclass(frozen=True)
class Comparison:
    """Per-dataset Blend numbers paired with their paper references.

    Built by ``compare_one`` so the plot script and the printed panels share
    one source of truth on the rerank-mode mapping (Table 2 -> rerank=off,
    Fig. 6 -> rerank=on)."""
    dataset: str
    ours_mixed_k1: float | None
    ours_mixed_k5: float | None
    ours_content_k1: float | None
    ours_context_k1: float | None


def compare_one(dataset: str, run_dir: Path) -> Comparison:
    """Extract Blend's four headline numbers for one dataset.

    The mode mapping mirrors ``_print_panel_1``: content/context_k1 come from
    the rerank=off rows (paper Table 2 setup, no judge); mixed_k1/k5 come
    from rerank=on rows (paper Fig. 6 setup, full Pneuma with judge).
    """
    rows = _read_summary(run_dir / "summary.csv")

    off = _filter(rows, rerank_mode="off", k_in={1})
    bc2_off_k1 = next((r for r in off if r["family"] == "BC2"), None)
    bx2_off_k1 = next((r for r in off if r["family"] == "BX2"), None)

    on = _filter(rows, rerank_mode="on", k_in={1, 5})
    bc2_on = [r for r in on if r["family"] == "BC2"]
    bx2_on = [r for r in on if r["family"] == "BX2"]

    def _mixed_on(k: int) -> float | None:
        return _weighted_mean_hit_rate(
            [r for r in bc2_on if int(r["k"]) == k]
            + [r for r in bx2_on if int(r["k"]) == k]
        )

    return Comparison(
        dataset=dataset,
        ours_mixed_k1=_mixed_on(1),
        ours_mixed_k5=_mixed_on(5),
        ours_content_k1=float(bc2_off_k1["hit_rate"]) if bc2_off_k1 else None,
        ours_context_k1=float(bx2_off_k1["hit_rate"]) if bx2_off_k1 else None,
    )


def main_print_panels(*, results_root: Path | None = None) -> None:
    """Load all datasets from results_root and print the four comparison panels."""
    if results_root is None:
        results_root = _project_root() / "benchmark-data" / "results"

    rows_by_ds: dict[str, list[dict]] = {}

    if not results_root.exists():
        print(f"(no results root found at {results_root})", file=sys.stderr)
        _print_panel_1({})
        _print_panel_2({})
        _print_panel_3({})
        _print_panel_4({})
        return

    for ds_dir in sorted(results_root.iterdir()):
        if not ds_dir.is_dir():
            continue
        latest = _latest_run_dir(ds_dir)
        if not latest:
            continue
        ds = _normalise(ds_dir.name)
        summary_rows = _read_summary(latest / "summary.csv")
        # Inject run_dir into each row so Panel 4 can find run_meta.json
        for r in summary_rows:
            r["_run_dir"] = str(latest)
        rows_by_ds[ds] = summary_rows

    _print_panel_1(rows_by_ds)
    _print_panel_2(rows_by_ds)
    _print_panel_3(rows_by_ds)
    _print_panel_4(rows_by_ds)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--results-root",
        type=Path,
        default=_project_root() / "benchmark-data" / "results",
        help="Root containing <dataset>/<timestamp>/summary.csv (default: benchmark-data/results).",
    )
    args = p.parse_args(argv)

    main_print_panels(results_root=args.results_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
