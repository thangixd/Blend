"""Compare Blend benchmark hit-rates against the PNEUMA paper figures."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PaperRef:
    mixed_k1: float | None    # Fig. 6a — BC2+BX2 weighted at k=1
    mixed_k5: float | None    # Fig. 6b — BC2+BX2 weighted at k=5
    content_k1: float | None  # Table 2 — BC2 only at k=1
    context_k1: float | None  # Table 2 — BX2 only at k=1


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


@dataclass(frozen=True)
class FamilyRow:
    family: str
    k: int
    n_questions: int
    hit_rate: float


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


def _load_summary(summary_csv: Path) -> list[FamilyRow]:
    rows: list[FamilyRow] = []
    with summary_csv.open() as f:
        for r in csv.DictReader(f):
            rows.append(FamilyRow(
                family=r["family"],
                k=int(r["k"]),
                n_questions=int(r["n_questions"]),
                hit_rate=float(r["hit_rate"]),
            ))
    return rows


def _weighted(content: FamilyRow | None, context: FamilyRow | None) -> float | None:
    """BC2+BX2 weighted mean, weighted by n_questions (the paper's own scheme)."""
    if content is None or context is None:
        return None
    total = content.n_questions + context.n_questions
    if total == 0:
        return None
    return (content.hit_rate * content.n_questions
            + context.hit_rate * context.n_questions) / total

@dataclass(frozen=True)
class Comparison:
    dataset: str
    run_dir: Path
    n_bc2: int
    n_bx2: int
    ours_mixed_k1: float | None
    ours_mixed_k5: float | None
    ours_content_k1: float | None
    ours_context_k1: float | None


def compare_one(dataset: str, run_dir: Path) -> Comparison:
    rows = _load_summary(run_dir / "summary.csv")
    by_fk = {(r.family, r.k): r for r in rows}
    bc2_k1 = by_fk.get(("BC2", 1))
    bx2_k1 = by_fk.get(("BX2", 1))
    bc2_k5 = by_fk.get(("BC2", 5))
    bx2_k5 = by_fk.get(("BX2", 5))
    return Comparison(
        dataset=dataset,
        run_dir=run_dir,
        n_bc2=bc2_k1.n_questions if bc2_k1 else 0,
        n_bx2=bx2_k1.n_questions if bx2_k1 else 0,
        ours_mixed_k1=_weighted(bc2_k1, bx2_k1),
        ours_mixed_k5=_weighted(bc2_k5, bx2_k5),
        ours_content_k1=bc2_k1.hit_rate if bc2_k1 else None,
        ours_context_k1=bx2_k1.hit_rate if bx2_k1 else None,
    )

def _fmt(v: float | None) -> str:
    return "  ---" if v is None else f"{v:6.2f}"


def _fmt_delta(ours: float | None, paper: float | None) -> str:
    if ours is None or paper is None:
        return "   ---"
    d = ours - paper
    sign = "+" if d >= 0 else "−"
    return f"{sign}{abs(d):5.2f}"


def print_table(cmps: list[Comparison]) -> None:
    print()
    print("PNEUMA paper vs. our NLSeeker run — per-dataset, BC2+BX2 weighted by n_questions")
    print("=" * 110)
    header = (
        f"{'dataset':<18} {'k':<2} {'metric':<11} "
        f"{'ours':>6}  {'paper':>6}  {'Δ (pp)':>7}   {'n_BC2':>5} {'n_BX2':>5}  source"
    )
    print(header)
    print("-" * 110)
    for c in cmps:
        ref = PAPER.get(c.dataset)
        if ref is None:
            print(f"{c.dataset:<18}   (no paper reference; skipping)")
            continue
        rows = [
            (1, "mixed",   c.ours_mixed_k1,   ref.mixed_k1,   "Fig. 6a"),
            (5, "mixed",   c.ours_mixed_k5,   ref.mixed_k5,   "Fig. 6b"),
            (1, "content", c.ours_content_k1, ref.content_k1, "Table 2"),
            (1, "context", c.ours_context_k1, ref.context_k1, "Table 2"),
        ]
        for k, metric, ours, paper, src in rows:
            print(
                f"{c.dataset:<18} {k:<2} {metric:<11} "
                f"{_fmt(ours)}  {_fmt(paper)}  {_fmt_delta(ours, paper)}   "
                f"{c.n_bc2:>5} {c.n_bx2:>5}  {src}"
            )
        print(f"{'':<18}   run: {c.run_dir}")
        print("-" * 110)


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
    p.add_argument(
        "--run",
        type=Path,
        action="append",
        default=[],
        help="Pin a specific run directory (repeatable). Overrides auto-discovery for that dataset.",
    )
    p.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Restrict to dataset(s). Aliases (chicago, public, ...) accepted.",
    )
    args = p.parse_args(argv)

    pinned: dict[str, Path] = {}
    for run in args.run:
        run = run.resolve()
        if not (run / "summary.csv").exists():
            print(f"warning: pinned run has no summary.csv: {run}", file=sys.stderr)
            continue
        # Dataset name = parent directory name.
        pinned[_normalise(run.parent.name)] = run

    requested = {_normalise(d) for d in args.dataset} if args.dataset else None

    comparisons: list[Comparison] = []
    if not args.results_root.exists() and not pinned:
        print(f"error: results root does not exist: {args.results_root}", file=sys.stderr)
        return 2

    if args.results_root.exists():
        for ds_dir in sorted(args.results_root.iterdir()):
            if not ds_dir.is_dir():
                continue
            ds = _normalise(ds_dir.name)
            if requested is not None and ds not in requested:
                continue
            run_dir = pinned.get(ds) or _latest_run_dir(ds_dir)
            if run_dir is None:
                continue  # Empty dataset directory; skip silently.
            comparisons.append(compare_one(ds, run_dir))

    # Pinned runs whose dataset directory wasn't in results-root.
    seen = {c.dataset for c in comparisons}
    for ds, run_dir in pinned.items():
        if ds in seen:
            continue
        if requested is not None and ds not in requested:
            continue
        comparisons.append(compare_one(ds, run_dir))

    if not comparisons:
        print("no benchmark runs found.", file=sys.stderr)
        return 1

    print_table(comparisons)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
