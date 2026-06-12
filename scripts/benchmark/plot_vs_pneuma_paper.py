"""Plot Blend benchmark hit-rates side-by-side with PNEUMA paper figures.

Reuses compare_to_pneuma_paper's data-loading so the numbers in the chart
are exactly the ones printed by `python -m scripts.benchmark.compare_to_pneuma_paper`.

Usage:
    python -m scripts.benchmark.plot_vs_pneuma_paper                # writes plot to benchmark-data/results/vs_pneuma.png
    python -m scripts.benchmark.plot_vs_pneuma_paper --out fig.pdf  # any matplotlib-supported extension
    python -m scripts.benchmark.plot_vs_pneuma_paper --show         # also opens an interactive window
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.benchmark.compare_to_pneuma_paper import (
    PAPER,
    Comparison,
    _latest_run_dir,
    _normalise,
    _project_root,
    compare_one,
)


# (k, metric_key, attr_on_Comparison, attr_on_PaperRef, source_label)
METRICS = [
    (1, "mixed",   "ours_mixed_k1",   "mixed_k1",   "Fig. 6a"),
    (5, "mixed",   "ours_mixed_k5",   "mixed_k5",   "Fig. 6b"),
    (1, "content", "ours_content_k1", "content_k1", "Table 2 (BC2)"),
    (1, "context", "ours_context_k1", "context_k1", "Table 2 (BX2)"),
]


def _collect(results_root: Path) -> list[Comparison]:
    """One Comparison per dataset that has both paper reference and a run dir."""
    cmps: list[Comparison] = []
    for ds_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        ds = _normalise(ds_dir.name)
        if ds not in PAPER:
            continue
        run = _latest_run_dir(ds_dir)
        if run is None:
            continue
        cmps.append(compare_one(ds, run))
    return cmps


def plot(cmps: list[Comparison], out_path: Path, show: bool = False) -> None:
    datasets = [c.dataset for c in cmps]
    n_ds = len(datasets)
    if n_ds == 0:
        raise SystemExit("no datasets to plot — did the run directories survive the rsync?")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharey=True)
    axes = axes.ravel()

    x = np.arange(n_ds)
    width = 0.38

    ours_color = "#1f6feb"   # blue
    paper_color = "#bdbdbd"  # neutral grey for the reference

    for ax, (k, metric, ours_attr, paper_attr, source) in zip(axes, METRICS):
        ours_vals = [getattr(c, ours_attr) or 0.0 for c in cmps]
        paper_vals = [getattr(PAPER[c.dataset], paper_attr) or 0.0 for c in cmps]

        b_ours = ax.bar(x - width / 2, ours_vals, width,
                        label="Ours (NLSeeker)", color=ours_color, edgecolor="black", linewidth=0.4)
        b_paper = ax.bar(x + width / 2, paper_vals, width,
                         label="PNEUMA paper", color=paper_color, edgecolor="black", linewidth=0.4)

        # Numeric labels above each bar
        for bars, vals in ((b_ours, ours_vals), (b_paper, paper_vals)):
            for rect, v in zip(bars, vals):
                ax.text(rect.get_x() + rect.get_width() / 2, v + 0.6, f"{v:.1f}",
                        ha="center", va="bottom", fontsize=8)

        ax.set_title(f"{metric} @ k={k}   ({source})", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0, 105)
        ax.set_ylabel("hit-rate (%)")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 0.98), frameon=False, fontsize=11)
    fig.suptitle("Blend (NLSeeker) vs. PNEUMA paper — hit-rate by dataset",
                 y=1.0, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--results-root",
        type=Path,
        default=_project_root() / "benchmark-data" / "results",
        help="Root containing <dataset>/<timestamp>/summary.csv (default: benchmark-data/results).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output image path (default: <results-root>/vs_pneuma.png). Extension picks the format.",
    )
    p.add_argument("--show", action="store_true", help="Also open an interactive window.")
    args = p.parse_args(argv)

    cmps = _collect(args.results_root)
    out = args.out or (args.results_root / "vs_pneuma.png")
    plot(cmps, out, show=args.show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
