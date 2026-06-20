"""Plot Blend ↔ PNEUMA results, one PNG per paper figure.

Reuses compare_v2's data-loading so the plotted numbers match the printed
side-by-side report exactly.

Usage:
    python -m scripts.benchmark.plot_v2                    # writes PNGs to benchmark-data/results/plots/
    python -m scripts.benchmark.plot_v2 --out-dir my-plots
    python -m scripts.benchmark.plot_v2 --show             # also pop up windows

Figures produced (one PNG each):
    fig06_overall_hitrate.png         — Fig. 6, full pipeline, k=1 and k=5
    table02_hybrid_no_judge.png       — Table 2, hybrid no judge, per lane
    fig13_judge_uplift.png            — Fig. 13, judge uplift, per lane
    fig15_k_sweep.png                 — Fig. 15, k-sweep ablation, per lane
    fig09_throughput_rerank_on.png    — Fig. 9, full pipeline q/s only (paper-faithful)
    fig09_throughput_rerank_both.png  — Fig. 9 + hybrid-only baseline side-by-side
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.benchmark.compare_v2 import (
    PAPER,
    _filter,
    _latest_run_dir,
    _normalise,
    _project_root,
    _read_summary,
    _weighted_mean_hit_rate,
)


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

BLEND_COLOR = "#1f6feb"   # blue
PNEUMA_COLOR = "#bdbdbd"  # neutral grey for the reference

DATASET_ORDER = ["chembl", "adventure_works", "public_bi", "chicago_open", "fetaqa"]


def _load(results_root: Path) -> dict[str, list[dict]]:
    """Return rows_by_ds, dataset → list-of-rows, only datasets the paper covers."""
    rows_by_ds: dict[str, list[dict]] = {}
    if not results_root.exists():
        return rows_by_ds
    for ds_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        latest = _latest_run_dir(ds_dir)
        if not latest:
            continue
        ds = _normalise(ds_dir.name)
        if ds not in PAPER:
            continue
        rows_by_ds[ds] = _read_summary(latest / "summary.csv")
    return rows_by_ds


def _ordered(rows_by_ds: dict[str, list[dict]]) -> list[str]:
    """Return datasets actually present, in the canonical paper order."""
    return [d for d in DATASET_ORDER if d in rows_by_ds]


def _grouped_bars(
    ax,
    datasets: list[str],
    blend_vals: list[float | None],
    pneuma_vals: list[float | None],
    *,
    title: str,
    ylabel: str,
    ymax: float | None = None,
    fmt: str = "{:.1f}",
) -> None:
    """Two-bar grouped chart: Blend on the left, PNEUMA on the right."""
    x = np.arange(len(datasets))
    width = 0.38

    b = [v if v is not None else 0.0 for v in blend_vals]
    p = [v if v is not None else 0.0 for v in pneuma_vals]

    bars_b = ax.bar(
        x - width / 2, b, width,
        label="Blend (NLSeeker)", color=BLEND_COLOR,
        edgecolor="black", linewidth=0.4,
    )
    bars_p = ax.bar(
        x + width / 2, p, width,
        label="PNEUMA", color=PNEUMA_COLOR,
        edgecolor="black", linewidth=0.4,
    )

    for bars, raw in ((bars_b, blend_vals), (bars_p, pneuma_vals)):
        for rect, v in zip(bars, raw):
            if v is None:
                continue
            ax.text(
                rect.get_x() + rect.get_width() / 2, v + 0.6,
                fmt.format(v), ha="center", va="bottom", fontsize=8,
            )

    ax.set_title(title, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    if ymax is not None:
        ax.set_ylim(0, ymax)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Fig. 6 — full pipeline overall hit rate
# ---------------------------------------------------------------------------

def plot_fig6(rows_by_ds: dict[str, list[dict]], out: Path) -> None:
    datasets = _ordered(rows_by_ds)
    blend_k1, blend_k5, paper_k1, paper_k5 = [], [], [], []
    for ds in datasets:
        on = _filter(rows_by_ds[ds], rerank_mode="on", k_in={1, 5})
        bc = [r for r in on if r["family"] == "BC2"]
        bx = [r for r in on if r["family"] == "BX2"]
        blend_k1.append(_weighted_mean_hit_rate(
            [r for r in bc if int(r["k"]) == 1] + [r for r in bx if int(r["k"]) == 1]
        ))
        blend_k5.append(_weighted_mean_hit_rate(
            [r for r in bc if int(r["k"]) == 5] + [r for r in bx if int(r["k"]) == 5]
        ))
        paper_k1.append(PAPER[ds].mixed_k1)
        paper_k5.append(PAPER[ds].mixed_k5)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    _grouped_bars(axes[0], datasets, blend_k1, paper_k1,
                  title="k = 1 (Fig. 6a)", ylabel="hit-rate (%)", ymax=105)
    _grouped_bars(axes[1], datasets, blend_k5, paper_k5,
                  title="k = 5 (Fig. 6b)", ylabel="hit-rate (%)", ymax=105)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 0.98), frameon=False, fontsize=11)
    fig.suptitle(
        "Overall hit rate, full pipeline (judge on, BC2 + BX2 weighted mean)",
        y=1.02, fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Table 2 — hybrid retrieval, no judge, per lane
# ---------------------------------------------------------------------------

def plot_table2(rows_by_ds: dict[str, list[dict]], out: Path) -> None:
    datasets = _ordered(rows_by_ds)
    blend_c, blend_x, paper_c, paper_x = [], [], [], []
    for ds in datasets:
        off = _filter(rows_by_ds[ds], rerank_mode="off", k_in={1})
        bc = next((r for r in off if r["family"] == "BC2"), None)
        bx = next((r for r in off if r["family"] == "BX2"), None)
        blend_c.append(float(bc["hit_rate"]) if bc else None)
        blend_x.append(float(bx["hit_rate"]) if bx else None)
        paper_c.append(PAPER[ds].content_k1)
        paper_x.append(PAPER[ds].context_k1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    _grouped_bars(axes[0], datasets, blend_c, paper_c,
                  title="content (BC2)", ylabel="hit-rate (%)", ymax=105)
    _grouped_bars(axes[1], datasets, blend_x, paper_x,
                  title="context (BX2)", ylabel="hit-rate (%)", ymax=105)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 0.98), frameon=False, fontsize=11)
    fig.suptitle(
        "Hybrid retrieval (no judge), per lane, k = 1",
        y=1.02, fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Fig. 13 — judge uplift over hybrid, per lane
# ---------------------------------------------------------------------------

def plot_fig13(rows_by_ds: dict[str, list[dict]], out: Path) -> None:
    datasets = _ordered(rows_by_ds)
    blend_c, blend_x, paper_c, paper_x = [], [], [], []
    for ds in datasets:
        on = _filter(rows_by_ds[ds], rerank_mode="on", k_in={1})
        off = _filter(rows_by_ds[ds], rerank_mode="off", k_in={1})
        bc_on = next((r for r in on if r["family"] == "BC2"), None)
        bx_on = next((r for r in on if r["family"] == "BX2"), None)
        bc_off = next((r for r in off if r["family"] == "BC2"), None)
        bx_off = next((r for r in off if r["family"] == "BX2"), None)
        blend_c.append(
            float(bc_on["hit_rate"]) - float(bc_off["hit_rate"])
            if bc_on and bc_off else None
        )
        blend_x.append(
            float(bx_on["hit_rate"]) - float(bx_off["hit_rate"])
            if bx_on and bx_off else None
        )
        ref = PAPER[ds]
        paper_c.append(
            ref.fig13_content_k1_qwen - ref.content_k1
            if ref.fig13_content_k1_qwen is not None else None
        )
        paper_x.append(
            ref.fig13_context_k1_qwen - ref.context_k1
            if ref.fig13_context_k1_qwen is not None else None
        )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    _grouped_bars(axes[0], datasets, blend_c, paper_c,
                  title="content (BC2)", ylabel="judge uplift (pp)", fmt="{:+.2f}")
    _grouped_bars(axes[1], datasets, blend_x, paper_x,
                  title="context (BX2)", ylabel="judge uplift (pp)", fmt="{:+.2f}")
    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 0.98), frameon=False, fontsize=11)
    fig.suptitle(
        "LLM Judge uplift over hybrid retrieval, per lane (k = 1)",
        y=1.02, fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Fig. 15 — k-sweep ablation, hybrid no judge, per lane
# ---------------------------------------------------------------------------

def plot_fig15(rows_by_ds: dict[str, list[dict]], out: Path) -> None:
    datasets = _ordered(rows_by_ds)
    ks = [1, 5, 10, 30, 50]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, lane_label, lane_attr, family in (
        (axes[0], "content (BC2)", "fig15_content_k", "BC2"),
        (axes[1], "context (BX2)", "fig15_context_k", "BX2"),
    ):
        for ds in datasets:
            off = _filter(rows_by_ds[ds], rerank_mode="off", k_in=set(ks))
            blend_by_k = {
                int(r["k"]): float(r["hit_rate"])
                for r in off if r["family"] == family
            }
            paper_by_k = getattr(PAPER[ds], lane_attr) or {}

            blend_xs = [k for k in ks if k in blend_by_k]
            blend_ys = [blend_by_k[k] for k in blend_xs]
            paper_xs = [k for k in ks if k in paper_by_k]
            paper_ys = [paper_by_k[k] for k in paper_xs]

            line, = ax.plot(blend_xs, blend_ys, marker="o", linewidth=1.7, label=ds)
            ax.plot(paper_xs, paper_ys, marker="x", linestyle="--",
                    linewidth=1.2, color=line.get_color(), alpha=0.7)

        ax.set_title(lane_label, fontsize=11)
        ax.set_xlabel("k")
        ax.set_xticks(ks)
        ax.set_ylabel("hit-rate (%)")
        ax.set_ylim(0, 105)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, loc="lower right", frameon=True)

    # Solid = Blend, dashed = PNEUMA estimate.
    style_handles = [
        plt.Line2D([0], [0], color="black", marker="o", linewidth=1.7, label="Blend"),
        plt.Line2D([0], [0], color="black", marker="x", linestyle="--",
                   linewidth=1.2, alpha=0.7, label="PNEUMA"),
    ]
    fig.legend(handles=style_handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 0.98), frameon=False, fontsize=11)
    fig.suptitle(
        "k-sweep, hybrid retrieval (no judge), per lane",
        y=1.02, fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Fig. 9 — online query throughput
# ---------------------------------------------------------------------------

def _qps(row: dict | None) -> float | None:
    if not row:
        return None
    try:
        return 1000.0 / float(row.get("latency_ms_mean", "nan"))
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _qps_by_ds(rows_by_ds: dict[str, list[dict]], *, mode: str) -> list[float | None]:
    """Mean q/s across BC2+BX2 at k=1 for the given rerank mode."""
    out: list[float | None] = []
    for ds in _ordered(rows_by_ds):
        rows = rows_by_ds[ds]
        bc = next((r for r in rows
                   if r["family"] == "BC2"
                   and r["rerank_mode"] == mode
                   and int(r["k"]) == 1), None)
        bx = next((r for r in rows
                   if r["family"] == "BX2"
                   and r["rerank_mode"] == mode
                   and int(r["k"]) == 1), None)
        vals = [v for v in (_qps(bc), _qps(bx)) if v is not None]
        out.append(sum(vals) / len(vals) if vals else None)
    return out


def plot_fig9_rerank_on(rows_by_ds: dict[str, list[dict]], out: Path) -> None:
    """Paper-faithful Fig. 9: only rerank=on (full pipeline)."""
    datasets = _ordered(rows_by_ds)
    blend_qps = _qps_by_ds(rows_by_ds, mode="on")
    paper_qps = [PAPER[d].fig9_qps for d in datasets]

    fig, ax = plt.subplots(figsize=(11, 5))
    _grouped_bars(
        ax, datasets, blend_qps, paper_qps,
        title="full pipeline (rerank = on, k = 1)",
        ylabel="queries / second",
        ymax=max(
            max((v or 0.0 for v in blend_qps), default=0.0),
            max((v or 0.0 for v in paper_qps), default=0.0),
        ) * 1.25 + 0.5,
        fmt="{:.2f}",
    )
    ax.legend(loc="upper right", fontsize=10, frameon=True)
    fig.suptitle(
        "Online query throughput",
        y=1.0, fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_fig9_rerank_both(rows_by_ds: dict[str, list[dict]], out: Path) -> None:
    """Side-by-side: rerank=off (hybrid only) vs rerank=on (full pipeline).

    Two subplots, one per rerank mode, so you can see how much latency the
    LLM judge owns. PNEUMA's Fig. 9 q/s only makes sense for rerank=on, so
    the rerank=off subplot has no paper bars (only Blend).
    """
    datasets = _ordered(rows_by_ds)
    blend_off = _qps_by_ds(rows_by_ds, mode="off")
    blend_on = _qps_by_ds(rows_by_ds, mode="on")
    paper_qps = [PAPER[d].fig9_qps for d in datasets]
    paper_blank: list[float | None] = [None] * len(datasets)

    # Independent y-axes — off is ~7-10× faster than on.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _grouped_bars(
        axes[0], datasets, blend_off, paper_blank,
        title="rerank = off (hybrid only)",
        ylabel="queries / second",
        ymax=max((v or 0.0 for v in blend_off), default=1.0) * 1.2 + 0.5,
        fmt="{:.2f}",
    )
    _grouped_bars(
        axes[1], datasets, blend_on, paper_qps,
        title="rerank = on (full pipeline)",
        ylabel="queries / second",
        ymax=max(
            max((v or 0.0 for v in blend_on), default=0.0),
            max((v or 0.0 for v in paper_qps), default=0.0),
        ) * 1.25 + 0.5,
        fmt="{:.2f}",
    )
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 0.98), frameon=False, fontsize=11)
    fig.suptitle(
        "Online query throughput, rerank = off vs on (k = 1, BC2 + BX2 mean)",
        y=1.02, fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Latency breakdowns — matvec (brute-force) and judge cost
# ---------------------------------------------------------------------------

def _row(rows: list[dict], *, family: str, mode: str, k: int) -> dict | None:
    return next(
        (r for r in rows
         if r["family"] == family and r["rerank_mode"] == mode and int(r["k"]) == k),
        None,
    )


def plot_matvec_cost(rows_by_ds: dict[str, list[dict]], out: Path) -> None:
    """Per-query brute-force matvec cost, p50 and p95, from rerank=off rows.

    rerank=off is the only mode where the matvec actually runs end-to-end;
    rerank=on rows serve the second pass from cache, so their vector_* are
    sub-millisecond ghost numbers and not representative of brute-force cost.
    BC2 + BX2 are averaged because both lanes hit the same vector index.
    """
    datasets = _ordered(rows_by_ds)
    p50_vals: list[float | None] = []
    p95_vals: list[float | None] = []
    for ds in datasets:
        rows = rows_by_ds[ds]
        bc = _row(rows, family="BC2", mode="off", k=1)
        bx = _row(rows, family="BX2", mode="off", k=1)
        p50s = [
            float(r["vector_ms_p50"])
            for r in (bc, bx)
            if r and r.get("vector_ms_p50") not in (None, "", "---")
        ]
        p95s = [
            float(r["vector_ms_p95"])
            for r in (bc, bx)
            if r and r.get("vector_ms_p95") not in (None, "", "---")
        ]
        p50_vals.append(sum(p50s) / len(p50s) if p50s else None)
        p95_vals.append(sum(p95s) / len(p95s) if p95s else None)

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(datasets))
    width = 0.38

    p50 = [v if v is not None else 0.0 for v in p50_vals]
    p95 = [v if v is not None else 0.0 for v in p95_vals]
    bars_50 = ax.bar(x - width / 2, p50, width,
                     label="p50 (median)", color="#1f6feb",
                     edgecolor="black", linewidth=0.4)
    bars_95 = ax.bar(x + width / 2, p95, width,
                     label="p95 (slow tail)", color="#7aa9f7",
                     edgecolor="black", linewidth=0.4)
    for bars, raw in ((bars_50, p50_vals), (bars_95, p95_vals)):
        for rect, v in zip(bars, raw):
            if v is None:
                continue
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.3,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ymax = max(max(p50, default=0.0), max(p95, default=0.0)) * 1.25 + 1.0
    ax.set_title("Per-query matvec cost = brute-force exact NN (rerank = off, k = 1)",
                 fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("milliseconds")
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=10, frameon=True)
    fig.suptitle(
        "Brute-force vector retrieval cost (numpy matvec, BC2 + BX2 mean)",
        y=1.0, fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_judge_cost(rows_by_ds: dict[str, list[dict]], out: Path) -> None:
    """LLM judge cost per query, read DIRECTLY from ``judge_ms_*`` and
    ``judge_calls_mean`` columns in summary.csv.

    Earlier versions of this panel approximated judge cost by
    ``latency(rerank=on) − latency(rerank=off)``, which conflated judge
    LLM time with rerank-driven candidate-set differences.  The bench now
    instruments the judge call directly via ``JUDGE_TIMER`` (Blend's
    NLSeeker rerank path) and via the patched
    ``HybridRetriever._llm_rerank`` (PNEUMA side).  We surface the
    measured columns instead of the subtraction.

    BC2 + BX2 averaged at k=1 (matches the prior subtraction-based panel
    for backward continuity in the figure).
    """
    datasets = _ordered(rows_by_ds)

    def _direct(stat_key: str) -> list[float | None]:
        """Read judge_ms_* directly from rerank=on rows.  rerank=off rows
        are zero by construction (JUDGE_TIMER never fires there); we
        average across BC2 + BX2 at k=1.
        """
        vals: list[float | None] = []
        for ds in datasets:
            rows = rows_by_ds[ds]
            bc_on = _row(rows, family="BC2", mode="on", k=1)
            bx_on = _row(rows, family="BX2", mode="on", k=1)
            measurements: list[float] = []
            for on_r in (bc_on, bx_on):
                if not on_r:
                    continue
                raw = on_r.get(stat_key)
                if raw in (None, "", "None"):
                    continue
                try:
                    measurements.append(float(raw))
                except (TypeError, ValueError):
                    pass
            vals.append(
                sum(measurements) / len(measurements) if measurements else None
            )
        return vals

    mean_vals = _direct("judge_ms_mean")
    p50_vals = _direct("judge_ms_p50")
    p95_vals = _direct("judge_ms_p95")
    calls_vals = _direct("judge_calls_mean")

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(datasets))
    width = 0.27

    def _draw(offset: float, vals: list[float | None], color: str, label: str):
        raw = [v if v is not None else 0.0 for v in vals]
        bars = ax.bar(x + offset, raw, width, label=label, color=color,
                      edgecolor="black", linewidth=0.4)
        for rect, v in zip(bars, vals):
            if v is None:
                continue
            ax.text(rect.get_x() + rect.get_width() / 2, v + 8,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=8)
        return bars

    _draw(-width, mean_vals, "#1f6feb", "mean")
    _draw(0.0, p50_vals, "#7aa9f7", "p50")
    _draw(+width, p95_vals, "#cfe1fb", "p95")

    # Annotate each dataset with the average judge_calls_mean (HTTP
    # roundtrips per query) so the reader can divide judge_ms by calls
    # to get per-call latency.
    all_vals = [v for v in mean_vals + p50_vals + p95_vals if v is not None]
    ymax = (max(all_vals) if all_vals else 1.0) * 1.2 + 50
    for xi, calls in zip(x, calls_vals):
        if calls is None:
            continue
        ax.text(xi, ymax * 0.95, f"calls/query: {calls:.1f}",
                ha="center", va="top", fontsize=8, color="#444",
                style="italic")

    ax.set_title(
        "Per-query LLM judge cost (direct measurement via JUDGE_TIMER), k = 1",
        fontsize=11,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("milliseconds")
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=10, frameon=True)
    fig.suptitle(
        "LLM judge cost (BC2 + BX2 mean, rerank=on, direct timer)",
        y=1.0, fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Cross-runner cost panels (Task 14)
# ---------------------------------------------------------------------------

def render_vector_cost_panel(rows: list[dict], *, out_path: Path) -> None:
    """Scatter: vector_ms_p50 (x, log scale) vs recall_at_k (y), color by
    vector_index_kind, shape by runner.

    rows fields: runner, family, k, vector_ms_p50, recall_at_k,
    vector_index_kind.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    color_map = {"npy_brute_force": "C0", "chromadb_hnsw_M48": "C1"}
    marker_map = {"blend": "o", "pneuma": "s"}
    for r in rows:
        ax.scatter(
            r["vector_ms_p50"], r["recall_at_k"],
            color=color_map.get(r["vector_index_kind"], "gray"),
            marker=marker_map.get(r["runner"], "x"),
            s=80, alpha=0.8,
            label=f"{r['runner']} · {r['vector_index_kind']}",
        )
    ax.set_xlabel("vector_ms_p50 (log)")
    ax.set_ylabel("recall_at_k")
    ax.set_xscale("log")
    ax.set_title("Vector-search cost vs recall — brute-force matvec vs HNSW")
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen[l] = h
    ax.legend(seen.values(), seen.keys(), loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_judge_cost_panel(rows: list[dict], *, out_path: Path) -> None:
    """Bar plot: judge_ms_p50 by (runner, family, k) for rerank_mode=on."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r for r in rows if r.get("rerank_mode") == "on"]
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = [f"{r['runner']}·{r['family']}·k={r['k']}" for r in rows]
    values = [r["judge_ms_p50"] for r in rows]
    ax.bar(range(len(rows)), values)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("judge_ms_p50")
    ax.set_title("LLM-judge cost per query (rerank=on)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--results-root",
        type=Path,
        default=_project_root() / "benchmark-data" / "results",
        help="Root containing <dataset>/<timestamp>/summary.csv "
             "(default: benchmark-data/results).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for output PNGs (default: <results-root>/plots/).",
    )
    p.add_argument("--show", action="store_true",
                   help="Also open interactive windows (one per figure).")
    args = p.parse_args(argv)

    rows_by_ds = _load(args.results_root)
    if not rows_by_ds:
        raise SystemExit(
            f"no benchmark data found under {args.results_root} — "
            "did you run `make bench`?"
        )

    out_dir = args.out_dir or (args.results_root / "plots")
    plot_fig6(rows_by_ds, out_dir / "fig06_overall_hitrate.png")
    plot_table2(rows_by_ds, out_dir / "table02_hybrid_no_judge.png")
    plot_fig13(rows_by_ds, out_dir / "fig13_judge_uplift.png")
    plot_fig15(rows_by_ds, out_dir / "fig15_k_sweep.png")
    plot_fig9_rerank_on(rows_by_ds, out_dir / "fig09_throughput_rerank_on.png")
    plot_fig9_rerank_both(rows_by_ds, out_dir / "fig09_throughput_rerank_both.png")
    plot_matvec_cost(rows_by_ds, out_dir / "matvec_cost.png")
    plot_judge_cost(rows_by_ds, out_dir / "judge_cost.png")

    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
