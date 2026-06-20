"""Side-by-side Blend ↔ PNEUMA paper comparison, one section per paper figure.

Replacement for compare_to_pneuma_paper.py with a different layout:
each section corresponds to ONE paper figure or table and shows
Blend's number next to PNEUMA's published number, with a Δ pp column
and a paragraph describing what the experiment is.

Sections:
  - Fig. 6  (§7.1.1) — Overall hit rate, full pipeline, k ∈ {1, 5}.
  - Table 2 (§7.3.2) — Hybrid retrieval (no judge), per lane, k=1.
  - Fig. 13 (§7.3.3) — LLM Judge uplift over hybrid, per lane.
  - Fig. 15 (§7.4.2) — k-sweep ablation, hybrid (no judge), per lane.
  - Fig. 9  (§7.2.1) — Online query throughput on FeTaQA.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PaperRef:
    # Fig. 6 (§7.1.1) — full PNEUMA, weighted BC2+BX2, k ∈ {1, 5}.
    mixed_k1: float | None
    mixed_k5: float | None
    # Table 2 (§7.3.2) — Hybrid Retrieval, no judge, k=1, per lane.
    content_k1: float | None
    context_k1: float | None
    # Fig. 13 (§7.3.3) — LLM Judge (Qwen) bars, per lane, k=1. Bar-chart only;
    # values estimated from the figure to the nearest pp.
    fig13_content_k1_qwen: float | None
    fig13_context_k1_qwen: float | None
    # Fig. 15 (§7.4.2) — k-sweep, hybrid no judge, per lane. Bar-chart only;
    # estimated from the figure.
    fig15_content_k: dict[int, float] | None
    fig15_context_k: dict[int, float] | None
    # Fig. 9 (§7.2.1) — Online query throughput at full FeTaQA size (10330
    # tables), full pipeline. Reported only for FeTaQA.
    fig9_qps: float | None


# Numbers read off the paper. Table 2 is exact; Fig. 6 uses the 2-decimal
# values from the paper text; Fig. 13 / Fig. 15 are estimated from bar
# heights since the paper publishes no table for them.
PAPER: dict[str, PaperRef] = {
    "chembl": PaperRef(
        mixed_k1=68.07, mixed_k5=83.74,
        content_k1=83.00, context_k1=50.88,
        fig13_content_k1_qwen=83.0, fig13_context_k1_qwen=51.0,
        fig15_content_k={1: 83.0, 5: 95.0, 10: 96.0, 30: 97.0, 50: 97.0},
        fig15_context_k={1: 51.0, 5: 79.0, 10: 84.0, 30: 87.0, 50: 88.0},
        fig9_qps=None,
    ),
    "adventure_works": PaperRef(
        mixed_k1=71.53, mixed_k5=90.45,
        content_k1=81.40, context_k1=56.08,
        fig13_content_k1_qwen=82.0, fig13_context_k1_qwen=58.0,
        fig15_content_k={1: 81.0, 5: 96.0, 10: 97.0, 30: 98.0, 50: 98.0},
        fig15_context_k={1: 56.0, 5: 79.0, 10: 84.0, 30: 88.0, 50: 89.0},
        fig9_qps=None,
    ),
    "public_bi": PaperRef(
        mixed_k1=65.89, mixed_k5=70.90,
        content_k1=71.10, context_k1=57.75,
        fig13_content_k1_qwen=72.0, fig13_context_k1_qwen=58.0,
        fig15_content_k={1: 71.0, 5: 84.0, 10: 86.0, 30: 88.0, 50: 89.0},
        fig15_context_k={1: 58.0, 5: 73.0, 10: 78.0, 30: 81.0, 50: 82.0},
        fig9_qps=None,
    ),
    "chicago_open": PaperRef(
        mixed_k1=56.38, mixed_k5=76.76,
        content_k1=55.09, context_k1=52.25,
        fig13_content_k1_qwen=56.0, fig13_context_k1_qwen=53.0,
        fig15_content_k={1: 55.0, 5: 73.0, 10: 80.0, 30: 86.0, 50: 88.0},
        fig15_context_k={1: 52.0, 5: 69.0, 10: 75.0, 30: 82.0, 50: 84.0},
        fig9_qps=None,
    ),
    "fetaqa": PaperRef(
        mixed_k1=53.71, mixed_k5=59.37,
        content_k1=56.14, context_k1=46.96,
        fig13_content_k1_qwen=58.0, fig13_context_k1_qwen=47.0,
        fig15_content_k={1: 56.0, 5: 71.0, 10: 76.0, 30: 81.0, 50: 83.0},
        fig15_context_k={1: 47.0, 5: 60.0, 10: 65.0, 30: 71.0, 50: 73.0},
        # "Pneuma" bar at 10330 tables in Fig. 9, full pipeline.
        fig9_qps=2.48,
    ),
}


DATASET_ALIASES: dict[str, str] = {
    "chembl_10k": "chembl",
    "public": "public_bi",
    "chicago": "chicago_open",
    "chicago_10k": "chicago_open",
    "adventure": "adventure_works",
}


def _normalise(name: str) -> str:
    return DATASET_ALIASES.get(name, name)


def _latest_run_dir(dataset_dir: Path) -> Path | None:
    candidates = [
        d for d in dataset_dir.iterdir()
        if d.is_dir() and (d / "summary.csv").exists()
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.name)[-1]


def _read_summary(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _filter(rows: list[dict], *, rerank_mode: str, k_in: set[int]) -> list[dict]:
    return [
        r for r in rows
        if r.get("rerank_mode") == rerank_mode and int(r["k"]) in k_in
    ]


def _weighted_mean_hit_rate(rows: list[dict]) -> float | None:
    if not rows:
        return None
    total = sum(float(r["hit_rate"]) * int(r["n_questions"]) for r in rows)
    n = sum(int(r["n_questions"]) for r in rows)
    return total / n if n else None


def _fmt(value: float | None, width: int = 8, decimals: int = 2) -> str:
    if value is None:
        return "---".rjust(width)
    return f"{value:>{width}.{decimals}f}"


def _delta(blend: float | None, paper: float | None, width: int = 7) -> str:
    if blend is None or paper is None:
        return "---".rjust(width)
    return f"{blend - paper:+{width}.2f}"


def _signed(value: float | None, width: int) -> str:
    if value is None:
        return "---".rjust(width)
    return f"{value:+{width}.2f}"


def _section(title: str, what: list[str]) -> None:
    print()
    print("=" * 100)
    print(title)
    print("-" * 100)
    print("What we did:")
    for line in what:
        print(f"  {line}")
    print("=" * 100)


def _print_intro() -> None:
    print()
    print("#" * 100)
    print("# Blend ↔ PNEUMA — side-by-side comparison, one section per paper figure")
    print("#" * 100)
    print(
        "Datasets: ChEMBL, Adventure Works, Public BI, Chicago Open, FeTaQA.\n"
        "PNEUMA splits questions into two lanes:\n"
        "  - content (BC) — answerable from columns + sampled rows.\n"
        "  - context (BX) — answerable only from an LLM-generated table description.\n"
        "Each section: paper figure name → what the experiment did → a table with\n"
        "Blend's number next to PNEUMA's published number, plus Δ pp = (Blend − PNEUMA).\n"
        "'---' means the paper does not publish a comparable for that cell."
    )


def _print_fig6(rows_by_ds: dict[str, list[dict]]) -> None:
    _section(
        "§ Fig. 6 — Overall Hit Rate at k ∈ {1, 5}   (paper §7.1.1)",
        [
            "Run the FULL PNEUMA pipeline (BM25 ⊕ vector retrieval ⊕ LLM judge) on",
            "every question in both lanes, count the % that have at least one relevant",
            "table in the top-k. The paper reports a single number per dataset, weighted",
            "across both lanes by n_questions. We do exactly the same on Blend with",
            "rerank_mode=on at k=1 and k=5.",
        ],
    )
    print(
        f"{'dataset':18} | "
        f"{'blend_k1':>9} | {'pneuma_k1':>9} | {'Δ pp':>7} | "
        f"{'blend_k5':>9} | {'pneuma_k5':>9} | {'Δ pp':>7}"
    )
    print("-" * 100)
    any_data = False
    for ds, rows in sorted(rows_by_ds.items()):
        if ds not in PAPER:
            continue
        ref = PAPER[ds]
        on = _filter(rows, rerank_mode="on", k_in={1, 5})
        bc2 = [r for r in on if r["family"] == "BC2"]
        bx2 = [r for r in on if r["family"] == "BX2"]

        def mixed(k: int) -> float | None:
            return _weighted_mean_hit_rate(
                [r for r in bc2 if int(r["k"]) == k]
                + [r for r in bx2 if int(r["k"]) == k]
            )

        m1, m5 = mixed(1), mixed(5)
        if m1 is None and m5 is None:
            continue
        print(
            f"{ds:18} | "
            f"{_fmt(m1, width=9)} | {_fmt(ref.mixed_k1, width=9)} | "
            f"{_delta(m1, ref.mixed_k1)} | "
            f"{_fmt(m5, width=9)} | {_fmt(ref.mixed_k5, width=9)} | "
            f"{_delta(m5, ref.mixed_k5)}"
        )
        any_data = True
    if not any_data:
        print("(no data — run `make benchmark DATASET=<ds>`)")


def _print_table2(rows_by_ds: dict[str, list[dict]]) -> None:
    _section(
        "§ Table 2 — Hybrid Retrieval (no judge), per lane, k=1   (paper §7.3.2)",
        [
            "Same retriever as Fig. 6 (BM25 ⊕ vector fusion at k=1, n=5, α=0.5) but with",
            "the LLM judge OFF, to isolate retrieval-engine quality from judge quality.",
            "The paper reports two numbers per dataset, one per lane:",
            "  - content (BC2): question answerable from column names + sampled rows.",
            "  - context (BX2): question answerable only from the LLM-generated description.",
            "We mirror this on Blend with rerank_mode=off at k=1, broken out by family.",
        ],
    )
    print(
        f"{'dataset':18} | "
        f"{'blend_content':>13} | {'pneuma_content':>14} | {'Δ pp':>7} | "
        f"{'blend_context':>13} | {'pneuma_context':>14} | {'Δ pp':>7}"
    )
    print("-" * 100)
    any_data = False
    for ds, rows in sorted(rows_by_ds.items()):
        if ds not in PAPER:
            continue
        ref = PAPER[ds]
        off = _filter(rows, rerank_mode="off", k_in={1})
        bc2 = next((r for r in off if r["family"] == "BC2"), None)
        bx2 = next((r for r in off if r["family"] == "BX2"), None)
        if not (bc2 or bx2):
            continue
        c1 = float(bc2["hit_rate"]) if bc2 else None
        x1 = float(bx2["hit_rate"]) if bx2 else None
        print(
            f"{ds:18} | "
            f"{_fmt(c1, width=13)} | {_fmt(ref.content_k1, width=14)} | "
            f"{_delta(c1, ref.content_k1)} | "
            f"{_fmt(x1, width=13)} | {_fmt(ref.context_k1, width=14)} | "
            f"{_delta(x1, ref.context_k1)}"
        )
        any_data = True
    if not any_data:
        print("(no data — run `make benchmark DATASET=<ds>`)")


def _print_fig13(rows_by_ds: dict[str, list[dict]]) -> None:
    _section(
        "§ Fig. 13 — LLM Judge uplift over hybrid retrieval, per lane, k=1   (paper §7.3.3)",
        [
            "How many percentage points does the LLM judge add on top of hybrid",
            "retrieval alone? uplift = hit_rate(hybrid + judge) − hit_rate(hybrid only).",
            "Reported per lane. The paper publishes Fig. 13 only as bar charts; the",
            "PNEUMA values shown here are derived as (Fig. 13 'LLM Judge (Qwen)' bar",
            "− Table 2 hybrid value), so treat them as sign + scale, not exact pp.",
            "Blend uplift = hit_rate(rerank=on) − hit_rate(rerank=off) at k=1.",
        ],
    )
    print(
        f"{'dataset':18} | "
        f"{'blend_content':>13} | {'pneuma_content':>14} | {'Δ pp':>7} | "
        f"{'blend_context':>13} | {'pneuma_context':>14} | {'Δ pp':>7}"
    )
    print("-" * 100)
    any_data = False
    for ds, rows in sorted(rows_by_ds.items()):
        ref = PAPER.get(ds)
        on = _filter(rows, rerank_mode="on", k_in={1})
        off = _filter(rows, rerank_mode="off", k_in={1})
        if not on or not off:
            continue
        bc2_on = next((r for r in on if r["family"] == "BC2"), None)
        bx2_on = next((r for r in on if r["family"] == "BX2"), None)
        bc2_off = next((r for r in off if r["family"] == "BC2"), None)
        bx2_off = next((r for r in off if r["family"] == "BX2"), None)

        b_content = (
            float(bc2_on["hit_rate"]) - float(bc2_off["hit_rate"])
            if bc2_on and bc2_off else None
        )
        b_context = (
            float(bx2_on["hit_rate"]) - float(bx2_off["hit_rate"])
            if bx2_on and bx2_off else None
        )
        p_content = (
            ref.fig13_content_k1_qwen - ref.content_k1
            if ref and ref.fig13_content_k1_qwen is not None else None
        )
        p_context = (
            ref.fig13_context_k1_qwen - ref.context_k1
            if ref and ref.fig13_context_k1_qwen is not None else None
        )
        print(
            f"{ds:18} | "
            f"{_signed(b_content, 13)} | {_signed(p_content, 14)} | "
            f"{_delta(b_content, p_content)} | "
            f"{_signed(b_context, 13)} | {_signed(p_context, 14)} | "
            f"{_delta(b_context, p_context)}"
        )
        any_data = True
    if not any_data:
        print("(no data — run `make benchmark DATASET=<ds>`)")


def _print_fig15(rows_by_ds: dict[str, list[dict]]) -> None:
    _section(
        "§ Fig. 15 — k-sweep ablation, hybrid retrieval (no judge)   (paper §7.4.2)",
        [
            "Ablation: the LLM judge is deliberately turned OFF here so we can study",
            "retrieval alone as a function of k ∈ {1, 5, 10, 30, 50}. The full PNEUMA",
            "pipeline still uses the judge — Fig. 15 just isolates the retrieval stage.",
            "Why: in the full pipeline, hybrid retrieval picks top-k candidates and the",
            "judge re-ranks them down to top-1. So `recall@k of hybrid retrieval' bounds",
            "what the judge can possibly find. Fig. 15 shows that recall saturates",
            "around k=10–30, which justifies the paper's small candidate-set choice.",
            "Blend setting: rerank_mode=off, per family/k. Cells showing '---' for",
            "k ≥ 10 mean the benchmark only ran k ∈ {1, 5} (default `make bench`);",
            "rerun with k_values=[1,5,10,30,50] to populate the full curve.",
            "PNEUMA values estimated from the Fig. 15 bar charts (paper has no table).",
        ],
    )
    for lane_label, lane_attr, family in (
        ("content (BC2)", "fig15_content_k", "BC2"),
        ("context (BX2)", "fig15_context_k", "BX2"),
    ):
        print()
        print(f"  Lane: {lane_label}")
        print(
            f"  {'dataset':16} | "
            + " | ".join(
                f"{f'blend_k{k}':>9} | {f'pneuma_k{k}':>10}"
                for k in (1, 5, 10, 30, 50)
            )
        )
        print("  " + "-" * 98)
        any_data = False
        for ds, rows in sorted(rows_by_ds.items()):
            off = _filter(rows, rerank_mode="off", k_in={1, 5, 10, 30, 50})
            blend_by_k = {
                int(r["k"]): float(r["hit_rate"])
                for r in off if r["family"] == family
            }
            if not blend_by_k:
                continue
            ref = PAPER.get(ds)
            paper_by_k = getattr(ref, lane_attr) if ref else None
            cells = []
            for k in (1, 5, 10, 30, 50):
                cells.append(_fmt(blend_by_k.get(k), width=9))
                cells.append(_fmt(paper_by_k.get(k) if paper_by_k else None, width=10))
            print(
                f"  {ds:16} | "
                + " | ".join(f"{cells[2 * i]} | {cells[2 * i + 1]}" for i in range(5))
            )
            any_data = True
        if not any_data:
            print("  (no data — run `make benchmark DATASET=<ds>`)")


def _print_fig9(rows_by_ds: dict[str, list[dict]]) -> None:
    _section(
        "§ Fig. 9 — Online query throughput (q/s)   (paper §7.2.1)",
        [
            "End-to-end latency for the full pipeline at k=1 on a 100-question subset,",
            "averaged over 10 runs, reported in queries/second. PNEUMA only runs Fig. 9",
            "on FeTaQA at table counts {625, 1250, 2500, 5000, 10330}; for the other",
            "four datasets the paper publishes no comparable q/s.",
            "Blend q/s = 1000 / latency_ms_mean per (family, rerank_mode, k=1).",
            "We additionally include rerank=off (NOT in paper) so you can see how",
            "much latency the LLM judge owns on each dataset.",
        ],
    )
    print(
        f"{'dataset':18} | {'family':6} | {'rerank':6} | "
        f"{'blend_qps':>9} | {'pneuma_qps':>10} | {'Δ q/s':>7} | "
        f"{'k=1 mean ms':>11} | {'k=1 p50':>7} | {'k=1 p95':>7} | "
        f"{'vector_p50':>10} | {'vector_p95':>10}"
    )
    print("-" * 100)
    any_data = False
    for ds, rows in sorted(rows_by_ds.items()):
        ref = PAPER.get(ds)
        for fam in ("BC2", "BX2"):
            for mode in ("off", "on"):
                k1 = next(
                    (r for r in rows
                     if r.get("family") == fam
                     and r.get("rerank_mode") == mode
                     and int(r["k"]) == 1),
                    None,
                )
                if not k1:
                    continue
                lat_mean = k1.get("latency_ms_mean", "---")
                lat_p50 = k1.get("latency_ms_p50", "---")
                lat_p95 = k1.get("latency_ms_p95", "---")
                vec_p50 = k1.get("vector_ms_p50", "---")
                vec_p95 = k1.get("vector_ms_p95", "---")

                try:
                    blend_qps = 1000.0 / float(lat_mean)
                    blend_qps_str = f"{blend_qps:9.2f}"
                except (ValueError, TypeError, ZeroDivisionError):
                    blend_qps = None
                    blend_qps_str = "     ----"

                if mode == "on" and ref is not None and ref.fig9_qps is not None:
                    pneuma_qps_str = f"{ref.fig9_qps:10.2f}"
                    delta_str = (
                        f"{blend_qps - ref.fig9_qps:+7.2f}"
                        if blend_qps is not None else "---".rjust(7)
                    )
                else:
                    pneuma_qps_str = "---".rjust(10)
                    delta_str = "---".rjust(7)

                def _f1(v: str) -> str:
                    try:
                        return f"{float(v):7.1f}"
                    except (ValueError, TypeError):
                        return "    ---"

                def _f1m(v: str) -> str:
                    try:
                        return f"{float(v):11.1f}"
                    except (ValueError, TypeError):
                        return "        ---"

                def _f2(v: str) -> str:
                    try:
                        return f"{float(v):10.2f}"
                    except (ValueError, TypeError):
                        return "       ---"

                print(
                    f"{ds:18} | {fam:6} | {mode:6} | "
                    f"{blend_qps_str} | {pneuma_qps_str} | {delta_str} | "
                    f"{_f1m(lat_mean)} | {_f1(lat_p50)} | {_f1(lat_p95)} | "
                    f"{_f2(vec_p50)} | {_f2(vec_p95)}"
                )
                any_data = True
    if not any_data:
        print("(no data — run `make benchmark DATASET=<ds>`)")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main_print(*, results_root: Path | None = None) -> None:
    if results_root is None:
        results_root = _project_root() / "benchmark-data" / "results"

    rows_by_ds: dict[str, list[dict]] = {}

    if not results_root.exists():
        print(f"(no results root found at {results_root})", file=sys.stderr)
        _print_intro()
        _print_fig6({})
        _print_table2({})
        _print_fig13({})
        _print_fig15({})
        _print_fig9({})
        return

    for ds_dir in sorted(results_root.iterdir()):
        if not ds_dir.is_dir():
            continue
        latest = _latest_run_dir(ds_dir)
        if not latest:
            continue
        ds = _normalise(ds_dir.name)
        rows_by_ds[ds] = _read_summary(latest / "summary.csv")

    _print_intro()
    _print_fig6(rows_by_ds)
    _print_table2(rows_by_ds)
    _print_fig13(rows_by_ds)
    _print_fig15(rows_by_ds)
    _print_fig9(rows_by_ds)


def _gather_panel_rows(results_root: Path, runner_name: str) -> list[dict]:
    rows: list[dict] = []
    if not results_root.exists():
        return rows
    for ds_dir in results_root.iterdir():
        if not ds_dir.is_dir():
            continue
        runs = sorted([d for d in ds_dir.iterdir() if d.is_dir()])
        if not runs:
            continue
        latest = runs[-1]
        meta_path = latest / "run_meta.json"
        sum_path = latest / "summary.csv"
        if not (meta_path.exists() and sum_path.exists()):
            continue
        meta = json.loads(meta_path.read_text())
        kind = meta.get("deployment", {}).get("vector_index_kind", "unknown")
        with sum_path.open() as f:
            for row in csv.DictReader(f):
                rows.append(dict(
                    runner=runner_name,
                    family=row["family"],
                    k=int(row["k"]),
                    rerank_mode=row["rerank_mode"],
                    vector_ms_p50=float(row.get("vector_ms_p50") or 0.0),
                    recall_at_k=float(row.get("recall_at_k") or row.get("hit_rate", 0.0)),
                    vector_index_kind=kind,
                    judge_ms_p50=float(row.get("judge_ms_p50") or 0.0),
                ))
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--source",
        choices=("blend", "pneuma", "both"),
        default="blend",
        help="Which results tree(s) to read (default: blend).",
    )
    p.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Override results root explicitly. When set, --source is ignored.",
    )
    args = p.parse_args(argv)

    if args.results_root is not None:
        main_print(results_root=args.results_root)
        return 0

    project_root = _project_root()
    roots: list[tuple[str, Path]] = []
    if args.source in ("blend", "both"):
        roots.append(("blend", project_root / "benchmark-data" / "results"))
    if args.source in ("pneuma", "both"):
        roots.append(("pneuma", project_root / "pneumaBenchdata" / "results"))

    for label, root in roots:
        if len(roots) > 1:
            print()
            print("=" * 60)
            print(f"  source: {label} → {root}")
            print("=" * 60)
        if not root.exists():
            print(f"  ⚠ {root} not present — skipping (run the corresponding bench first)")
            continue
        main_print(results_root=root)

    # Gather rows once per root we visited, render the cross-runner panels.
    from scripts.benchmark.plot_v2 import render_vector_cost_panel, render_judge_cost_panel
    all_rows: list[dict] = []
    for label, root in roots:
        all_rows.extend(_gather_panel_rows(root, label))
    if all_rows:
        plots_dir = _project_root() / "benchmark-data" / "results" / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        render_vector_cost_panel(all_rows, out_path=plots_dir / "vector_cost.png")
        render_judge_cost_panel(all_rows, out_path=plots_dir / "judge_cost.png")
        print(f"  wrote {plots_dir / 'vector_cost.png'}")
        print(f"  wrote {plots_dir / 'judge_cost.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

