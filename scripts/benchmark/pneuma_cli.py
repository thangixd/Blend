"""PNEUMA-side benchmark CLI. Mirrors ``scripts.benchmark.cli`` and
invokes the same prepare() with out_root pointing at pneumaBenchdata/."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.benchmark.datasets import DATASETS
from scripts.benchmark.prepare import prepare
from scripts.benchmark.pneuma_patches import apply_patches
from scripts.benchmark.pneuma_build import build_pneuma_index
from scripts.benchmark.pneuma_run import run_pneuma_benchmark

PNEUMA_ROOT = Path("pneumaBenchdata")
LAKE_ROOT = PNEUMA_ROOT / "lakes"
INDEX_ROOT = PNEUMA_ROOT / "indexes"
RESULTS_ROOT = PNEUMA_ROOT / "results"


def _parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _parse_str_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _do_prepare(args: argparse.Namespace) -> int:
    spec = DATASETS[args.dataset]
    prepare(spec, out_root=PNEUMA_ROOT)
    return 0


def _do_build(args: argparse.Namespace) -> int:
    build_pneuma_index(args.dataset, force=args.force)
    return 0


def _do_run(args: argparse.Namespace) -> int:
    import datetime as dt
    ds = args.dataset
    lake_dir = LAKE_ROOT / ds
    index_dir = INDEX_ROOT / ds
    # Resolve ``content.jsonl`` from the DatasetSpec (points into
    # ``EvaluationDataFromPneuma/``) — same source Blend's ``cli.py`` uses
    # at line 104.  ``prepare()`` does not copy this file into the lake;
    # both runners read it from the spec so the question text is byte-
    # identical across Blend / PNEUMA bench output.
    content_jsonl = DATASETS[ds].content_jsonl
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = RESULTS_ROOT / ds / ts

    run_pneuma_benchmark(
        dataset=ds,
        index_dir=index_dir,
        lake_dir=lake_dir,
        content_jsonl=content_jsonl,
        run_dir=run_dir,
        k_values=_parse_int_list(args.k_values),
        rerank_modes=_parse_str_list(args.rerank_modes),
        families=_parse_str_list(args.families),
        max_questions=args.max_questions,
        n=args.n,
        alpha=args.alpha,
    )
    return 0


def _do_all(args: argparse.Namespace) -> int:
    rc = _do_prepare(args)
    if rc != 0:
        return rc
    rc = _do_build(args)
    if rc != 0:
        return rc
    return _do_run(args)


def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--max-questions", type=int, default=None)
    p.add_argument("--k-values", default="1,5,10,30,50")
    p.add_argument("--rerank-modes", default="off,on")
    p.add_argument("--families", default="BC1,BC2,BX1,BX2")
    # Hybrid-fusion knobs.  Defaults match the PNEUMA paper §7.4.1/§7.4.3
    # (``α=0.5``, ``n=5``).  Override to reproduce paper Fig. 14-16
    # ablations:
    #   make pneuma-bench DATASET=adventure_works PNEUMA_ALPHA=0.7 PNEUMA_N=15
    p.add_argument(
        "--n", type=int, default=5,
        help="Hybrid retrieval k×n fan-out (paper Fig.15 sweeps {1,5,15,20}).",
    )
    p.add_argument(
        "--alpha", type=float, default=0.5,
        help="BM25↔vector mixing weight in hybrid fusion (paper Fig.14 sweeps 0..1).",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pneuma_cli")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prepare")
    p_prep.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    p_prep.set_defaults(func=_do_prepare)

    p_build = sub.add_parser("build")
    p_build.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    p_build.add_argument("--force", action="store_true")
    p_build.set_defaults(func=_do_build)

    p_run = sub.add_parser("run")
    p_run.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    _add_run_args(p_run)
    p_run.set_defaults(func=_do_run)

    p_all = sub.add_parser("all")
    p_all.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    p_all.add_argument("--force", action="store_true")
    _add_run_args(p_all)
    p_all.set_defaults(func=_do_all)

    args = parser.parse_args(argv)
    apply_patches(
        llm_endpoint_url="http://127.0.0.1:8001/v1",
        llm_model_id="Qwen2.5-7B-Instruct",
        embed_model_id="BAAI/bge-base-en-v1.5",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
