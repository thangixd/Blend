from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from scripts.benchmark.datasets import DATASETS


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BENCH_DATA = _PROJECT_ROOT / "benchmark-data"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scripts.benchmark.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    def _add_dataset(p: argparse.ArgumentParser) -> None:
        p.add_argument("--dataset", required=True)

    p_prep = sub.add_parser("prepare")
    _add_dataset(p_prep)

    p_build = sub.add_parser("build")
    _add_dataset(p_build)
    p_build.add_argument("--force", action="store_true")

    p_run = sub.add_parser("run")
    _add_dataset(p_run)
    p_run.add_argument("--max-questions", type=int, default=None)

    p_all = sub.add_parser("all")
    _add_dataset(p_all)
    p_all.add_argument("--force", action="store_true")
    p_all.add_argument("--max-questions", type=int, default=None)

    return parser


def _resolve_spec(dataset: str):
    if dataset not in DATASETS:
        raise KeyError(f"unknown dataset {dataset!r}; "
                       f"known: {sorted(DATASETS)}")
    return DATASETS[dataset]


def _do_prepare(spec) -> "PrepareResult":
    from scripts.benchmark.prepare import prepare
    return prepare(spec, _BENCH_DATA)


def _do_build(spec, force: bool) -> "BuildResult":
    from scripts.benchmark.build_index import build_index
    lake_dir = _BENCH_DATA / "lakes" / spec.name
    return build_index(
        dataset_name=spec.name,
        lake_dir=lake_dir,
        out_root=_BENCH_DATA,
        force=force,
    )


def _do_run(spec, max_questions: int | None) -> Path:
    from scripts.benchmark.endpoints import probe_endpoints
    from scripts.benchmark.run import run_benchmark, FAMILY_LLM_URLS, EMBED_URL
    from scripts.benchmark.build_index import build_index

    lake_dir = _BENCH_DATA / "lakes" / spec.name
    # rebuild config.ini header even if we reuse the index
    build = build_index(
        dataset_name=spec.name,
        lake_dir=lake_dir,
        out_root=_BENCH_DATA,
        force=False,
    )
    endpoints_meta = probe_endpoints(
        llm_urls=list(FAMILY_LLM_URLS.values()),
        llm_model="Qwen2.5-7B-Instruct",
        embed_url=EMBED_URL,
        embed_model="bge-base-en-v1.5",
    )
    # Recover unresolvable/dropped counts from a fresh sweep over the manifest.
    import json
    manifest = json.loads((lake_dir / "_manifest.json").read_text())
    return run_benchmark(
        dataset=spec.name,
        lake_dir=lake_dir,
        config_path=build.config_path,
        index_name=build.index_name,
        content_jsonl=spec.content_jsonl,
        results_dir=_BENCH_DATA / "results",
        endpoints_meta=endpoints_meta,
        build_wall_clock_s=build.wall_clock_s,
        unresolvable=manifest.get("health", {}).get("unresolvable_questions", 0),
        dropped_context_rows=manifest.get("health", {}).get("dropped_context_rows", 0),
        max_questions=max_questions,
    )


def dispatch(ns: argparse.Namespace) -> int:
    spec = _resolve_spec(ns.dataset)
    if ns.command == "prepare":
        result = _do_prepare(spec)
        print(f"prepare ok: {result.lake_dir} ({result.n_tables} tables, "
              f"{result.n_context_rows} context rows, "
              f"dropped={result.dropped_context_rows}, "
              f"unresolvable={result.unresolvable_questions})")
        return 0
    if ns.command == "build":
        result = _do_build(spec, ns.force)
        print(f"build ok: {result.duckdb_path} (wall={result.wall_clock_s:.1f}s)")
        return 0
    if ns.command == "run":
        run_dir = _do_run(spec, ns.max_questions)
        print(f"run ok: {run_dir}")
        return 0
    if ns.command == "all":
        _do_prepare(spec)
        _do_build(spec, ns.force)
        run_dir = _do_run(spec, ns.max_questions)
        print(f"all ok: {run_dir}")
        return 0
    raise AssertionError(f"unreachable: {ns.command}")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    parser = build_parser()
    ns = parser.parse_args(argv)
    try:
        return dispatch(ns)
    except SystemExit:
        raise
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["build_parser", "dispatch", "main"]
