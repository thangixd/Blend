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

    def _add_axis_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--rerank-modes", default="off,on",
                       help="Comma-separated rerank modes (default: off,on)")
        p.add_argument("--k-values", default="1,5,10,30,50",
                       help="Comma-separated k values (default: 1,5,10,30,50)")
        p.add_argument("--families", default="BC1,BC2,BX1,BX2",
                       help="Comma-separated families (default: BC1,BC2,BX1,BX2)")

    p_prep = sub.add_parser("prepare")
    _add_dataset(p_prep)

    p_build = sub.add_parser("build")
    _add_dataset(p_build)
    p_build.add_argument("--force", action="store_true")

    p_run = sub.add_parser("run")
    _add_dataset(p_run)
    p_run.add_argument("--max-questions", type=int, default=None)
    _add_axis_flags(p_run)

    p_all = sub.add_parser("all")
    _add_dataset(p_all)
    p_all.add_argument("--force", action="store_true")
    p_all.add_argument("--max-questions", type=int, default=None)
    _add_axis_flags(p_all)

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


def _do_run(spec, max_questions: int | None,
            rerank_modes: tuple[str, ...] = ("off", "on"),
            k_values: tuple[int, ...] = (1, 5, 10, 30, 50),
            families: tuple[str, ...] = ("BC1", "BC2", "BX1", "BX2")) -> Path:
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
        rerank_modes=rerank_modes,
        k_values=k_values,
        families=families,
    )


def _parse_axis_flags(ns: argparse.Namespace, parser: argparse.ArgumentParser
                      ) -> tuple[tuple[str, ...], tuple[int, ...], tuple[str, ...]]:
    """Strip whitespace, drop empty fragments, error on empty result.

    Tolerates ``--k-values ""`` and stray commas like ``--k-values 1,,5``.
    """
    rerank_modes = tuple(s.strip() for s in ns.rerank_modes.split(",") if s.strip())
    if not rerank_modes:
        parser.error("--rerank-modes must contain at least one mode")
    try:
        k_values = tuple(int(k.strip()) for k in ns.k_values.split(",") if k.strip())
    except ValueError as exc:
        parser.error(f"--k-values must be comma-separated integers: {exc}")
    if not k_values:
        parser.error("--k-values must contain at least one integer")
    families = tuple(s.strip() for s in ns.families.split(",") if s.strip())
    if not families:
        parser.error("--families must contain at least one family")
    return rerank_modes, k_values, families


def dispatch(ns: argparse.Namespace, parser: argparse.ArgumentParser | None = None
             ) -> int:
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
    # `parser` is required from here on for argparse-style error reporting.
    if parser is None:
        parser = build_parser()
    if ns.command == "run":
        rerank_modes, k_values, families = _parse_axis_flags(ns, parser)
        run_dir = _do_run(
            spec, ns.max_questions,
            rerank_modes=rerank_modes,
            k_values=k_values,
            families=families,
        )
        print(f"run ok: {run_dir}")
        return 0
    if ns.command == "all":
        rerank_modes, k_values, families = _parse_axis_flags(ns, parser)
        _do_prepare(spec)
        _do_build(spec, ns.force)
        run_dir = _do_run(
            spec, ns.max_questions,
            rerank_modes=rerank_modes,
            k_values=k_values,
            families=families,
        )
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
        return dispatch(ns, parser)
    except SystemExit:
        raise
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["build_parser", "dispatch", "main"]
