from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BENCH_DATA = _PROJECT_ROOT / "benchmark-data"

_DEFAULT_LLM_BASE_URL = "http://127.0.0.1:8001/v1"
_DEFAULT_LLM_MODEL = "Qwen2.5-7B-Instruct"
_DEFAULT_EMBED_BASE_URL = "http://127.0.0.1:8002/v1"
_DEFAULT_EMBED_MODEL = "bge-base-en-v1.5"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scripts.benchmark.autoddg_cli")
    sub = parser.add_subparsers(dest="command", required=True)

    def _add_common_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--base", required=True,
                       help="Base dataset name (e.g. chembl)")
        p.add_argument("--lake-dir", required=True, type=Path,
                       help="Path to the lake directory for the base dataset")
        p.add_argument("--out-root", default=str(_BENCH_DATA), type=Path,
                       help="Root output directory (default: benchmark-data)")
        p.add_argument("--force", action="store_true",
                       help="Rebuild even if output already exists")
        p.add_argument("--llm-base-url", default=_DEFAULT_LLM_BASE_URL,
                       help=f"LLM endpoint base URL (default: {_DEFAULT_LLM_BASE_URL})")
        p.add_argument("--llm-model", default=_DEFAULT_LLM_MODEL,
                       help=f"LLM model name (default: {_DEFAULT_LLM_MODEL})")
        p.add_argument("--embed-base-url", default=_DEFAULT_EMBED_BASE_URL,
                       help=f"Embedder endpoint base URL (default: {_DEFAULT_EMBED_BASE_URL})")
        p.add_argument("--embed-model", default=_DEFAULT_EMBED_MODEL,
                       help=f"Embedder model name (default: {_DEFAULT_EMBED_MODEL})")

    p_generate = sub.add_parser(
        "generate",
        help="Run AutoDDG artifact generation for a base dataset (shared DuckDB only).",
    )
    _add_common_flags(p_generate)

    p_build_cell = sub.add_parser(
        "build-cell",
        help="Build the NL index for one AutoDDG ablation cell.",
    )
    _add_common_flags(p_build_cell)
    p_build_cell.add_argument("--cell", required=True,
                               help="Cell key (e.g. B0, B1, ...)")

    return parser


def _do_generate(
    base: str,
    lake_dir: Path,
    out_root: Path,
    llm_base_url: str,
    llm_model: str,
    embed_base_url: str,
    embed_model: str,
    force: bool,
) -> Path:
    from scripts.benchmark.build_index import build_autoddg_shared_duckdb
    return build_autoddg_shared_duckdb(
        base_dataset=base,
        lake_dir=lake_dir,
        out_root=out_root,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        embed_base_url=embed_base_url,
        embed_model=embed_model,
        force=force,
    )


def _do_build_cell(
    base: str,
    cell: str,
    lake_dir: Path,
    out_root: Path,
    llm_base_url: str,
    llm_model: str,
    embed_base_url: str,
    embed_model: str,
    force: bool,
) -> "BuildResult":
    from scripts.benchmark.build_index import build_autoddg_shared_duckdb, build_autoddg_cell

    # Guarantee the shared DuckDB exists before building the cell index.
    build_autoddg_shared_duckdb(
        base_dataset=base,
        lake_dir=lake_dir,
        out_root=out_root,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        embed_base_url=embed_base_url,
        embed_model=embed_model,
        force=False,
    )
    return build_autoddg_cell(
        base_dataset=base,
        cell=cell,
        lake_dir=lake_dir,
        out_root=out_root,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        embed_base_url=embed_base_url,
        embed_model=embed_model,
        force=force,
    )


def dispatch(ns: argparse.Namespace, parser: argparse.ArgumentParser | None = None
             ) -> int:
    if ns.command == "generate":
        duckdb_path = _do_generate(
            base=ns.base,
            lake_dir=ns.lake_dir,
            out_root=ns.out_root,
            llm_base_url=ns.llm_base_url,
            llm_model=ns.llm_model,
            embed_base_url=ns.embed_base_url,
            embed_model=ns.embed_model,
            force=ns.force,
        )
        print(f"generate ok: {duckdb_path}")
        return 0
    if ns.command == "build-cell":
        result = _do_build_cell(
            base=ns.base,
            cell=ns.cell,
            lake_dir=ns.lake_dir,
            out_root=ns.out_root,
            llm_base_url=ns.llm_base_url,
            llm_model=ns.llm_model,
            embed_base_url=ns.embed_base_url,
            embed_model=ns.embed_model,
            force=ns.force,
        )
        print(f"build-cell ok: {result.duckdb_path} (wall={result.wall_clock_s:.1f}s)")
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
