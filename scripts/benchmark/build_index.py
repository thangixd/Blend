from __future__ import annotations

import configparser
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path


LOG = logging.getLogger(__name__)


LAKE_GLOB_PATTERN = "[!_]*.csv"


@dataclass(frozen=True)
class BuildResult:
    duckdb_path: Path
    nl_out_path: Path
    index_name: str
    config_path: Path
    wall_clock_s: float


def write_benchmark_config(
    *,
    out_path: Path,
    duckdb_path: Path,
    nl_out_path: Path,
    index_name: str,
    llm_base_url: str,
    llm_model: str,
    embed_base_url: str,
    embed_model: str,
    embed_max_input_tokens: int = 512,
    embed_tokenizer_id: str = "BAAI/bge-base-en-v1.5",
    llm_max_input_tokens: int = 32768,
    llm_tokenizer_id: str = "Qwen/Qwen2.5-7B-Instruct",
) -> None:
    """Render the per-dataset config.ini consumed by run_pipeline()."""
    parser = configparser.ConfigParser()
    parser["Database"] = {
        "dbms": "duckdb",
        "path": str(duckdb_path),
        "index_table": "blend_index",
    }
    parser["NLSeeker"] = {
        "out_path": str(nl_out_path),
        "index_name": index_name,
        "use_local_model": "false",
        "openai_api_key": "vllm-local",
        "openai_base_url": llm_base_url,
        "openai_llm_model": llm_model,
        "openai_llm_max_input_tokens": str(llm_max_input_tokens),
        "openai_llm_tokenizer_id": llm_tokenizer_id,
        "openai_embed_base_url": embed_base_url,
        "openai_embed_model": embed_model,
        "openai_embed_max_input_tokens": str(embed_max_input_tokens),
        "openai_embed_tokenizer_id": embed_tokenizer_id,
        "alpha": "0.5",
        "n": "5",
        "default_k": "10",
        "max_llm_batch_size": "50",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        parser.write(f)


def build_index(
    *,
    dataset_name: str,
    lake_dir: Path,
    out_root: Path,
    llm_base_url: str = "http://127.0.0.1:8001/v1",
    llm_model: str = "Qwen2.5-7B-Instruct",
    embed_base_url: str = "http://127.0.0.1:8002/v1",
    embed_model: str = "bge-base-en-v1.5",
    embed_max_input_tokens: int = 512,
    force: bool = False,
) -> BuildResult:
    """Render a per-dataset config.ini and call run_pipeline() over the lake.

    Reuses an existing index unless ``force`` is True.
    """
    index_dir = out_root / "indexes" / dataset_name
    duckdb_path = index_dir / "blend.duckdb"
    nl_out_path = index_dir / "nl-out"
    config_path = index_dir / "config.ini"
    index_name = f"benchmark_{dataset_name}"

    nl_vector_path = nl_out_path / "indexes" / "vector" / index_name
    nl_fulltext_path = nl_out_path / "indexes" / "fulltext" / index_name
    index_complete = (
        duckdb_path.exists()
        and nl_vector_path.is_dir()
        and nl_fulltext_path.is_dir()
    )

    if index_complete and not force:
        LOG.info("build skipped: index exists at %s", index_dir)
        # Still rewrite config.ini in case URLs changed.
        write_benchmark_config(
            out_path=config_path,
            duckdb_path=duckdb_path,
            nl_out_path=nl_out_path,
            index_name=index_name,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            embed_base_url=embed_base_url,
            embed_model=embed_model,
            embed_max_input_tokens=embed_max_input_tokens,
        )
        return BuildResult(duckdb_path, nl_out_path, index_name, config_path, 0.0)

    if duckdb_path.exists():
        duckdb_path.unlink()
    if nl_out_path.exists():
        import shutil
        shutil.rmtree(nl_out_path)

    index_dir.mkdir(parents=True, exist_ok=True)
    write_benchmark_config(
        out_path=config_path,
        duckdb_path=duckdb_path,
        nl_out_path=nl_out_path,
        index_name=index_name,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        embed_base_url=embed_base_url,
        embed_model=embed_model,
        embed_max_input_tokens=embed_max_input_tokens,
    )

    metadata_path = lake_dir / "_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"build_index: {metadata_path} missing — run prepare first."
        )

    from scripts.create_blend_index import run_pipeline  # delayed to avoid heavy imports at module load

    # Match the CLI's default (scripts/create_blend_index.py --workers): leaves
    # one core for the main process / DuckDB sink, capped at 16. This only
    # parallelises the value-index compute.
    t0 = time.perf_counter()
    run_pipeline(
        lake_path=str(lake_dir / LAKE_GLOB_PATTERN),
        config_path=config_path,
        nl_index=True,
        metadata_path=metadata_path,
        workers=min((os.cpu_count() or 2) - 1, 16),
    )
    wall = time.perf_counter() - t0
    return BuildResult(duckdb_path, nl_out_path, index_name, config_path, wall)


__all__ = ["build_index", "write_benchmark_config", "BuildResult", "LAKE_GLOB_PATTERN"]
