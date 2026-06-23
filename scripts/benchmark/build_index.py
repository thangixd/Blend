from __future__ import annotations

import configparser
import logging
import time
from dataclasses import dataclass
from pathlib import Path


LOG = logging.getLogger(__name__)


LAKE_GLOB_PATTERN = "*.csv"
LAKE_BOOKKEEPING_NAMES = frozenset({"_metadata.csv"})


def _dir_nonempty(p: Path) -> bool:
    """True iff ``p`` is a directory with at least one entry. Used to
    distinguish a populated index dir from an empty shell left by a
    crashed build."""
    if not p.is_dir():
        return False
    return any(p.iterdir())


@dataclass(frozen=True)
class BuildResult:
    duckdb_path: Path
    nl_out_path: Path
    index_name: str
    config_path: Path
    wall_clock_s: float


def _nlseeker_section_dict(
    *,
    nl_out_path: Path,
    index_name: str,
    llm_base_url: str,
    llm_model: str,
    embed_base_url: str,
    embed_model: str,
    embed_max_input_tokens: int,
    embed_tokenizer_id: str,
    llm_max_input_tokens: int,
    llm_tokenizer_id: str,
    local_embedder: bool,
) -> dict:
    """Return the dict used as parser["NLSeeker"] in config.ini writers.

    Factored out of write_benchmark_config() so that write_autoddg_cell_config()
    can share the same keys without duplication.
    """
    return {
        "out_path": str(nl_out_path),
        "index_name": index_name,
        "use_local_model": "false",
        "local_embedder": "true" if local_embedder else "false",
        "openai_api_key": "vllm-local",
        "openai_base_url": llm_base_url,
        "openai_llm_model": llm_model,
        "openai_llm_max_input_tokens": str(llm_max_input_tokens),
        "openai_llm_tokenizer_id": llm_tokenizer_id,
        "openai_embed_base_url": embed_base_url,
        "openai_embed_model": embed_model,
        "openai_embed_max_input_tokens": str(embed_max_input_tokens),
        "openai_embed_tokenizer_id": embed_tokenizer_id,
        "embed_path": embed_tokenizer_id,
        "alpha": "0.5",
        "n": "5",
        "default_k": "10",
        "max_llm_batch_size": "50",
    }


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
    local_embedder: bool = True,
) -> None:
    """Render the per-dataset config.ini consumed by run_pipeline().

    ``local_embedder=True`` (the benchmark default) loads the embedder via
    sentence-transformers locally rather than through vLLM's /v1/embeddings.
    The embed_base_url / embed_model knobs are
    written either way for config completeness; they're ignored at runtime
    when local_embedder is True.
    """
    parser = configparser.ConfigParser()
    parser["Database"] = {
        "dbms": "duckdb",
        "path": str(duckdb_path),
        "index_table": "blend_index",
    }
    parser["NLSeeker"] = _nlseeker_section_dict(
        nl_out_path=nl_out_path,
        index_name=index_name,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        embed_base_url=embed_base_url,
        embed_model=embed_model,
        embed_max_input_tokens=embed_max_input_tokens,
        embed_tokenizer_id=embed_tokenizer_id,
        llm_max_input_tokens=llm_max_input_tokens,
        llm_tokenizer_id=llm_tokenizer_id,
        local_embedder=local_embedder,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        parser.write(f)


def write_autoddg_cell_config(
    *,
    out_path: Path,
    shared_duckdb_path: Path,
    cell_nl_out_path: Path,
    index_name: str,
    llm_base_url: str,
    llm_model: str,
    embed_base_url: str,
    embed_model: str,
    embed_max_input_tokens: int = 512,
    embed_tokenizer_id: str = "BAAI/bge-base-en-v1.5",
    llm_max_input_tokens: int = 32768,
    llm_tokenizer_id: str = "Qwen/Qwen2.5-7B-Instruct",
    local_embedder: bool = True,
) -> None:
    """Render a per-cell config.ini for an AutoDDG ablation cell.

    Like write_benchmark_config() but:
    - [Database].path points to the **shared** blend.duckdb (one per base dataset).
    - [NLSeeker].out_path points to the **per-cell** nl-out directory.

    This separation lets all cells within a base dataset share the expensive
    DuckDB artifact generation step while keeping their NL indexes isolated.
    """
    parser = configparser.ConfigParser()
    parser["Database"] = {
        "dbms": "duckdb",
        "path": str(shared_duckdb_path),
        "index_table": "blend_index",
    }
    parser["NLSeeker"] = _nlseeker_section_dict(
        nl_out_path=cell_nl_out_path,
        index_name=index_name,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        embed_base_url=embed_base_url,
        embed_model=embed_model,
        embed_max_input_tokens=embed_max_input_tokens,
        embed_tokenizer_id=embed_tokenizer_id,
        llm_max_input_tokens=llm_max_input_tokens,
        llm_tokenizer_id=llm_tokenizer_id,
        local_embedder=local_embedder,
    )
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
    local_embedder: bool = True,
    force: bool = False,
) -> BuildResult:
    """Render a per-dataset config.ini and call run_pipeline() over the lake.

    ``local_embedder=True`` (the default) routes the embedder through
    sentence-transformers locally rather than the vLLM /v1/embeddings
    endpoint - the paper-faithful path. The vLLM embedder remains
    available for callers that pass ``local_embedder=False``.

    Reuses an existing index unless ``force`` is True.
    """
    index_dir = out_root / "indexes" / dataset_name
    duckdb_path = index_dir / "blend.duckdb"
    nl_out_path = index_dir / "nl-out"
    config_path = index_dir / "config.ini"
    index_name = f"benchmark_{dataset_name}"

    nl_vector_path = nl_out_path / "indexes" / "vector" / index_name
    nl_fulltext_path = nl_out_path / "indexes" / "fulltext" / index_name
    wall_clock_path = index_dir / "_build_wall_clock_s.txt"
    # value_index is disabled in benchmark mode; don't gate on the
    # blend_index DuckDB table; only the NL sub-dirs need to be present.
    index_complete = (
        _dir_nonempty(nl_vector_path)
        and _dir_nonempty(nl_fulltext_path)
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
            local_embedder=local_embedder,
        )
        # Recover the wall-clock from the previous build so run_meta.json
        # carries the real cost rather than 0.0. Falls back to 0.0 only when
        # the index pre-dates this sidecar (e.g. an older artifact).
        prior_wall = 0.0
        if wall_clock_path.exists():
            try:
                prior_wall = float(wall_clock_path.read_text().strip())
            except ValueError:
                LOG.warning("Unparseable %s; reporting 0.0", wall_clock_path)
        return BuildResult(duckdb_path, nl_out_path, index_name, config_path, prior_wall)

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
        local_embedder=local_embedder,
    )

    metadata_path = lake_dir / "_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"build_index: {metadata_path} missing - run prepare first."
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
        workers=1,
        value_index=False,
        pre_chunked_contexts=True,
        lake_exclude_names=LAKE_BOOKKEEPING_NAMES,
    )
    wall = time.perf_counter() - t0
    # Persist alongside the index so a subsequent reuse-path build can recover
    # the real cost for run_meta.json instead of reporting 0.0.
    wall_clock_path.write_text(f"{wall:.3f}\n")
    return BuildResult(duckdb_path, nl_out_path, index_name, config_path, wall)


def build_autoddg_shared_duckdb(
    *,
    base_dataset: str,
    lake_dir: Path,
    out_root: Path,
    llm_base_url: str = "http://127.0.0.1:8001/v1",
    llm_model: str = "Qwen2.5-7B-Instruct",
    embed_base_url: str = "http://127.0.0.1:8002/v1",
    embed_model: str = "bge-base-en-v1.5",
    embed_max_input_tokens: int = 512,
    local_embedder: bool = True,
    force: bool = False,
) -> Path:
    """Run AutoDDG artifact generation into a shared per-base DuckDB.

    Creates benchmark-data/indexes/autoddg/<base>/ if missing.  Writes a
    bootstrap config.ini at that directory (used only to drive the generate
    step) pointing at .../blend.duckdb.  Calls
    _run_generation_only() to populate artifact tables.

    Idempotent: skips if blend.duckdb already exists and is non-empty, unless
    ``force=True``.  Writes _shared_build_wall_clock_s.txt alongside the DB.
    Returns the path to the shared DuckDB.
    """
    base_index_dir = out_root / "indexes" / "autoddg" / base_dataset
    duckdb_path = base_index_dir / "blend.duckdb"
    bootstrap_cfg = base_index_dir / "config.ini"
    wall_clock_path = base_index_dir / "_shared_build_wall_clock_s.txt"

    db_nonempty = duckdb_path.exists() and duckdb_path.stat().st_size > 0
    if db_nonempty and not force:
        LOG.info(
            "build_autoddg_shared_duckdb: skipping; %s already populated", duckdb_path
        )
        return duckdb_path

    if duckdb_path.exists():
        duckdb_path.unlink()

    base_index_dir.mkdir(parents=True, exist_ok=True)

    # Bootstrap config: out_path / index_name are placeholders - they are not
    # used by _run_generation_only(), but NLSeekerConfig.load() requires the
    # keys to be present.
    write_autoddg_cell_config(
        out_path=bootstrap_cfg,
        shared_duckdb_path=duckdb_path,
        cell_nl_out_path=base_index_dir / "_bootstrap_nl_out",
        index_name=f"_bootstrap_{base_dataset}",
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        embed_base_url=embed_base_url,
        embed_model=embed_model,
        embed_max_input_tokens=embed_max_input_tokens,
        local_embedder=local_embedder,
    )

    metadata_path = lake_dir / "_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"build_autoddg_shared_duckdb: {metadata_path} missing - run prepare first."
        )

    from scripts.create_blend_index import _run_generation_only  # delayed import

    t0 = time.perf_counter()
    _run_generation_only(
        lake_path=str(lake_dir / LAKE_GLOB_PATTERN),
        config_path=bootstrap_cfg,
        metadata_path=metadata_path,
        nl_config_overrides=None,
    )
    wall = time.perf_counter() - t0
    wall_clock_path.write_text(f"{wall:.3f}\n")
    return duckdb_path


def build_autoddg_cell(
    *,
    base_dataset: str,
    cell: str,
    lake_dir: Path,
    out_root: Path,
    llm_base_url: str = "http://127.0.0.1:8001/v1",
    llm_model: str = "Qwen2.5-7B-Instruct",
    embed_base_url: str = "http://127.0.0.1:8002/v1",
    embed_model: str = "bge-base-en-v1.5",
    embed_max_input_tokens: int = 512,
    local_embedder: bool = True,
    force: bool = False,
) -> BuildResult:
    """Build the per-cell NL index for one AutoDDG ablation cell.

    Paths:
    - index_dir  = out_root/indexes/autoddg/<base>/<cell>/
    - shared DB  = out_root/indexes/autoddg/<base>/blend.duckdb
    - nl_out     = index_dir/nl-out/
    - config.ini = index_dir/config.ini

    The cell preset from scripts/benchmark/cells.py is merged with
    ``{"index_name": "benchmark_autoddg_<base>_<cell>"}`` and forwarded as
    nl_config_overrides to run_pipeline().

    Skip logic mirrors build_index(): the two NL sub-dirs (vector + fulltext
    under nl_out_path/indexes/) must both be non-empty to be considered
    complete.  On a skip, config.ini is still rewritten (URL changes) and the
    wall-clock is recovered from _build_wall_clock_s.txt.
    """
    from scripts.benchmark.cells import preset_for_cell

    index_dir = out_root / "indexes" / "autoddg" / base_dataset / cell
    shared_duckdb = out_root / "indexes" / "autoddg" / base_dataset / "blend.duckdb"
    nl_out_path = index_dir / "nl-out"
    config_path = index_dir / "config.ini"
    index_name = f"benchmark_autoddg_{base_dataset}_{cell}"
    wall_clock_path = index_dir / "_build_wall_clock_s.txt"

    nl_vector_path = nl_out_path / "indexes" / "vector" / index_name
    nl_fulltext_path = nl_out_path / "indexes" / "fulltext" / index_name
    index_complete = _dir_nonempty(nl_vector_path) and _dir_nonempty(nl_fulltext_path)

    if index_complete and not force:
        LOG.info("build_autoddg_cell: index exists at %s; skipping build", index_dir)
        # Still rewrite config.ini in case URLs changed.
        write_autoddg_cell_config(
            out_path=config_path,
            shared_duckdb_path=shared_duckdb,
            cell_nl_out_path=nl_out_path,
            index_name=index_name,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            embed_base_url=embed_base_url,
            embed_model=embed_model,
            embed_max_input_tokens=embed_max_input_tokens,
            local_embedder=local_embedder,
        )
        prior_wall = 0.0
        if wall_clock_path.exists():
            try:
                prior_wall = float(wall_clock_path.read_text().strip())
            except ValueError:
                LOG.warning("Unparseable %s; reporting 0.0", wall_clock_path)
        return BuildResult(shared_duckdb, nl_out_path, index_name, config_path, prior_wall)

    if nl_out_path.exists():
        import shutil
        shutil.rmtree(nl_out_path)

    index_dir.mkdir(parents=True, exist_ok=True)
    write_autoddg_cell_config(
        out_path=config_path,
        shared_duckdb_path=shared_duckdb,
        cell_nl_out_path=nl_out_path,
        index_name=index_name,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        embed_base_url=embed_base_url,
        embed_model=embed_model,
        embed_max_input_tokens=embed_max_input_tokens,
        local_embedder=local_embedder,
    )

    metadata_path = lake_dir / "_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"build_autoddg_cell: {metadata_path} missing - run prepare first."
        )

    from scripts.create_blend_index import run_pipeline  # delayed import

    nl_overrides = preset_for_cell(cell) | {"index_name": index_name}

    t0 = time.perf_counter()
    run_pipeline(
        lake_path=str(lake_dir / LAKE_GLOB_PATTERN),
        config_path=config_path,
        nl_index=True,
        nl_config_overrides=nl_overrides,
        metadata_path=metadata_path,
        workers=1,
        value_index=False,
        pre_chunked_contexts=True,
        lake_exclude_names=LAKE_BOOKKEEPING_NAMES,
    )
    wall = time.perf_counter() - t0
    wall_clock_path.write_text(f"{wall:.3f}\n")
    return BuildResult(shared_duckdb, nl_out_path, index_name, config_path, wall)


__all__ = [
    "build_index",
    "build_autoddg_shared_duckdb",
    "build_autoddg_cell",
    "write_benchmark_config",
    "write_autoddg_cell_config",
    "BuildResult",
    "LAKE_GLOB_PATTERN",
    "LAKE_BOOKKEEPING_NAMES",
]
