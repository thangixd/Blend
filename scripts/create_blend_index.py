"""Build Blend's value-index, and (by default) NLSeeker's NL-index, in one pass.

Both indexes are written from a single ``sorted(glob(lake))`` enumeration so
they share the same integer ``TableId`` namespace.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import multiprocessing
import os
import sys
import threading
from configparser import ConfigParser
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.NLSeeker.config import NLSeekerConfig
from src.NLSeeker.index_build import NLIndexBuilder
from src.NLSeeker.llm import build_backends
from src.utils import df_to_index
from scripts import _blend_ingest_worker

LOG = logging.getLogger("create_blend_index")

_DEFAULT_METADATA = Path("data/metadata.csv")

# DB

def _read_db_section(config_path: Path) -> dict[str, str]:
    parser = ConfigParser()
    parser.read(config_path)
    if not parser.has_section("Database"):
        raise ValueError(f"[Database] section missing in {config_path}")
    return dict(parser.items("Database"))


def _open_writer(cfg: dict[str, str]):
    """Open a writeable connection in the configured DBMS."""
    dbms = cfg["dbms"].lower()
    if dbms == "duckdb":
        import duckdb

        con = duckdb.connect(database=cfg["path"], read_only=False)
    elif dbms == "postgres":
        import psycopg as pg

        con = pg.connect(
            host=cfg["host"],
            port=cfg["port"],
            user=cfg["user"],
            password=cfg["password"],
            dbname=cfg["dbname"],
        )
    elif dbms == "vertica":
        import vertica_python

        con = vertica_python.connect(
            host=cfg["host"],
            port=cfg["port"],
            user=cfg["user"],
            password=cfg["password"],
            database=cfg["dbname"],
            session_label="create_blend_index",
            read_timeout=60000,
            unicode_error="strict",
            ssl=False,
            use_prepared_statements=False,
        )
    else:
        raise ValueError(f"Unsupported dbms {dbms!r}")
    return con, con.cursor(), dbms


def _create_value_index_table(cursor, dbms: str, table_name: str) -> None:
    """Create the AllTables value index."""
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    if dbms == "duckdb":
        cursor.execute(
            f"""
            CREATE TABLE {table_name} (
                tokenized VARCHAR,
                tableid INTEGER,
                colid INTEGER,
                rowid INTEGER,
                super_key VARCHAR,
                quadrant BOOLEAN
            )
            """
        )
    elif dbms == "postgres":
        cursor.execute(
            f"""
            CREATE TABLE {table_name} (
                tokenized VARCHAR(200),
                tableid INTEGER,
                colid INTEGER,
                rowid INTEGER,
                super_key VARCHAR(64),
                quadrant BOOLEAN
            )
            """
        )
        cursor.execute(f"CREATE INDEX ON {table_name} (tokenized)")
        cursor.execute(f"CREATE INDEX ON {table_name} (tableid)")
    elif dbms == "vertica":
        cursor.execute(
            f"""
            CREATE TABLE {table_name} (
                tokenized varchar(200),
                tableid INT,
                colid INT,
                rowid INT,
                superkey BINARY(16),
                quadrant BOOLEAN
            )
            """
        )


def _insert_index_rows(cursor, dbms: str, table_name: str, df: pd.DataFrame) -> None:
    """Insert the per-table df_to_index() output rows into ``table_name``."""
    if df.empty:
        return

    def _to_pybool(x):
        if x is None:
            return None
        return bool(x)

    rows = [
        (cell, int(tid), int(cid), int(rid), sk, _to_pybool(q))
        for cell, tid, cid, rid, sk, q in df[
            ["CellValue", "TableId", "ColumnId", "RowId", "SuperKey", "Quadrant"]
        ].itertuples(index=False, name=None)
    ]
    if dbms == "duckdb":
        cursor.executemany(
            f"INSERT INTO {table_name} (tokenized, tableid, colid, rowid, super_key, quadrant) "
            f"VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
    elif dbms == "postgres":
        cursor.executemany(
            f"INSERT INTO {table_name} (tokenized, tableid, colid, rowid, super_key, quadrant) "
            f"VALUES (%s, %s, %s, %s, %s, %s)",
            rows,
        )
    elif dbms == "vertica":
        cursor.executemany(
            f"INSERT INTO {table_name} (tokenized, tableid, colid, rowid, superkey, quadrant) "
            f"VALUES (%s, %s, %s, %s, %s, %s)",
            rows,
        )


def _sink_one(cursor, con, dbms: str, table_name: str, result) -> None:
    """Insert one ``WorkerResult`` into ``table_name``.

    For DuckDB we register the Arrow table as a temporary view and ingest it
    via ``INSERT INTO ... SELECT FROM <view>`` - zero-copy, vectorised, and
    far faster than ``executemany``. The explicit column list is the
    byte-identity firewall: the worker's Arrow column names come from
    df_to_index's ``CellValue, TableId, ColumnId, RowId, SuperKey, Quadrant``
    and we map them to the existing schema's
    ``(tokenized, tableid, colid, rowid, super_key, quadrant)``.

    For Postgres / Vertica we convert the Arrow table back to a DataFrame
    and delegate to the existing ``_insert_index_rows`` executemany sink -
    those backends benefit from the parallel CPU compute but keep their
    current bulk-load path.
    """
    if result.value_shard is None:
        # Empty-table skip mirrors the sequential `if df.empty: continue`.
        return
    if dbms == "duckdb":
        # Register the view on `cursor`, not `con`: DuckDB's con.cursor()
        # returns an independent DuckDBPyConnection whose registered-view
        # namespace is separate from the parent connection's. Registering on
        # `con` would leave the view invisible to the cursor that runs the
        # INSERT, raising "Table with name blend_shard does not exist".
        cursor.register("blend_shard", result.value_shard)
        try:
            cursor.execute(
                f"INSERT INTO {table_name} "
                "(tokenized, tableid, colid, rowid, super_key, quadrant) "
                "SELECT CellValue, TableId, ColumnId, RowId, SuperKey, Quadrant "
                "FROM blend_shard"
            )
        finally:
            cursor.unregister("blend_shard")
    else:
        df = result.value_shard.to_pandas()
        _insert_index_rows(cursor, dbms, table_name, df)


def _run_parallel_ingest(
    files: list[tuple[int, Path]],
    cursor,
    con,
    dbms: str,
    table_name: str,
    workers: int,
    nl_builder,
    contexts_by_tid: dict[int, list[str]],
    value_index: bool = True,
    pre_chunked_contexts: bool = False,
) -> None:
    """Run per-table value-index compute in a process pool, sink each shard
    in submission order, and drive the NL builder in lockstep.

    Submission order is ``_iter_lake()`` order, so ``nl_builder.add_table()``
    is called in exactly the same order as the sequential pipeline.

    The transaction boundary lives here: we BEGIN before the first shard
    and COMMIT after the last; on any exception we ROLLBACK and DROP the
    partially-built ``table_name`` so the on-disk database is left clean.

    When ``value_index=False`` the cursor/con are None; the value-index
    transaction, _sink_one calls, and want_value compute are all skipped.
    NL builder calls are unconditional.
    """
    want_raw = nl_builder is not None

    if value_index and dbms == "duckdb":
        cursor.execute("BEGIN TRANSACTION")

    try:
        if workers <= 1:
            LOG.info(
                "Value-index ingest in single-process mode (workers=%d, lake=%d)",
                workers, len(files),
            )
            for table_id, file_path in tqdm(files, desc="Tables"):
                result = _blend_ingest_worker.build_value_shard(
                    table_id, str(file_path), want_raw, want_value=value_index,
                )
                if value_index:
                    _sink_one(cursor, con, dbms, table_name, result)
                if nl_builder is not None and result.raw_table is not None:
                    nl_builder.add_table(
                        table_id,
                        result.raw_table,
                        contexts=contexts_by_tid.get(table_id),
                        pre_chunked_contexts=pre_chunked_contexts,
                    )
                elif nl_builder is not None and contexts_by_tid.get(table_id):
                    LOG.warning(
                        "TableId=%d: %d contexts present but raw_table is None; "
                        "contexts will not be indexed for this table.",
                        table_id, len(contexts_by_tid[table_id]),
                    )
        else:
            LOG.info(
                "Parallel value-index ingest with %d workers (lake=%d)",
                workers, len(files),
            )
            # Bounded in-flight cap: at most workers*2 pending futures.
            # Without this the executor would happily queue every file in the
            # lake, and memory would scale with lake size, not pool size.
            inflight = threading.Semaphore(workers * 2)
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
            ) as pool:
                try:
                    futures: list[concurrent.futures.Future] = []
                    for table_id, file_path in files:
                        inflight.acquire()
                        fut = pool.submit(
                            _blend_ingest_worker.build_value_shard,
                            table_id, str(file_path), want_raw,
                            want_value=value_index,
                        )
                        fut.add_done_callback(lambda _f: inflight.release())
                        futures.append(fut)

                    for tid_path, fut in zip(files, tqdm(futures, desc="Tables")):
                        table_id = tid_path[0]
                        result = fut.result()  # raises in submission order
                        if value_index:
                            _sink_one(cursor, con, dbms, table_name, result)
                        if nl_builder is not None and result.raw_table is not None:
                            nl_builder.add_table(
                                table_id,
                                result.raw_table,
                                contexts=contexts_by_tid.get(table_id),
                                pre_chunked_contexts=pre_chunked_contexts,
                            )
                        elif nl_builder is not None and contexts_by_tid.get(table_id):
                            LOG.warning(
                                "TableId=%d: %d contexts present but raw_table is None; "
                                "contexts will not be indexed for this table.",
                                table_id, len(contexts_by_tid[table_id]),
                            )
                except BaseException:
                    # Drop queued-but-not-started work promptly; let the with
                    # block's exit join the workers.
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise

        if value_index:
            if dbms == "duckdb":
                cursor.execute("COMMIT")
            else:
                con.commit()
    except BaseException:
        if value_index:
            if dbms == "duckdb":
                try:
                    cursor.execute("ROLLBACK")
                except Exception:
                    pass
                try:
                    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                except Exception:
                    pass
            else:
                try:
                    con.rollback()
                except Exception:
                    pass
        raise


def _iter_lake(
    lake_path: str,
    exclude_names: Optional[frozenset] = None,
) -> list[tuple[int, Path]]:
    """Enumerate the lake into ``[(TableId, Path), ...]``.

    ``sorted(glob(lake_path))`` is the canonical TableId source. Reordering it
    would break the shared TableId namespace between the value-index and the
    NL-index, so no sort key is exposed.

    ``exclude_names`` filters out files by basename before TableIds are
    assigned. The benchmark uses this to skip ``_metadata.csv`` (written into
    the lake dir by prepare) without resorting to a glob pattern that also
    drops legitimate tables whose titles start with ``_``.
    """
    files = sorted(glob(lake_path, recursive=True))
    if exclude_names:
        files = [p for p in files if Path(p).name not in exclude_names]
    return [(idx, Path(p)) for idx, p in enumerate(files)]


def _read_table(file_path: Path) -> pd.DataFrame:
    """Read a CSV/Parquet table into pandas. Detection is by extension."""
    suffix = file_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(file_path)
    return pd.read_csv(file_path, low_memory=False)


def _df_for_value_index(table_id: int, df: pd.DataFrame) -> pd.DataFrame:
    """Stamp ``df.columns.name = TableId`` so utils.df_to_index can read it."""
    indexed = df.copy()
    indexed.columns.name = str(table_id)
    return df_to_index(indexed)


def _read_metadata_csv(path: Path) -> dict[int, list[str]]:
    """Load ``(TableId, Context)`` rows into ``{TableId: [Context, ...]}``.

    Multiple rows per ``TableId`` are concatenated in file order, so a table
    can carry several distinct context entries that get indexed separately.
    """
    df = pd.read_csv(path)
    required = {"TableId", "Context"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"Metadata CSV {path!r} missing columns: {sorted(missing)}. "
            f"Expected at least {sorted(required)}."
        )
    out: dict[int, list[str]] = {}
    for tid, ctx in df[["TableId", "Context"]].itertuples(index=False, name=None):
        if pd.isna(ctx):
            continue
        out.setdefault(int(tid), []).append(str(ctx))
    return out


def run_pipeline(
    lake_path: str,
    config_path: Path,
    nl_index: bool,
    nl_config_overrides: Optional[dict] = None,
    metadata_path: Optional[Path] = None,
    workers: int = 1,
    value_index: bool = True,
    pre_chunked_contexts: bool = False,
    lake_exclude_names: Optional[frozenset] = None,
) -> None:
    """End-to-end driver shared by the CLI and any programmatic caller.

    When ``metadata_path`` is supplied, each context row is indexed alongside
    the LLM-generated content summaries. Tables without a context row fall
    through to the content-only path.

    When ``value_index=False`` the value-index DuckDB table (``blend_index``)
    is not created or populated. NLSeeker's NL index is still built when
    ``nl_index=True``. Benchmark mode uses this to avoid building the
    100 MB-1 GB value-index that the legacy Union/SC/MC operators require.

    When ``pre_chunked_contexts=True`` each context row in ``_metadata.csv``
    is treated as exactly one chunk rather than being passed through
    ``block_texts``.  Benchmark mode enables this so that each merged-context
    record (one per table in ``contexts_<ds>_merged.jsonl``) becomes a single
    retrieval unit, mirroring PNEUMA's retriever behaviour.
    """
    db_cfg = _read_db_section(config_path)
    table_name = db_cfg["index_table"]
    files = _iter_lake(lake_path, exclude_names=lake_exclude_names)
    if not files:
        raise SystemExit(f"No files matched lake pattern {lake_path!r}")

    LOG.info("Lake has %d files; building value-index '%s' on %s", len(files), table_name, db_cfg["dbms"])

    contexts_by_tid: dict[int, list[str]] = {}
    if metadata_path is not None:
        contexts_by_tid = _read_metadata_csv(metadata_path)
        LOG.info(
            "Loaded contexts for %d tables from %s (%d total rows)",
            len(contexts_by_tid),
            metadata_path,
            sum(len(v) for v in contexts_by_tid.values()),
        )

    # Build NL up first so a misconfigured backend (missing key, unreachable
    # endpoint) fails before we open the value-index transaction.
    nl_builder: Optional[NLIndexBuilder] = None
    if nl_index:
        nl_cfg = NLSeekerConfig.load(config_path=config_path, overrides=nl_config_overrides)
        LOG.info("NLSeeker enabled: out_path=%s index=%s", nl_cfg.out_path, nl_cfg.index_name)
        # Eagerly construct the backends to surface auth/connectivity errors now.
        build_backends(nl_cfg)
        nl_builder = NLIndexBuilder(nl_cfg)
        nl_builder.start()

    if value_index:
        con, cursor, dbms = _open_writer(db_cfg)
    else:
        con, cursor, dbms = None, None, db_cfg.get("dbms", "duckdb").lower()

    try:
        if value_index:
            _create_value_index_table(cursor, dbms, table_name)
        _run_parallel_ingest(
            files=files,
            cursor=cursor,
            con=con,
            dbms=dbms,
            table_name=table_name,
            workers=workers,
            nl_builder=nl_builder,
            contexts_by_tid=contexts_by_tid,
            value_index=value_index,
            pre_chunked_contexts=pre_chunked_contexts,
        )
    finally:
        if value_index:
            cursor.close()
            con.close()

    if nl_builder is not None:
        result = nl_builder.finalize()
        LOG.info(
            "NLSeeker index '%s' built: %d documents across %d tables",
            result.index_name,
            result.total_documents,
            result.total_tables,
        )

    if value_index:
        LOG.info("Value-index '%s' built across %d tables", table_name, len(files))
    else:
        LOG.info("Value-index skipped (value_index=False); NL-only build across %d tables", len(files))



# CLI

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--lake",
        default="data/lake/*.csv",
        help="Glob pattern for the lake (default: 'data/lake/*.csv').",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT / "config" / "config.ini",
        help="Path to Blend's config.ini (default: ./config/config.ini).",
    )
    parser.add_argument(
        "--nl-index",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Build NLSeeker's vector + BM25 index alongside the value-index "
            "(default: on). Pass --no-nl-index to skip "
        ),
    )
    parser.add_argument(
        "--value-index",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Build the value-index (blend_index DuckDB table) used by the "
            "legacy Union/SC/MC operators (default: on). Pass --no-value-index "
            "to skip - useful for NL-only benchmark builds."
        ),
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=_DEFAULT_METADATA,
        help=(
            "CSV with columns 'TableId, Context' adding free-text metadata "
            "to the NL index (default: data/metadata.csv if present; "
            "silently ignored if missing). One row per context entry; "
            "multiple rows per TableId allowed. Ignored when --no-nl-index."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min((os.cpu_count() or 2) - 1, 16),
        help=(
            "Process-pool size for parallel value-index ingestion. "
            "Default: min(cpu_count - 1, 16). Pass 1 to disable the pool "
            "(single-process fast path; identical to the pre-parallel build)."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    return args


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    metadata_path: Optional[Path] = args.metadata_path
    # Default path is tolerated as missing; an explicit --metadata-path that
    # does not exist still raises inside _read_metadata_csv.
    if metadata_path == _DEFAULT_METADATA and not metadata_path.exists():
        metadata_path = None
    run_pipeline(
        lake_path=args.lake,
        config_path=args.config,
        nl_index=args.nl_index,
        metadata_path=metadata_path if args.nl_index else None,
        workers=args.workers,
        value_index=args.value_index,
    )


if __name__ == "__main__":
    main()
