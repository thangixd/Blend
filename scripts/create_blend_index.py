"""Build Blend's value-index, and (by default) NLSeeker's NL-index, in one pass.

Both indexes are written from a single ``sorted(glob(lake))`` enumeration so
they share the same integer ``TableId`` namespace.
"""

from __future__ import annotations

import argparse
import logging
import sys
from configparser import ConfigParser
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


def _iter_lake(lake_path: str) -> list[tuple[int, Path]]:
    """Enumerate the lake into ``[(TableId, Path), ...]``.

    ``sorted(glob(lake_path))`` is the canonical TableId source. Reordering it
    would break the shared TableId namespace between the value-index and the
    NL-index, so no sort key is exposed.
    """
    files = sorted(glob(lake_path, recursive=True))
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
) -> None:
    """End-to-end driver shared by the CLI and any programmatic caller.

    When ``metadata_path`` is supplied, each context row is indexed alongside
    the LLM-generated content summaries. Tables without a context row fall
    through to the content-only path.
    """
    db_cfg = _read_db_section(config_path)
    table_name = db_cfg["index_table"]
    files = _iter_lake(lake_path)
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

    con, cursor, dbms = _open_writer(db_cfg)
    try:
        _create_value_index_table(cursor, dbms, table_name)
        for table_id, file_path in tqdm(files, desc="Tables"):
            df = _read_table(file_path)
            if df.empty:
                LOG.warning("TableId=%d (%s) is empty - skipping", table_id, file_path.name)
                continue

            # _df_for_value_index copies before stamping columns.name, so df
            # is unmutated and safe to hand to the NL builder afterwards.
            value_rows = _df_for_value_index(table_id, df)
            _insert_index_rows(cursor, dbms, table_name, value_rows)

            if nl_builder is not None:
                nl_builder.add_table(
                    table_id,
                    df,
                    contexts=contexts_by_tid.get(table_id),
                )

        con.commit() if dbms != "duckdb" else None
    except Exception:
        if dbms != "duckdb":
            con.rollback()
        raise
    finally:
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

    LOG.info("Value-index '%s' built across %d tables", table_name, len(files))



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
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args(argv)


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
    )


if __name__ == "__main__":
    main()
