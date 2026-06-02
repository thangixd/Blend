import contextlib
import enum
from configparser import ConfigParser

# Typing imports
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class SummaryType(str, enum.Enum):
    COLUMN_NARRATION = "COLUMN_NARRATION"
    ROW_SAMPLE = "ROW_SAMPLE"
    CONTEXT = "CONTEXT"


def _read_db_section(config_path: Optional[Path] = None) -> dict:
    path = Path(config_path) if config_path else _PROJECT_ROOT / "config" / "config.ini"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path}")
    parser = ConfigParser()
    parser.read(path)
    if not parser.has_section("Database"):
        raise ValueError(f"[Database] section missing in {path}")
    return dict(parser.items("Database"))


@contextlib.contextmanager
def open_writer(config_path: Optional[Path] = None) -> Iterator:
    """Yield ``(cursor, dbms)`` against a writeable connection."""

    cfg = _read_db_section(config_path)
    dbms = cfg["dbms"].lower()

    if dbms == "duckdb":
        import duckdb

        # DuckDB rejects opening the same file rw + ro from one process.
        con = duckdb.connect(database=cfg["path"], read_only=False)
        cursor = con.cursor()
        try:
            yield cursor, dbms
        finally:
            cursor.close()
            con.close()
    elif dbms == "postgres":
        import psycopg as pg

        con = pg.connect(
            host=cfg["host"],
            port=cfg["port"],
            user=cfg["user"],
            password=cfg["password"],
            dbname=cfg["dbname"],
        )
        cursor = con.cursor()
        try:
            yield cursor, dbms
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            cursor.close()
            con.close()
    elif dbms == "vertica":
        import vertica_python

        con = vertica_python.connect(
            host=cfg["host"],
            port=cfg["port"],
            user=cfg["user"],
            password=cfg["password"],
            database=cfg["dbname"],
            session_label="nlseeker_writer",
            read_timeout=60000,
            unicode_error="strict",
            ssl=False,
            use_prepared_statements=False,
        )
        cursor = con.cursor()
        try:
            yield cursor, dbms
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            cursor.close()
            con.close()
    else:
        raise ValueError(f"Unsupported dbms {dbms!r} for NLSeeker writer")


def _summaries_ddl(dbms: str) -> str:
    return """
    CREATE TABLE IF NOT EXISTS blend_nl_summaries (
        TableId INTEGER NOT NULL,
        BlockIdx INTEGER NOT NULL,
        SummaryType VARCHAR(32) NOT NULL,
        Text VARCHAR NOT NULL,
        PRIMARY KEY (TableId, BlockIdx, SummaryType)
    )
    """


def _contexts_ddl(dbms: str) -> str:
    return """
    CREATE TABLE IF NOT EXISTS blend_nl_contexts (
        TableId INTEGER NOT NULL,
        ContextIdx INTEGER NOT NULL,
        Text VARCHAR NOT NULL,
        PRIMARY KEY (TableId, ContextIdx)
    )
    """


def _indexes_ddl(dbms: str) -> str:
    return """
    CREATE TABLE IF NOT EXISTS blend_nl_indexes (
        name VARCHAR PRIMARY KEY,
        vector_path VARCHAR NOT NULL,
        fulltext_path VARCHAR NOT NULL
    )
    """


def ensure_schema(cursor, dbms: str) -> None:
    """Create the auxiliary tables if absent."""
    cursor.execute(_summaries_ddl(dbms))
    cursor.execute(_indexes_ddl(dbms))
    cursor.execute(_contexts_ddl(dbms))


def insert_summaries(
    cursor,
    dbms: str,
    rows: Sequence[tuple],
) -> None:
    """Bulk-insert ``(TableId, BlockIdx, SummaryType, Text)`` rows."""
    if not rows:
        return
    payload = [
        (
            int(tid),
            int(idx),
            stype.value if isinstance(stype, SummaryType) else str(stype),
            text,
        )
        for tid, idx, stype, text in rows
    ]
    if dbms == "duckdb":
        cursor.executemany(
            "INSERT OR REPLACE INTO blend_nl_summaries "
            "(TableId, BlockIdx, SummaryType, Text) VALUES (?, ?, ?, ?)",
            payload,
        )
    elif dbms == "postgres":
        cursor.executemany(
            "INSERT INTO blend_nl_summaries (TableId, BlockIdx, SummaryType, Text) "
            "VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (TableId, BlockIdx, SummaryType) DO UPDATE SET Text = EXCLUDED.Text",
            payload,
        )
    else:  # vertica - no UPSERT, fall back to delete-then-insert
        keys = {(tid, idx, stype) for tid, idx, stype, _ in payload}
        for tid, idx, stype in keys:
            cursor.execute(
                "DELETE FROM blend_nl_summaries WHERE TableId = %s AND BlockIdx = %s AND SummaryType = %s",
                (tid, idx, stype),
            )
        cursor.executemany(
            "INSERT INTO blend_nl_summaries (TableId, BlockIdx, SummaryType, Text) "
            "VALUES (%s, %s, %s, %s)",
            payload,
        )


def insert_contexts(
    cursor,
    dbms: str,
    rows: Sequence[tuple],
) -> None:
    """Bulk-insert raw ``(TableId, ContextIdx, Text)`` context rows."""
    if not rows:
        return
    payload = [(int(tid), int(idx), text) for tid, idx, text in rows]
    if dbms == "duckdb":
        cursor.executemany(
            "INSERT OR REPLACE INTO blend_nl_contexts "
            "(TableId, ContextIdx, Text) VALUES (?, ?, ?)",
            payload,
        )
    elif dbms == "postgres":
        cursor.executemany(
            "INSERT INTO blend_nl_contexts (TableId, ContextIdx, Text) "
            "VALUES (%s, %s, %s) "
            "ON CONFLICT (TableId, ContextIdx) DO UPDATE SET Text = EXCLUDED.Text",
            payload,
        )
    else:  # vertica - no UPSERT, fall back to delete-then-insert
        keys = {(tid, idx) for tid, idx, _ in payload}
        for tid, idx in keys:
            cursor.execute(
                "DELETE FROM blend_nl_contexts WHERE TableId = %s AND ContextIdx = %s",
                (tid, idx),
            )
        cursor.executemany(
            "INSERT INTO blend_nl_contexts (TableId, ContextIdx, Text) "
            "VALUES (%s, %s, %s)",
            payload,
        )


def fetch_table_contexts(
    cursor,
    dbms: str,
    table_ids: Optional[Iterable[int]] = None,
) -> dict:
    """Return ``{TableId: [Text, ... ordered by ContextIdx]}``."""
    base = "SELECT TableId, ContextIdx, Text FROM blend_nl_contexts"
    params: tuple = ()
    if table_ids is not None:
        ids = tuple(int(t) for t in table_ids)
        if not ids:
            return {}
        placeholders = ",".join("?" if dbms == "duckdb" else "%s" for _ in ids)
        base += f" WHERE TableId IN ({placeholders})"
        params = ids
    base += " ORDER BY TableId, ContextIdx"
    cursor.execute(base, params) if params else cursor.execute(base)
    out: dict = {}
    for tid, _idx, text in cursor.fetchall():
        out.setdefault(int(tid), []).append(text)
    return out


def upsert_index(
    cursor,
    dbms: str,
    name: str,
    vector_path: Path,
    fulltext_path: Path,
) -> None:
    """Record (or update) the on-disk paths of an NL index by name."""
    payload = (str(name), str(Path(vector_path).resolve()), str(Path(fulltext_path).resolve()))
    if dbms == "duckdb":
        cursor.execute(
            "INSERT OR REPLACE INTO blend_nl_indexes (name, vector_path, fulltext_path) "
            "VALUES (?, ?, ?)",
            payload,
        )
    elif dbms == "postgres":
        cursor.execute(
            "INSERT INTO blend_nl_indexes (name, vector_path, fulltext_path) VALUES (%s, %s, %s) "
            "ON CONFLICT (name) DO UPDATE SET vector_path = EXCLUDED.vector_path, "
            "fulltext_path = EXCLUDED.fulltext_path",
            payload,
        )
    else:  # vertica
        cursor.execute("DELETE FROM blend_nl_indexes WHERE name = %s", (name,))
        cursor.execute(
            "INSERT INTO blend_nl_indexes (name, vector_path, fulltext_path) VALUES (%s, %s, %s)",
            payload,
        )


def fetch_index_paths(
    cursor,
    dbms: str,
    name: str,
):
    """Return ``(vector_path, fulltext_path)`` for ``name``, or ``None``."""
    if dbms == "postgres" or dbms == "vertica":
        cursor.execute(
            "SELECT vector_path, fulltext_path FROM blend_nl_indexes WHERE name = %s",
            (name,),
        )
    else:
        cursor.execute(
            "SELECT vector_path, fulltext_path FROM blend_nl_indexes WHERE name = ?",
            (name,),
        )
    row = cursor.fetchone()
    if row is None:
        return None
    return Path(row[0]), Path(row[1])


def fetch_table_summaries(
    cursor,
    dbms: str,
    table_ids: Optional[Iterable[int]] = None,
) -> dict:
    """Return ``{TableId: {SummaryType: [Text, ... ordered by BlockIdx]}}``."""
    base = "SELECT TableId, BlockIdx, SummaryType, Text FROM blend_nl_summaries"
    params: tuple = ()
    if table_ids is not None:
        ids = tuple(int(t) for t in table_ids)
        if not ids:
            return {}
        placeholders = ",".join("?" if dbms == "duckdb" else "%s" for _ in ids)
        base += f" WHERE TableId IN ({placeholders})"
        params = ids
    base += " ORDER BY TableId, SummaryType, BlockIdx"
    cursor.execute(base, params) if params else cursor.execute(base)
    out: dict = {}
    for tid, _idx, stype, text in cursor.fetchall():
        bucket = out.setdefault(int(tid), {})
        bucket.setdefault(SummaryType(stype), []).append(text)
    return out


__all__ = [
    "SummaryType",
    "open_writer",
    "ensure_schema",
    "insert_summaries",
    "insert_contexts",
    "upsert_index",
    "fetch_index_paths",
    "fetch_table_summaries",
    "fetch_table_contexts",
]
