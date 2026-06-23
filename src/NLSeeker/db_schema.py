import contextlib
import enum
from configparser import ConfigParser

# Typing imports
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class SummaryType(str, enum.Enum):
    COLUMN_NARRATION = "COLUMN_NARRATION"
    COLUMN_NARRATION_PROFILED = "COLUMN_NARRATION_PROFILED"  # spec §3.2
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
    # Spec §3.1: per-(TableId, Source) ContextIdx; default Source='real' for
    # backward compat with pre-AutoDDG rows.
    return """
    CREATE TABLE IF NOT EXISTS blend_nl_contexts (
        TableId    INTEGER NOT NULL,
        Source     VARCHAR NOT NULL DEFAULT 'real',
        ContextIdx INTEGER NOT NULL,
        Text       VARCHAR NOT NULL,
        PRIMARY KEY (TableId, Source, ContextIdx)
    )
    """


def _autoddg_profiles_ddl(dbms: str) -> str:
    # NB: dbms param accepted for signature consistency with _summaries_ddl;
    # DDL is DuckDB/Postgres syntax. Vertica needs DEFAULT NOW() and no
    # CREATE TABLE IF NOT EXISTS - extend when Vertica support is needed.
    return """
    CREATE TABLE IF NOT EXISTS blend_nl_autoddg_profiles (
        TableId     INTEGER NOT NULL,
        ProfileKind VARCHAR NOT NULL,
        ProfileJSON VARCHAR NOT NULL,
        GeneratedAt TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (TableId, ProfileKind)
    )
    """


def _autoddg_topics_ddl(dbms: str) -> str:
    # NB: dbms param accepted for signature consistency with _summaries_ddl;
    # DDL is DuckDB/Postgres syntax. Vertica needs DEFAULT NOW() and no
    # CREATE TABLE IF NOT EXISTS - extend when Vertica support is needed.
    return """
    CREATE TABLE IF NOT EXISTS blend_nl_autoddg_topics (
        TableId     INTEGER NOT NULL PRIMARY KEY,
        Topic       VARCHAR NOT NULL,
        GeneratedAt TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
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
    """Create (or migrate) the auxiliary tables.

    Migration: a pre-existing ``blend_nl_contexts`` without a ``Source``
    column is rebuilt in-place with the new PK. Existing rows are preserved
    with ``Source='real'``.
    """
    cursor.execute(_summaries_ddl(dbms))
    cursor.execute(_indexes_ddl(dbms))

    # Detect existing contexts table.
    legacy_exists = False
    needs_migration = False
    if dbms == "duckdb":
        rows = cursor.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='blend_nl_contexts'"
        ).fetchall()
        if rows:
            legacy_exists = True
            needs_migration = "Source" not in {r[0] for r in rows}
    elif dbms == "postgres":
        rows = cursor.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='blend_nl_contexts'"
        ).fetchall() or cursor.fetchall()
        if rows:
            legacy_exists = True
            needs_migration = "source" not in {r[0].lower() for r in rows}
    else:  # vertica
        cursor.execute(
            "SELECT column_name FROM v_catalog.columns "
            "WHERE table_name='blend_nl_contexts'"
        )
        rows = cursor.fetchall()
        if rows:
            legacy_exists = True
            needs_migration = "source" not in {r[0].lower() for r in rows}

    if legacy_exists and needs_migration:
        # Rebuild in-place: rename old -> new schema -> copy -> drop.
        # DuckDB auto-commits each DDL statement; wrap in an explicit
        # transaction so that a mid-migration crash leaves the database
        # recoverable (ROLLBACK restores the original table name).
        if dbms == "duckdb":
            cursor.execute("BEGIN TRANSACTION")
            try:
                cursor.execute("ALTER TABLE blend_nl_contexts RENAME TO _blend_nl_contexts_legacy")
                cursor.execute(_contexts_ddl(dbms))
                cursor.execute(
                    "INSERT INTO blend_nl_contexts (TableId, Source, ContextIdx, Text) "
                    "SELECT TableId, 'real' AS Source, ContextIdx, Text "
                    "FROM _blend_nl_contexts_legacy"
                )
                cursor.execute("DROP TABLE _blend_nl_contexts_legacy")
                cursor.execute("COMMIT")
            except Exception:
                cursor.execute("ROLLBACK")
                raise
        else:
            # Postgres / Vertica: DDL runs inside the connection-level transaction
            # managed by open_writer(), so no explicit BEGIN is needed.
            cursor.execute("ALTER TABLE blend_nl_contexts RENAME TO _blend_nl_contexts_legacy")
            cursor.execute(_contexts_ddl(dbms))
            cursor.execute(
                "INSERT INTO blend_nl_contexts (TableId, Source, ContextIdx, Text) "
                "SELECT TableId, 'real' AS Source, ContextIdx, Text "
                "FROM _blend_nl_contexts_legacy"
            )
            cursor.execute("DROP TABLE _blend_nl_contexts_legacy")
    else:
        cursor.execute(_contexts_ddl(dbms))

    cursor.execute(_autoddg_profiles_ddl(dbms))
    cursor.execute(_autoddg_topics_ddl(dbms))

    # Spec §3.1: standalone index on Source for filter-by-source queries.
    if dbms == "duckdb":
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_blend_nl_contexts_source "
            "ON blend_nl_contexts (Source)"
        )
    elif dbms == "postgres":
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_blend_nl_contexts_source "
            "ON blend_nl_contexts (Source)"
        )
    else:  # vertica
        # Vertica uses projections, not classical indexes - skip; the PK already
        # contains Source as the second key, which is what Vertica would use anyway.
        pass


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
    """Bulk-insert context rows. Accepts ``(TableId, ContextIdx, Text)`` for
    backward compatibility (Source defaults to ``'real'``) or
    ``(TableId, Source, ContextIdx, Text)`` for the AutoDDG-aware path.
    """
    if not rows:
        return
    payload: list[tuple] = []
    for entry in rows:
        if len(entry) == 3:
            tid, idx, text = entry
            src = "real"
        elif len(entry) == 4:
            tid, src, idx, text = entry
        else:
            raise ValueError(
                f"insert_contexts expects (TableId, ContextIdx, Text) or "
                f"(TableId, Source, ContextIdx, Text); got {len(entry)}-tuple"
            )
        payload.append((int(tid), str(src), int(idx), text))

    if dbms == "duckdb":
        cursor.executemany(
            "INSERT OR REPLACE INTO blend_nl_contexts "
            "(TableId, Source, ContextIdx, Text) VALUES (?, ?, ?, ?)",
            payload,
        )
    elif dbms == "postgres":
        cursor.executemany(
            "INSERT INTO blend_nl_contexts (TableId, Source, ContextIdx, Text) "
            "VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (TableId, Source, ContextIdx) DO UPDATE SET Text = EXCLUDED.Text",
            payload,
        )
    else:  # vertica - no UPSERT, fall back to delete-then-insert
        keys = {(tid, src, idx) for tid, src, idx, _ in payload}
        for tid, src, idx in keys:
            cursor.execute(
                "DELETE FROM blend_nl_contexts WHERE TableId=%s AND Source=%s AND ContextIdx=%s",
                (tid, src, idx),
            )
        cursor.executemany(
            "INSERT INTO blend_nl_contexts (TableId, Source, ContextIdx, Text) "
            "VALUES (%s, %s, %s, %s)",
            payload,
        )


def fetch_table_contexts(
    cursor,
    dbms: str,
    table_ids: Optional[Iterable[int]] = None,
    sources: Optional[Iterable[str]] = None,
) -> dict:
    """Return ``{TableId: [Text, ... ordered by Source, ContextIdx]}``.

    ``sources`` filters by the new Source tag (spec §3.1). When omitted,
    all sources are returned, preserving the legacy call site's behavior on
    a real-only database.
    """
    base = "SELECT TableId, Source, ContextIdx, Text FROM blend_nl_contexts"
    where: list[str] = []
    params: list = []
    qmark = "?" if dbms == "duckdb" else "%s"
    if table_ids is not None:
        ids = tuple(int(t) for t in table_ids)
        if not ids:
            return {}
        placeholders = ",".join(qmark for _ in ids)
        where.append(f"TableId IN ({placeholders})")
        params.extend(ids)
    if sources is not None:
        srcs = tuple(str(s) for s in sources)
        if not srcs:
            return {}
        placeholders = ",".join(qmark for _ in srcs)
        where.append(f"Source IN ({placeholders})")
        params.extend(srcs)
    if where:
        base += " WHERE " + " AND ".join(where)
    base += " ORDER BY TableId, Source, ContextIdx"
    cursor.execute(base, tuple(params)) if params else cursor.execute(base)
    out: dict = {}
    for tid, _src, _idx, text in cursor.fetchall():
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


def upsert_autoddg_profile(cursor, dbms: str, table_id: int, kind: str, payload_json: str) -> None:
    """Upsert one ProfileKind row for a table. ``kind`` ∈ {'content', 'semantic'}."""
    if dbms == "duckdb":
        cursor.execute(
            "INSERT OR REPLACE INTO blend_nl_autoddg_profiles "
            "(TableId, ProfileKind, ProfileJSON) VALUES (?, ?, ?)",
            (int(table_id), kind, payload_json),
        )
    elif dbms == "postgres":
        cursor.execute(
            "INSERT INTO blend_nl_autoddg_profiles (TableId, ProfileKind, ProfileJSON) "
            "VALUES (%s, %s, %s) "
            "ON CONFLICT (TableId, ProfileKind) DO UPDATE SET ProfileJSON=EXCLUDED.ProfileJSON",
            (int(table_id), kind, payload_json),
        )
    else:
        cursor.execute(
            "DELETE FROM blend_nl_autoddg_profiles WHERE TableId=%s AND ProfileKind=%s",
            (int(table_id), kind),
        )
        cursor.execute(
            "INSERT INTO blend_nl_autoddg_profiles (TableId, ProfileKind, ProfileJSON) "
            "VALUES (%s, %s, %s)",
            (int(table_id), kind, payload_json),
        )


def fetch_autoddg_profile(cursor, dbms: str, table_id: int, kind: str) -> Optional[str]:
    """Return the raw ProfileJSON for ``(table_id, kind)``, or ``None`` if absent."""
    if dbms == "duckdb":
        cursor.execute(
            "SELECT ProfileJSON FROM blend_nl_autoddg_profiles "
            "WHERE TableId=? AND ProfileKind=?",
            (int(table_id), kind),
        )
    else:
        cursor.execute(
            "SELECT ProfileJSON FROM blend_nl_autoddg_profiles "
            "WHERE TableId=%s AND ProfileKind=%s",
            (int(table_id), kind),
        )
    row = cursor.fetchone()
    return None if row is None else row[0]


def upsert_autoddg_topic(cursor, dbms: str, table_id: int, topic: str) -> None:
    """Upsert the 2-3 word topic for a table."""
    if dbms == "duckdb":
        cursor.execute(
            "INSERT OR REPLACE INTO blend_nl_autoddg_topics (TableId, Topic) VALUES (?, ?)",
            (int(table_id), topic),
        )
    elif dbms == "postgres":
        cursor.execute(
            "INSERT INTO blend_nl_autoddg_topics (TableId, Topic) VALUES (%s, %s) "
            "ON CONFLICT (TableId) DO UPDATE SET Topic=EXCLUDED.Topic",
            (int(table_id), topic),
        )
    else:
        cursor.execute("DELETE FROM blend_nl_autoddg_topics WHERE TableId=%s", (int(table_id),))
        cursor.execute(
            "INSERT INTO blend_nl_autoddg_topics (TableId, Topic) VALUES (%s, %s)",
            (int(table_id), topic),
        )


def fetch_autoddg_topic(cursor, dbms: str, table_id: int) -> Optional[str]:
    """Return the cached topic string for ``table_id``, or ``None`` if absent."""
    if dbms == "duckdb":
        cursor.execute(
            "SELECT Topic FROM blend_nl_autoddg_topics WHERE TableId=?",
            (int(table_id),),
        )
    else:
        cursor.execute(
            "SELECT Topic FROM blend_nl_autoddg_topics WHERE TableId=%s",
            (int(table_id),),
        )
    row = cursor.fetchone()
    return None if row is None else row[0]


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
    "upsert_autoddg_profile",
    "fetch_autoddg_profile",
    "upsert_autoddg_topic",
    "fetch_autoddg_topic",
]
