import argparse
import threading
from pathlib import Path

import duckdb

from scripts.create_index_duckdb import BATCH_SIZE, assign_table_ids, create_index
from src.NLSeeker.Clients import EmbeddingClient, LLMClient
from src.NLSeeker.Config import NLSeekerConfig
from src.NLSeeker.IndexGenerator import IndexGenerator
from src.NLSeeker.Registrar import Registrar
from src.NLSeeker.Summarizer import Summarizer

# Typing imports
from typing import List, Tuple


class LockedConnection:
    """A DuckDB connection whose statements are serialized across the builder threads.

    Each thread takes its own cursor so that `SET search_path` and registered DataFrames stay
    thread-local, while the shared lock keeps two threads out of the catalog at the same time.
    """

    def __init__(self, connection, lock=None) -> None:
        self._connection = connection
        self._lock = lock if lock is not None else threading.Lock()

    def cursor(self) -> 'LockedConnection':
        with self._lock:
            return LockedConnection(self._connection.cursor(), self._lock)

    def execute(self, *args, **kwargs) -> 'LockedConnection':
        with self._lock:
            self._connection.execute(*args, **kwargs)
        return self

    def executemany(self, *args, **kwargs) -> 'LockedConnection':
        with self._lock:
            self._connection.executemany(*args, **kwargs)
        return self

    def fetchall(self):
        with self._lock:
            return self._connection.fetchall()

    def fetchone(self):
        with self._lock:
            return self._connection.fetchone()

    def df(self):
        with self._lock:
            return self._connection.df()

    def register(self, *args, **kwargs) -> None:
        with self._lock:
            self._connection.register(*args, **kwargs)

    def unregister(self, *args, **kwargs) -> None:
        with self._lock:
            self._connection.unregister(*args, **kwargs)

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    @property
    def description(self):
        return self._connection.description


def resolve_assignment(dbcon, lake_dir: Path, table_name: str, rebuild_blend_index: bool) -> List[Tuple[int, str]]:
    """Assigns TableIds, or recovers the existing assignment when the Blend index is not rebuilt."""
    if rebuild_blend_index:
        return assign_table_ids(str(lake_dir / '*.csv'))

    exists = dbcon.execute('SELECT count(*) FROM duckdb_tables() WHERE table_name = ?',
                           [f'{table_name}_tables']).fetchone()[0]
    if not exists:
        raise LookupError(f'"{table_name}_tables" does not exist; build the Blend index before '
                          f'reusing its TableIds.')

    stored = dbcon.execute(f'SELECT tableid, filename FROM "{table_name}_tables" ORDER BY tableid').fetchall()
    on_disk = {path.name: path for path in sorted(lake_dir.glob('*.csv'))}

    missing = [filename for _, filename in stored if filename not in on_disk]
    extra = sorted(set(on_disk) - {filename for _, filename in stored})
    if missing or extra:
        raise LookupError(f'{lake_dir} no longer matches "{table_name}_tables". '
                          f'Indexed but absent: {missing}. Present but not indexed: {extra}. '
                          f'Rebuild the Blend index instead of reusing its TableIds.')

    return [(table_id, str(on_disk[filename])) for table_id, filename in stored]


def verify_ids(dbcon, table_name: str, schema: str) -> None:
    """Fails unless the Blend and NL table mappings are the same set of (TableId, filename) pairs."""
    mismatches = dbcon.execute(f"""
        (SELECT tableid, filename FROM "{table_name}_tables"
         EXCEPT
         SELECT CAST(id AS INTEGER), table_name FROM "{schema}".table_status)
        UNION ALL
        (SELECT CAST(id AS INTEGER), table_name FROM "{schema}".table_status
         EXCEPT
         SELECT tableid, filename FROM "{table_name}_tables")
    """).fetchall()

    if mismatches:
        raise AssertionError(f'TableId mismatch between "{table_name}_tables" and '
                             f'"{schema}".table_status: {mismatches}')
    print(f'TableIds verified: "{table_name}_tables" and "{schema}".table_status agree.')


def build_nl_index(dbcon, config: NLSeekerConfig, assignment, metadata: str, accept_duplicates: bool) -> None:
    llm = LLMClient(config)
    embedder = EmbeddingClient(config)

    registrar = Registrar(dbcon, config.schema)
    registrar.setup()
    registered = registrar.add_tables(assignment, creator='blend', accept_duplicates=accept_duplicates)
    print(f'Registered {len(registered)} of {len(assignment)} tables.')
    if metadata:
        print(f'Registered {registrar.add_metadata(metadata)} metadata entries.')

    Summarizer(dbcon, config.schema, llm, embedder).summarize()

    generator = IndexGenerator(dbcon, config.schema, embedder, config.vector_path, config.fulltext_path)
    print(f'Indexed {generator.generate_index(config.index_name, replace=True)} documents.')


def parse_args():
    repo_root = Path(__file__).parent.parent

    parser = argparse.ArgumentParser(
        description='Build the BLEND inverted index and the NL index over one CSV data lake.')
    parser.add_argument('--lake', default='adventure_works',
                        help='Lake directory inside --datalake, also used as the index table name.')
    parser.add_argument('--datalake', default=str(repo_root / 'datalake'), help='Directory holding the lakes.')
    parser.add_argument('--db', default=str(repo_root / 'blend_duckdb.db'), help='DuckDB database file to write.')
    parser.add_argument('--config', default=str(repo_root / 'config' / 'config.ini'),
                        help='Ini file carrying the [NLSeeker] section.')
    parser.add_argument('--sep', default=',', help='CSV separator.')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Index rows to buffer before writing to DuckDB.')
    parser.add_argument('--metadata', default=None, help='Metadata file or directory to register as context.')
    parser.add_argument('--no-blend-index', action='store_true', help='Skip the inverted index.')
    parser.add_argument('--no-nl-index', action='store_true', help='Skip the NL index.')
    parser.add_argument('--reject-duplicates', action='store_true',
                        help='Skip tables whose content already exists under another TableId.')
    parser.add_argument('--sequential', action='store_true', help='Build the two indexes one after the other.')
    parser.add_argument('--verify-ids', action='store_true',
                        help='Only check that both indexes agree on TableIds, then exit.')

    return parser.parse_args()


def main():
    args = parse_args()

    lake_dir = Path(args.datalake) / args.lake
    if not lake_dir.is_dir():
        raise NotADirectoryError(f'Lake directory not found: {lake_dir}')
    if args.no_blend_index and args.no_nl_index:
        raise ValueError('Nothing to build: --no-blend-index and --no-nl-index are both set.')

    config = NLSeekerConfig.load(args.config)

    # DBHandler opens DuckDB read-only, so index creation needs its own connection.
    dbcon = LockedConnection(duckdb.connect(database=args.db, read_only=False))
    try:
        if args.verify_ids:
            verify_ids(dbcon, args.lake, config.schema)
            return

        assignment = resolve_assignment(dbcon, lake_dir, args.lake, not args.no_blend_index)

        blend = (lambda: create_index(dbcon.cursor(), assignment, args.lake,
                                      sep=args.sep, batch_size=args.batch_size)) \
            if not args.no_blend_index else None
        nl = (lambda: build_nl_index(dbcon.cursor(), config, assignment,
                                     args.metadata, not args.reject_duplicates)) \
            if not args.no_nl_index else None

        run([step for step in (blend, nl) if step is not None], args.sequential)

        if blend is not None and nl is not None:
            verify_ids(dbcon, args.lake, config.schema)
    finally:
        dbcon.close()

    print(f'\nIndexes written to {args.db} and {config.index_path}. To query them, set in {args.config}:')
    print(f'  dbms=duckdb\n  path={args.db}\n  index_table={args.lake}')
    print(f'  [NLSeeker] index_name={config.index_name}  schema={config.schema}')


def run(steps, sequential: bool) -> None:
    if sequential or len(steps) == 1:
        for step in steps:
            step()
        return

    failures = []

    def guarded(step):
        try:
            step()
        except BaseException as error:
            failures.append(error)

    threads = [threading.Thread(target=guarded, args=(step,)) for step in steps]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if failures:
        raise failures[0]


if __name__ == '__main__':
    main()
