import json
from pathlib import Path

import pandas as pd

from src.NLSeeker import Schema

# Typing imports
from duckdb import DuckDBPyConnection
from typing import Iterable, List, Optional, Tuple, Union


class Registrar:
    """Copies lake tables and their metadata into the NL schema, keyed by Blend TableId."""

    def __init__(self, connection: DuckDBPyConnection, schema: str) -> None:
        self.connection = connection
        self.schema = schema
        Schema.use_schema(connection, schema)

    def setup(self) -> None:
        """Creates the NL tables."""
        Schema.setup(self.connection, self.schema)

    def add_tables(self, assignment: Iterable[Tuple[int, Union[str, Path]]], creator: str,
                   accept_duplicates: bool = True) -> List[int]:
        """Registers every (TableId, path) pair and returns the ids that were registered."""
        registered = []
        for table_id, path in assignment:
            if self._register(table_id, Path(path), creator, accept_duplicates):
                registered.append(table_id)
        return registered

    def add_metadata(self, metadata_path: Union[str, Path], table_id: Optional[int] = None) -> int:
        """Registers context from a .txt file (named `<tableid>.txt` unless table_id is given) or a
        .csv with table_id and value columns."""
        metadata_path = Path(metadata_path)
        if metadata_path.is_dir():
            return sum(self.add_metadata(child) for child in sorted(metadata_path.rglob('*'))
                       if child.suffix in ('.txt', '.csv'))

        if metadata_path.suffix == '.txt':
            if table_id is None:
                if not metadata_path.stem.isdigit():
                    raise ValueError(f'{metadata_path} needs an explicit table_id '
                                     f'or a TableId filename like 12.txt')
                table_id = int(metadata_path.stem)
            self._insert_context(table_id, metadata_path.read_text())
            return 1

        if metadata_path.suffix != '.csv':
            raise ValueError(f'Unsupported metadata file type: {metadata_path}')

        entries = pd.read_csv(metadata_path)
        for _, row in entries.iterrows():
            self._insert_context(int(row['table_id']), str(row['value']))
        return len(entries)

    def _register(self, table_id: int, path: Path, creator: str, accept_duplicates: bool) -> bool:
        reader = self._reader(path)
        # Row order from a parallel CSV scan is not deterministic, so the aggregate
        # orders by the row text to keep the hash stable across runs.
        table_hash = self.connection.execute(
            f'SELECT md5(coalesce(string_agg(source::text, \'\' ORDER BY source::text), \'\')) '
            f'FROM {reader} AS source',
            [str(path)]
        ).fetchone()[0]

        if not accept_duplicates:
            duplicate = self.connection.execute(
                'SELECT id FROM table_status WHERE hash = ?', [table_hash]
            ).fetchone()
            if duplicate is not None and duplicate[0] != str(table_id):
                print(f'Skipping {path.name}: same content as TableId {duplicate[0]}')
                return False

        # Dropping by id, not by table_name: the physical copy is named after the id.
        self.connection.execute(f'DROP TABLE IF EXISTS "{table_id}"')
        self._purge(table_id)

        self.connection.execute(f'CREATE TABLE "{table_id}" AS SELECT * FROM {reader}', [str(path)])
        self.connection.execute(
            'INSERT INTO table_status (id, table_name, status, creator, hash) VALUES (?, ?, ?, ?, ?)',
            [str(table_id), path.name, Schema.REGISTERED, creator, table_hash],
        )
        return True

    def _purge(self, table_id: int) -> None:
        # DuckDB has no ON DELETE CASCADE, so the referencing rows go first.
        for table in ('index_table_mappings', 'table_summaries', 'table_contexts', 'table_status'):
            column = 'id' if table == 'table_status' else 'table_id'
            self.connection.execute(f'DELETE FROM {table} WHERE {column} = ?', [str(table_id)])

    def _insert_context(self, table_id: int, context: str) -> None:
        self.connection.execute(
            'INSERT INTO table_contexts (table_id, context) VALUES (?, ?)',
            [str(table_id), json.dumps({'payload': context.strip()})],
        )

    @staticmethod
    def _reader(path: Path) -> str:
        if path.suffix == '.csv':
            return 'read_csv(?, auto_detect=True, header=True, ignore_errors=True)'
        if path.suffix == '.parquet':
            return 'read_parquet(?)'
        raise ValueError(f'Unsupported table file type: {path}')
