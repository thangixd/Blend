"""DuckDB schema, status literals and the document-id grammar shared by the NL index."""

# Typing imports
from duckdb import DuckDBPyConnection
from typing import Callable, List

REGISTERED = 'TableStatus.REGISTERED'
SUMMARIZED = 'TableStatus.SUMMARIZED'

COLUMN_NARRATION = 'SummaryType.COLUMN_NARRATION'
ROW_SAMPLE = 'SummaryType.ROW_SAMPLE'

SEPARATOR = '_SEP_'
SUMMARY_JOINER = ' || '
CONTEXT_JOINER = ' | '

DDL = [
    """CREATE TABLE IF NOT EXISTS table_status (
        id VARCHAR PRIMARY KEY,
        table_name VARCHAR NOT NULL,
        status VARCHAR NOT NULL,
        time_created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        creator VARCHAR NOT NULL,
        hash VARCHAR NOT NULL
        )""",
    "CREATE SEQUENCE IF NOT EXISTS id_seq START 1",
    """CREATE TABLE IF NOT EXISTS table_contexts (
        id INTEGER DEFAULT nextval('id_seq') PRIMARY KEY,
        table_id VARCHAR NOT NULL REFERENCES table_status(id),
        context JSON NOT NULL
        )""",
    """CREATE TABLE IF NOT EXISTS table_summaries (
        id INTEGER DEFAULT nextval('id_seq') PRIMARY KEY,
        table_id VARCHAR NOT NULL REFERENCES table_status(id),
        summary JSON NOT NULL,
        summary_type VARCHAR NOT NULL
        )""",
    """CREATE TABLE IF NOT EXISTS indexes (
        id INTEGER DEFAULT nextval('id_seq') PRIMARY KEY,
        name VARCHAR NOT NULL,
        location VARCHAR NOT NULL
        )""",
    """CREATE TABLE IF NOT EXISTS index_table_mappings (
        index_id INTEGER NOT NULL REFERENCES indexes(id),
        table_id VARCHAR NOT NULL REFERENCES table_status(id),
        PRIMARY KEY (index_id, table_id)
        )""",
]


def use_schema(connection: DuckDBPyConnection, schema: str) -> None:
    """Creates the NL schema and makes it the default for this connection."""
    connection.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
    connection.execute(f"SET search_path='{schema}'")


def setup(connection: DuckDBPyConnection, schema: str) -> None:
    """Creates the NL schema and its tables."""
    use_schema(connection, schema)
    for statement in DDL:
        connection.execute(statement)


def content_schema_id(table_id: int, index: int) -> str:
    return f'{table_id}{SEPARATOR}contents{SEPARATOR}schema-{index}'


def content_row_id(table_id: int, index: int) -> str:
    return f'{table_id}{SEPARATOR}contents{SEPARATOR}row-{index}'


def context_id(table_id: int, index: int) -> str:
    return f'{table_id}{SEPARATOR}contexts-{index}'


def parse_table_id(document_id: str) -> int:
    return int(document_id.split(SEPARATOR)[0])


def is_content(document_id: str) -> bool:
    return document_id.split(SEPARATOR)[1].startswith('contents')


def block(items: List[str], joiner: str, budget: int, count_tokens: Callable[[str], int]) -> List[str]:
    """Greedily concatenates items into blocks that stay under the embedding budget."""
    blocks = []
    current = ''
    for item in items:
        candidate = item if not current else current + joiner + item
        if current and count_tokens(candidate) >= budget:
            blocks.append(current)
            current = item
        else:
            current = candidate
    if current:
        blocks.append(current)
    return blocks
