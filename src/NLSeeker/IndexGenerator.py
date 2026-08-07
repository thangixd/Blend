import json

import pandas as pd

from src.NLSeeker import Schema, Tokenizer

# Typing imports
from duckdb import DuckDBPyConnection
from src.NLSeeker.Clients import EmbeddingClient
from typing import Iterable, List, Optional, Tuple


EMBEDDING_CHUNK_SIZE = 2000


class IndexGenerator:
    """Builds the document and token tables from the stored summaries and contexts."""

    def __init__(self, connection: DuckDBPyConnection, schema: str, embedder: EmbeddingClient,
                 documents_table: str, tokens_table: str) -> None:
        self.connection = connection
        self.embedder = embedder
        self.documents_table = documents_table
        self.tokens_table = tokens_table
        Schema.use_schema(connection, schema)

    def generate_index(self, index_name: str, table_ids: Optional[Iterable[int]] = None,
                       replace: bool = False) -> int:
        """Indexes the summaries and contexts of the given tables and returns the document count."""
        if replace:
            self._drop_index(index_name)

        if table_ids is None:
            rows = self.connection.execute(
                'SELECT id FROM table_status ORDER BY CAST(id AS INTEGER)').fetchall()
            table_ids = [int(row[0]) for row in rows]
        else:
            table_ids = list(table_ids)

        documents = []
        missing = []
        for table_id in table_ids:
            table_documents = self._documents(table_id)
            if not table_documents:
                missing.append(table_id)
            documents.extend(table_documents)
        if missing:
            raise ValueError(f'Tables {missing} have no summaries or contexts; run the summarizer first.')
        if not documents:
            raise ValueError('No summaries or contexts to index; run the summarizer first.')

        self._write_documents(documents)
        self._write_tokens(documents)
        self._record(index_name, self.documents_table, table_ids)
        self._record(index_name, self.tokens_table, table_ids)
        return len(documents)

    def _drop_index(self, index_name: str) -> None:
        """Removes a previous build of this index so the script can be re-run."""
        self.connection.execute(f'DROP TABLE IF EXISTS main."{self.documents_table}"')
        self.connection.execute(f'DROP TABLE IF EXISTS main."{self.tokens_table}"')

        index_ids = [row[0] for row in self.connection.execute(
            'SELECT id FROM indexes WHERE name = ?', [index_name]).fetchall()]
        for index_id in index_ids:
            self.connection.execute('DELETE FROM index_table_mappings WHERE index_id = ?', [index_id])
            self.connection.execute('DELETE FROM indexes WHERE id = ?', [index_id])

    def _documents(self, table_id: int) -> List[Tuple[str, str]]:
        documents = []
        for index, text in enumerate(self._payloads(
                'SELECT summary FROM table_summaries WHERE table_id = ? AND summary_type = ? ORDER BY id',
                [str(table_id), Schema.COLUMN_NARRATION])):
            documents.append((Schema.content_schema_id(table_id, index), text))

        for index, text in enumerate(self._payloads(
                'SELECT summary FROM table_summaries WHERE table_id = ? AND summary_type = ? ORDER BY id',
                [str(table_id), Schema.ROW_SAMPLE])):
            documents.append((Schema.content_row_id(table_id, index), text))

        contexts = self._payloads('SELECT context FROM table_contexts WHERE table_id = ? ORDER BY id',
                                  [str(table_id)])
        merged = Schema.block(contexts, Schema.CONTEXT_JOINER, self.embedder.content_budget,
                              self.embedder.count_tokens)
        for index, text in enumerate(merged):
            documents.append((Schema.context_id(table_id, index), text))

        return documents

    def _payloads(self, sql: str, parameters: List[str]) -> List[str]:
        return [json.loads(row[0])['payload']
                for row in self.connection.execute(sql, parameters).fetchall()]

    def _write_documents(self, documents: List[Tuple[str, str]]) -> None:
        dimension = None
        for start in range(0, len(documents), EMBEDDING_CHUNK_SIZE):
            chunk = documents[start:start + EMBEDDING_CHUNK_SIZE]
            texts = [text for _, text in chunk]
            embeddings = self.embedder.encode(texts)

            if dimension is None:
                dimension = len(embeddings[0])
                self.connection.execute(f"""CREATE TABLE main."{self.documents_table}" (
                    docid VARCHAR PRIMARY KEY,
                    tableid INTEGER NOT NULL,
                    text VARCHAR NOT NULL,
                    length INTEGER NOT NULL,
                    embedding FLOAT[{dimension}] NOT NULL
                    )""")

            frame = pd.DataFrame({
                'docid': [document_id for document_id, _ in chunk],
                'tableid': [Schema.parse_table_id(document_id) for document_id, _ in chunk],
                'text': texts,
                'length': [len(Tokenizer.tokenize(text)) for text in texts],
                'embedding': embeddings,
            })
            self._insert(f'main."{self.documents_table}"', frame,
                         f'docid, tableid, text, length, CAST(embedding AS FLOAT[{dimension}])')

        self.connection.execute(
            f'CREATE INDEX "{self.documents_table}_to_tableid" '
            f'ON main."{self.documents_table}" (tableid)')

    def _write_tokens(self, documents: List[Tuple[str, str]]) -> None:
        rows = [(document_id, token, frequency)
                for document_id, text in documents
                for token, frequency in Tokenizer.term_frequencies(text).items()]
        frame = pd.DataFrame(rows, columns=['docid', 'token', 'tf'])

        self.connection.execute(f"""CREATE TABLE main."{self.tokens_table}" (
            docid VARCHAR NOT NULL,
            token VARCHAR NOT NULL,
            tf INTEGER NOT NULL
            )""")
        self._insert(f'main."{self.tokens_table}"', frame, 'docid, token, tf')
        self.connection.execute(
            f'CREATE INDEX "{self.tokens_table}_to_token" ON main."{self.tokens_table}" (token)')

    def _insert(self, table: str, frame: pd.DataFrame, projection: str) -> None:
        self.connection.register('index_df', frame)
        try:
            self.connection.execute(f'INSERT INTO {table} SELECT {projection} FROM index_df')
        finally:
            self.connection.unregister('index_df')

    def _record(self, index_name: str, location: str, table_ids: List[int]) -> None:
        index_id = self.connection.execute(
            'INSERT INTO indexes (name, location) VALUES (?, ?) RETURNING id',
            [index_name, location],
        ).fetchone()[0]
        self.connection.executemany(
            'INSERT INTO index_table_mappings (index_id, table_id) VALUES (?, ?)',
            [(index_id, str(table_id)) for table_id in table_ids],
        )
