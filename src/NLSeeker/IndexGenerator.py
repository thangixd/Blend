import json
import shutil

import bm25s
import chromadb
import Stemmer

from src.NLSeeker import Schema

# Typing imports
from duckdb import DuckDBPyConnection
from pathlib import Path
from src.NLSeeker.Clients import EmbeddingClient
from typing import Iterable, List, Optional, Tuple

VECTOR_CHUNK_SIZE = 30000

# chromadb 1.x dropped hnsw:random_seed, so graph construction is no longer seeded and
# near-ties may reorder between rebuilds.
HNSW_CONFIGURATION = {'hnsw': {'space': 'cosine', 'max_neighbors': 48}}


class IndexGenerator:
    """Builds the vector and full-text indexes from the stored summaries and contexts."""

    def __init__(self, connection: DuckDBPyConnection, schema: str, embedder: EmbeddingClient,
                 vector_path: Path, fulltext_path: Path) -> None:
        self.connection = connection
        self.embedder = embedder
        self.vector_path = vector_path
        self.fulltext_path = fulltext_path
        self.stemmer = Stemmer.Stemmer('english')
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

        documents = [document for table_id in table_ids for document in self._documents(table_id)]
        if not documents:
            raise ValueError('No summaries or contexts to index; run the summarizer first.')

        self._build_vector_index(index_name, documents)
        self._build_fulltext_index(index_name, documents)
        self._record(index_name, str(self.vector_path), table_ids)
        self._record(index_name, str(self.fulltext_path), table_ids)
        return len(documents)

    def _drop_index(self, index_name: str) -> None:
        """Removes a previous build of this index so the script can be re-run."""
        if self.vector_path.exists():
            client = chromadb.PersistentClient(str(self.vector_path))
            if index_name in [collection.name for collection in client.list_collections()]:
                client.delete_collection(index_name)
        shutil.rmtree(self.fulltext_path / index_name, ignore_errors=True)

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

    def _build_vector_index(self, index_name: str, documents: List[Tuple[str, str]]) -> None:
        self.vector_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(str(self.vector_path))
        collection = client.create_collection(name=index_name, configuration=HNSW_CONFIGURATION)

        try:
            for start in range(0, len(documents), VECTOR_CHUNK_SIZE):
                chunk = documents[start:start + VECTOR_CHUNK_SIZE]
                texts = [text for _, text in chunk]
                collection.add(ids=[document_id for document_id, _ in chunk],
                               documents=texts,
                               embeddings=self.embedder.encode(texts))
        except Exception:
            client.delete_collection(index_name)
            raise

    def _build_fulltext_index(self, index_name: str, documents: List[Tuple[str, str]]) -> None:
        corpus = [{'text': text, 'metadata': {'table': document_id}} for document_id, text in documents]
        tokens = bm25s.tokenize([text for _, text in documents], stopwords='en',
                                stemmer=self.stemmer, show_progress=False)

        retriever = bm25s.BM25(corpus=corpus)
        retriever.index(tokens, show_progress=False)
        retriever.save(str(self.fulltext_path / index_name), corpus=corpus)

    def _record(self, index_name: str, location: str, table_ids: List[int]) -> None:
        index_id = self.connection.execute(
            'INSERT INTO indexes (name, location) VALUES (?, ?) RETURNING id',
            [index_name, location],
        ).fetchone()[0]
        self.connection.executemany(
            'INSERT INTO index_table_mappings (index_id, table_id) VALUES (?, ?)',
            [(index_id, str(table_id)) for table_id in table_ids],
        )
