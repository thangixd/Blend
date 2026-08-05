import json

from src.NLSeeker import Schema

# Typing imports
from duckdb import DuckDBPyConnection
from src.NLSeeker.Clients import EmbeddingClient, LLMClient
from typing import Iterable, List, Optional

NARRATION_PROMPT = '''A table has the following columns:
/*
{columns}
*/
Describe briefly what the {column} column represents. If not possible, simply state "No description."'''

NARRATION_MAX_TOKENS = 400
ROW_SAMPLE_SIZE = 5


class Summarizer:
    """Turns each registered table into column narrations and row samples."""

    def __init__(self, connection: DuckDBPyConnection, schema: str,
                 llm: LLMClient, embedder: EmbeddingClient) -> None:
        self.connection = connection
        self.llm = llm
        self.embedder = embedder
        Schema.use_schema(connection, schema)

    def summarize(self, table_ids: Optional[Iterable[int]] = None) -> List[int]:
        """Summarizes the still-REGISTERED tables among the given ones, or all of them."""
        sql = 'SELECT id FROM table_status WHERE status = ?'
        parameters = [Schema.REGISTERED]
        if table_ids is not None:
            table_ids = [str(table_id) for table_id in table_ids]
            if not table_ids:
                return []
            sql += f" AND id IN ({', '.join('?' * len(table_ids))})"
            parameters += table_ids
        rows = self.connection.execute(sql + ' ORDER BY CAST(id AS INTEGER)', parameters).fetchall()
        table_ids = [int(row[0]) for row in rows]

        if not table_ids:
            return []

        columns = {table_id: self._columns(table_id) for table_id in table_ids}
        conversations = [[{'role': 'user', 'content': NARRATION_PROMPT.format(
                              columns=' | '.join(columns[table_id]), column=column)}]
                         for table_id in table_ids for column in columns[table_id]]

        answers = iter(self.llm.complete(conversations, max_new_tokens=NARRATION_MAX_TOKENS,
                                         desc='Narrating columns'))

        for table_id in table_ids:
            narrations = [f'{column}: {next(answers)}'.strip() for column in columns[table_id]]
            self._insert(table_id, Schema.COLUMN_NARRATION, self._block(narrations))
            self._insert(table_id, Schema.ROW_SAMPLE, self._block(self._row_samples(table_id)))
            self.connection.execute('UPDATE table_status SET status = ? WHERE id = ?',
                                    [Schema.SUMMARIZED, str(table_id)])

        return table_ids

    def _columns(self, table_id: int) -> List[str]:
        return [column[0] for column in
                self.connection.execute(f'SELECT * FROM "{table_id}" LIMIT 0').description]

    def _row_samples(self, table_id: int) -> List[str]:
        df = self.connection.execute(f'SELECT * FROM "{table_id}"').df()
        if df.empty:
            return []
        sample = df.sample(n=min(len(df), ROW_SAMPLE_SIZE), random_state=0)
        return [' | '.join(f'{column}: {value}' for column, value in row.items()).strip()
                for _, row in sample.iterrows()]

    def _block(self, items: List[str]) -> List[str]:
        return Schema.block(items, Schema.SUMMARY_JOINER, self.embedder.content_budget,
                            self.embedder.count_tokens)

    def _insert(self, table_id: int, summary_type: str, blocks: List[str]) -> None:
        for text in blocks:
            self.connection.execute(
                'INSERT INTO table_summaries (table_id, summary, summary_type) VALUES (?, ?, ?)',
                [str(table_id), json.dumps({'payload': text}), summary_type],
            )
