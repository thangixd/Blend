from collections import Counter

from src.NLSeeker import Schema, Tokenizer

# Typing imports
from src.DBHandler import DBHandler
from src.NLSeeker.Clients import EmbeddingClient, LLMClient
from src.NLSeeker.Config import NLSeekerConfig
from typing import Dict, List, Sequence


RELEVANCE_PROMPTS = {
    'content': """Given a table with the following columns:
*/
{desc}
*/
and this question:
/*
{query}
*/
Is the table relevant to answer the question? Begin your answer with yes/no.""",
    'context': """Given this context describing a table:
*/
{desc}
*/
and this question:
/*
{query}
*/
Is the table relevant to answer the question? Begin your answer with yes/no.""",
}

RERANK_MAX_TOKENS = 2

# bm25s' Lucene-variant defaults. The SQL BM25 below must match them and bm25s' scoring exactly.
K1 = 1.5
B = 0.75


def bm25_sql(documents_table: str, tokens_table: str, tokens: Sequence[str],
             additionals: str = '') -> str:
    """Scores every document containing at least one query token, as bm25s' Lucene variant does."""
    # Tokens come out of \\w\\w+ matching and stemming, so they cannot carry a quote.
    # q has one row per distinct token, so df below is a true document frequency even
    # when the query repeats a token; occurrences scales that token's contribution instead.
    occurrences = Counter(tokens)
    values = ', '.join(f"('{token}', {count})" for token, count in occurrences.items()) if occurrences else None
    query_tokens = f'VALUES {values}' if values else 'SELECT NULL::VARCHAR, 0 WHERE FALSE'

    return f"""
        WITH q(token, occurrences) AS ({query_tokens}),
        stats AS (SELECT count(*) AS n, avg(length) AS avgdl FROM "{documents_table}"),
        df AS (SELECT t.token, count(*) AS df
               FROM "{tokens_table}" t JOIN q USING (token) GROUP BY 1)
        SELECT t.docid,
               sum(q.occurrences * ln(1 + (s.n - df.df + 0.5) / (df.df + 0.5))
                   * t.tf / (t.tf + {K1} * ((1 - {B}) + {B} * d.length / s.avgdl))) AS score
        FROM "{tokens_table}" t
        JOIN q ON q.token = t.token
        JOIN df ON df.token = t.token
        JOIN "{documents_table}" d ON d.docid = t.docid
        CROSS JOIN stats s
        WHERE 1=1 {additionals}
        GROUP BY t.docid
    """


class Retriever:
    """Hybrid retrieval over the document and token tables, refined by an LLM relevance judgement."""

    def __init__(self, config: NLSeekerConfig, llm: LLMClient, embedder: EmbeddingClient) -> None:
        self.config = config
        self.llm = llm
        self.embedder = embedder

    def retrieve(self, db: DBHandler, query: str, k: int, n: int, alpha: float,
                 additionals: str = '', rerank: bool = True) -> List[int]:
        """Returns the TableIds of the top-k tables for the query, in rank order."""
        embedding = self.embedder.encode([query])[0]
        rows = db.execute_and_fetchall(
            self._sql(Tokenizer.tokenize(query), len(embedding), alpha, k * n, additionals),
            [embedding])

        texts = {document_id: text for document_id, text in rows}
        ranked = self._rerank(query, list(texts), texts) if rerank else list(texts)
        tables = dict.fromkeys(Schema.parse_table_id(document_id) for document_id in ranked)
        return list(tables)[:k]

    def _sql(self, tokens: Sequence[str], dimension: int, alpha: float,
             pooled: int, additionals: str) -> str:
        documents = f'"{self.config.documents_table}"'
        bm25 = bm25_sql(self.config.documents_table, self.config.tokens_table, tokens, additionals)

        # The pool is the union of both top-lists; the joins below guarantee every pooled
        # document has a score on both sides, even one only the other side ranked highly.
        return f"""
        WITH bm25 AS ({bm25}),
        vec AS (
            SELECT docid, array_cosine_similarity(embedding, ?::FLOAT[{dimension}]) AS score
            FROM {documents}
            WHERE 1=1 {additionals}),
        pool AS (
            SELECT docid FROM (SELECT docid, score FROM bm25 ORDER BY score DESC, docid LIMIT {pooled})
            UNION
            SELECT docid FROM (SELECT docid, score FROM vec ORDER BY score DESC, docid LIMIT {pooled})),
        scored AS (
            SELECT p.docid, coalesce(bm25.score, 0) AS b, vec.score AS v
            FROM pool p
            LEFT JOIN bm25 ON bm25.docid = p.docid
            JOIN vec ON vec.docid = p.docid),
        normalized AS (
            SELECT docid,
                   CASE WHEN max(b) OVER () = min(b) OVER () THEN 1.0
                        ELSE (b - min(b) OVER ()) / (max(b) OVER () - min(b) OVER ()) END AS b,
                   CASE WHEN max(v) OVER () = min(v) OVER () THEN 1.0
                        ELSE (v - min(v) OVER ()) / (max(v) OVER () - min(v) OVER ()) END AS v
            FROM scored)
        SELECT normalized.docid, d.text
        FROM normalized JOIN {documents} d ON d.docid = normalized.docid
        ORDER BY {alpha} * normalized.b + (1 - {alpha}) * normalized.v DESC, normalized.docid
        LIMIT {pooled}
        """

    def _rerank(self, query: str, document_ids: List[str], texts: Dict[str, str]) -> List[str]:
        """Stably partitions the documents into relevant-first, without dropping any."""
        if not document_ids:
            return []

        conversations = [[{'role': 'user', 'content': RELEVANCE_PROMPTS[
            'content' if Schema.is_content(document_id) else 'context'
        ].format(desc=texts[document_id], query=query)}] for document_id in document_ids]

        answers = self.llm.complete(conversations, max_new_tokens=RERANK_MAX_TOKENS, desc='Reranking')
        relevant = {document_id: answer.lower().startswith('yes')
                    for document_id, answer in zip(document_ids, answers)}

        return ([i for i in document_ids if relevant[i]] +
                [i for i in document_ids if not relevant[i]])
