import bm25s
import chromadb
import Stemmer
from bm25s.tokenization import convert_tokenized_to_string_list
from scipy.spatial.distance import cosine

from src.NLSeeker import Schema
from src.NLSeeker.Predicate import EMPTY_FILTER, TableFilter

# Typing imports
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


class Retriever:
    """Hybrid retrieval over the vector and full-text indexes, refined by an LLM relevance judgement."""

    def __init__(self, config: NLSeekerConfig, llm: LLMClient, embedder: EmbeddingClient) -> None:
        self.config = config
        self.llm = llm
        self.embedder = embedder
        self.stemmer = Stemmer.Stemmer('english')

        index_name = config.index_name
        if not config.vector_path.exists():
            raise FileNotFoundError(f'No vector index at {config.vector_path}')
        if not (config.fulltext_path / index_name).exists():
            raise FileNotFoundError(f'No full-text index at {config.fulltext_path / index_name}')

        self._collection = chromadb.PersistentClient(str(config.vector_path)).get_collection(index_name)
        self._bm25 = bm25s.BM25.load(str(config.fulltext_path / index_name), load_corpus=True)
        self._corpus_positions = {document['metadata']['table']: position
                                  for position, document in enumerate(self._bm25.corpus)}

    def retrieve(self, query: str, k: int, n: int, alpha: float,
                 table_filter: TableFilter = EMPTY_FILTER, rerank: bool = True) -> List[int]:
        """Returns the TableIds of the top-k tables for the query, in rank order."""
        query_tokens = bm25s.tokenize(query, stopwords='en', stemmer=self.stemmer, show_progress=False)
        query_embedding = self.embedder.encode([query])[0]

        texts = {}
        bm25_raw = {}
        vector_raw = {}
        if table_filter.allow is None:
            # A deny-list only needs the global pool widened by its size; an allow-list
            # would push the top-k outside the permitted set, so only that case skips it.
            fetched = min(k * n + len(table_filter.deny or ()), len(self._bm25.corpus))
            if fetched == 0:
                return []
            bm25_documents, bm25_scores = self._bm25.retrieve(query_tokens, k=fetched, show_progress=False)
            vector_results = self._collection.query(query_embeddings=[query_embedding], n_results=fetched)

            for document, score in zip(bm25_documents[0], bm25_scores[0]):
                document_id = document['metadata']['table']
                if not table_filter.permits(Schema.parse_table_id(document_id)):
                    continue
                texts[document_id] = document['text']
                bm25_raw[document_id] = float(score)

            for document_id, document, distance in zip(vector_results['ids'][0],
                                                       vector_results['documents'][0],
                                                       vector_results['distances'][0]):
                if not table_filter.permits(Schema.parse_table_id(document_id)):
                    continue
                texts.setdefault(document_id, document)
                vector_raw[document_id] = 1 - float(distance)

            document_ids = list(bm25_raw) + [i for i in vector_raw if i not in bm25_raw]
        else:
            document_ids = [i for i in self._corpus_positions
                            if table_filter.permits(Schema.parse_table_id(i))]
        pooled = min(k * n, len(document_ids))
        if pooled == 0:
            return []

        self._backfill_bm25(query_tokens, bm25_raw, document_ids)
        self._backfill_vector(query_embedding, vector_raw, texts, document_ids)

        bm25_norm = _min_max(bm25_raw, document_ids)
        vector_norm = _min_max(vector_raw, document_ids)
        fused = sorted(document_ids,
                       key=lambda i: (-(alpha * bm25_norm[i] + (1 - alpha) * vector_norm[i]), i))[:pooled]

        ranked = self._rerank(query, fused, texts) if rerank else fused
        tables = dict.fromkeys(Schema.parse_table_id(document_id) for document_id in ranked)
        return list(tables)[:k]

    def _backfill_bm25(self, query_tokens, scores: Dict[str, float], document_ids: Sequence[str]) -> None:
        missing = [i for i in document_ids if i not in scores]
        if not missing:
            return
        all_scores = self._bm25.get_scores(convert_tokenized_to_string_list(query_tokens)[0])
        for document_id in missing:
            scores[document_id] = float(all_scores[self._corpus_positions[document_id]])

    def _backfill_vector(self, query_embedding: List[float], scores: Dict[str, float],
                         texts: Dict[str, str], document_ids: Sequence[str]) -> None:
        missing = [i for i in document_ids if i not in scores]
        if not missing:
            return
        found = self._collection.get(ids=missing, include=['documents', 'embeddings'])
        for document_id, document, embedding in zip(found['ids'], found['documents'], found['embeddings']):
            texts.setdefault(document_id, document)
            scores[document_id] = 1 - float(cosine(query_embedding, embedding))

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


def _min_max(scores: Dict[str, float], document_ids: Sequence[str]) -> Dict[str, float]:
    values = [scores[document_id] for document_id in document_ids]
    low, high = min(values), max(values)
    if low == high:
        return {document_id: 1.0 for document_id in document_ids}
    return {document_id: (scores[document_id] - low) / (high - low) for document_id in document_ids}
