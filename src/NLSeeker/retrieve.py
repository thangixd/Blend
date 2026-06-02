import logging
from dataclasses import dataclass

import bm25s
from bm25s.tokenization import convert_tokenized_to_string_list
import chromadb_deterministic as chromadb
from chromadb_deterministic.api.shared_system_client import SharedSystemClient
from chromadb_deterministic.config import Settings
import numpy as np
import Stemmer

from src.NLSeeker.index_build import parse_doc_id
from src.NLSeeker.llm import EmbedBackend, LLMBackend

# Typing imports
from pathlib import Path
from typing import Sequence

LOG = logging.getLogger(__name__)


# Two prompts: content blocks (column narrations / row samples) vs context
# blocks (external metadata). Prompt is picked per candidate by doc-id kind.
_CONTENT_RERANK_PROMPT = (
    "Given a table with the following columns:\n"
    "*/\n"
    "{desc}\n"
    "*/\n"
    "and this question:\n"
    "/*\n"
    "{query}\n"
    "*/\n"
    "Is the table relevant to answer the question? Begin your answer with yes/no."
)

_CONTEXT_RERANK_PROMPT = (
    "Given this context describing a table:\n"
    "*/\n"
    "{desc}\n"
    "*/\n"
    "and this question:\n"
    "/*\n"
    "{query}\n"
    "*/\n"
    "Is the table relevant to answer the question? Begin your answer with yes/no."
)


def _rerank_prompt_for(doc_id: str, desc: str, query: str) -> str:
    try:
        _, kind, _ = parse_doc_id(doc_id)
    except ValueError:
        kind = "schema"
    template = _CONTEXT_RERANK_PROMPT if kind == "contexts" else _CONTENT_RERANK_PROMPT
    return template.format(desc=desc, query=query)


@dataclass
class HybridResult:
    doc_id: str
    text: str
    bm25_score: float
    vector_score: float
    fused_score: float


@dataclass
class RetrievalResult:
    table_id: int
    fused_score: float
    document: str
    relevant: bool  # LLM-judge verdict


def _minmax(scores: dict) -> dict:
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi <= lo:
        # All-equal (e.g. BM25 zero scores on a one-term query): map to 1.0
        # so docs still contribute to the fusion.
        return {k: 1.0 for k in scores}
    span = hi - lo
    return {k: (v - lo) / span for k, v in scores.items()}


class _BM25Index:
    """BM25s index loaded from disk plus a per-process Stemmer."""

    def __init__(self, fulltext_path: Path) -> None:
        self._retriever = bm25s.BM25.load(str(fulltext_path), load_corpus=True)
        self._stemmer = Stemmer.Stemmer("english")
        self._corpus_size = len(self._retriever.corpus)
        self._doc_id_to_corpus_idx = {
            entry["doc_id"]: idx
            for idx, entry in enumerate(self._retriever.corpus)
        }

    def _tokenize(self, query: str):
        return bm25s.tokenize(
            [query],
            stopwords="en",
            stemmer=self._stemmer,
            show_progress=False,
        )

    def retrieve(self, query: str, k: int) -> tuple:
        """Return ``({doc_id: (bm25_score, doc_text)}, query_tokens)``."""
        if k <= 0 or self._corpus_size == 0:
            return {}, None
        effective_k = min(k, self._corpus_size)
        q_tokens = self._tokenize(query)
        docs, scores = self._retriever.retrieve(q_tokens, k=effective_k, show_progress=False)
        out = {}
        for doc, score in zip(docs[0], scores[0]):
            out[doc["doc_id"]] = (float(score), doc["text"])
        return out, q_tokens

    def score_for_ids(self, q_tokens, doc_ids):
        """Score ``doc_ids`` not present in the top-k list."""
        if not doc_ids:
            return {}
        query_terms = convert_tokenized_to_string_list(q_tokens)[0]
        all_scores = self._retriever.get_scores(query_terms)
        out = {}
        for did in doc_ids:
            idx = self._doc_id_to_corpus_idx.get(did)
            if idx is None:  # pragma: no cover
                continue
            corpus_entry = self._retriever.corpus[idx]
            out[did] = (float(all_scores[idx]), corpus_entry["text"])
        return out


class _VectorIndex:
    """ChromaDB collection wrapper. Borrows the embedder from the engine."""

    def __init__(self, vector_path: Path, collection_name: str, embedder: EmbedBackend) -> None:
        # See NLIndexBuilder.start for why the cache is cleared here.
        SharedSystemClient.clear_system_cache()
        client = chromadb.PersistentClient(
            path=str(vector_path),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = client.get_collection(collection_name)
        self._embedder = embedder
        self._collection_size = self._collection.count()

    def retrieve(self, query: str, k: int) -> tuple:
        """Return ``({doc_id: (cos_similarity, doc_text)}, query_embedding)``."""
        if k <= 0 or self._collection_size == 0:
            return {}, None
        effective_k = min(k, self._collection_size)
        vec = self._embedder.encode([query])
        result = self._collection.query(
            query_embeddings=vec.tolist(),
            n_results=effective_k,
            include=["documents", "distances"],
        )
        ids = result["ids"][0]
        distances = result["distances"][0]
        documents = result["documents"][0]
        out = {}
        for did, dist, doc in zip(ids, distances, documents):
            # Cosine distance → similarity, clamped against fp noise.
            sim = max(0.0, 1.0 - float(dist))
            out[did] = (sim, doc or "")
        return out, vec[0]

    def score_for_ids(self, query_vec, doc_ids):
        """Score ``doc_ids`` not present in the top-k list."""
        if not doc_ids:
            return {}
        ids = list(doc_ids)
        result = self._collection.get(
            ids=ids,
            include=["documents", "embeddings"],
        )
        # Chroma may reorder; rebuild the parallel arrays by id.
        got_ids = result["ids"]
        embeddings = result["embeddings"]
        documents = result["documents"]
        q = np.asarray(query_vec, dtype=np.float64)
        qn = np.linalg.norm(q)
        out = {}
        for did, emb, doc in zip(got_ids, embeddings, documents):
            v = np.asarray(emb, dtype=np.float64)
            denom = qn * np.linalg.norm(v)
            sim = float(q @ v / denom) if denom > 0 else 0.0
            sim = max(0.0, sim)
            out[did] = (sim, doc or "")
        return out


def hybrid_retrieve(
    query: str,
    bm25: _BM25Index,
    vector: _VectorIndex,
    k: int,
    n: int,
    alpha: float,
) -> list:
    """Fuse BM25 and vector signals into a ranked list of ``HybridResult``.

    Each retriever fetches ``k * n`` candidates; ids found by only one are
    backfilled against the other; scores are min-max normalised per modality
    and combined as ``alpha * bm25 + (1 - alpha) * vector``. The full pool
    is returned (rerank truncates to ``k`` later).
    """
    fetch = max(1, k * n)
    bm25_raw, q_tokens = bm25.retrieve(query, fetch)
    vec_raw, query_vec = vector.retrieve(query, fetch)

    bm25_missing = set(vec_raw) - set(bm25_raw)
    vec_missing = set(bm25_raw) - set(vec_raw)
    if bm25_missing and q_tokens is not None:
        bm25_raw = {**bm25_raw, **bm25.score_for_ids(q_tokens, bm25_missing)}
    if vec_missing and query_vec is not None:
        vec_raw = {**vec_raw, **vector.score_for_ids(query_vec, vec_missing)}

    bm25_norm = _minmax({did: s for did, (s, _) in bm25_raw.items()})
    vec_norm = _minmax({did: s for did, (s, _) in vec_raw.items()})

    all_ids = set(bm25_raw) | set(vec_raw)
    fused = []
    for did in all_ids:
        bm = bm25_norm.get(did, 0.0)
        ve = vec_norm.get(did, 0.0)
        text = bm25_raw.get(did, (0.0, ""))[1] or vec_raw.get(did, (0.0, ""))[1]
        fused.append(
            HybridResult(
                doc_id=did,
                text=text,
                bm25_score=bm,
                vector_score=ve,
                fused_score=alpha * bm + (1.0 - alpha) * ve,
            )
        )
    # Sort by fused score desc, then doc_id asc to stay deterministic across
    # processes (PYTHONHASHSEED randomises set iteration).
    fused.sort(key=lambda r: (-r.fused_score, r.doc_id))
    return fused[:fetch]


def _is_yes(answer: str) -> bool:
    return answer.strip().lower().startswith("yes")


def llm_rerank(
    query: str,
    candidates: Sequence[HybridResult],
    llm: LLMBackend,
) -> list:
    """Run the LLM yes/no judge once per candidate; relevant ones first."""
    if not candidates:
        return []
    prompts = [_rerank_prompt_for(c.doc_id, c.text, query) for c in candidates]
    answers = llm.generate(prompts, max_new_tokens=2)
    judged = [(cand, _is_yes(ans)) for cand, ans in zip(candidates, answers)]
    relevant = [pair for pair in judged if pair[1]]
    others = [pair for pair in judged if not pair[1]]
    return relevant + others


def search(
    query: str,
    bm25: _BM25Index,
    vector: _VectorIndex,
    llm: LLMBackend,
    k: int,
    n: int,
    alpha: float,
) -> list:
    """Hybrid retrieve, rerank with the LLM, truncate to ``k``, dedupe by table.

    Truncation happens before deduping, so the result can hold fewer than
    ``k`` entries when one table dominates several top positions.
    """
    fused = hybrid_retrieve(query, bm25, vector, k=k, n=n, alpha=alpha)
    if not fused:
        return []

    reranked = llm_rerank(query, fused, llm)

    top_k_positions = reranked[:k]

    seen = {}
    for hit, relevant in top_k_positions:
        try:
            table_id, _, _ = parse_doc_id(hit.doc_id)
        except ValueError:
            LOG.warning("Skipping malformed doc_id %r", hit.doc_id)
            continue
        if table_id in seen:
            continue
        seen[table_id] = RetrievalResult(
            table_id=table_id,
            fused_score=hit.fused_score,
            document=hit.text,
            relevant=relevant,
        )
    return list(seen.values())


def open_indexes(
    vector_path: Path,
    fulltext_path: Path,
    collection_name: str,
    embedder: EmbedBackend,
) -> tuple:
    """Load both indexes from disk."""
    return _BM25Index(fulltext_path), _VectorIndex(vector_path, collection_name, embedder)


__all__ = [
    "HybridResult",
    "RetrievalResult",
    "hybrid_retrieve",
    "llm_rerank",
    "search",
    "open_indexes",
]
