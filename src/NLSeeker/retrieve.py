import logging
import time
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
from src.NLSeeker.predicate import EMPTY_FILTER, TableFilter

# Typing imports
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

LOG = logging.getLogger(__name__)


# Two prompts: content blocks (column narrations / row samples) vs context
# blocks (external metadata). Picked per candidate by doc-id kind.
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
    """Pick the rerank template canonically by doc-kind."""
    if "_SEP_contents_SEP_" in doc_id:
        template = _CONTENT_RERANK_PROMPT
    elif "_SEP_contexts-" in doc_id:
        template = _CONTEXT_RERANK_PROMPT
    else:
        raise ValueError(f"Unrecognised doc_id kind: {doc_id!r}")
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
        # All-equal (e.g. BM25 zeroes on a one-term query): map to 1.0 so
        # docs still contribute to the fusion.
        return {k: 1.0 for k in scores}
    span = hi - lo
    return {k: (v - lo) / span for k, v in scores.items()}


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    n = scores.shape[0]
    if k <= 0 or n == 0:
        return np.empty(0, dtype=np.int64)
    k = min(k, n)
    if k == n:
        return np.argsort(-scores, kind="stable")
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.argsort(-scores[part], kind="stable")]


class _BM25Index:
    """BM25s index loaded from disk plus a per-process Stemmer."""

    def __init__(self, fulltext_path: Path) -> None:
        self._retriever = bm25s.BM25.load(str(fulltext_path), load_corpus=True)
        self._stemmer = Stemmer.Stemmer("english")
        self._corpus_size = len(self._retriever.corpus)

        self._doc_id_to_corpus_idx: dict = {}
        self._doc_id_to_table_id: dict = {}
        self._table_id_to_doc_ids: dict = {}
        for idx, entry in enumerate(self._retriever.corpus):
            did = entry["doc_id"]
            tid = int(entry["table_id"])
            self._doc_id_to_corpus_idx[did] = idx
            self._doc_id_to_table_id[did] = tid
            self._table_id_to_doc_ids.setdefault(tid, []).append(did)

    def _tokenize(self, query: str):
        return bm25s.tokenize(
            [query],
            stopwords="en",
            stemmer=self._stemmer,
            show_progress=False,
        )

    def doc_ids_for_tables(self, table_ids: Iterable[int]) -> set:
        out: set = set()
        for tid in table_ids:
            out.update(self._table_id_to_doc_ids.get(int(tid), ()))
        return out

    def all_doc_ids(self) -> set:
        return set(self._doc_id_to_corpus_idx)

    def retrieve(
        self,
        query: str,
        k: int,
        allowed_doc_ids: Optional[set] = None,
    ) -> tuple:
        """With ``allowed_doc_ids`` set, top-k is taken inside the allow-list
        via ``get_scores`` over the full corpus. Without, defer to native top-k.
        """
        if k <= 0 or self._corpus_size == 0:
            return {}, None
        q_tokens = self._tokenize(query)

        if allowed_doc_ids is None:
            effective_k = min(k, self._corpus_size)
            docs, scores = self._retriever.retrieve(
                q_tokens, k=effective_k, show_progress=False
            )
            out = {}
            for doc, score in zip(docs[0], scores[0]):
                out[doc["doc_id"]] = (float(score), doc["text"])
            return out, q_tokens

        if not allowed_doc_ids:
            return {}, q_tokens
        query_terms = convert_tokenized_to_string_list(q_tokens)[0]
        all_scores = self._retriever.get_scores(query_terms)
        allowed_idxs = []
        allowed_dids = []
        for did in allowed_doc_ids:
            idx = self._doc_id_to_corpus_idx.get(did)
            if idx is None:
                continue
            allowed_idxs.append(idx)
            allowed_dids.append(did)
        if not allowed_idxs:
            return {}, q_tokens
        sub_scores = np.asarray([all_scores[i] for i in allowed_idxs], dtype=np.float64)
        order = _topk_indices(sub_scores, k)
        out = {}
        for j in order:
            did = allowed_dids[j]
            corpus_entry = self._retriever.corpus[allowed_idxs[j]]
            out[did] = (float(sub_scores[j]), corpus_entry["text"])
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
    """ChromaDB-backed embeddings, queried via exact numpy cosine.

    Chroma is the on-disk persistence layer only. At open time we read
    the full ``(N, D)`` matrix into memory once, L2-normalize it, and
    answer every retrieval with a single dense matvec. HNSW is not
    consulted at query time.
    """

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

        self._all_doc_ids: list = []
        self._all_documents: list = []
        self._all_embeddings: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self._doc_id_to_row: dict = {}
        self._last_vector_ms: float = 0.0

        if self._collection_size == 0:
            return
        bundle = self._collection.get(include=["documents", "embeddings"])
        ids = bundle["ids"]
        documents = bundle["documents"] or [""] * len(ids)
        embeddings = bundle["embeddings"]
        mat = np.asarray(embeddings, dtype=np.float32)
        if mat.ndim != 2 or mat.shape[0] != len(ids):
            raise RuntimeError(
                f"NLSeeker vector collection {collection_name!r}: expected a "
                f"2-D embedding matrix with {len(ids)} rows, got shape {mat.shape}."
            )
        # Normalize once so ``mat @ q_norm`` is the cosine similarity.
        # Zero-vectors stay zero; clamp denominator against fp noise.
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        self._all_doc_ids = list(ids)
        self._all_documents = [d or "" for d in documents]
        self._all_embeddings = mat / norms
        self._doc_id_to_row = {did: i for i, did in enumerate(self._all_doc_ids)}

    def _normalize_query(self, query: str) -> Optional[np.ndarray]:
        if self._collection_size == 0:
            return None
        vec = self._embedder.encode([query])
        q = np.asarray(vec[0], dtype=np.float32)
        n = float(np.linalg.norm(q))
        if n <= 0:
            return q
        return q / n

    def reset_last_vector_ms(self) -> None:
        """Reset the accumulated vector timing counter to zero."""
        self._last_vector_ms = 0.0

    def retrieve(
        self,
        query: str,
        k: int,
        allowed_doc_ids: Optional[set] = None,
    ) -> tuple:
        if k <= 0 or self._collection_size == 0:
            return {}, None
        q = self._normalize_query(query)
        if q is None:
            return {}, None

        if allowed_doc_ids is None:
            _t0 = time.perf_counter()
            scores = self._all_embeddings @ q
            order = _topk_indices(scores, k)
            out = {}
            for i in order:
                did = self._all_doc_ids[i]
                sim = max(0.0, float(scores[i]))
                out[did] = (sim, self._all_documents[i])
            self._last_vector_ms += (time.perf_counter() - _t0) * 1000.0
            return out, q

        if not allowed_doc_ids:
            return {}, q
        rows = []
        dids = []
        for did in allowed_doc_ids:
            r = self._doc_id_to_row.get(did)
            if r is None:
                continue
            rows.append(r)
            dids.append(did)
        if not rows:
            return {}, q
        sub = self._all_embeddings[rows]
        _t0 = time.perf_counter()
        scores = sub @ q
        order = _topk_indices(scores, k)
        out = {}
        for j in order:
            did = dids[j]
            sim = max(0.0, float(scores[j]))
            out[did] = (sim, self._all_documents[rows[j]])
        self._last_vector_ms += (time.perf_counter() - _t0) * 1000.0
        return out, q

    def score_for_ids(self, query_vec, doc_ids):
        """Score ``doc_ids`` not present in the top-k list."""
        if not doc_ids:
            return {}
        # Resolve rows BEFORE the timer - dict lookups are not matvec work.
        pairs = [(d, self._doc_id_to_row.get(d)) for d in doc_ids]
        valid = [(d, r) for d, r in pairs if r is not None]
        if not valid:
            return {}
        rows = [r for _, r in valid]
        q = np.asarray(query_vec, dtype=np.float32)
        n = float(np.linalg.norm(q))
        q_norm = q if n <= 0 else q / n

        # Time the matvec + topk + dict build 
        _t0 = time.perf_counter()
        sims = self._all_embeddings[rows] @ q_norm
        sims = np.maximum(sims, 0.0)
        result = {
            d: (float(sims[i]), self._all_documents[rows[i]])
            for i, (d, _r) in enumerate(valid)
        }
        self._last_vector_ms += (time.perf_counter() - _t0) * 1000.0
        return result


def hybrid_retrieve(
    query: str,
    bm25: _BM25Index,
    vector: _VectorIndex,
    k: int,
    n: int,
    alpha: float,
    table_filter: TableFilter = EMPTY_FILTER,
) -> List[HybridResult]:
    """Fuse BM25 and vector signals into a ranked candidate pool.

    Each retriever fetches ``k * n``; ids found by only one are backfilled
    against the other; scores are min-max normalised per modality and
    combined as ``alpha * bm25 + (1 - alpha) * vector``. The full pool is
    returned - rerank truncates to ``k`` later.

    A non-empty ``table_filter`` restricts both retrievers to the
    corresponding doc-id allow-set. The canonical doc-id universe is
    BM25's, since BM25 and vector are populated from the same triples.
    """
    fetch = max(1, k * n)

    allowed_doc_ids: Optional[set] = None
    if not table_filter.is_empty():
        universe = bm25.all_doc_ids()
        if table_filter.allow is not None:
            universe &= bm25.doc_ids_for_tables(table_filter.allow)
        if table_filter.deny is not None:
            universe -= bm25.doc_ids_for_tables(table_filter.deny)
        allowed_doc_ids = universe
        if not allowed_doc_ids:
            return []

    bm25_raw, q_tokens = bm25.retrieve(query, fetch, allowed_doc_ids=allowed_doc_ids)
    vec_raw, query_vec = vector.retrieve(query, fetch, allowed_doc_ids=allowed_doc_ids)

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
    # doc_id tiebreak keeps results deterministic across processes
    # (PYTHONHASHSEED randomises set iteration).
    fused.sort(key=lambda r: (-r.fused_score, r.doc_id))
    return fused[:fetch]


def _is_yes(answer: str) -> bool:
    return answer.strip().lower().startswith("yes")


def llm_rerank(
    query: str,
    candidates: Sequence[HybridResult],
    llm: LLMBackend,
    *,
    concurrency: int | None = None,
) -> List[tuple]:
    """Yes/no judge per candidate; relevant ones bubble to the front."""
    if not candidates:
        return []
    prompts = [_rerank_prompt_for(c.doc_id, c.text, query) for c in candidates]
    from scripts.benchmark._judge_timer import JUDGE_TIMER
    with JUDGE_TIMER.judging():
        answers = llm.generate(prompts, max_new_tokens=2, concurrency=concurrency)
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
    table_filter: TableFilter = EMPTY_FILTER,
    *,
    rerank: bool = True,
    judge_concurrency: int | None = None, 
) -> List[RetrievalResult]:
    fused = hybrid_retrieve(
        query, bm25, vector, k=k, n=n, alpha=alpha, table_filter=table_filter
    )
    if not fused:
        return []

    if rerank:
        candidates = llm_rerank(query, fused, llm, concurrency=judge_concurrency)
    else:
        # Wrap each candidate as (cand, True) to keep downstream tuple shape stable.
        candidates = [(c, True) for c in fused]

    seen = {}
    for hit, relevant in candidates:
        if len(seen) == k:
            break
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
    return _BM25Index(fulltext_path), _VectorIndex(vector_path, collection_name, embedder)


def search_with_metrics(
    query: str,
    bm25: _BM25Index,
    vector: _VectorIndex,
    llm: LLMBackend,
    k: int,
    n: int,
    alpha: float,
    table_filter: TableFilter = EMPTY_FILTER,
    *,
    rerank: bool = True,
    judge_concurrency: int | None = None,
) -> tuple:
    """Like ``search``, but also returns a metrics dict with ``vector_ms``.

    Returns
    -------
    (results, {"vector_ms": float})
        *results* is the same ``List[RetrievalResult]`` that ``search`` returns.
        *vector_ms* is the wall-clock time (ms) spent inside numpy matvec
        operations on the ``_VectorIndex`` during this query.

    Production callers should continue to use the plain ``search()`` function.
    This peer function exists for the benchmark runner, which needs per-query
    timing breakdowns without modifying the public ``search`` signature.
    """
    vector.reset_last_vector_ms()
    results = search(
        query=query,
        bm25=bm25,
        vector=vector,
        llm=llm,
        k=k,
        n=n,
        alpha=alpha,
        table_filter=table_filter,
        rerank=rerank,
        judge_concurrency=judge_concurrency,
    )
    return results, {"vector_ms": vector._last_vector_ms}


__all__ = [
    "HybridResult",
    "RetrievalResult",
    "hybrid_retrieve",
    "llm_rerank",
    "search",
    "search_with_metrics",
    "open_indexes",
]
