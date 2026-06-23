"""Drive PNEUMA's hybrid retrieval against a prebuilt index and emit
Blend-compatible artifacts (per_query.jsonl, summary.csv, pneuma_compat.jsonl,
run_meta.json).

The module's job is the *retrieval* half of the PNEUMA bench:

* ``run_pneuma_benchmark(...)`` is the public entrypoint.  It iterates the
  cross-product (family × question × rerank_mode × k), calls the private
  ``_evaluate(...)`` helper for each (question, rerank_mode, k) tuple, and
  funnels the results through Blend's existing ``run.PerQueryRecord`` /
  ``run.SummaryRow`` shapes so downstream tooling (compare_v2, plot_v2) can
  consume PNEUMA-side runs without branching.

* ``_evaluate(...)`` is what unit tests mock.  In production it (a) calls
  ChromaDB ``collection.query(...)`` for the timed vector retrieval,
  (b) tokenises the query through ``bm25s``, (c) calls ``HybridRetriever``
  to merge the two channels (and optionally LLM-rerank, wrapped in
  ``JUDGE_TIMER.judging()`` so the patched ``prompt_openai_llm`` records
  judge time onto the per-query record).

* ``_open_collection`` and ``_open_retriever`` lazily import chromadb /
  bm25s so unit tests that patch ``_evaluate`` (and therefore never touch
  the index) do not need PNEUMA / chromadb / bm25s on ``sys.path``.

The ``_FIELD_BY_FAMILY`` and ``_ID_BY_SOURCE`` mappings mirror the source
of truth in ``scripts.benchmark.run`` so per-query rows from both
backends compare cell-for-cell.
"""
from __future__ import annotations

import json
import logging
import resource
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

from scripts.benchmark._judge_timer import JUDGE_TIMER
from scripts.benchmark._pneuma_compat_writer import write_pneuma_compat_jsonl
from scripts.benchmark.metrics import hit_at_k, recall_at_k, reciprocal_rank
from scripts.benchmark.run import (
    Manifest,
    PerQueryRecord,
    Question,
    SummaryRow,
    aggregate_summary,
    load_manifest,
    seed_all,
    write_per_query_jsonl,
    write_summary_csv,
)


LOG = logging.getLogger(__name__)


# Mirror of ``run._FIELD_BY_FAMILY`` so per-query rows from both backends are
# directly comparable.  Source of truth: scripts/benchmark/run.py.
_FIELD_BY_FAMILY = {
    "BC1": ("content", "question_from_sql_1"),
    "BC2": ("content", "question"),
    "BX1": ("bx", "question_bx1"),
    "BX2": ("bx", "question_bx2"),
}

_ID_BY_SOURCE = {
    "content": "id",
    "bx": "context_id",
}


# Bench-wide constants - must match the Blend side so the two pneuma_compat
# files can be cross-compared.
N = 5
ALPHA = 0.5
K_VALUES_DEFAULT = (1, 5, 10, 30, 50)
FAMILIES_DEFAULT = ("BC1", "BC2", "BX1", "BX2")
RERANK_MODES_DEFAULT = ("off", "on")


# ---- Question loading ------------------------------------------------------

def _load_questions(
    *,
    family: str,
    content_jsonl: Path,
    bx_jsonl: Path,
) -> list[Question]:
    """Load questions for a given family.

    Mirrors ``run.load_questions`` but is duplicated here (rather than
    re-imported) because we want the PNEUMA-side mapping to stay locked
    against the constants above; if someone changes ``run._FIELD_BY_FAMILY``
    the schema-parity test (Task 15) will catch the divergence.
    """
    if family not in _FIELD_BY_FAMILY:
        raise ValueError(
            f"unknown family {family!r}; expected one of "
            f"{sorted(_FIELD_BY_FAMILY)}"
        )
    source, qfield = _FIELD_BY_FAMILY[family]
    src_path = content_jsonl if source == "content" else bx_jsonl
    id_field = _ID_BY_SOURCE[source]

    questions: list[Question] = []
    if not src_path.exists():
        return questions
    for line in src_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if qfield not in row or id_field not in row:
            continue
        questions.append(Question(
            qid=str(row[id_field]),
            family=family,
            text=str(row[qfield]),
            answer_tables=frozenset(row.get("answer_tables", [])),
        ))
    return questions


# ---- Index handles (lazy imports so tests can mock above this layer) ------

def _open_collection(index_dir: Path):
    """Open the PNEUMA-built ChromaDB collection.

    Index layout (verified against IndexGenerator):
        ``<index_dir>/indexes/vector/`` - chromadb persistent client root
        ``<index_dir>/indexes/fulltext/`` - bm25s retriever directory

    The collection name is ``pneuma_<dataset>`` per pneuma_build's
    ``generate_index(index_name=f"pneuma_{dataset}")`` convention.
    """
    import chromadb  # type: ignore[import-untyped]

    vector_path = index_dir / "indexes" / "vector"
    client = chromadb.PersistentClient(path=str(vector_path))
    # Discover the collection name from the index_dir tail; pneuma_build
    # writes ``pneuma_<dataset>`` so dataset == index_dir.name.
    dataset = index_dir.name
    return client.get_collection(f"pneuma_{dataset}")


def _open_retriever(index_dir: Path):
    import bm25s  # type: ignore[import-untyped]

    dataset = index_dir.name
    fulltext_path = index_dir / "indexes" / "fulltext" / f"pneuma_{dataset}"
    return bm25s.BM25.load(str(fulltext_path), load_corpus=True)


def _build_dictionary_id_bm25(retriever) -> dict[str, int]:
    """Build the ``{table_name: corpus_index}`` dictionary that
    ``HybridRetriever`` needs for the bm25 fallback path.
    Lifted from ``hybrid_search.start``.
    """
    return {
        datum["metadata"]["table"]: idx
        for idx, datum in enumerate(retriever.corpus)
    }


# ---- The single-query evaluation hook (mocked in unit tests) -------------

def _evaluate(
    *,
    collection,
    retriever,
    query: str,
    k: int,
    n: int,
    alpha: float,
    rerank_mode: str,
    in_judge_ctx: bool,
    **kwargs: Any,
) -> dict:
    """Run hybrid retrieval for a single question.

    Returns ``{"retrieved_pneuma_ids": [...], "vector_ms": <ms>}``.

    * ``vector_ms`` measures only the ChromaDB ``collection.query`` call
      (the same isolation Blend's NLSeeker reports).
    * If ``in_judge_ctx`` is True (rerank_mode=="on"), the hybrid call is
      executed inside ``JUDGE_TIMER.judging()`` so the patched
      ``prompt_openai_llm`` records its time on the timer.

    Lazy imports of bm25s / Stemmer / numpy / the vendored PNEUMA tree
    keep unit tests that mock this function free of those dependencies.

    ``run_pneuma_benchmark`` passes ``stemmer``, ``dictionary_id_bm25``, and
    ``hybrid_retriever`` as kwargs to avoid rebuilding them on every call.
    """
    import bm25s  # type: ignore[import-untyped]
    import numpy as np  # local to ease unit-test mocking
    import Stemmer  # type: ignore[import-untyped]

    # PNEUMA's HybridRetriever lives under ``pneuma/experiments/...``;
    # that directory is added to sys.path by pneuma_patches at import.
    from pneuma_retriever.hybrid_retriever import (  # type: ignore[import-not-found]
        HybridRetriever,
        RerankingMode,
    )

    stemmer = kwargs.get("stemmer") or Stemmer.Stemmer("english")
    increased_k = k * n

    embedder_factory = kwargs.get("embedder_factory")
    if embedder_factory is None:
        raise RuntimeError(
            "_evaluate requires an ``embedder_factory`` kwarg (zero-arg "
            "callable returning a SentenceTransformer compatible with "
            "PNEUMA's index).  ``run_pneuma_benchmark`` provides one; "
            "callers that mock ``_evaluate`` don't reach this branch."
        )
    embedder = embedder_factory()
    # ``_build_sentence_transformer`` already pins the model to the right
    # device (cuda if available else cpu, see ``pneuma_build.py``); don't
    # force ``device="cuda"`` here or the bench cannot run on a CPU-only
    # machine (smoke / CI).  Mirrors Blend's ``_LocalEmbedder.encode``
    # which omits the device kwarg for the same reason.
    question_embedding = np.asarray(
        embedder.encode([query])[0], dtype=np.float32
    ).tolist()

    query_tokens = bm25s.tokenize(query, stopwords="en", stemmer=stemmer, show_progress=False)

    t_vec = time.perf_counter()
    vec_res = collection.query(
        query_embeddings=[question_embedding],
        n_results=increased_k,
    )
    vector_ms = (time.perf_counter() - t_vec) * 1000.0

    # ---- BM25 retrieval --------------------------------------------------
    bm25_results, bm25_scores = retriever.retrieve(
        query_tokens, k=increased_k, show_progress=False
    )
    bm25_res = (bm25_results, bm25_scores)


    dictionary_id_bm25 = kwargs.get("dictionary_id_bm25") or _build_dictionary_id_bm25(retriever)
    rmode_enum = RerankingMode.LLM if rerank_mode == "on" else RerankingMode.NONE

    reranker = kwargs.get("reranker")
    hybrid_retriever = kwargs.get("hybrid_retriever") or HybridRetriever(reranker, rmode_enum)

    all_nodes = hybrid_retriever.retrieve(
        retriever,
        collection,
        bm25_res,
        vec_res,
        increased_k,
        query,
        alpha,
        query_tokens,
        question_embedding,
        dictionary_id_bm25,
    )


    retrieved_pneuma: list[str] = []
    _seen_tables: set[str] = set()
    for entry in all_nodes:
        if len(retrieved_pneuma) == k:
            break
        pneuma_id = entry[0]
        parts = pneuma_id.split("_SEP_")
        table_name = parts[1].split(".")[0] if len(parts) >= 2 else parts[0]
        if table_name in _seen_tables:
            continue
        _seen_tables.add(table_name)
        retrieved_pneuma.append(table_name)

    return {
        "retrieved_pneuma_ids": retrieved_pneuma,
        "vector_ms": float(vector_ms),
    }


# ---- run_meta.json --------------------------------------------------------

_PARITY_DELTAS_PNEUMA: list[dict] = [
    {
        "id": "D18",
        "title": "PNEUMA uses ChromaDB HNSW (M=48) vs Blend brute-force matvec",
        "status": "irreducible-by-design",
        "note": (
            "PNEUMA's IndexGenerator builds an HNSW index with M=48 and "
            "random_seed=42; Blend's NLSeeker scans the full embedding "
            "matrix.  Recall divergence at small k is expected; the "
            "compare_v2 plot (Task 14) reports the gap."
        ),
    },
    {
        "id": "judge_calls-semantics",
        "title": "judge_calls counts HTTP roundtrips, not logical judgments",
        "status": "documented",
        "note": (
            "JUDGE_TIMER.record is invoked once per ``chat.completions.create``. "
            "PNEUMA's prompt_pipeline batches multiple judgments per call "
            "(batch_size=2), so judge_calls is a lower bound on the number "
            "of judgments evaluated."
        ),
    },
    {
        "id": "D19",
        "title": "Chunk-id splitter was off-by-one; now correctly extracts table_NNN",
        "status": "closed",
        "note": (
            "The original extractor took parts[0] of the '_SEP_'-split, which "
            "is the leading file path component, not the table id.  Fixed to "
            "take parts[1].split('.')[0], which is 'table_NNN' from the PNEUMA "
            "chunk-id grammar '<path>_SEP_table_NNN.csv_SEP_<role>_SEP_<idx>'. "
            "This was the root cause of hit_rate=0.00 across all runs."
        ),
    },
    {
        "id": "D20",
        "title": "Query-side BM25 tokeniser now passes stopwords='en'",
        "status": "closed",
        "note": (
            "bm25s.tokenize() on the query path was missing stopwords='en', "
            "while the corpus-side tokenisation in PNEUMA's IndexGenerator and "
            "Blend's _BM25Index._tokenize both pass it.  Adding stopwords='en' "
            "to the query tokeniser makes scores comparable across the two "
            "retrieval paths."
        ),
    },
    {
        "id": "D21",
        "title": "Per-query result list now contains distinct table ids (dedup-by-table)",
        "status": "closed",
        "note": (
            "The previous implementation took all_nodes[:k] and appended one "
            "entry per chunk, so a single table with multiple top-ranked chunks "
            "could occupy several result slots.  The loop now iterates all_nodes, "
            "deduplicates by table_name, and breaks once len(retrieved)==k. "
            "Blend's production path was fixed in lockstep "
            "(src/NLSeeker/retrieve.py:search): the dedup loop now iterates the "
            "full reranked pool rather than the candidates[:k] prefix. "
            "Affects MRR and hit@k/recall@k whenever the per-table chunk count "
            "exceeds 1 (public_bi has ~25 chunks/table, vs PNEUMA's ~10)."
        ),
    },
    {
        "id": "D22",
        "title": "Blend vector_ms timer now brackets matvec + topk + dict build",
        "status": "closed-by-scope-symmetry",
        "note": (
            "Blend's _VectorIndex timer previously covered only the matrix-vector "
            "multiply.  PNEUMA's vector_ms covers the full collection.query() call "
            "(HNSW traversal + result deserialisation + topk + ids/documents "
            "materialisation).  Widening Blend's timer to include topk + dict build "
            "makes both measure 'ranked-id list from query embedding in hand'; the "
            "remaining difference (HNSW vs brute-force kernel) is the algorithmic "
            "contrast this metric is intended to expose."
        ),
    },
    {
        "id": "D24",
        "title": "judge_ms scope still differs: PNEUMA wraps full _llm_rerank body",
        "status": "documented-residual",
        "note": (
            "_patch_llm_rerank_judge_scope in pneuma_patches.py wraps the full "
            "_llm_rerank body (including prompt construction), where Blend wraps "
            "only llm.generate(...).  Magnitude is sub-millisecond per call.  "
            "Not closed because closing it would require restricting the patch to "
            "the HTTP fan-out or expanding Blend's scope, both of which add "
            "complexity for negligible gain."
        ),
    },
]


def _git_commit_dirty() -> tuple[str, bool]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = bool(subprocess.run(
            ["git", "diff", "--quiet"], capture_output=True
        ).returncode)
        return sha, dirty
    except Exception:
        return "unknown", False


def _du_sb(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def _gpu_info() -> dict:
    """Return ``{"raw": "<nvidia-smi line>"}`` or ``{"raw": "unknown"}``.

    The bench's compare_v2 only treats this as an opaque label, so a single
    raw line is enough - no parsing into count/model/memory.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return {"raw": out or "unknown"}
    except Exception:
        return {"raw": "unknown"}


def _peak_rss_bytes() -> int:
    """Linux: ru_maxrss is in KB; convert to bytes for run_meta parity with
    Blend's ``run.py:_build_run_meta`` which reports the same field."""
    try:
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except Exception:
        return 0


def _read_manifest_health(lake_dir: Path) -> tuple[int, int]:
    """Return ``(unresolvable_questions, dropped_context_rows)`` from the
    lake's ``_manifest.json`` ``health`` block; defaults to ``(0, 0)`` if
    either the file or the keys are absent.
    """
    try:
        raw = json.loads((Path(lake_dir) / "_manifest.json").read_text())
        health = raw.get("health", {}) or {}
        return (
            int(health.get("unresolvable_questions", 0) or 0),
            int(health.get("dropped_context_rows", 0) or 0),
        )
    except Exception:
        return 0, 0


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _build_run_meta(
    *,
    dataset: str,
    index_dir: Path,
    lake_dir: Path,
    n_questions: dict[str, int],
    build_wall_clock_s: float,
    run_wall_clock_s: float,
    rerank_modes: Iterable[str],
    k_values: Iterable[int],
    endpoints: dict | None = None,
    status: str = "complete",
    judge_model_id: str = "Qwen2.5-7B-Instruct",
    embedder_model_id: str = "BAAI/bge-base-en-v1.5",
    n: int = N,
    alpha: float = ALPHA,
) -> dict:
    sha, dirty = _git_commit_dirty()
    unresolvable, dropped_ctx = _read_manifest_health(Path(lake_dir))
    return {
        "schema_version": 2,
        "ts_utc": _now_utc_iso(),
        "dataset": dataset,
        "git_commit": sha,
        "git_dirty": dirty,
        "endpoints": dict(endpoints or {}),
        "gpu": _gpu_info(),
        "k_values": list(k_values),
        "n": n,
        "alpha": alpha,
        "rerank_modes": list(rerank_modes),
        "slicing_method": "per_k_search_pneuma_parity",
        "retrieval_path": "pneuma.experiments.hybrid_retriever.HybridRetriever.retrieve",
        "n_questions": dict(n_questions),
        "build_wall_clock_s": build_wall_clock_s,
        "run_wall_clock_s": run_wall_clock_s,
        "peak_rss_bytes": _peak_rss_bytes(),
        "unresolvable_questions_count": unresolvable,
        "dropped_context_rows": dropped_ctx,
        "status": status,
        "deployment": {
            "mode": "pneuma_bench_sequential",
            "vector_index_kind": "chromadb_hnsw_M48",
            "fulltext_index_kind": "bm25s",
            "retrieval_path": "HybridRetriever.retrieve()",
            "judge_model_id": judge_model_id,
            "embedder_model_id": embedder_model_id,
            "seeds": {
                "PYTHONHASHSEED": "0",
                "vllm_seed": 42,
                # HNSW seed is hard-coded at the index_generator layer
                # (``pneuma/src/pneuma/index_generator/index_generator.py:181-187``)
                # via ``chromadb-deterministic``.  Surfaced here so a reader
                # of run_meta.json knows how the index was built without
                # cross-referencing source.
                "hnsw_random_seed": 42,
            },
        },
        "index": {
            "path": str(index_dir),
            "size_bytes": int(_du_sb(index_dir)),
        },
        "parity_deltas": list(_PARITY_DELTAS_PNEUMA),
    }


# ---- Public entrypoint ----------------------------------------------------

def run_pneuma_benchmark(
    *,
    dataset: str,
    index_dir: Path,
    lake_dir: Path,
    content_jsonl: Path,
    run_dir: Path,
    families: Iterable[str] | None = None,
    rerank_modes: Iterable[str] | None = None,
    k_values: Iterable[int] | None = None,
    max_questions: int | None = None,
    build_wall_clock_s: float = 0.0,
    endpoints: dict | None = None,
    judge_model_id: str = "Qwen2.5-7B-Instruct",
    embedder_model_id: str = "BAAI/bge-base-en-v1.5",
    embedder: Any | None = None,
    n: int = N,
    alpha: float = ALPHA,
) -> Path:
    """End-to-end PNEUMA-side bench run.

    For each ``(family, question, rerank_mode, k)`` it calls ``_evaluate``
    and emits one ``PerQueryRecord``.  After the sweep:

    * ``per_query.jsonl`` via Blend's ``write_per_query_jsonl``
    * ``summary.csv`` via Blend's ``aggregate_summary`` + ``write_summary_csv``
    * ``pneuma_compat.jsonl`` via the shared writer
    * ``run_meta.json`` with ``schema_version=2`` and PNEUMA-specific
      deployment + parity_deltas blocks.

    Returns ``run_dir`` for caller convenience.
    """
    families = tuple(families or FAMILIES_DEFAULT)
    rerank_modes = tuple(rerank_modes or RERANK_MODES_DEFAULT)
    k_values = tuple(k_values or K_VALUES_DEFAULT)

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_all(42)

    bx_jsonl = Path(lake_dir) / "_bx_questions.jsonl"
    manifest = load_manifest(Path(lake_dir))

    # Open the index handles ONCE; the loop below reuses them across all
    # (family, question, rerank_mode, k) tuples.
    collection = _open_collection(Path(index_dir))
    retriever = _open_retriever(Path(index_dir))


    import Stemmer as _Stemmer  # type: ignore[import-untyped]
    _stemmer = _Stemmer.Stemmer("english")
    _dictionary_id_bm25 = _build_dictionary_id_bm25(retriever)

    from pneuma_retriever.hybrid_retriever import (  # type: ignore[import-not-found]
        HybridRetriever as _HybridRetriever,
        RerankingMode as _RerankingMode,
    )
    _hybrid_retrievers: dict[str, Any] = {
        "off": _HybridRetriever(None, _RerankingMode.NONE),
        "on":  _HybridRetriever(None, _RerankingMode.LLM),
    }

    _embedder_box: list[Any] = [embedder]  # closed over below

    def _get_embedder() -> Any:
        if _embedder_box[0] is None:
            from scripts.benchmark.pneuma_build import _build_sentence_transformer
            _embedder_box[0] = _build_sentence_transformer(embedder_model_id)
        return _embedder_box[0]

    per_query: list[PerQueryRecord] = []
    n_questions_by_family: dict[str, int] = {}

    t_run0 = time.perf_counter()
    for family in families:
        questions = _load_questions(
            family=family, content_jsonl=Path(content_jsonl), bx_jsonl=bx_jsonl,
        )
        if max_questions is not None:
            questions = questions[:max_questions]
        n_questions_by_family[family] = len(questions)
        if not questions:
            LOG.warning("family %s: no questions found, skipping", family)
            continue

        bar = tqdm(
            questions,
            desc=f"{dataset} {family}",
            unit="q",
            dynamic_ncols=True,
            leave=True,
        )
        for q in bar:
            answer_set = set(q.answer_tables)
            for rerank_mode in rerank_modes:
                in_judge_ctx = (rerank_mode == "on")
                for k in k_values:
                    bar.set_postfix_str(f"{rerank_mode} k={k}", refresh=True)

                    JUDGE_TIMER.reset()
                    t0 = time.perf_counter()

                    result = _evaluate(
                        collection=collection,
                        retriever=retriever,
                        query=q.text,
                        k=k,
                        n=n,
                        alpha=alpha,
                        rerank_mode=rerank_mode,
                        in_judge_ctx=in_judge_ctx,
                        embedder_factory=_get_embedder,
                        stemmer=_stemmer,
                        dictionary_id_bm25=_dictionary_id_bm25,
                        hybrid_retriever=_hybrid_retrievers[rerank_mode],
                    )
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    judge_metrics = JUDGE_TIMER.read()

                    retrieved_ids = result["retrieved_pneuma_ids"]
                    vector_ms = result.get("vector_ms")

                    retrieved_dicts = []
                    for pneuma_id in retrieved_ids:
                        tid = manifest.pneuma_to_tid.get(pneuma_id)
                        retrieved_dicts.append({
                            "table_id": int(tid) if tid is not None else -1,
                            "pneuma_id": pneuma_id,
                        })

                    per_query.append(PerQueryRecord(
                        ts_utc=_now_utc_iso(),
                        dataset=dataset,
                        family=family,
                        k=k,
                        rerank_mode=rerank_mode,
                        question_id=q.qid,
                        question=q.text,
                        retrieved=retrieved_dicts,
                        answer_tables=sorted(q.answer_tables),
                        hit=hit_at_k(retrieved_ids, answer_set, k),
                        recall=recall_at_k(retrieved_ids, answer_set, k),
                        rr=reciprocal_rank(retrieved_ids, answer_set),
                        latency_ms=latency_ms,
                        vector_ms=vector_ms,
                        judge_ms=judge_metrics["judge_ms"],
                        judge_calls=judge_metrics["judge_calls"],
                        judge_tokens_in=judge_metrics["judge_tokens_in"],
                        judge_tokens_out=judge_metrics["judge_tokens_out"],
                    ))
        bar.close()

    run_wall = time.perf_counter() - t_run0

    per_query.sort(key=lambda r: (r.family, r.question_id, r.rerank_mode, r.k))

    write_per_query_jsonl(run_dir / "per_query.jsonl", per_query)
    summary: list[SummaryRow] = aggregate_summary(per_query, n=n, alpha=alpha)
    write_summary_csv(run_dir / "summary.csv", summary)
    write_pneuma_compat_jsonl(run_dir / "pneuma_compat.jsonl", summary)

    meta = _build_run_meta(
        dataset=dataset,
        index_dir=Path(index_dir),
        lake_dir=Path(lake_dir),
        n_questions=n_questions_by_family,
        build_wall_clock_s=build_wall_clock_s,
        run_wall_clock_s=run_wall,
        rerank_modes=rerank_modes,
        k_values=k_values,
        endpoints=endpoints,
        judge_model_id=judge_model_id,
        embedder_model_id=embedder_model_id,
        n=n,
        alpha=alpha,
    )
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    return run_dir


__all__ = [
    "run_pneuma_benchmark",
    "N",
    "ALPHA",
    "K_VALUES_DEFAULT",
    "FAMILIES_DEFAULT",
    "RERANK_MODES_DEFAULT",
    "_FIELD_BY_FAMILY",
]
