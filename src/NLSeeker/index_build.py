import logging
import shutil
from dataclasses import dataclass

import bm25s
import chromadb_deterministic as chromadb
from chromadb_deterministic.api.shared_system_client import SharedSystemClient
from chromadb_deterministic.config import Settings
import pandas as pd
import Stemmer

from src.NLSeeker.config import NLSeekerConfig
from src.NLSeeker.db_schema import (
    SummaryType,
    ensure_schema,
    insert_contexts,
    insert_summaries,
    open_writer,
    upsert_index,
)
from src.NLSeeker.llm import EmbedBackend, LLMBackend, build_backends
from src.NLSeeker.summarize import SummarizedTable, summarize_table

# Typing imports
from pathlib import Path
from typing import Iterable, Optional, Sequence

LOG = logging.getLogger(__name__)

# Doc-id separator (kept compatible with the on-disk format).
DOC_ID_SEP = "_SEP_"

# Pinned for deterministic HNSW indexing.
_HNSW_METADATA = {
    "hnsw:space": "cosine",
    "hnsw:M": 48,
    "hnsw:random_seed": 42,
}

_BM25_STOPWORDS = "en"


@dataclass
class IndexBuildResult:
    index_name: str
    vector_path: Path
    fulltext_path: Path
    total_documents: int
    total_tables: int


def make_doc_id(table_id: int, kind: str, index: int, *, source: str = "real") -> str:
    """Build a doc id.

    - Contents (column narrations / row samples): ``<tid>_SEP_contents_SEP_<kind>-<i>``.
    - Contexts: ``<tid>_SEP_contexts-<source>-<i>`` (spec §3.1).

    The substring ``_SEP_contexts-`` is preserved so retrieve.py routes to
    the context-rerank prompt for both real and synthetic contexts.
    """
    if kind == "contexts":
        return f"{table_id}{DOC_ID_SEP}contexts-{source}-{index}"
    return f"{table_id}{DOC_ID_SEP}contents{DOC_ID_SEP}{kind}-{index}"


def parse_doc_id(doc_id: str) -> tuple:
    """Inverse of make_doc_id(). Returns ``(table_id, kind, index)``."""
    head, _, tail = doc_id.partition(DOC_ID_SEP)
    if not tail:
        raise ValueError(f"Malformed NLSeeker doc id: {doc_id!r}")
    if DOC_ID_SEP in tail:
        _, _, body = tail.partition(DOC_ID_SEP)
        kind, _, index_str = body.rpartition("-")
    else:
        kind, _, index_str = tail.rpartition("-")
    return int(head), kind, int(index_str)


@dataclass
class _TableDocuments:
    table_id: int
    triples: list


def _flatten_summary(
    summary: SummarizedTable, context_source: str = "real"
) -> _TableDocuments:
    """Flatten a SummarizedTable into (doc_id, text, SummaryType) triples.

    ``context_source`` is the Source tag written into context doc ids. 
    Defaults to ``"real"`` for backward compatibility with
    the ``add_table`` call path.

    Note: this function serves the ``add_table`` path and emits all summary
    types unconditionally. Cell-aware filtering by ``enabled_summary_types``
    happens in ``run()``.
    """
    triples = []
    for i, block in enumerate(summary.column_narration_blocks):
        triples.append((make_doc_id(summary.table_id, "schema", i), block, SummaryType.COLUMN_NARRATION))
    for i, block in enumerate(summary.row_sample_blocks):
        triples.append((make_doc_id(summary.table_id, "row", i), block, SummaryType.ROW_SAMPLE))
    for i, block in enumerate(summary.context_blocks):
        triples.append((
            make_doc_id(summary.table_id, "contexts", i, source=context_source),
            block,
            SummaryType.CONTEXT,
        ))
    return _TableDocuments(table_id=summary.table_id, triples=triples)


class NLIndexBuilder:
    """Drive table summarisation and persist the Chroma + BM25s indexes."""

    def __init__(
        self,
        cfg: NLSeekerConfig,
        llm: Optional[LLMBackend] = None,
        embedder: Optional[EmbedBackend] = None,
    ) -> None:
        self.cfg = cfg
        if llm is None or embedder is None:
            llm, embedder = build_backends(cfg)
        self.llm = llm
        self.embedder = embedder

        self._vector_path = cfg.vector_index_path()
        self._fulltext_path = cfg.fulltext_index_path()

        self._chroma_client = None
        self._collection = None
        self._bm25_corpus = []
        self._bm25_texts = []
        self._stemmer = None

        self._documents_total = 0
        self._tables_total = 0
        self._summaries_to_persist = []
        self._contexts_to_persist = []

    def start(self) -> None:
        """Wipe any existing index artifacts and create empty Chroma + BM25 stores."""
        self._vector_path.parent.mkdir(parents=True, exist_ok=True)
        self._fulltext_path.parent.mkdir(parents=True, exist_ok=True)

        if self._vector_path.exists():
            shutil.rmtree(self._vector_path)
        if self._fulltext_path.exists():
            shutil.rmtree(self._fulltext_path)
        self._vector_path.mkdir(parents=True, exist_ok=True)

        # ChromaDB caches System instances by path across clients in the same
        # process; after wiping the directory a stale entry would still
        # reference the deleted sqlite file and add() would fail with
        # "attempt to write a readonly database".
        SharedSystemClient.clear_system_cache()

        chroma_settings = Settings(anonymized_telemetry=False, allow_reset=True)
        self._chroma_client = chromadb.PersistentClient(
            path=str(self._vector_path),
            settings=chroma_settings,
        )
        self._collection = self._chroma_client.get_or_create_collection(
            name=self.cfg.index_name,
            metadata=_HNSW_METADATA,
            embedding_function=None,
        )
        self._stemmer = Stemmer.Stemmer("english")

    def add_table(
        self,
        table_id: int,
        df: pd.DataFrame,
        contexts: Optional[Sequence[str]] = None,
        pre_chunked_contexts: bool = False,
    ) -> None:
        """Summarise one table and stage its documents for both indexes."""
        if self._collection is None:
            raise RuntimeError("NLIndexBuilder.start() must be called first")

        summary = summarize_table(table_id, df, self.llm, self.embedder, contexts=contexts,
                                  pre_chunked_contexts=pre_chunked_contexts)
        flat = _flatten_summary(summary)
        if not flat.triples:
            LOG.warning("TableId=%d produced no summary documents - skipping", table_id)
            return

        for block_idx, block in enumerate(summary.column_narration_blocks):
            self._summaries_to_persist.append(
                (table_id, block_idx, SummaryType.COLUMN_NARRATION, block)
            )
        for block_idx, block in enumerate(summary.row_sample_blocks):
            self._summaries_to_persist.append(
                (table_id, block_idx, SummaryType.ROW_SAMPLE, block)
            )
        for block_idx, block in enumerate(summary.context_blocks):
            self._summaries_to_persist.append(
                (table_id, block_idx, SummaryType.CONTEXT, block)
            )
        if contexts:
            for ctx_idx, raw in enumerate(contexts):
                self._contexts_to_persist.append((table_id, ctx_idx, raw))

        ids = [t[0] for t in flat.triples]
        texts = [t[1] for t in flat.triples]
        embeddings = self.embedder.encode(texts)
        if embeddings.shape[0] != len(texts):
            raise RuntimeError(
                f"Embedder returned {embeddings.shape[0]} vectors for {len(texts)} texts"
            )

        metadatas = [{"table_id": str(table_id), "kind": kind_for_doc(d)} for d in ids]

        self._collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
        )

        for doc_id, text, _ in flat.triples:
            self._bm25_texts.append(text)
            self._bm25_corpus.append({"text": text, "doc_id": doc_id, "table_id": table_id})

        self._documents_total += len(flat.triples)
        self._tables_total += 1
        LOG.info(
            "Indexed TableId=%d: %d narration blocks, %d row blocks, %d context blocks "
            "(running total: %d docs, %d tables)",
            table_id,
            len(summary.column_narration_blocks),
            len(summary.row_sample_blocks),
            len(summary.context_blocks),
            self._documents_total,
            self._tables_total,
        )

    def finalize(self) -> IndexBuildResult:
        """Persist both indexes and register their paths in ``blend_nl_indexes``."""
        if self._collection is None:
            raise RuntimeError("NLIndexBuilder.start() must be called first")
        if self._documents_total == 0:
            raise RuntimeError("No documents added - refusing to persist an empty index")

        assert self._stemmer is not None
        tokens = bm25s.tokenize(
            self._bm25_texts,
            stopwords=_BM25_STOPWORDS,
            stemmer=self._stemmer,
            show_progress=False,
        )
        retriever = bm25s.BM25(corpus=self._bm25_corpus)
        retriever.index(tokens, show_progress=False)
        retriever.save(str(self._fulltext_path), corpus=self._bm25_corpus)

        # Drop the client and clear the path-keyed cache so retrieval (or a
        # follow-up build) opens a fresh handle.
        self._collection = None
        self._chroma_client = None
        SharedSystemClient.clear_system_cache()

        with open_writer() as (cur, dbms):
            ensure_schema(cur, dbms)
            insert_summaries(cur, dbms, self._summaries_to_persist)
            insert_contexts(cur, dbms, self._contexts_to_persist)
            upsert_index(
                cur,
                dbms,
                name=self.cfg.index_name,
                vector_path=self._vector_path,
                fulltext_path=self._fulltext_path,
            )

        result = IndexBuildResult(
            index_name=self.cfg.index_name,
            vector_path=self._vector_path,
            fulltext_path=self._fulltext_path,
            total_documents=self._documents_total,
            total_tables=self._tables_total,
        )
        LOG.info(
            "NLSeeker index '%s' built: %d documents across %d tables (vector=%s, fulltext=%s)",
            result.index_name,
            result.total_documents,
            result.total_tables,
            result.vector_path,
            result.fulltext_path,
        )
        return result


    @classmethod
    def from_storage(
        cls,
        *,
        cfg: NLSeekerConfig,
        duckdb_path: str,
        embedder,
        llm=None,
    ) -> "NLIndexBuilder":
        """Construct a builder whose data source is an existing DuckDB.

        Use ``builder.run()`` to build the cell-specific Chroma + BM25s
        indexes from the filtered subset defined by ``cfg.enabled_summary_types``
        and ``cfg.context_sources`` (spec §4.2).
        """
        builder = cls(cfg=cfg, llm=llm or _NoopLLM(), embedder=embedder)
        builder._duckdb_path = duckdb_path  # type: ignore[attr-defined]
        return builder

    def staged_doc_ids(self) -> list[str]:
        """Return doc IDs collected by the last ``run()`` call. Useful in tests."""
        return [t[0] for t in getattr(self, "_collected_triples", [])]

    def staged_doc_id_kinds(self) -> list[str]:
        """Return the *kind* segment of each staged doc id (for test assertions)."""
        kinds: list = []
        for did in self.staged_doc_ids():
            try:
                _, kind, _ = parse_doc_id(did)
                kinds.append(kind)
            except Exception:
                kinds.append("?")
        return kinds

    def run(self) -> "IndexBuildResult":
        """Build a cell-aware index end-to-end from storage.

        Reads rows filtered by ``cfg.enabled_summary_types`` and
        ``cfg.context_sources`` from the source DuckDB (set via
        ``from_storage``), embeds them, persists Chroma + BM25s under the
        cell-specific ``cfg.index_name``, and returns the result.
        """
        import duckdb as _duckdb

        from src.NLSeeker.db_schema import fetch_table_summaries

        duckdb_path = getattr(self, "_duckdb_path", None)
        if duckdb_path is None:
            raise RuntimeError(
                "NLIndexBuilder.run() requires a source DB. "
                "Construct via NLIndexBuilder.from_storage(...)."
            )

        # --- load filtered rows from storage ---
        con = _duckdb.connect(duckdb_path, read_only=True)
        cur = con.cursor()
        try:
            summaries = fetch_table_summaries(cur, "duckdb")
            # Discover which tables have contexts under this cell's source filter.
            # We only need the set of TableIds; per-table text is re-fetched below
            # to preserve the (source, idx) pairs needed for doc-id construction.
            if self.cfg.context_sources:
                placeholders = ",".join("?" for _ in self.cfg.context_sources)
                cur.execute(
                    f"SELECT DISTINCT TableId FROM blend_nl_contexts WHERE Source IN ({placeholders})",
                    tuple(self.cfg.context_sources),
                )
                ctx_tids = {int(row[0]) for row in cur.fetchall()}
            else:
                ctx_tids = set()
            contexts = {tid: True for tid in ctx_tids}  # values unused; keys used for union
        finally:
            cur.close()
            con.close()

        self.start()
        self._collected_triples: list = []

        all_tables = set(summaries.keys()) | set(contexts.keys())
        for tid in sorted(all_tables):
            per_type = summaries.get(tid, {})
            triples: list = []

            # --- content-side: summaries filtered by enabled_summary_types ---
            for st in self.cfg.enabled_summary_types:
                blocks = per_type.get(st, [])
                kind = (
                    "schema"
                    if st in (SummaryType.COLUMN_NARRATION, SummaryType.COLUMN_NARRATION_PROFILED)
                    else "row"
                )
                for i, txt in enumerate(blocks):
                    triples.append((make_doc_id(tid, kind, i), txt, st))

            # --- context-side: per-source doc ids from storage ---
            if self.cfg.context_sources:
                # Fresh read-only connection per table: DuckDB allows concurrent readers;
                # no write lock is held since both the outer and inner connections are read_only=True.
                ctx_con = _duckdb.connect(duckdb_path, read_only=True)
                ctx_cur = ctx_con.cursor()
                try:
                    sources_list = list(self.cfg.context_sources)
                    placeholders = ",".join("?" for _ in sources_list)
                    ctx_cur.execute(
                        f"SELECT Source, ContextIdx, Text FROM blend_nl_contexts "
                        f"WHERE TableId=? AND Source IN ({placeholders}) "
                        f"ORDER BY Source, ContextIdx",
                        (int(tid), *sources_list),
                    )
                    for src, idx, text in ctx_cur.fetchall():
                        triples.append(
                            (
                                make_doc_id(tid, "contexts", int(idx), source=src),
                                text,
                                SummaryType.CONTEXT,
                            )
                        )
                finally:
                    ctx_cur.close()
                    ctx_con.close()

            if not triples:
                LOG.warning(
                    "TableId=%d produced no documents for this cell; skipping.", tid
                )
                continue

            self._collected_triples.extend(triples)

            ids = [t[0] for t in triples]
            texts = [t[1] for t in triples]
            embeddings = self.embedder.encode(texts)

            metadatas = [{"table_id": str(tid), "kind": kind_for_doc(d)} for d in ids]
            self._collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
            )
            for doc_id, text, _ in triples:
                self._bm25_texts.append(text)
                self._bm25_corpus.append({"text": text, "doc_id": doc_id, "table_id": tid})

            self._documents_total += len(triples)
            self._tables_total += 1

        return self.finalize()


class _NoopLLM:
    """Placeholder LLM backend for ``from_storage`` builds.

    No generation happens in the indexing phase; this sentinel satisfies the
    constructor signature without requiring a real backend.
    """

    max_input_tokens = 1

    def generate(self, prompts, max_new_tokens, *, concurrency=None):
        return [""] * len(prompts)


def kind_for_doc(doc_id: str) -> str:
    _, kind, _ = parse_doc_id(doc_id)
    return kind


def build_index(
    tables: Iterable[tuple],
    cfg: Optional[NLSeekerConfig] = None,
    llm: Optional[LLMBackend] = None,
    embedder: Optional[EmbedBackend] = None,
) -> IndexBuildResult:
    """One-shot helper for ``(table_id, df)`` or ``(table_id, df, contexts)`` entries."""
    cfg = cfg or NLSeekerConfig.load()
    builder = NLIndexBuilder(cfg, llm=llm, embedder=embedder)
    builder.start()
    for entry in tables:
        if len(entry) == 2:
            table_id, df = entry
            builder.add_table(table_id, df)
        elif len(entry) == 3:
            table_id, df, contexts = entry
            builder.add_table(table_id, df, contexts=contexts)
        else:
            raise ValueError(
                f"build_index expects (table_id, df) pairs or "
                f"(table_id, df, contexts) triples; got {len(entry)} elements"
            )
    return builder.finalize()


__all__ = [
    "DOC_ID_SEP",
    "IndexBuildResult",
    "NLIndexBuilder",
    "_NoopLLM",
    "build_index",
    "kind_for_doc",
    "make_doc_id",
    "parse_doc_id",
]
