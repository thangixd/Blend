import logging
import threading
from dataclasses import dataclass

from src.NLSeeker.config import NLSeekerConfig
from src.NLSeeker.db_schema import _read_db_section
from src.NLSeeker.llm import EmbedBackend, LLMBackend, build_backends, reset_backend_cache
from src.NLSeeker.predicate import EMPTY_FILTER, TableFilter
from src.NLSeeker.retrieve import RetrievalResult, _BM25Index, _VectorIndex, open_indexes, search, search_with_metrics

# Typing imports
from pathlib import Path
from typing import List

LOG = logging.getLogger(__name__)

_ENGINE_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()


class IndexNotFoundError(RuntimeError):
    pass


@dataclass
class _NLEngine:
    cfg: NLSeekerConfig
    llm: LLMBackend
    embedder: EmbedBackend
    _bm25: _BM25Index
    _vector: _VectorIndex

    @classmethod
    def get(cls, cfg: NLSeekerConfig, index_name: str = None) -> "_NLEngine":
        name = index_name or cfg.index_name
        key = (cfg.signature(), name)
        with _CACHE_LOCK:
            cached = _ENGINE_CACHE.get(key)
            if cached is not None:
                return cached
            engine = cls._build(cfg, name)
            _ENGINE_CACHE[key] = engine
            return engine

    @classmethod
    def _build(cls, cfg: NLSeekerConfig, index_name: str) -> "_NLEngine":
        vector_path, fulltext_path = _resolve_index_paths(cfg, index_name)
        llm, embedder = build_backends(cfg)
        bm25, vector = open_indexes(vector_path, fulltext_path, index_name, embedder)
        return cls(cfg=cfg, llm=llm, embedder=embedder, _bm25=bm25, _vector=vector)

    def search(
        self,
        query: str,
        k: int = None,
        n: int = None,
        alpha: float = None,
        table_filter: TableFilter = EMPTY_FILTER,
        *,
        rerank: bool = True,
        judge_concurrency: int | None = None,
    ) -> List[RetrievalResult]:
        eff_k = self.cfg.default_k if k is None else int(k)
        eff_n = self.cfg.n if n is None else int(n)
        eff_alpha = self.cfg.alpha if alpha is None else float(alpha)
        return search(
            query=query,
            bm25=self._bm25,
            vector=self._vector,
            llm=self.llm,
            k=eff_k,
            n=eff_n,
            alpha=eff_alpha,
            table_filter=table_filter,
            rerank=rerank,
            judge_concurrency=judge_concurrency,
        )

    def search_with_metrics(
        self,
        query: str,
        k: int = None,
        n: int = None,
        alpha: float = None,
        table_filter: TableFilter = EMPTY_FILTER,
        *,
        rerank: bool = True,
        judge_concurrency: int | None = None,
    ) -> tuple:
        """Like ``search``, but returns ``(results, {"vector_ms": float})``.

        Delegates to ``retrieve.search_with_metrics`` which resets and reads
        ``_VectorIndex._last_vector_ms`` around the search call.
        """
        eff_k = self.cfg.default_k if k is None else int(k)
        eff_n = self.cfg.n if n is None else int(n)
        eff_alpha = self.cfg.alpha if alpha is None else float(alpha)
        return search_with_metrics(
            query=query,
            bm25=self._bm25,
            vector=self._vector,
            llm=self.llm,
            k=eff_k,
            n=eff_n,
            alpha=eff_alpha,
            table_filter=table_filter,
            rerank=rerank,
            judge_concurrency=judge_concurrency,
        )


def _resolve_index_paths(cfg: NLSeekerConfig, index_name: str) -> tuple:
    """Look up paths in blend_nl_indexes; fall back to cfg defaults."""
    try:
        paths = _fetch_index_paths_readonly(cfg, index_name)
        if paths is not None and paths[0].exists() and paths[1].exists():
            return paths
    except Exception as e:  # pragma: no cover
        LOG.warning("Could not query blend_nl_indexes (%s); falling back to cfg paths.", e)

    vp = cfg.vector_index_path()
    fp = cfg.fulltext_index_path()
    if not vp.exists() or not fp.exists():
        raise IndexNotFoundError(
            f"NLSeeker index {index_name!r} not found. Build it first with "
            f"`python scripts/create_blend_index.py --lake ... --nl-index`. "
            f"Looked at vector_path={vp} fulltext_path={fp}."
        )
    return vp, fp


def _fetch_index_paths_readonly(cfg: NLSeekerConfig, index_name: str):
    # DuckDB allows multiple read-only handles on the same file, so we open
    # our own connection rather than borrowing DBHandler's writer.
    db_cfg = _read_db_section()
    dbms = db_cfg["dbms"].lower()
    if dbms == "duckdb":
        import duckdb

        con = duckdb.connect(database=db_cfg["path"], read_only=True)
        try:
            cur = con.cursor()
            cur.execute(
                "SELECT vector_path, fulltext_path FROM blend_nl_indexes WHERE name = ?",
                (index_name,),
            )
            row = cur.fetchone()
            cur.close()
        finally:
            con.close()
        if row is None:
            return None
        return Path(row[0]), Path(row[1])

    from src.NLSeeker.db_schema import fetch_index_paths, open_writer

    with open_writer() as (cur, dialect):
        return fetch_index_paths(cur, dialect, index_name)


def reset_engine_cache() -> None:
    with _CACHE_LOCK:
        _ENGINE_CACHE.clear()
    reset_backend_cache()


__all__ = ["IndexNotFoundError", "_NLEngine", "reset_engine_cache"]
