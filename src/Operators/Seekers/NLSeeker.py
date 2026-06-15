from src.Operators.Seekers.SeekerBase import Seeker
from src.NLSeeker.config import NLSeekerConfig
from src.NLSeeker.engine import _NLEngine
from src.NLSeeker.predicate import parse_additionals
from src.NLSeeker.sql_builder import values_select_with_rank

# Typing imports
from src.DBHandler import DBHandler


class NLSeeker(Seeker):
    # The 3-feature runtime regressor (distinct rows, geomean token freq, ncols)
    # has nothing meaningful to feed for an NL query, so we opt out of the XGB
    # model load in Seeker.__init__ and use a constant cost.
    HAS_ML_COST_MODEL = False

    def __init__(
        self,
        query: str,
        k: int = 10,
        n: int = None,
        alpha: float = None,
        index_name: str = None,
        config_overrides: dict = None,
        *,
        rerank: bool = True,
        judge_concurrency: int | None = None,
    ) -> None:
        super().__init__(k)

        self.query = str(query)
        self._cfg = NLSeekerConfig.load(overrides=config_overrides)
        self._index_name = index_name or self._cfg.index_name
        self._n = self._cfg.n if n is None else int(n)
        self._alpha = self._cfg.alpha if alpha is None else float(alpha)
        self._rerank = bool(rerank)
        self._judge_concurrency = judge_concurrency

        self._cached_table_ids = None
        self._last_metrics: dict | None = None

    def create_sql_query(self, db: DBHandler, additionals: str = "") -> str:
        # Reset metrics so a cache hit yields None (no retrieval ran this call).
        # _last_metrics reflects the most recent retrieval that actually ran.
        self._last_metrics = None
        # Combiner contract: ``additionals`` is a TableId IN/NOT IN predicate.
        # Parse and push it into retrieval so top-k is taken inside the
        # allow-list (not post-hoc filtered), then wrap the ranked id list
        # as a synthetic VALUES table for Operator.run. The cache key
        # includes the parsed filter so a different ``additionals`` re-fetches.
        table_filter = parse_additionals(additionals)
        cache_key = (
            self.query,
            self.k,
            self._n,
            self._alpha,
            self._index_name,
            self._rerank,
            self._judge_concurrency,
            table_filter.cache_key(),
        )
        if self._cached_table_ids is None or self._cached_table_ids[0] != cache_key:
            engine = _NLEngine.get(self._cfg, index_name=self._index_name)
            results, metrics = engine.search_with_metrics(
                self.query,
                k=self.k,
                n=self._n,
                alpha=self._alpha,
                table_filter=table_filter,
                rerank=self._rerank,
                judge_concurrency=self._judge_concurrency,
            )
            self._last_metrics = metrics
            ids = [r.table_id for r in results]
            self._cached_table_ids = (cache_key, ids)
        _, ids = self._cached_table_ids
        return values_select_with_rank(db, ids, k=self.k)

    def cost(self) -> int:
        # Blend hard-codes the rule-based ordering:
        # KW=3 < SC=4 < C=6 < MC=10 ("MC always executes last"). LLM rerank
        # dwarfs every in-DB seeker, so NLSeeker sits one above MC and
        # therefore sorts last in any Intersection.
        return 11

    def ml_cost(self, db: DBHandler) -> float:
        # Tiebreaker between same-cost seekers; NLSeeker has no peer at
        # cost==11, so this is unreachable in practice. The paper's
        # cardinality / column-count / value-frequency regression doesn't
        # apply to a natural-language query.
        return 1.0
