from src.Operators.Seekers.SeekerBase import Seeker
from src.NLSeeker.config import NLSeekerConfig
from src.NLSeeker.engine import _NLEngine
from src.NLSeeker.sql_builder import values_select_with_rank

# Typing imports
from src.DBHandler import DBHandler


class NLSeeker(Seeker):
    # The 3-feature runtime regressor (distinct rows, geomean token freq, ncols)
    # has no meaningful inputs for a natural-language query, so NLSeeker opts
    # out of the XGB model load in Seeker.__init__ and uses a constant cost.
    HAS_ML_COST_MODEL = False

    def __init__(
        self,
        query: str,
        k: int = 10,
        n: int = None,
        alpha: float = None,
        index_name: str = None,
        config_overrides: dict = None,
    ) -> None:
        super().__init__(k)

        self.query = str(query)
        self._cfg = NLSeekerConfig.load(overrides=config_overrides)
        self._index_name = index_name or self._cfg.index_name
        self._n = self._cfg.n if n is None else int(n)
        self._alpha = self._cfg.alpha if alpha is None else float(alpha)

        self._cached_table_ids = None

    def create_sql_query(self, db: DBHandler, additionals: str = "") -> str:
        # Retrieval (BM25 + vector + LLM rerank) runs in Python; the ranked
        # TableId list is wrapped as a synthetic VALUES table so Operator.run
        # can execute it and $ADDITIONALS$ from a Combiner still filters.
        cache_key = (self.query, self.k, self._n, self._alpha, self._index_name)
        if self._cached_table_ids is None or self._cached_table_ids[0] != cache_key:
            engine = _NLEngine.get(self._cfg, index_name=self._index_name)
            ids = engine.search(self.query, k=self.k, n=self._n, alpha=self._alpha)
            self._cached_table_ids = (cache_key, ids)
        _, ids = self._cached_table_ids
        return values_select_with_rank(db, ids, additionals=additionals, k=self.k)

    def cost(self) -> int:
        # Blend's paper (§VII) hard-codes the rule-based ordering as
        # KW=3 < SC=4 < C=6 < MC=10 ("MC always executes last"). The
        # LLM-rerank cost dwarfs every in-DB seeker, so NLSeeker sits one
        # above MC and sorts last in any Intersection.
        return 11

    def ml_cost(self, db: DBHandler) -> float:
        # Tiebreaker between same-cost seekers; NLSeeker has no peer at
        # cost==11 so this is unreachable in practice. The paper's 3-feature
        # regression (cardinality, columns, value frequency in lake) does
        # not apply to a natural-language query.
        return 1.0
