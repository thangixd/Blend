from src.Operators.Seekers.SeekerBase import Seeker

# Typing imports
from src.DBHandler import DBHandler
from typing import List, Optional

_retrievers = {}


class NaturalLanguage(Seeker):
    """Answers a natural-language question against the NL index."""

    def __init__(self, query: str, k: int = 10, n: Optional[int] = None,
                 alpha: Optional[float] = None, index_name: Optional[str] = None,
                 rerank: bool = True) -> None:
        super().__init__(k)
        self.query = str(query)
        self.n = n
        self.alpha = alpha
        self.index_name = index_name
        self.rerank = rerank

    def create_sql_query(self, db: DBHandler, additionals: str = "") -> str:
        # A combiner's TableId predicate is pushed into retrieval rather than spliced into the
        # SQL below, so the top-k is taken inside the allow-list instead of after the cut.
        table_ids = self._retrieve(db, additionals)
        if len(table_ids) == 0:
            return "SELECT TableId FROM AllTables WHERE 1=0"

        # table_ids_to_sql drops the ordering, and retrieval rank is the point of this seeker.
        rows = ' UNION ALL '.join(f'SELECT {table_id} AS TableId, {rank} AS nl_rank'
                                  for rank, table_id in enumerate(table_ids))
        return f"""
            SELECT TableId FROM (
            {rows}
            ) AS {db.random_subquery_name()}
            ORDER BY nl_rank
            LIMIT {self.k}
        """

    def cost(self) -> int:
        return 11

    def _features(self, db: DBHandler) -> list:
        n = self.n
        if n is None:
            from src.NLSeeker.Config import NLSeekerConfig
            n = NLSeekerConfig.load(db.config_path).n

        return [len(self.query.split()), self.k * n, int(self.rerank)]

    def _retrieve(self, db: DBHandler, additionals: str) -> List[int]:
        from dataclasses import replace
        from src.NLSeeker.Config import NLSeekerConfig
        from src.NLSeeker.Predicate import parse_additionals

        config = NLSeekerConfig.load(db.config_path)
        if self.index_name is not None:
            config = replace(config, index_name=self.index_name)

        retriever = _load_retriever(config)
        return retriever.retrieve(self.query, k=self.k,
                                  n=config.n if self.n is None else self.n,
                                  alpha=config.alpha if self.alpha is None else self.alpha,
                                  table_filter=parse_additionals(additionals),
                                  rerank=self.rerank)


def _load_retriever(config):
    """Keeps one Retriever per index so a plan does not reload Chroma and BM25s per operator."""
    from src.NLSeeker.Clients import EmbeddingClient, LLMClient
    from src.NLSeeker.Retriever import Retriever

    key = (str(config.out_path), config.index_name)
    if key not in _retrievers:
        _retrievers[key] = Retriever(config, LLMClient(config), EmbeddingClient(config))
    return _retrievers[key]
