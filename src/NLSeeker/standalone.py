from src.NLSeeker.config import NLSeekerConfig
from src.NLSeeker.engine import _NLEngine
from src.NLSeeker.retrieve import RetrievalResult

# Typing imports
from typing import List


class NLSeekerStandalone:
    """Run NLSeeker retrieval without going through a Plan."""

    def __init__(
        self,
        index_name: str = None,
        config_overrides: dict = None,
    ) -> None:
        self._cfg = NLSeekerConfig.load(overrides=config_overrides)
        self._index_name = index_name or self._cfg.index_name

    def search(
        self,
        query: str,
        k: int = None,
        n: int = None,
        alpha: float = None,
    ) -> List[int]:
        engine = _NLEngine.get(self._cfg, index_name=self._index_name)
        results = engine.search(query=query, k=k, n=n, alpha=alpha)
        return [r.table_id for r in results]

    def search_raw(
        self,
        query: str,
        k: int = None,
        n: int = None,
        alpha: float = None,
    ) -> List[RetrievalResult]:
        """Like ``search`` but returns the full ``RetrievalResult`` per table."""
        engine = _NLEngine.get(self._cfg, index_name=self._index_name)
        return engine.search(query=query, k=k, n=n, alpha=alpha)


def nl_search(query: str, k: int = 10) -> List[int]:
    return NLSeekerStandalone().search(query, k=k)


__all__ = ["NLSeekerStandalone", "nl_search"]
