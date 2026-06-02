from src.NLSeeker.config import NLSeekerConfig
from src.NLSeeker.engine import _NLEngine
from src.NLSeeker.retrieve import RetrievalResult


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
    ) -> list:
        """Return up to ``k`` ranked TableIds. Defaults from cfg."""
        engine = _NLEngine.get(self._cfg, index_name=self._index_name)
        return engine.search(query=query, k=k, n=n, alpha=alpha)

    def search_raw(
        self,
        query: str,
        k: int = None,
        n: int = None,
        alpha: float = None,
    ) -> list:
        """Like search() but returns ``RetrievalResult`` per table."""
        engine = _NLEngine.get(self._cfg, index_name=self._index_name)
        return engine.search_raw(query=query, k=k, n=n, alpha=alpha)


def nl_search(query: str, k: int = 10) -> list:
    """One-shot wrapper around NLSeekerStandalone."""
    return NLSeekerStandalone().search(query, k=k)


__all__ = ["NLSeekerStandalone", "nl_search"]
