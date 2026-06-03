import re
from dataclasses import dataclass

# Typing imports
from typing import Optional


_CLAUSE_RE = re.compile(
    r"^\s*"
    r"(?:[A-Za-z_][A-Za-z0-9_]*\.)?"
    r"TableId\s+(NOT\s+)?IN\s*\(\s*"
    r"(\d+(?:\s*,\s*\d+)*)"
    r"\s*\)\s*$",
    re.IGNORECASE,
)

# Word-boundaried so we don't split on e.g. ``ANDREW``. The grammar has no
# parentheses outside the IN-list, so a flat scan is sufficient.
_AND_SPLIT_RE = re.compile(r"\s+AND\s+", re.IGNORECASE)

# Combiners always prepend `" AND "`; OR is tolerated for symmetry. Bare
# connectives must collapse so input like ``" AND "`` parses to empty.
_LEADING_CONN_RE = re.compile(r"^(?:\s*\b(?:AND|OR)\b\s*)+", re.IGNORECASE)
_TRAILING_CONN_RE = re.compile(r"(?:\s*\b(?:AND|OR)\b\s*)+$", re.IGNORECASE)


@dataclass(frozen=True)
class TableFilter:
    """``allow=None`` is unconstrained; otherwise an explicit allow-list. Same for ``deny``."""

    allow: Optional[frozenset]
    deny: Optional[frozenset]

    def is_empty(self) -> bool:
        return self.allow is None and self.deny is None

    def permits(self, table_id: int) -> bool:
        if self.allow is not None and table_id not in self.allow:
            return False
        if self.deny is not None and table_id in self.deny:
            return False
        return True

    def cache_key(self) -> tuple:
        return (self.allow, self.deny)


EMPTY_FILTER: TableFilter = TableFilter(allow=None, deny=None)


def _parse_id_list(group: str) -> frozenset:
    return frozenset(int(tok) for tok in group.split(","))


def parse_additionals(s: str) -> TableFilter:
    """Multiple ``IN`` clauses intersect into ``allow``; ``NOT IN`` clauses union into ``deny``."""
    if s is None:
        return EMPTY_FILTER
    text = s.strip()
    if not text:
        return EMPTY_FILTER
    text = _LEADING_CONN_RE.sub("", text)
    text = _TRAILING_CONN_RE.sub("", text)
    if not text.strip():
        return EMPTY_FILTER

    allow: Optional[frozenset] = None
    deny: Optional[frozenset] = None

    for clause in _AND_SPLIT_RE.split(text):
        clause = clause.strip()
        if not clause:
            continue
        m = _CLAUSE_RE.match(clause)
        if m is None:
            raise ValueError(
                f"Only TableId IN/NOT IN (...) predicates are supported in "
                f"NLSeeker additionals; got: {clause!r}"
            )
        is_not = m.group(1) is not None
        ids = _parse_id_list(m.group(2))
        if is_not:
            deny = ids if deny is None else (deny | ids)
        else:
            allow = ids if allow is None else (allow & ids)

    return TableFilter(allow=allow, deny=deny)


__all__ = ["TableFilter", "EMPTY_FILTER", "parse_additionals"]
