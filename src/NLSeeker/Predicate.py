"""Parses the combiner `additionals` predicate into a table filter for retrieval."""

import re
from dataclasses import dataclass

# Typing imports
from typing import FrozenSet, Optional

_CLAUSE = re.compile(r'TableId\s+(NOT\s+)?IN\s*\(\s*(\d+(?:\s*,\s*\d+)*)\s*\)', re.IGNORECASE)
_SPLIT = re.compile(r'\s+AND\s+', re.IGNORECASE)
_LEADING_AND = re.compile(r'^AND\s+', re.IGNORECASE)


@dataclass(frozen=True)
class TableFilter:
    """`None` is unconstrained; a set is an explicit allow- or deny-list."""

    allow: Optional[FrozenSet[int]] = None
    deny: Optional[FrozenSet[int]] = None

    def is_empty(self) -> bool:
        return self.allow is None and self.deny is None

    def permits(self, table_id: int) -> bool:
        return ((self.allow is None or table_id in self.allow)
                and (self.deny is None or table_id not in self.deny))


EMPTY_FILTER = TableFilter()


def parse_additionals(additionals: str) -> TableFilter:
    """Intersects the `IN` clauses into an allow-list and unions the `NOT IN` ones into a deny-list."""
    text = _LEADING_AND.sub('', additionals.strip())
    if not text:
        return EMPTY_FILTER

    allow = None
    deny = None
    for clause in _SPLIT.split(text):
        match = _CLAUSE.fullmatch(clause.strip())
        if match is None:
            raise ValueError(f'NLSeeker only understands TableId IN/NOT IN predicates, got {clause!r}')

        table_ids = frozenset(int(table_id) for table_id in match.group(2).split(','))
        if match.group(1) is None:
            allow = table_ids if allow is None else allow & table_ids
        else:
            deny = table_ids if deny is None else deny | table_ids

    return TableFilter(allow, deny)
