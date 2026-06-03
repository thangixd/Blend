import random

from src.DBHandler import DBHandler

# Typing imports
from typing import Sequence


# Filter pushdown happens inside retrieval (see NLSeeker.create_sql_query and
# predicate.py), so the wrapper is just a rank-preserving SELECT plus the
# LIMIT-k invariant Operators rely on. Some dialects don't preserve VALUES
# row order - hence the explicit ORDER BY rank.
_OUTER_TEMPLATE = """
SELECT TableId
FROM ({inner}) AS {alias}
ORDER BY rank ASC
LIMIT $TOPK$
"""

_EMPTY_TEMPLATE = "SELECT TableId FROM (SELECT 0 AS TableId, 0 AS rank WHERE 1=0) AS {alias}"


def _alias() -> str:
    return f"nl_subquery{random.random() * 1_000_000:.0f}"


def _values_inner(dbms: str, rows: Sequence, alias: str) -> str:
    if dbms == "postgres":
        body = ", ".join(f"({tid}, {rank})" for tid, rank in rows)
        return f"SELECT * FROM (VALUES {body}) AS {alias}_v(TableId, rank)"
    if dbms == "vertica":
        # Vertica has no VALUES-table literal; zip parallel arrays via Explode.
        ids = ", ".join(str(tid) for tid, _ in rows)
        ranks = ", ".join(str(r) for _, r in rows)
        return (
            f"SELECT TableId, rank FROM ("
            f"  SELECT Explode(Array[{ids}]) OVER (Partition Best) AS (i1, TableId)"
            f") {alias}_t1 "
            f"JOIN ("
            f"  SELECT Explode(Array[{ranks}]) OVER (Partition Best) AS (i2, rank)"
            f") {alias}_t2 ON {alias}_t1.i1 = {alias}_t2.i2"
        )
    body = " UNION ALL ".join(
        f"SELECT {tid} AS TableId, {rank} AS rank" for tid, rank in rows
    )
    return body


def values_select_with_rank(
    db: DBHandler,
    table_ids: Sequence[int],
    k: int,
) -> str:
    """Wrap a ranked TableId list as a SELECT compatible with Blend's operators.

    Filtering is done inside retrieval (see ``parse_additionals``), so no
    ``$ADDITIONALS$`` substitution happens here. Empty inputs emit a
    query that returns zero rows but parses.
    """
    seen = {}
    for i, tid in enumerate(table_ids):
        if tid not in seen:
            seen[int(tid)] = i
    rows = sorted(seen.items(), key=lambda kv: kv[1])

    alias = _alias()
    if not rows:
        sql = _EMPTY_TEMPLATE.format(alias=alias)
    else:
        inner = _values_inner(db.dbms, rows, alias)
        sql = _OUTER_TEMPLATE.format(inner=inner, alias=alias)

    sql = sql.replace("$TOPK$", str(int(k)))
    return sql


__all__ = ["values_select_with_rank"]
