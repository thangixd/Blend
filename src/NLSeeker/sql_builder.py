import random

from src.DBHandler import DBHandler

# Typing imports
from typing import Sequence


_OUTER_TEMPLATE = """
SELECT TableId
FROM ({inner}) AS {alias}
WHERE 1=1 $ADDITIONALS$
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
    additionals: str,
    k: int,
) -> str:
    """Wrap a ranked TableId list as a SELECT compatible with Blend's operators.

    NLSeeker retrieves in Python; this preserves rank order via an outer
    ``ORDER BY rank`` and lets ``$ADDITIONALS$`` from a Combiner filter the
    result. Empty inputs emit a query that returns zero rows but parses.
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
    sql = sql.replace("$ADDITIONALS$", additionals or "")
    return sql


__all__ = ["values_select_with_rank"]
