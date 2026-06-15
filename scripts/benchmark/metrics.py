from __future__ import annotations

from typing import Iterable, Sequence


def hit_at_k(retrieved_pneuma_ids: list[str], answer_ids: set[str], k: int) -> bool:
    """True if any of the first ``k`` retrieved ids is in ``answer_ids``."""
    if not answer_ids:
        return False
    for tid in retrieved_pneuma_ids[:k]:
        if tid in answer_ids:
            return True
    return False


def recall_at_k(
    retrieved_pneuma_ids: list[str], answer_ids: set[str], k: int
) -> float:
    """Fraction of ``answer_ids`` that appear in the first ``k`` retrieved ids.

    The retrieved window is de-duplicated before counting matches so that
    a repeated id does not inflate the numerator.
    """
    if not answer_ids:
        return 0.0
    window = set(retrieved_pneuma_ids[:k])
    return len(window & answer_ids) / len(answer_ids)


def reciprocal_rank(
    retrieved_pneuma_ids: Iterable[str], answer_ids: set[str]
) -> float:
    """1 / (1-based rank of the first relevant id), or 0.0 if no hit."""
    if not answer_ids:
        return 0.0
    for i, tid in enumerate(retrieved_pneuma_ids, start=1):
        if tid in answer_ids:
            return 1.0 / i
    return 0.0


def latency_stats(latencies_ms: Sequence[float]) -> tuple[float, float, float]:
    if not latencies_ms:
        return 0.0, 0.0, 0.0
    n = len(latencies_ms)
    mean = sum(latencies_ms) / n
    sorted_ms = sorted(latencies_ms)
    p50 = sorted_ms[n // 2]
    p95_idx = max(0, min(n - 1, int(round(0.95 * (n - 1)))))
    p95 = sorted_ms[p95_idx]
    return float(mean), float(p50), float(p95)


__all__ = ["hit_at_k", "recall_at_k", "reciprocal_rank", "latency_stats"]
