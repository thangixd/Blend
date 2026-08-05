# Typing imports
from typing import List, Set, Tuple


def hit(retrieved: List[str], answers: Set[str]) -> bool:
    return any(table in answers for table in retrieved)


def recall(retrieved: List[str], answers: Set[str]) -> float:
    if not answers:
        return 0.0
    return len(set(retrieved) & answers) / len(answers)


def reciprocal_rank(retrieved: List[str], answers: Set[str]) -> float:
    for rank, table in enumerate(retrieved, start=1):
        if table in answers:
            return 1.0 / rank
    return 0.0


def latency_stats(latencies_ms: List[float]) -> Tuple[float, float, float]:
    """(mean, p50, p95) in milliseconds."""
    if not latencies_ms:
        return 0.0, 0.0, 0.0
    ordered = sorted(latencies_ms)
    count = len(ordered)
    p95_index = max(0, min(count - 1, round(0.95 * (count - 1))))
    return sum(ordered) / count, ordered[count // 2], ordered[p95_index]
