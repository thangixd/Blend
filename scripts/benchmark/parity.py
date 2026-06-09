from __future__ import annotations

import configparser
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


LOG = logging.getLogger(__name__)


class ParityMismatch(RuntimeError):
    pass


@dataclass(frozen=True)
class ParityResult:
    n_queries: int
    standalone_eq_engine: bool
    engine_ids_eq_plan_ids: bool
    status: str  # "ok" | "fail"


def compare_results(
    qid: str,
    a: Sequence[tuple[int, float]],
    b: Sequence[tuple[int, float]],
) -> None:
    """Exact equality on `(table_id, fused_score)` pairs. Raises on diff."""
    if len(a) != len(b):
        raise ParityMismatch(
            f"q={qid}: length mismatch {len(a)} vs {len(b)}: a={list(a)} b={list(b)}"
        )
    for i, (x, y) in enumerate(zip(a, b)):
        if x[0] != y[0]:
            raise ParityMismatch(
                f"q={qid}: table_id mismatch at position {i}: {x[0]} vs {y[0]}"
            )
        if x[1] != y[1]:
            raise ParityMismatch(
                f"q={qid}: fused_score mismatch at position {i} (table {x[0]}): "
                f"{x[1]!r} vs {y[1]!r}"
            )


def _overrides_from_config(config_path: Path) -> dict:
    parser = configparser.ConfigParser()
    parser.read(config_path)
    if not parser.has_section("NLSeeker"):
        return {}
    return dict(parser.items("NLSeeker"))


def _load_first_n_bc1(content_jsonl: Path, n: int = 5) -> list[tuple[str, str]]:
    """Pull the first n BC1 questions: list of (id, question_from_sql_1)."""
    out = []
    for line in content_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out.append((row["id"], row["question_from_sql_1"]))
        if len(out) >= n:
            break
    return out


def run_parity_check(
    *,
    config_path: Path,
    index_name: str,
    content_jsonl: Path,
    k: int = 50,
    n: int = 5,
    alpha: float = 0.5,
) -> ParityResult:

    from src.NLSeeker.config import NLSeekerConfig
    from src.NLSeeker.engine import _NLEngine, reset_engine_cache
    from src.NLSeeker.llm import reset_backend_cache
    from src.NLSeeker.standalone import NLSeekerStandalone
    from src.Operators.Seekers.NLSeeker import NLSeeker as PlanNLSeeker
    from src.Plan import Plan

    reset_engine_cache()
    reset_backend_cache()

    overrides = _overrides_from_config(config_path)
    cfg = NLSeekerConfig.load(config_path=config_path)
    queries = _load_first_n_bc1(content_jsonl, n=5)
    if not queries:
        raise ParityMismatch(f"parity: no BC1 questions found in {content_jsonl}")

    standalone = NLSeekerStandalone(
        index_name=index_name,
        config_overrides=overrides,
    )

    for qid, q in queries:
        std_results = standalone.search_raw(q, k=k, n=n, alpha=alpha)
        std_pairs = [(r.table_id, r.fused_score) for r in std_results]

        engine = _NLEngine.get(cfg, index_name=index_name)
        eng_results = engine.search(q, k=k, n=n, alpha=alpha)
        eng_pairs = [(r.table_id, r.fused_score) for r in eng_results]

        compare_results(qid, std_pairs, eng_pairs)

        plan = Plan()
        plan.add(
            "nl",
            PlanNLSeeker(
                query=q,
                k=k,
                n=n,
                alpha=alpha,
                index_name=index_name,
                config_overrides=overrides,
            ),
        )
        plan_ids = plan.run()
        engine_ids = [tid for tid, _ in eng_pairs]
        if list(plan_ids) != engine_ids:
            raise ParityMismatch(
                f"q={qid}: plan ids != engine ids: plan={list(plan_ids)} engine={engine_ids}"
            )

    LOG.info("parity OK: %d/%d queries, standalone == engine, engine ids == plan ids",
             len(queries), len(queries))
    return ParityResult(
        n_queries=len(queries),
        standalone_eq_engine=True,
        engine_ids_eq_plan_ids=True,
        status="ok",
    )


__all__ = ["run_parity_check", "compare_results", "ParityMismatch", "ParityResult"]
