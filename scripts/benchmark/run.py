from __future__ import annotations

import csv as _csv
import hashlib
import json
import logging
import resource
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from tqdm import tqdm

from scripts.benchmark.parity import _overrides_from_config


LOG = logging.getLogger(__name__)


_FIELD_BY_FAMILY = {
    "BC1": ("content", "question_from_sql_1"),
    "BC2": ("content", "question"),
    "BX1": ("bx", "question_bx1"),
    "BX2": ("bx", "question_bx2"),
}

_ID_BY_SOURCE = {
    "content": "id",
    "bx": "context_id",
}


@dataclass(frozen=True)
class Manifest:
    tid_to_pneuma: dict[int, str]
    pneuma_to_tid: dict[str, int]
    n_tables: int

    @classmethod
    def from_dict(cls, raw: dict) -> "Manifest":
        return cls(
            tid_to_pneuma={int(k): v for k, v in raw["table_id_to_pneuma_id"].items()},
            pneuma_to_tid={k: int(v) for k, v in raw["pneuma_id_to_table_id"].items()},
            n_tables=int(raw["n_tables"]),
        )


@dataclass(frozen=True)
class Question:
    qid: str
    family: str
    text: str
    answer_tables: frozenset[str] = field(default_factory=frozenset)


def load_manifest(lake_dir: Path) -> Manifest:
    raw = json.loads((lake_dir / "_manifest.json").read_text())
    return Manifest.from_dict(raw)


def load_questions(
    *,
    family: str,
    content_jsonl: Path,
    bx_jsonl: Path,
) -> list[Question]:
    if family not in _FIELD_BY_FAMILY:
        raise ValueError(f"unknown family {family!r}; expected one of "
                         f"{sorted(_FIELD_BY_FAMILY)}")
    source, qfield = _FIELD_BY_FAMILY[family]
    src_path = content_jsonl if source == "content" else bx_jsonl
    id_field = _ID_BY_SOURCE[source]

    questions: list[Question] = []
    if not src_path.exists():
        return questions
    for line in src_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if qfield not in row or id_field not in row:
            continue
        questions.append(Question(
            qid=str(row[id_field]),
            family=family,
            text=str(row[qfield]),
            answer_tables=frozenset(row.get("answer_tables", [])),
        ))
    return questions


@dataclass(frozen=True)
class PerQueryRecord:
    ts_utc: str
    dataset: str
    family: str
    k: int
    question_id: str
    question: str
    retrieved: list[dict]
    answer_tables: list[str]
    hit: bool
    recall: float
    rr: float
    latency_ms: float


@dataclass(frozen=True)
class SummaryRow:
    dataset: str
    family: str
    k: int
    n: int
    alpha: float
    n_questions: int
    hit_rate: float       # percentage, 2 decimals (e.g. 73.50)
    recall_at_k: float    # 0..1, 4 decimals
    mrr: float            # 0..1, 4 decimals


def write_per_query_jsonl(path: Path, rows: Iterable[PerQueryRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            obj = {
                "ts_utc": r.ts_utc, "dataset": r.dataset, "family": r.family,
                "k": r.k, "question_id": r.question_id, "question": r.question,
                "retrieved": r.retrieved, "answer_tables": list(r.answer_tables),
                "hit": r.hit, "recall": r.recall, "rr": r.rr,
                "latency_ms": r.latency_ms,
            }
            f.write(json.dumps(obj) + "\n")


def write_summary_csv(path: Path, rows: Iterable[SummaryRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["dataset", "family", "k", "n", "alpha",
                    "n_questions", "hit_rate", "recall_at_k", "mrr"])
        for r in rows:
            w.writerow([
                r.dataset, r.family, r.k, r.n, f"{r.alpha:.1f}",
                r.n_questions, f"{r.hit_rate:.2f}",
                f"{r.recall_at_k:.4f}", f"{r.mrr:.4f}",
            ])


def write_pneuma_compat_jsonl(path: Path, rows: Iterable[SummaryRow]) -> None:
    """One line per (dataset, family, k) in PNEUMA's hybrid-...jsonl shape."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            n_hits = round(r.hit_rate * r.n_questions / 100)
            obj = {
                "dataset": r.dataset, "benchmark_name": r.family,
                "k": r.k, "n": r.n, "alpha": r.alpha,
                "hitrate": r.hit_rate, "sum": n_hits,
            }
            f.write(json.dumps(obj) + "\n")


def aggregate_summary(
    records: Sequence[PerQueryRecord], *, n: int, alpha: float
) -> list[SummaryRow]:
    """Aggregate per-query records into one row per (dataset, family, k)."""
    buckets: dict[tuple[str, str, int], list[PerQueryRecord]] = defaultdict(list)
    for r in records:
        buckets[(r.dataset, r.family, r.k)].append(r)
    out: list[SummaryRow] = []
    for (dataset, family, k), group in sorted(buckets.items()):
        n_q = len(group)
        if n_q == 0:
            continue
        hits = sum(1 for r in group if r.hit)
        out.append(SummaryRow(
            dataset=dataset, family=family, k=k, n=n, alpha=alpha,
            n_questions=n_q,
            hit_rate=round(hits / n_q * 100, 2),
            recall_at_k=round(sum(r.recall for r in group) / n_q, 4),
            mrr=round(sum(r.rr for r in group) / n_q, 4),
        ))
    return out


K_VALUES = [1, 5, 10, 30, 50]
N = 5
ALPHA = 0.5
FAMILIES = ["BC1", "BC2", "BX1", "BX2"]

# Each family runs in its own subprocess against its own vLLM container.
# Order in this dict matters: matches FAMILIES order so tqdm position=
# slots are stable. The embedder is shared across families on :8002.
FAMILY_LLM_URLS = {
    "BC1": "http://127.0.0.1:8001/v1",
    "BC2": "http://127.0.0.1:8003/v1",
    "BX1": "http://127.0.0.1:8004/v1",
    "BX2": "http://127.0.0.1:8005/v1",
}
EMBED_URL = "http://127.0.0.1:8002/v1"


def _git_commit_dirty() -> tuple[str, bool]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = bool(subprocess.run(
            ["git", "diff", "--quiet"], capture_output=True
        ).returncode)
        return sha, dirty
    except Exception:
        return "unknown", False


def _gpu_info() -> str:
    try:
        return subprocess.check_output(["nvidia-smi", "-L"], text=True).strip()
    except Exception:
        return "no-nvidia-smi"


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _config_ini_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_run_meta(
    *,
    dataset: str,
    config_path: Path,
    endpoints_meta: dict,
    k_values: list[int],
    n: int,
    alpha: float,
    n_questions: dict[str, int],
    build_wall_clock_s: float,
    run_wall_clock_s: float,
    unresolvable: int,
    dropped_context_rows: int,
) -> dict:
    sha, dirty = _git_commit_dirty()
    return {
        "schema_version": 1,
        "ts_utc": _now_utc_iso(),
        "dataset": dataset,
        "git_commit": sha,
        "git_dirty": dirty,
        "config_ini_sha256": _config_ini_sha256(config_path),
        "endpoints": endpoints_meta,
        "gpu": _gpu_info(),
        "k_values": k_values,
        "n": n,
        "alpha": alpha,
        "slicing_method": "per_k_search_pneuma_parity",
        "retrieval_path": "Plan(PlanNLSeeker).run()",
        "concurrency_mode": f"per_family_subprocess (4× vLLM @ {','.join(FAMILY_LLM_URLS.values())})",
        "n_questions": n_questions,
        "build_wall_clock_s": build_wall_clock_s,
        "run_wall_clock_s": run_wall_clock_s,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "unresolvable_questions_count": unresolvable,
        "dropped_context_rows": dropped_context_rows,
        "parity_deltas": [
            {"id": "D2", "name": "Reranker runtime",
             "nlseeker": "vLLM v0.6.6.post1 HTTP at http://127.0.0.1:8001/v1",
             "pneuma": "HuggingFace transformers prompt_pipeline in-process",
             "rationale": "Same Qwen2.5-7B-Instruct weights, same greedy decode "
                          "(temp=0). Sub-percent yes/no decision divergence "
                          "expected and disclosed."},
            {"id": "D4", "name": "Vector retrieval at query time",
             "nlseeker": "brute-force exact cosine via numpy matmul over all "
                         "embeddings (retrieve.py:198-228)",
             "pneuma": "Chroma HNSW approximate cosine (hybrid_search.py:96-98)",
             "rationale": "Intended NLSeeker design — exact retrieval at NL "
                          "query time on small/medium lakes; HNSW artifact "
                          "retained for compatibility."},
        ],
        "vllm_seed_note": (
            "NLSeeker's _OpenAILLM omits the seed= field when base_url is "
            "non-empty (src/NLSeeker/llm.py:215-217); greedy decode at "
            "temperature=0 is still deterministic given identical input prompts."
        ),
        "status": "ok",
    }


def _run_family(args: dict) -> dict:
    family = args["family"]
    dataset = args["dataset"]
    lake_dir = Path(args["lake_dir"])
    config_path = Path(args["config_path"])
    index_name = args["index_name"]
    content_jsonl = Path(args["content_jsonl"])
    bx_jsonl = Path(args["bx_jsonl"])
    base_overrides = dict(args["base_overrides"])
    llm_base_url = args["llm_base_url"]
    embed_base_url = args["embed_base_url"]
    position = int(args["position"])
    max_questions = args["max_questions"]

    from src.NLSeeker.engine import reset_engine_cache
    from src.NLSeeker.llm import reset_backend_cache
    from src.Operators.Seekers.NLSeeker import NLSeeker as PlanNLSeeker
    from src.Plan import Plan

    from scripts.benchmark.metrics import hit_at_k, recall_at_k, reciprocal_rank

    reset_engine_cache()
    reset_backend_cache()

    manifest = load_manifest(lake_dir)
    questions = load_questions(
        family=family,
        content_jsonl=content_jsonl,
        bx_jsonl=bx_jsonl,
    )
    if max_questions is not None:
        questions = questions[:max_questions]

    if not questions:
        LOG.warning("family %s: no questions found, skipping", family)
        return {"family": family, "n_questions": 0, "records": []}

    overrides = {
        **base_overrides,
        "openai_base_url": llm_base_url,
        "openai_embed_base_url": embed_base_url,
    }

    per_query: list[PerQueryRecord] = []
    family_hits = {k: 0 for k in K_VALUES}
    bar = tqdm(
        questions,
        desc=f"{dataset} {family}",
        unit="q",
        dynamic_ncols=True,
        leave=True,
        position=position,
    )
    for q in bar:
        answer_set = set(q.answer_tables)
        # Production path: build a Plan with one NLSeeker operator
        # per (question, k) and call plan.run(). This is exactly how
        # Blend is invoked in production with other seekers.
        #
        # PNEUMA-parity: one search per k, so the rerank pool is k*n at
        # each reported k (PNEUMA's hybrid_search.py:71 loops with
        # increased_k = k * n; our search() reranks the full k*n pool
        # before truncating to k).
        for k in K_VALUES:
            bar.set_postfix_str(f"k={k}", refresh=True)
            plan = Plan()
            plan.add(
                "nl",
                PlanNLSeeker(
                    query=q.text,
                    k=k,
                    n=N,
                    alpha=ALPHA,
                    index_name=index_name,
                    config_overrides=overrides,
                ),
            )
            t0 = time.perf_counter()
            table_ids = plan.run()
            latency_ms = (time.perf_counter() - t0) * 1000.0

            retrieved_dicts = []
            retrieved_pneuma: list[str] = []
            for tid in table_ids:
                pneuma_id = manifest.tid_to_pneuma.get(int(tid))
                if pneuma_id is None:
                    LOG.warning(
                        "q=%s: retrieved table_id %d not in manifest",
                        q.qid, tid,
                    )
                    continue
                retrieved_dicts.append({
                    "table_id": int(tid),
                    "pneuma_id": pneuma_id,
                })
                retrieved_pneuma.append(pneuma_id)

            hit = hit_at_k(retrieved_pneuma, answer_set, k)
            if hit:
                family_hits[k] += 1
            per_query.append(PerQueryRecord(
                ts_utc=_now_utc_iso(), dataset=dataset, family=family, k=k,
                question_id=q.qid, question=q.text,
                retrieved=retrieved_dicts,
                answer_tables=sorted(q.answer_tables),
                hit=hit,
                recall=recall_at_k(retrieved_pneuma, answer_set, k),
                rr=reciprocal_rank(retrieved_pneuma, answer_set),
                latency_ms=latency_ms,
            ))
        n_done = bar.n + 1  # this question is finishing
        bar.set_postfix_str(
            " ".join(
                f"hit@{k}={family_hits[k] / n_done:.2f}" for k in K_VALUES
            ),
            refresh=False,
        )
    bar.close()

    return {
        "family": family,
        "n_questions": len(questions),
        "records": per_query,
    }


def run_benchmark(
    *,
    dataset: str,
    lake_dir: Path,
    config_path: Path,
    index_name: str,
    content_jsonl: Path,
    results_dir: Path,
    endpoints_meta: dict,
    build_wall_clock_s: float,
    unresolvable: int,
    dropped_context_rows: int,
    max_questions: int | None = None,
) -> Path:
    """End-to-end run for one prepared+indexed dataset.

    Fans out one subprocess per family (BC1/BC2/BX1/BX2), each pinned to
    its own vLLM container via ``FAMILY_LLM_URLS``. Returns the
    timestamped results directory it wrote to.
    """
    import multiprocessing as mp

    # Validate the manifest exists before spawning workers (each worker
    # also loads it, but failing fast here gives a single, clean error).
    load_manifest(lake_dir)
    bx_jsonl = lake_dir / "_bx_questions.jsonl"

    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = results_dir / dataset / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    overrides = _overrides_from_config(config_path)

    worker_args = [
        {
            "family": family,
            "dataset": dataset,
            "lake_dir": str(lake_dir),
            "config_path": str(config_path),
            "index_name": index_name,
            "content_jsonl": str(content_jsonl),
            "bx_jsonl": str(bx_jsonl),
            "base_overrides": overrides,
            "llm_base_url": FAMILY_LLM_URLS[family],
            "embed_base_url": EMBED_URL,
            "position": idx,
            "max_questions": max_questions,
        }
        for idx, family in enumerate(FAMILIES)
    ]

    t_run0 = time.perf_counter()
    # 'spawn' (vs the Linux default 'fork') gives each worker a clean
    # interpreter — no inherited engine cache, no shared httpx clients.
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(FAMILIES)) as pool:
        family_results = pool.map(_run_family, worker_args)
    run_wall = time.perf_counter() - t_run0

    per_query: list[PerQueryRecord] = []
    n_questions: dict[str, int] = {}
    for fr in family_results:
        per_query.extend(fr["records"])
        n_questions[fr["family"]] = fr["n_questions"]
    # Sort so per_query.jsonl ordering is stable across runs.
    per_query.sort(key=lambda r: (r.family, r.question_id, r.k))

    write_per_query_jsonl(run_dir / "per_query.jsonl", per_query)
    summary = aggregate_summary(per_query, n=N, alpha=ALPHA)
    write_summary_csv(run_dir / "summary.csv", summary)
    write_pneuma_compat_jsonl(run_dir / "pneuma_compat.jsonl", summary)

    meta = _build_run_meta(
        dataset=dataset,
        config_path=config_path,
        endpoints_meta=endpoints_meta,
        k_values=K_VALUES,
        n=N,
        alpha=ALPHA,
        n_questions=n_questions,
        build_wall_clock_s=build_wall_clock_s,
        run_wall_clock_s=run_wall,
        unresolvable=unresolvable,
        dropped_context_rows=dropped_context_rows,
    )
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    LOG.info("benchmark complete: %s", run_dir)
    return run_dir


__all__ = [
    "Manifest", "Question", "load_manifest", "load_questions",
    "PerQueryRecord", "SummaryRow",
    "write_per_query_jsonl", "write_summary_csv", "write_pneuma_compat_jsonl",
    "aggregate_summary",
    "K_VALUES", "N", "ALPHA", "FAMILIES", "run_benchmark",
]
