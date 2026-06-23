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

from scripts.benchmark._pneuma_compat_writer import write_pneuma_compat_jsonl
from scripts.benchmark.metrics import latency_stats
from scripts.benchmark.parity import _overrides_from_config


LOG = logging.getLogger(__name__)


def seed_all(seed: int = 42) -> None:
    """Seed every RNG the benchmark touches.

    PYTHONHASHSEED only affects hash(); python random / numpy / torch /
    transformers are independent. Call once per family worker startup.
    Optional torch / transformers imports are tolerated (test envs).
    """
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    # CUDA init failures (OOM, driver issues) are silently caught here;
    # determinism is best-effort when torch/transformers are unavailable or broken.
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    try:
        from transformers import set_seed as _hf_set_seed
        _hf_set_seed(seed)
    except Exception:
        pass


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
class _IndexProbe:
    """Lightweight snapshot of the loaded embedding matrix for run_meta.json.

    Constructed from a live ``_VectorIndex`` instance in ``_run_family`` and
    serialised back to the parent process via multiprocessing pickle.
    All fields are plain Python scalars / strings so pickle round-trips safely.
    """
    embedding_matrix_dtype: str
    embedding_matrix_shape: list   # list, not tuple - JSON-serialisable
    embedding_matrix_bytes: int

    @classmethod
    def from_vector_index(cls, vector_index) -> "_IndexProbe":
        mat = getattr(vector_index, "_all_embeddings", None)
        if mat is None:
            return cls(
                embedding_matrix_dtype="unknown",
                embedding_matrix_shape=[0, 0],
                embedding_matrix_bytes=0,
            )
        return cls(
            embedding_matrix_dtype=str(mat.dtype),
            embedding_matrix_shape=list(mat.shape),
            embedding_matrix_bytes=int(mat.nbytes),
        )


@dataclass(frozen=True)
class PerQueryRecord:
    ts_utc: str
    dataset: str
    family: str
    k: int
    rerank_mode: str          # "off" | "on"
    question_id: str
    question: str
    retrieved: list[dict]
    answer_tables: list[str]
    hit: bool
    recall: float
    rr: float
    latency_ms: float
    vector_ms: float | None = None
    # Judge timing - populated when rerank_mode=="on", zeros when "off".
    judge_ms: float = 0.0
    judge_calls: int = 0
    judge_tokens_in: int = 0
    judge_tokens_out: int = 0


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
    latency_ms_mean: float = 0.0   # ms, 1 decimal
    latency_ms_p50: float = 0.0    # ms, 1 decimal
    latency_ms_p95: float = 0.0    # ms, 1 decimal
    vector_ms_mean: float = 0.0    # ms, 2 decimals
    vector_ms_p50: float = 0.0     # ms, 2 decimals
    vector_ms_p95: float = 0.0     # ms, 2 decimals
    judge_ms_mean: float = 0.0
    judge_ms_p50: float = 0.0
    judge_ms_p95: float = 0.0
    judge_calls_mean: float = 0.0
    rerank_mode: str = "on"        # default preserves single-mode legacy runs


def write_per_query_jsonl(path: Path, rows: Iterable[PerQueryRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            obj = {
                "ts_utc": r.ts_utc, "dataset": r.dataset, "family": r.family,
                "k": r.k, "rerank_mode": r.rerank_mode,
                "question_id": r.question_id, "question": r.question,
                "retrieved": r.retrieved, "answer_tables": list(r.answer_tables),
                "hit": r.hit, "recall": r.recall, "rr": r.rr,
                "latency_ms": r.latency_ms,
                "vector_ms": r.vector_ms,
                "judge_ms": r.judge_ms,
                "judge_calls": r.judge_calls,
                "judge_tokens_in": r.judge_tokens_in,
                "judge_tokens_out": r.judge_tokens_out,
            }
            f.write(json.dumps(obj) + "\n")


def write_summary_csv(path: Path, rows: Iterable[SummaryRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["dataset", "family", "k", "n", "alpha",
                    "n_questions", "hit_rate", "recall_at_k", "mrr",
                    "latency_ms_mean", "latency_ms_p50", "latency_ms_p95",
                    "vector_ms_mean", "vector_ms_p50", "vector_ms_p95",
                    "judge_ms_mean", "judge_ms_p50", "judge_ms_p95",
                    "judge_calls_mean",
                    "rerank_mode"])
        for r in rows:
            w.writerow([
                r.dataset, r.family, r.k, r.n, f"{r.alpha:.1f}",
                r.n_questions, f"{r.hit_rate:.2f}",
                f"{r.recall_at_k:.4f}", f"{r.mrr:.4f}",
                f"{r.latency_ms_mean:.1f}", f"{r.latency_ms_p50:.1f}", f"{r.latency_ms_p95:.1f}",
                f"{r.vector_ms_mean:.2f}", f"{r.vector_ms_p50:.2f}", f"{r.vector_ms_p95:.2f}",
                r.judge_ms_mean, r.judge_ms_p50, r.judge_ms_p95,
                r.judge_calls_mean,
                r.rerank_mode,
            ])



def aggregate_summary(
    records: Sequence[PerQueryRecord], *, n: int, alpha: float
) -> list[SummaryRow]:
    """Aggregate per-query records into one row per (dataset, family, k, rerank_mode)."""
    buckets: dict[tuple[str, str, int, str], list[PerQueryRecord]] = defaultdict(list)
    for r in records:
        buckets[(r.dataset, r.family, r.k, r.rerank_mode)].append(r)
    out: list[SummaryRow] = []
    for (dataset, family, k, rerank_mode), group in sorted(buckets.items()):
        n_q = len(group)
        if n_q == 0:
            continue
        hits = sum(1 for r in group if r.hit)
        lat_mean, lat_p50, lat_p95 = latency_stats([r.latency_ms for r in group])
        vec_samples = [r.vector_ms for r in group if r.vector_ms is not None]
        v_mean, v_p50, v_p95 = latency_stats(vec_samples)
        j_mean, j_p50, j_p95 = latency_stats([r.judge_ms for r in group])
        judge_calls_mean = sum(r.judge_calls for r in group) / n_q
        out.append(SummaryRow(
            dataset=dataset, family=family, k=k, n=n, alpha=alpha,
            n_questions=n_q,
            hit_rate=round(hits / n_q * 100, 2),
            recall_at_k=round(sum(r.recall for r in group) / n_q, 4),
            mrr=round(sum(r.rr for r in group) / n_q, 4),
            latency_ms_mean=round(lat_mean, 1),
            latency_ms_p50=round(lat_p50, 1),
            latency_ms_p95=round(lat_p95, 1),
            vector_ms_mean=round(v_mean, 2),
            vector_ms_p50=round(v_p50, 2),
            vector_ms_p95=round(v_p95, 2),
            judge_ms_mean=round(j_mean, 2),
            judge_ms_p50=round(j_p50, 2),
            judge_ms_p95=round(j_p95, 2),
            judge_calls_mean=round(judge_calls_mean, 2),
            rerank_mode=rerank_mode,
        ))
    return out


K_VALUES = [1, 5, 10, 30, 50]
N = 5
ALPHA = 0.5
FAMILIES = ["BC1", "BC2", "BX1", "BX2"]
RERANK_MODES_DEFAULT = ("off", "on")

# Each family runs in its own subprocess against its own vLLM container.
# Sequential mode: all four families share one vLLM container on :8001.
# Pre-Phase-8 layout used :8001/:8003/:8004/:8005 with one container per family
# (mp.spawn Pool); Phase 8 dropped the pool, so a single endpoint is sufficient
# and matches the reduced-deployment requirement (boot one Qwen on :8001 + one
# embedder on :8002, nothing else).
FAMILY_LLM_URLS = {
    "BC1": "http://127.0.0.1:8001/v1",
    "BC2": "http://127.0.0.1:8001/v1",
    "BX1": "http://127.0.0.1:8001/v1",
    "BX2": "http://127.0.0.1:8001/v1",
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


def _gpu_info_dict() -> dict:
    """Return GPU info as a dict with count/model keys (schema_version=2)."""
    try:
        lines = subprocess.check_output(["nvidia-smi", "-L"], text=True).strip().splitlines()
        count = len(lines)
        model = lines[0].split(":")[1].strip().split("(")[0].strip() if lines else ""
        return {"count": count, "model": model}
    except Exception:
        return {"count": 0, "model": "no-nvidia-smi"}


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _config_ini_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_PARITY_DELTAS: list[dict] = [
    {"id": "D9", "title": "Context source", "status": "closed"},
    {"id": "D10", "title": "Context blocker tokenizer", "status": "closed"},
    {"id": "D11", "title": "PNEUMA OpenAI judge inherits T=0.7", "status": "closed-by-routing"},
    {"id": "D12", "title": "PNEUMA tiktoken/gpt-4o tokenizer for block packing", "status": "closed-by-routing"},
    {"id": "D13", "title": "PNEUMA OpenAI summarizer inherits T=0.7", "status": "closed-by-routing"},
    {"id": "D14", "title": "Blend always indexes contexts", "status": "documented"},
    {"id": "D15", "title": "Doc-kind canonical enum", "status": "closed"},
    {"id": "D16", "title": "vLLM seed=42 unconditional", "status": "closed"},
    {"id": "D17", "title": "Benchmark seeds transformers/torch/numpy", "status": "closed"},
    {"id": "D18", "title": "Brute-force vs HNSW", "status": "documented-with-latency-measured"},
    {"id": "D3", "title": "Block packing strict-< vs ≤max-8", "status": "closed"},
    {"id": "D6", "title": "Column-list prompt format", "status": "closed"},
    {"id": "D7", "title": "NaN row formatting", "status": "closed"},
    {"id": "stopwords-asymmetry", "title": "Query-time stopwords", "status": "documented-marginal"},
]


def _build_run_meta(*, dataset, endpoints, gpu, n_questions,
                    build_wall_clock_s, run_wall_clock_s, peak_rss_bytes,
                    unresolvable, dropped_context_rows, status,
                    index_probe=None,
                    rerank_modes=("off", "on"),
                    k_values=(1, 5, 10, 30, 50),
                    judge_model_id=None,
                    embedder_model_id=None,
                    config_path: Path | None = None,
                    embed_endpoint: str = "",
                    **kwargs):
    # TODO: probe importlib.metadata.version("vllm") instead of relying on caller.
    vllm_version = kwargs.get("vllm_version")
    deployment = {
        "mode": "benchmark_sequential",
        "gpu_count": gpu.get("count", 0),
        "gpu_model": gpu.get("model", ""),
        "gpus_in_use_concurrently": 1,
        "inference_engine": "vllm",
        "vllm_version": vllm_version,
        "judge_thread_concurrency": 1,
        "family_parallelism": 1,
        "family_to_endpoint": dict(endpoints),
        "embed_endpoint": embed_endpoint,
        "build_workers": 1,
        "value_index_built": False,
        # Renamed ``vector_index`` → ``vector_index_kind`` for parity with
        # the PNEUMA bench's ``run_meta.json``: both sides now expose the
        # same field name so ``compare_v2`` can pivot panels by it without
        # special-casing.  Value space:
        #   ``brute_force_exact``    - Blend NLSeeker (this side)
        #   ``chromadb_hnsw_M48``    - PNEUMA bench
        "vector_index_kind": "brute_force_exact",
        "context_source": "contexts_<ds>_merged.jsonl",
        "context_blocking": "one-merged-record-per-chunk",
        "judge_model_id": judge_model_id,
        "embedder_model_id": embedder_model_id,
        "seeds": {
            "PYTHONHASHSEED": 0, "set_seed": 42, "torch_manual_seed": 42,
            "numpy_random_seed": 42, "vllm_seed": 42,
            "hnsw_random_seed": 42, "row_sample_random_state": 0,
        },
    }
    if index_probe is not None:
        deployment["embedding_matrix_dtype"] = str(getattr(index_probe, "embedding_matrix_dtype", "unknown"))
        shape = getattr(index_probe, "embedding_matrix_shape", None)
        deployment["embedding_matrix_shape"] = list(shape) if shape is not None else None
        deployment["embedding_matrix_bytes"] = int(getattr(index_probe, "embedding_matrix_bytes", 0))

    sha, dirty = _git_commit_dirty()
    return {
        "schema_version": 2,
        "ts_utc": _now_utc_iso(),
        "dataset": dataset,
        "git_commit": sha,
        "git_dirty": dirty,
        "config_ini_sha256": _config_ini_sha256(config_path) if config_path is not None else "",
        "endpoints": endpoints,
        "gpu": gpu,
        "k_values": list(k_values),
        "n": N,
        "alpha": ALPHA,
        "rerank_modes": list(rerank_modes),
        "slicing_method": "per_k_search_pneuma_parity",
        "retrieval_path": "Plan(PlanNLSeeker).run()",
        "n_questions": n_questions,
        "build_wall_clock_s": build_wall_clock_s,
        "run_wall_clock_s": run_wall_clock_s,
        "peak_rss_bytes": peak_rss_bytes,
        "unresolvable_questions_count": unresolvable,
        "dropped_context_rows": dropped_context_rows,
        "vllm_seed_note": (
            "vLLM honours seed=42 over OpenAI-compat; greedy decode (T=0.0). "
            "Ollama (port 11434) does NOT honour seed= and is detected by URL."
        ),
        "status": status,
        "deployment": deployment,
        "parity_deltas": list(_PARITY_DELTAS),
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
    rerank_modes = args.get("rerank_modes") or RERANK_MODES_DEFAULT
    k_values = list(args.get("k_values") or K_VALUES)
    reset_caches = bool(args.get("reset_caches", True))

    from src.NLSeeker.engine import reset_engine_cache, _NLEngine
    from src.NLSeeker.llm import reset_backend_cache
    from src.Operators.Seekers.NLSeeker import NLSeeker as PlanNLSeeker
    from src.Plan import Plan

    from scripts.benchmark._judge_timer import JUDGE_TIMER
    from scripts.benchmark.metrics import hit_at_k, recall_at_k, reciprocal_rank

    if reset_caches:
        reset_engine_cache()
        reset_backend_cache()
    seed_all(42)  # D17

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
    family_hits = {k: 0 for k in k_values}
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
        # per (question, rerank_mode, k) and call plan.run(). This is exactly
        # how Blend is invoked in production with other seekers.
        #
        # PNEUMA-parity: one search per (rerank_mode, k), so the rerank pool
        # is k*n at each reported k (PNEUMA's hybrid_search.py:71 loops with
        # increased_k = k * n; our search() reranks the full k*n pool
        # before truncating to k).
        for rerank_mode in rerank_modes:
            for k in k_values:
                bar.set_postfix_str(f"{rerank_mode} k={k}", refresh=True)
                nl_op = PlanNLSeeker(
                    query=q.text,
                    k=k,
                    n=N,
                    alpha=ALPHA,
                    index_name=index_name,
                    rerank=(rerank_mode == "on"),
                    judge_concurrency=1,
                    config_overrides=overrides,
                )
                plan = Plan()
                plan.add("nl", nl_op)
                JUDGE_TIMER.reset()
                t0 = time.perf_counter()
                table_ids = plan.run()
                latency_ms = (time.perf_counter() - t0) * 1000.0
                judge_metrics = JUDGE_TIMER.read()
                metrics = nl_op._last_metrics or {}
                vector_ms = metrics.get("vector_ms")

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
                # Progress bar tracks rerank-on hit-rate as the headline
                # metric (rerank-off is the paper-comparison column;
                # rerank-on is Blend's production-class). Setting
                # rerank_modes=("off",) suppresses display hits; the
                # rerank_mode column in summary.csv has the full breakdown
                # either way.
                if hit and rerank_mode == "on":
                    family_hits[k] += 1
                per_query.append(PerQueryRecord(
                    ts_utc=_now_utc_iso(), dataset=dataset, family=family, k=k,
                    rerank_mode=rerank_mode,
                    question_id=q.qid, question=q.text,
                    retrieved=retrieved_dicts,
                    answer_tables=sorted(q.answer_tables),
                    hit=hit,
                    recall=recall_at_k(retrieved_pneuma, answer_set, k),
                    rr=reciprocal_rank(retrieved_pneuma, answer_set),
                    latency_ms=latency_ms,
                    vector_ms=vector_ms,
                    judge_ms=judge_metrics["judge_ms"],
                    judge_calls=judge_metrics["judge_calls"],
                    judge_tokens_in=judge_metrics["judge_tokens_in"],
                    judge_tokens_out=judge_metrics["judge_tokens_out"],
                ))
        n_done = bar.n + 1  # this question is finishing
        bar.set_postfix_str(
            " ".join(
                f"hit@{k}={family_hits[k] / n_done:.2f}" for k in k_values
            ),
            refresh=False,
        )
    bar.close()

    # Build an index probe from the cached engine (loaded during queries above).
    # _NLEngine.get() hits the in-process cache and costs nothing extra.
    index_probe: _IndexProbe | None = None
    try:
        from src.NLSeeker.config import NLSeekerConfig
        cfg = NLSeekerConfig.load(overrides=overrides)
        engine = _NLEngine.get(cfg, index_name=index_name)
        index_probe = _IndexProbe.from_vector_index(engine._vector)
    except Exception as exc:
        LOG.warning("Could not build index probe: %s", exc)

    return {
        "family": family,
        "n_questions": len(questions),
        "records": per_query,
        "index_probe": index_probe,
    }


def run_benchmark(
    *,
    dataset: str,
    lake_dir: Path | None = None,
    config_path: Path | None = None,
    index_name: str | None = None,
    content_jsonl: Path | None = None,
    results_dir: Path | None = None,
    run_dir: Path | None = None,
    endpoints_meta: dict | None = None,
    build_wall_clock_s: float = 0.0,
    unresolvable: int = 0,
    dropped_context_rows: int = 0,
    max_questions: int | None = None,
    families: tuple[str, ...] | None = None,
    rerank_modes: tuple[str, ...] | None = None,
    k_values: tuple[int, ...] | None = None,
) -> Path:
    """End-to-end run for one prepared+indexed dataset.

    Runs all families sequentially in a single Python process (Phase 8.1).
    Returns the results directory it wrote to.
    """
    from src.NLSeeker.engine import reset_engine_cache
    from src.NLSeeker.llm import reset_backend_cache

    _families = list(families) if families is not None else FAMILIES

    # Determine the output directory.
    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        if results_dir is None:
            raise ValueError("Either run_dir or results_dir must be provided")
        ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        run_dir = Path(results_dir) / dataset / ts
        run_dir.mkdir(parents=True, exist_ok=True)

    if lake_dir is not None:
        # Validate the manifest exists before running workers (each worker
        # also loads it, but failing fast here gives a single, clean error).
        load_manifest(lake_dir)
        bx_jsonl = Path(lake_dir) / "_bx_questions.jsonl"
        overrides = _overrides_from_config(config_path) if config_path is not None else {}
    else:
        bx_jsonl = Path("/dev/null")
        overrides = {}

    worker_args = [
        {
            "family": family,
            "dataset": dataset,
            "lake_dir": str(lake_dir) if lake_dir is not None else "",
            "config_path": str(config_path) if config_path is not None else "",
            "index_name": index_name or "",
            "content_jsonl": str(content_jsonl) if content_jsonl is not None else "",
            "bx_jsonl": str(bx_jsonl),
            "base_overrides": overrides,
            "llm_base_url": FAMILY_LLM_URLS.get(family, ""),
            "embed_base_url": EMBED_URL,
            "position": idx,
            "max_questions": max_questions,
            "rerank_modes": rerank_modes,
            "k_values": k_values,
        }
        for idx, family in enumerate(_families)
    ]

    t_run0 = time.perf_counter()
    family_results = []
    for i, args in enumerate(worker_args):
        args = {**args, "reset_caches": (i == 0)}
        family_results.append(_run_family(args))
    run_wall = time.perf_counter() - t_run0

    per_query: list[PerQueryRecord] = []
    n_questions: dict[str, int] = {}
    index_probe: _IndexProbe | None = None
    for fr in family_results:
        per_query.extend(fr["records"])
        n_questions[fr["family"]] = fr["n_questions"]
        if index_probe is None:
            index_probe = fr.get("index_probe")
    # Sort so per_query.jsonl ordering is stable across runs.
    per_query.sort(key=lambda r: (r.family, r.question_id, r.rerank_mode, r.k))

    write_per_query_jsonl(run_dir / "per_query.jsonl", per_query)
    summary = aggregate_summary(per_query, n=N, alpha=ALPHA)
    write_summary_csv(run_dir / "summary.csv", summary)
    write_pneuma_compat_jsonl(run_dir / "pneuma_compat.jsonl", summary)

    # `_build_run_meta` tolerates config_path=None (sha256 column is empty
    # in that case). Always emit run_meta.json so tooling can discover the
    # k_values / families / rerank_modes the user actually requested.
    meta = _build_run_meta(
        dataset=dataset,
        endpoints=endpoints_meta or {},
        gpu=_gpu_info_dict(),
        n_questions=n_questions,
        build_wall_clock_s=build_wall_clock_s,
        run_wall_clock_s=run_wall,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        unresolvable=unresolvable,
        dropped_context_rows=dropped_context_rows,
        status="ok",
        index_probe=index_probe,
        rerank_modes=rerank_modes if rerank_modes is not None else RERANK_MODES_DEFAULT,
        k_values=list(k_values) if k_values is not None else list(K_VALUES),
        config_path=config_path,
    )
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    LOG.info("benchmark complete: %s", run_dir)
    return run_dir


def _build_cli_parser() -> "argparse.ArgumentParser":
    """Build the argparse parser for benchmark/run.py.

    Used by ``main()`` and also exposed via ``__all__`` for tests and
    external callers. The parser declares all flags needed to run one
    ablation cell against one dataset.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="scripts.benchmark.run",
        description="Run the Blend NLSeeker benchmark for one prepared+indexed dataset.",
    )
    parser.add_argument("--dataset", required=True,
                        help="Dataset name (e.g. chembl, public_bi).")
    parser.add_argument(
        "--cell",
        default=None,
        help="Ablation cell key (spec §5). When set and --index-name is not given, "
             "the index name is derived as 'benchmark_autoddg_<dataset>_<cell>'.",
    )
    parser.add_argument("--index-name", default=None, dest="index_name",
                        help="Explicit index name. Overrides the --cell-derived name.")
    parser.add_argument("--questions", default=None,
                        help="Path to the content JSONL file with benchmark questions.")
    parser.add_argument("--lake-dir", default=None, dest="lake_dir",
                        help="Path to the prepared lake directory (resolves BX questions and "
                             "manifest; default: benchmark-data/lakes/<dataset>).")
    parser.add_argument("--out", default=None,
                        help="Results output directory (run_dir).")
    parser.add_argument("--max-questions", type=int, default=None, dest="max_questions",
                        help="Cap on number of questions per family (debugging).")
    parser.add_argument("--families", default=None,
                        help="Comma-separated list of families to run (default: all).")
    parser.add_argument("--rerank-modes", default=None, dest="rerank_modes",
                        help="Comma-separated rerank modes (default: off,on).")
    parser.add_argument("--k-values", default=None, dest="k_values",
                        help="Comma-separated k values (default: 1,5,10,30,50).")
    parser.add_argument(
        "--config",
        default=None,
        dest="config_path",
        help=(
            "Path to per-run config.ini "
            "(e.g. benchmark-data/indexes/autoddg/<base>/<cell>/config.ini). "
            "When omitted, the runner inherits config/config.ini at project root."
        ),
    )
    return parser


def _evaluate(args: dict) -> None:
    """Dispatch the benchmark run from a resolved args dict.

    Separated from ``main()`` so tests can monkeypatch this name without
    touching the argparse layer.  The ``args`` dict mirrors the resolved
    namespace produced by ``_build_cli_parser``:

    ``index_name``, ``dataset``, ``questions``, ``lake_dir``, ``out``,
    ``max_questions``, ``families``, ``rerank_modes``, ``k_values``.
    """
    from pathlib import Path

    questions_path = Path(args["questions"]) if args.get("questions") else None
    out_dir = Path(args["out"]) if args.get("out") else None

    # Resolve config_path: explicit --config wins; None means default project config.
    config_path = Path(args["config_path"]) if args.get("config_path") else None

    # Auto-compute autoddg results dir when --out not given and --cell was set.
    if out_dir is None and args.get("cell"):
        _project_root = Path(__file__).resolve().parents[2]
        _ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        out_dir = (
            _project_root
            / "benchmark-data"
            / "results"
            / "autoddg"
            / args["dataset"]
            / args["cell"]
            / _ts
        )

    # Resolve lake_dir: explicit > auto-derived from benchmark-data/lakes/<dataset>.
    if args.get("lake_dir"):
        lake_dir: Path | None = Path(args["lake_dir"])
    else:
        _project_root = Path(__file__).resolve().parents[2]
        _candidate = _project_root / "benchmark-data" / "lakes" / args["dataset"]
        lake_dir = _candidate if _candidate.is_dir() else None

    families_arg = args.get("families")
    families = (
        tuple(f.strip() for f in families_arg.split(",") if f.strip())
        if families_arg
        else None
    )
    rerank_arg = args.get("rerank_modes")
    rerank_modes = (
        tuple(r.strip() for r in rerank_arg.split(",") if r.strip())
        if rerank_arg
        else None
    )
    k_arg = args.get("k_values")
    k_values = (
        tuple(int(k.strip()) for k in k_arg.split(",") if k.strip())
        if k_arg
        else None
    )

    run_benchmark(
        dataset=args["dataset"],
        lake_dir=lake_dir,
        config_path=config_path,
        index_name=args.get("index_name"),
        content_jsonl=questions_path,
        run_dir=out_dir,
        results_dir=Path(__file__).resolve().parents[2] / "results" if out_dir is None else None,
        max_questions=args.get("max_questions"),
        families=families,
        rerank_modes=rerank_modes,
        k_values=k_values,
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for ``scripts/benchmark/run.py``.

    Usage example::

        python -m scripts.benchmark.run \\
            --dataset chembl \\
            --cell B8 \\
            --config benchmark-data/indexes/autoddg/chembl/B8/config.ini \\
            --questions /data/chembl/content.jsonl

    When ``--cell`` is given and ``--index-name`` is not, the index name
    is derived as ``benchmark_autoddg_<dataset>_<cell>``.
    """
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    parser = _build_cli_parser()
    ns = parser.parse_args(argv)

    # Resolve --cell → index_name (spec §3.5 / §7.1).
    if ns.cell:
        if not ns.index_name:
            ns.index_name = f"benchmark_autoddg_{ns.dataset}_{ns.cell}"

    args = {
        "dataset": ns.dataset,
        "cell": ns.cell,
        # Note: 'cell' is resolved above into index_name; it is forwarded
        # so _evaluate() can compute the autoddg results dir.
        "index_name": ns.index_name,
        "questions": ns.questions,
        "lake_dir": ns.lake_dir,
        "out": ns.out,
        "config_path": ns.config_path,
        "max_questions": ns.max_questions,
        "families": ns.families,
        "rerank_modes": ns.rerank_modes,
        "k_values": ns.k_values,
    }
    _evaluate(args)


__all__ = [
    "Manifest", "Question", "load_manifest", "load_questions",
    "PerQueryRecord", "SummaryRow",
    "write_per_query_jsonl", "write_summary_csv", "write_pneuma_compat_jsonl",
    "aggregate_summary",
    "K_VALUES", "N", "ALPHA", "FAMILIES", "RERANK_MODES_DEFAULT", "run_benchmark",
    "_evaluate", "_build_cli_parser", "main",
]

if __name__ == "__main__":
    main()
