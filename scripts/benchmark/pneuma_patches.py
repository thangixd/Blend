"""Monkeypatch the vendored PNEUMA tree for the local vLLM benchmark.

PNEUMA's ``prompt_openai_llm`` defaults to OpenAI's public endpoint, uses
tiktoken's ``gpt-4o`` encoder for chunking, and the summarizer/index_generator
fall back to that encoder whenever the embedding model is an ``OpenAI`` client
rather than a sentence-transformers model.  Those defaults are wrong for our
isolated benchmark: we serve Qwen2.5-7B-Instruct locally over vLLM and use the
bge-base 512-token embedder, so chunk sizing must respect bge's vocabulary,
not gpt-4o's.

This module patches at import time:

1. ``pneuma.utils.prompting_interface.prompt_openai_llm`` — rebuilds the
   OpenAI client to point at ``http://127.0.0.1:8001/v1`` with a dummy api
   key, and forces ``model='Qwen2.5-7B-Instruct'``, ``temperature=0.0``,
   ``seed=42`` regardless of what the caller passes.
2. ``tiktoken.encoding_for_model('gpt-4o')`` — returns a thin shim around
   the bge-base ``AutoTokenizer`` so chunking respects the 512-token
   embedder cap.  The real AutoTokenizer is also stashed on
   ``pneuma.summarizer.summarizer._BENCH_ENCODER`` for instrumentation.
3. Process-wide determinism env vars: ``PYTHONHASHSEED=0``,
   ``CUBLAS_WORKSPACE_CONFIG=:4096:8``.
4. Judge-call timing — when the patched ``prompt_openai_llm`` runs inside
   a ``JUDGE_TIMER.judging()`` block, it records (elapsed_ms, tokens_in,
   tokens_out) into the thread-local timer.
5. ``benchmark_generator.context.utils.prompting_interface.prompt_pipeline``
   — replaces the HuggingFace ``TextGenerationPipeline`` wrapper with a
   thin OpenAI-compat shim that routes each conversation through the same
   vLLM client (one HTTP roundtrip per candidate, judge-timed).  This lets
   ``HybridRetriever._llm_rerank`` (and PNEUMA's summarizer paths that use
   ``prompt_pipeline``) work without a real HF pipeline handle, mirroring
   how Blend's NLSeeker rerank issues yes/no judge calls.

Notes / OMITTED sub-patches
---------------------------
* ``pneuma.utils.pipeline_initializer.initialize_pipeline`` was NOT patched.
  Inspection of ``pneuma/src/pneuma/pneuma.py`` shows that ``Pneuma()``
  only calls ``initialize_pipeline`` when ``use_local_model=True``; with
  ``use_local_model=False`` it constructs an ``OpenAI`` client directly
  and never invokes the HF loader.  Callers driving the vLLM endpoint
  pass ``use_local_model=False, openai_api_key='dummy'`` and rely on this
  module's ``prompt_openai_llm`` patch to redirect the client at call time.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import openai
import tiktoken
from transformers import AutoTokenizer

from scripts.benchmark._judge_timer import JUDGE_TIMER

# ---- Configuration -----------------------------------------------------

VLLM_BASE_URL = "http://127.0.0.1:8001/v1"
VLLM_MODEL = "Qwen2.5-7B-Instruct"
VLLM_API_KEY = "dummy"  # vLLM ignores the key but the OpenAI SDK requires one.
BGE_TOKENIZER_NAME = "BAAI/bge-base-en-v1.5"

# The vendored PNEUMA tree lives at ``<repo>/pneuma/`` and is not installed
# as a regular package, so we splice it onto sys.path on first import.  Done
# at module import time (before apply_patches runs) so callers can
# ``import pneuma...`` themselves without bootstrapping.
#
# Three locations need to be on sys.path:
#   * ``pneuma/src``            — for ``pneuma.*`` (the public API).
#   * ``pneuma``                — for ``benchmark_generator.*``, the sibling
#                                  package vendored at ``pneuma/benchmark_generator``.
#   * ``pneuma/experiments``    — for ``pneuma_retriever.hybrid_retriever``
#                                  etc., which the experiments tree imports
#                                  as bare top-level names (no __init__.py
#                                  on the experiments/ directory).
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PNEUMA_SRC = _REPO_ROOT / "pneuma" / "src"
_PNEUMA_TOP = _REPO_ROOT / "pneuma"
_PNEUMA_EXPERIMENTS = _REPO_ROOT / "pneuma" / "experiments"
for _candidate in (_PNEUMA_SRC, _PNEUMA_TOP, _PNEUMA_EXPERIMENTS):
    if _candidate.is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

# PNEUMA's experiments tree does ``import chromadb`` and
# ``from chromadb.api.models.Collection import Collection``, but this
# project pins ``chromadb-deterministic`` (a fork with seeded HNSW) instead
# of upstream ``chromadb``.  Alias the package so PNEUMA's imports resolve
# to the deterministic fork without touching its source.
try:
    import chromadb_deterministic as _chromadb_det  # noqa: WPS433
    sys.modules.setdefault("chromadb", _chromadb_det)
except ImportError:  # not installed in unit-test envs
    pass

# Set once apply_patches() has run, so re-application is a no-op.
_APPLIED: bool = False


# ---- Tokenizer shim ----------------------------------------------------

class _BgeTiktokenShim:
    """Drop-in replacement for a tiktoken ``Encoding`` backed by an
    AutoTokenizer.

    PNEUMA only uses ``encode(text) -> list[int]`` and counts the length of
    the returned list to decide whether to merge another chunk, so a
    minimal shim suffices.  Crucially this shim does NOT expose
    ``encoding_name`` — callers can use that as a sentinel to detect that
    the bench-time tokenizer is not a real tiktoken Encoding.
    """

    __slots__ = ("_tok",)

    def __init__(self, tokenizer: Any) -> None:
        self._tok = tokenizer

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False)


# ---- Patch implementations ---------------------------------------------

def _patch_env() -> None:
    # PYTHONHASHSEED must be set BEFORE the Python interpreter starts —
    # CPython reads it during ``Py_Initialize`` and a runtime
    # ``os.environ`` write is a no-op for the current process.  The
    # ``make benchmark-*`` targets export it from the shell; we deliberately
    # do not set it here to avoid the false sense of security.
    #
    # ``CUBLAS_WORKSPACE_CONFIG`` is read by cuBLAS at first kernel launch,
    # which happens long after Python init, so a runtime write IS effective
    # and remains useful as belt-and-braces against callers that forget the
    # Makefile entrypoint.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _patch_tiktoken(bge_tokenizer: Any) -> None:
    """Replace ``tiktoken.encoding_for_model`` so the gpt-4o lookup PNEUMA
    performs (in summarizer + index_generator) returns the bge shim.

    The bge tokenizer is supplied by the caller (``apply_patches``) so the
    embedder identity stays a function parameter, not a module constant."""
    if getattr(tiktoken, "_bench_patched", False):
        return

    shim = _BgeTiktokenShim(bge_tokenizer)
    original = tiktoken.encoding_for_model

    def _patched_encoding_for_model(model_name: str):  # type: ignore[no-untyped-def]
        if model_name == "gpt-4o":
            return shim
        return original(model_name)

    tiktoken.encoding_for_model = _patched_encoding_for_model  # type: ignore[assignment]
    tiktoken._bench_patched = True  # type: ignore[attr-defined]


def _patch_summarizer_encoder(bge_tokenizer: Any) -> None:
    """Stash the real AutoTokenizer on the summarizer module for
    instrumentation/visibility (the tests assert this attribute exists)."""
    from pneuma.summarizer import summarizer as _summarizer

    _summarizer._BENCH_ENCODER = bge_tokenizer  # type: ignore[attr-defined]


def _patch_prompt_openai_llm(*, llm_endpoint_url: str, llm_model_id: str) -> None:
    """Wrap ``pneuma.utils.prompting_interface.prompt_openai_llm`` so it:

    * rebuilds the client against ``llm_endpoint_url`` (with a dummy api key),
    * forces ``model=llm_model_id``, ``temperature=0.0``, ``seed=42``,
    * records judge timing when called inside ``JUDGE_TIMER.judging()``.

    Endpoint and model identity are passed in as parameters so callers
    (e.g. ``pneuma_cli.py``) can drive the patch from CLI flags rather
    than baking the values in at module scope.
    """
    from pneuma.utils import prompting_interface

    # The OpenAI client (and its httpx connection pool) is shared across all
    # calls — PNEUMA's summarizer issues 1000+ chat.completions per dataset
    # and constructing a fresh client per call would spike file descriptors
    # and add connect overhead.  Built lazily on first call so apply_patches()
    # itself does not touch the network.
    _client_box: dict[str, openai.OpenAI] = {}

    def _get_client() -> openai.OpenAI:
        c = _client_box.get("c")
        if c is None:
            c = openai.OpenAI(base_url=llm_endpoint_url, api_key=VLLM_API_KEY)
            _client_box["c"] = c
        return c

    def patched_prompt_openai_llm(
        llm: Any,  # ignored — replaced with a vLLM-pointing client
        conversations: list[list[dict[str, str]]],
        model: str = llm_model_id,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        retry_attempts: int = 5,
    ):
        # Spec mandates forcing ``model`` to ``llm_model_id``; the caller's
        # ``model`` kwarg is intentionally absorbed.  Same for ``temperature``
        # (0.0) and ``seed`` (42) — bench determinism trumps caller intent.
        client = _get_client()
        for conv in conversations:
            # ``elapsed_ms`` accumulates total wall-clock from first attempt to
            # final success/failure, matching what the bench is trying to
            # measure (real time the call cost, retries included).
            t0 = time.monotonic()
            for attempt in range(retry_attempts):
                try:
                    response = client.chat.completions.create(
                        model=llm_model_id,
                        messages=conv,
                        max_tokens=max_new_tokens,
                        temperature=0.0,
                        top_p=top_p,
                        seed=42,
                    )
                except (openai.OpenAIError, httpx.HTTPError) as e:
                    # ``httpx.HTTPError`` covers the entire transport+timeout
                    # subtree (ConnectError, TimeoutException, ReadError, …),
                    # so a vLLM restart is retried instead of aborting the
                    # bench.
                    if attempt < retry_attempts - 1:
                        print(
                            f"OpenAI/HTTP error: {e}. Retrying "
                            f"({attempt + 1}/{retry_attempts})..."
                        )
                        continue
                    print(f"Failed after {retry_attempts} attempts.")
                    raise
                elapsed_ms = (time.monotonic() - t0) * 1000.0
                if JUDGE_TIMER.in_judge:
                    usage = getattr(response, "usage", None)
                    tokens_in = int(getattr(usage, "prompt_tokens", 0) or 0)
                    tokens_out = int(getattr(usage, "completion_tokens", 0) or 0)
                    JUDGE_TIMER.record(
                        elapsed_ms=elapsed_ms,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                    )
                conv.append(
                    {
                        "role": "assistant",
                        "content": response.choices[0].message.content,
                    }
                )
                break
        return conversations

    # Build the wrapped function once; rebinding the name on the call sites
    # is unconditional so a partially-mutated module state (e.g. test reload
    # that leaves ``_bench_patched`` truthy but clears the call sites) is
    # repaired on every apply_patches() invocation.
    if not getattr(prompting_interface, "_bench_patched", False):
        prompting_interface.prompt_openai_llm = patched_prompt_openai_llm  # type: ignore[assignment]
        prompting_interface._bench_patched = True  # type: ignore[attr-defined]

    # ``from pneuma.utils.prompting_interface import prompt_openai_llm`` binds
    # the name at import time, so the call sites need explicit rebinds.
    from pneuma.summarizer import summarizer as _summarizer
    from pneuma.query_processor import query_processor as _query_processor

    _summarizer.prompt_openai_llm = prompting_interface.prompt_openai_llm  # type: ignore[assignment]
    _query_processor.prompt_openai_llm = prompting_interface.prompt_openai_llm  # type: ignore[assignment]


def _patch_prompt_pipeline(*, llm_endpoint_url: str, llm_model_id: str) -> None:
    """Route ``prompt_pipeline`` (the HuggingFace ``TextGenerationPipeline``
    wrapper) through the same vLLM endpoint as ``prompt_openai_llm``.

    PNEUMA's ``HybridRetriever._llm_rerank`` (and several summarizer paths)
    call ``prompt_pipeline(self.reranker, conversations, ...)`` expecting
    a real HF ``TextGenerationPipeline`` object as the first arg.  Without
    this patch a benchmark with ``rerank_mode='on'`` and ``reranker=None``
    crashes with ``AttributeError: 'NoneType' object has no attribute
    'tokenizer'``.

    The patched function ignores the ``pipe`` argument, fans each
    conversation through the same ``client.chat.completions.create`` call
    pattern Blend's NLSeeker rerank path uses (one HTTP roundtrip per
    candidate, deterministic sampling, judge-timed when inside
    ``JUDGE_TIMER.judging()``), and returns conversations with an
    ``{"role": "assistant", "content": ...}`` message appended — matching
    the shape PNEUMA's downstream parsing expects.

    Idempotent: a sentinel on the source module gates the source-side
    rebind; the per-module rebinds run unconditionally so a partially
    mutated state self-repairs.
    """
    from benchmark_generator.context.utils import prompting_interface as _bg_pi

    _client_box: dict[str, openai.OpenAI] = {}

    def _get_client() -> openai.OpenAI:
        c = _client_box.get("c")
        if c is None:
            c = openai.OpenAI(base_url=llm_endpoint_url, api_key=VLLM_API_KEY)
            _client_box["c"] = c
        return c

    def patched_prompt_pipeline(
        pipe: Any,  # ignored — replaced with a vLLM-pointing client
        conversations: list[list[dict[str, str]]],
        batch_size: int = 2,  # noqa: ARG001 — accepted for signature parity
        context_length: int = 8192,  # noqa: ARG001 — same
        max_new_tokens: int = 512,
        do_sample: bool = False,  # noqa: ARG001
        top_k: int = 0,  # noqa: ARG001
        top_p: float = 1.0,
        penalty_alpha: float = 0.0,  # noqa: ARG001
        temperature: float = 0.0,  # noqa: ARG001 — bench forces 0.0 below
    ):
        # Spec: forces ``model``, ``temperature=0.0``, ``seed=42`` on every
        # call; caller-supplied generation knobs are intentionally absorbed
        # (PNEUMA's ``hybrid_retriever`` already passes top_p=None etc., and
        # Blend's NLSeeker rerank also pins these — see
        # ``src/NLSeeker/llm.py:_OpenAILLM.generate._one``).
        client = _get_client()
        for conv in conversations:
            t0 = time.monotonic()
            for attempt in range(5):
                try:
                    response = client.chat.completions.create(
                        model=llm_model_id,
                        messages=conv,
                        max_tokens=max_new_tokens,
                        temperature=0.0,
                        top_p=top_p,
                        seed=42,
                    )
                except (openai.OpenAIError, httpx.HTTPError) as e:
                    if attempt < 4:
                        print(
                            f"OpenAI/HTTP error (prompt_pipeline): {e}. "
                            f"Retrying ({attempt + 1}/5)..."
                        )
                        continue
                    print("Failed after 5 attempts.")
                    raise
                elapsed_ms = (time.monotonic() - t0) * 1000.0
                if JUDGE_TIMER.in_judge:
                    usage = getattr(response, "usage", None)
                    tokens_in = int(getattr(usage, "prompt_tokens", 0) or 0)
                    tokens_out = int(getattr(usage, "completion_tokens", 0) or 0)
                    JUDGE_TIMER.record(
                        elapsed_ms=elapsed_ms,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                    )
                conv.append(
                    {
                        "role": "assistant",
                        "content": response.choices[0].message.content,
                    }
                )
                break
        return conversations

    # Source rebind on the prompting_interface module — gated by sentinel
    # so we don't double-wrap if apply_patches() runs twice.
    if not getattr(_bg_pi, "_bench_pp_patched", False):
        _bg_pi.prompt_pipeline = patched_prompt_pipeline  # type: ignore[assignment]
        _bg_pi._bench_pp_patched = True  # type: ignore[attr-defined]

    # ``from ... import prompt_pipeline`` binds the name at import time,
    # so each consumer module needs an explicit rebind.  These are cheap
    # and idempotent — repeated apply_patches() calls always source the
    # current ``_bg_pi.prompt_pipeline`` so partial mutation self-repairs.
    #
    # ``pneuma_retriever.hybrid_retriever`` is imported via the bare
    # top-level name because ``pneuma/experiments/`` is on sys.path
    # (it has no ``__init__.py`` so it isn't a sub-package of ``pneuma``).
    # If chromadb / bm25s aren't installed (unit-test environment), the
    # import will fail; we catch ImportError per module so the patch still
    # reaches the rest.  Production environments have all deps installed
    # and every rebind succeeds.
    _consumers: list[tuple[str, str]] = [
        ("pneuma_retriever", "hybrid_retriever"),
        ("pneuma.summarizer", "summarizer"),
        ("pneuma.query_processor", "query_processor"),
        ("benchmark_generator.context.utils", "generators"),
    ]
    for pkg, mod in _consumers:
        try:
            module = __import__(f"{pkg}.{mod}", fromlist=[mod])
        except ImportError as e:
            # Production has chromadb/bm25s/etc; unit tests don't.  Skip
            # rebinds whose dependencies aren't importable.  The source
            # rebind on _bg_pi already routes any direct
            # ``prompting_interface.prompt_pipeline`` lookup through vLLM.
            print(
                f"  skip prompt_pipeline rebind on {pkg}.{mod}: {e}"
            )
            continue
        module.prompt_pipeline = _bg_pi.prompt_pipeline  # type: ignore[attr-defined]


def _patch_llm_rerank_judge_scope() -> None:
    """Wrap ``HybridRetriever._llm_rerank`` in ``JUDGE_TIMER.judging()`` so
    ``judge_ms`` brackets ONLY the rerank LLM call(s), byte-identical to
    Blend's NLSeeker scope at ``src/NLSeeker/retrieve.py:llm_rerank``.

    Without this patch, the runner would have to open ``judging()`` at a
    coarser scope (e.g. the whole ``HybridRetriever.retrieve(...)`` call,
    which also runs BM25 + hybrid merge before the LLM), inflating
    ``judge_ms`` by retrieval overhead.

    PNEUMA's vendored ``_llm_rerank`` issues exactly the LLM judge HTTP
    roundtrip(s) we want to time — wrapping its body matches Blend's
    ``llm.generate(...)``-only scope exactly.
    """
    try:
        from pneuma_retriever import hybrid_retriever as _hr  # type: ignore[import-not-found]
    except ImportError as e:
        # Unit-test environment without chromadb/bm25s: skip silently.
        # Production has all deps installed and the patch lands.
        print(f"  skip _llm_rerank judge-scope patch: {e}")
        return

    if getattr(_hr.HybridRetriever, "_bench_judge_scope_patched", False):
        return

    original_llm_rerank = _hr.HybridRetriever._llm_rerank

    def patched_llm_rerank(self, *args, **kwargs):
        # Time only the body of _llm_rerank — matches Blend's
        # ``with JUDGE_TIMER.judging(): answers = llm.generate(...)`` at
        # ``src/NLSeeker/retrieve.py:420-421``.
        with JUDGE_TIMER.judging():
            return original_llm_rerank(self, *args, **kwargs)

    _hr.HybridRetriever._llm_rerank = patched_llm_rerank  # type: ignore[method-assign]
    _hr.HybridRetriever._bench_judge_scope_patched = True  # type: ignore[attr-defined]


# ---- Public entrypoint -------------------------------------------------

def apply_patches(
    *,
    llm_endpoint_url: str = VLLM_BASE_URL,
    llm_model_id: str = VLLM_MODEL,
    embed_model_id: str = BGE_TOKENIZER_NAME,
) -> None:
    """Apply all PNEUMA-tree patches.  Idempotent.

    Parameters
    ----------
    llm_endpoint_url:
        OpenAI-compat base URL the patched ``prompt_openai_llm`` will rebuild
        its client against (e.g. the local vLLM server).
    llm_model_id:
        Model identifier the patched call site will force on every
        ``chat.completions.create`` regardless of caller-supplied ``model``.
    embed_model_id:
        HuggingFace tokenizer name to back both the tiktoken ``gpt-4o`` shim
        and the ``_BENCH_ENCODER`` attribute on the summarizer module.

    The keyword-only signature matches the spec contract; callers like
    ``pneuma_cli.py`` pass the three values from CLI flags instead of
    relying on module-level defaults.
    """
    global _APPLIED
    if _APPLIED:
        return

    _patch_env()
    bge_tokenizer = AutoTokenizer.from_pretrained(embed_model_id)
    _patch_tiktoken(bge_tokenizer)
    _patch_summarizer_encoder(bge_tokenizer)
    _patch_prompt_openai_llm(
        llm_endpoint_url=llm_endpoint_url,
        llm_model_id=llm_model_id,
    )
    _patch_prompt_pipeline(
        llm_endpoint_url=llm_endpoint_url,
        llm_model_id=llm_model_id,
    )
    _patch_llm_rerank_judge_scope()

    _APPLIED = True


__all__ = ["apply_patches", "JUDGE_TIMER"]
