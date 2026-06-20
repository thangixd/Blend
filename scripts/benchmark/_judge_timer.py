"""Thread-local LLM-judge timing accumulator.

Both the Blend NLSeeker rerank path and the PNEUMA-side patched
``prompt_openai_llm`` call ``JUDGE_TIMER.record(...)`` exactly once per
HTTP roundtrip to the judge. ``judge_calls`` therefore counts roundtrips,
not logical judgments — the spec's `parity_deltas` notes this.

Per-query usage:

    JUDGE_TIMER.reset()
    ... run retrieval ...
    metrics = JUDGE_TIMER.read()  # judge_ms, judge_calls, judge_tokens_in/out

The ``judging()`` context manager flips a thread-local flag so the
LLM-call wrapper can decide whether the *current* call is a judge call
(rerank=on path) or a non-judge call (e.g. summarizer). Without the flag
we'd time every LLM call, polluting judge metrics with summarization.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator


class _JudgeTimer:
    def __init__(self) -> None:
        self._tls = threading.local()

    def _state(self) -> dict:
        s = getattr(self._tls, "state", None)
        if s is None:
            s = {
                "judge_ms": 0.0,
                "judge_calls": 0,
                "judge_tokens_in": 0,
                "judge_tokens_out": 0,
                "in_judge": False,
            }
            self._tls.state = s
        return s

    def reset(self) -> None:
        s = self._state()
        s["judge_ms"] = 0.0
        s["judge_calls"] = 0
        s["judge_tokens_in"] = 0
        s["judge_tokens_out"] = 0
        s["in_judge"] = False

    def record(self, *, elapsed_ms: float, tokens_in: int, tokens_out: int) -> None:
        s = self._state()
        s["judge_ms"] += float(elapsed_ms)
        s["judge_calls"] += 1
        s["judge_tokens_in"] += int(tokens_in)
        s["judge_tokens_out"] += int(tokens_out)

    def read(self) -> dict:
        s = self._state()
        return {
            "judge_ms": s["judge_ms"],
            "judge_calls": s["judge_calls"],
            "judge_tokens_in": s["judge_tokens_in"],
            "judge_tokens_out": s["judge_tokens_out"],
        }

    @property
    def in_judge(self) -> bool:
        return self._state()["in_judge"]

    @contextmanager
    def judging(self) -> Iterator[None]:
        s = self._state()
        prev = s["in_judge"]
        s["in_judge"] = True
        try:
            yield
        finally:
            s["in_judge"] = prev


JUDGE_TIMER = _JudgeTimer()

__all__ = ["JUDGE_TIMER"]
