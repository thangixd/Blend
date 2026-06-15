import logging
from dataclasses import dataclass

import pandas as pd

from src.NLSeeker.llm import EmbedBackend, LLMBackend

# Typing imports
from typing import Iterable, Optional, Sequence

LOG = logging.getLogger(__name__)

# Tunables exposed as constants rather than config so the retrieval pipeline
# stays comparable across runs.
ROW_SAMPLE_SIZE = 5
ROW_SAMPLE_RANDOM_STATE = 0
COLUMN_NARRATION_MAX_NEW_TOKENS = 400
BLOCK_SEPARATOR = " || "
# Single pipe is used for context blocks (vs the double pipe for narration
# and row samples).
CONTEXT_BLOCK_SEPARATOR = " | "

_BLOCK_HEADROOM = 0  


@dataclass
class SummarizedTable:
    table_id: int
    column_narration_blocks: list
    row_sample_blocks: list
    context_blocks: list = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.context_blocks is None:
            self.context_blocks = []


def _column_list_block(columns: Sequence[str]) -> str:
    return " | ".join(columns)


def _build_narration_prompt(columns: Sequence[str], target: str) -> str:
    return (
        "A table has the following columns:\n"
        "/*\n"
        f"{_column_list_block(columns)}\n"
        "*/\n"
        f"Describe briefly what the {target} column represents. "
        'If not possible, simply state "No description."'
    )


def generate_column_narrations(
    df: pd.DataFrame,
    llm: LLMBackend,
) -> list:
    """Return one narration per column, in order, prefixed with ``"<col>: "``."""
    columns = [str(c) for c in df.columns]
    if not columns:
        return []
    prompts = [_build_narration_prompt(columns, c) for c in columns]
    narrations = llm.generate(prompts, max_new_tokens=COLUMN_NARRATION_MAX_NEW_TOKENS)
    return [f"{col}: {n}".strip() for col, n in zip(columns, narrations)]


def _format_row(row: pd.Series) -> str:
    """Render one row as 'col1: val1 | col2: val2 | ...'.

    NaN is stringified as 'nan'.
    """
    parts = [f"{col}: {val}" for col, val in row.items()]
    return " | ".join(parts)


def generate_row_samples(df: pd.DataFrame) -> list:
    """Sample up to :data:`ROW_SAMPLE_SIZE` rows deterministically."""
    if df.empty:
        return []
    n = min(len(df), ROW_SAMPLE_SIZE)
    sample = df.sample(n=n, random_state=ROW_SAMPLE_RANDOM_STATE)
    return [_format_row(row) for _, row in sample.iterrows()]


def _token_count(text: str, embedder: EmbedBackend) -> int:
    # _HFEmbedder exposes its sentence-transformers model as ._model with a
    # .tokenizer attribute; _OpenAIEmbedder exposes a .tokenizer property
    # (lazily loaded from openai_embed_tokenizer_id).
    tok = getattr(embedder, "tokenizer", None)
    if tok is None:
        inner = getattr(embedder, "_model", None)
        tok = getattr(inner, "tokenizer", None) if inner is not None else None
    if tok is not None:
        try:
            return len(tok.encode(text, add_special_tokens=False))
        except Exception:  # pragma: no cover
            pass
    return max(1, len(text) // 4)


def block_texts(
    texts: Iterable[str],
    embedder: EmbedBackend,
    separator: str = BLOCK_SEPARATOR,
) -> list:
    """Greedily concatenate ``texts`` into ``separator``-joined groups under the embedder's window."""
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        return []

    limit = max(1, embedder.max_input_tokens - _BLOCK_HEADROOM)
    blocks = []
    current = []
    current_tokens = 0
    sep_tokens = _token_count(separator, embedder)

    for text in texts:
        text_tokens = _token_count(text, embedder)
        # Over-long single texts are emitted as their own block; the
        # embedder will truncate. Splitting mid-sentence would produce nonsense.
        if text_tokens >= limit:
            if current:
                blocks.append(separator.join(current))
                current = []
                current_tokens = 0
            blocks.append(text)
            continue

        join_cost = (sep_tokens + text_tokens) if current else text_tokens
        if current and current_tokens + join_cost > limit:
            blocks.append(separator.join(current))
            current = [text]
            current_tokens = text_tokens
        else:
            current.append(text)
            current_tokens += join_cost

    if current:
        blocks.append(separator.join(current))
    return blocks


def summarize_table(
    table_id: int,
    df: pd.DataFrame,
    llm: LLMBackend,
    embedder: EmbedBackend,
    contexts: Optional[Sequence[str]] = None,
    pre_chunked_contexts: bool = False,
) -> SummarizedTable:
    """Produce blocked column narrations, row samples, and (optional) context blocks.

    When ``pre_chunked_contexts=True`` each item in ``contexts`` is treated as
    exactly one chunk - ``block_texts`` is bypassed entirely.  This matches
    PNEUMA's retrieval behaviour where each record in ``contexts_<ds>_merged.jsonl``
    is already the unit of retrieval.  The default (``False``) preserves the
    legacy greedy-pack behaviour.
    """
    LOG.info("Summarizing TableId=%d (%d cols, %d rows%s)",
             table_id, df.shape[1], df.shape[0],
             f", {len(contexts)} contexts" if contexts else "")
    narrations = generate_column_narrations(df, llm)
    samples = generate_row_samples(df)
    context_blocks = []
    if contexts:                      # also skips the None case (list(None) never runs)
        # Contexts skip the LLM and go straight to the blocker - the only
        # added LLM cost is at rerank time.
        if pre_chunked_contexts:
            # Each merged-context record is already one retrieval unit; skip
            # block_texts. list() materialises any generator and decouples
            # from the caller's mutable sequence.
            context_blocks = list(contexts)
        else:
            context_blocks = block_texts(
                contexts,
                embedder,
                separator=CONTEXT_BLOCK_SEPARATOR,
            )
    return SummarizedTable(
        table_id=table_id,
        column_narration_blocks=block_texts(narrations, embedder),
        row_sample_blocks=block_texts(samples, embedder),
        context_blocks=context_blocks,
    )


__all__ = [
    "BLOCK_SEPARATOR",
    "CONTEXT_BLOCK_SEPARATOR",
    "ROW_SAMPLE_SIZE",
    "ROW_SAMPLE_RANDOM_STATE",
    "SummarizedTable",
    "generate_column_narrations",
    "generate_row_samples",
    "block_texts",
    "summarize_table",
]
