from __future__ import annotations

import pandas as pd

# Typing imports
from typing import Iterable, Sequence

A1_CLOSING_LINE = (
    'Describe briefly what the {target} column represents. '
    'If not possible, simply state "No description."'
)

A4_SAMPLE_SIZE = 5
A4_SAMPLE_RANDOM_STATE = 9  # AutoDDG


def extract_column_stats_line(profile_text: str, target: str) -> str:
    marker = f"**{target}**:"
    for line in profile_text.splitlines():
        if line.startswith(marker):
            return line.strip()
    return ""


def extract_sample_values(df: pd.DataFrame, target: str) -> list:
    """5 string-coerced values from ``target`` using ``random_state=9``."""
    if target not in df.columns or len(df) == 0:
        return []
    n = min(A4_SAMPLE_SIZE, len(df))
    sampled = df.sample(n=n, random_state=A4_SAMPLE_RANDOM_STATE)
    return sampled[target].astype(str).tolist()


def build_profiled_narration_prompt(
    *,
    columns: Sequence[str],
    target: str,
    column_stats_line: str,
    sample_values: Iterable[str],
    semantic_sentence: str,
) -> str:
    column_list_block = " | ".join(str(c) for c in columns)
    samples = list(sample_values)

    lines: list[str] = [
        "A table has the following columns:",
        "/*",
        column_list_block,
        "*/",
    ]
    if column_stats_line:
        lines.extend([
            f"Structural profile of column {target}:",
            column_stats_line,
            "",
        ])
    if samples:
        lines.append(f"Sample values for column {target}: " + ", ".join(samples))
        lines.append("")
    if semantic_sentence:
        lines.extend([
            f"Semantic profile of column {target}:",
            semantic_sentence,
            "",
        ])
    lines.append(A1_CLOSING_LINE.replace("{target}", target))
    return "\n".join(lines)


__all__ = [
    "A1_CLOSING_LINE",
    "A4_SAMPLE_SIZE",
    "A4_SAMPLE_RANDOM_STATE",
    "extract_column_stats_line",
    "extract_sample_values",
    "build_profiled_narration_prompt",
]
