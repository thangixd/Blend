from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

# Verbatim from prompts.yaml topic_generation.system_message.
TOPIC_SYSTEM_MESSAGE = "You are an assistant for generating concise dataset topics."

# Verbatim from prompts.yaml topic_generation.user_prompt.
_TOPIC_USER_PROMPT = (
    "Using the dataset information provided, generate a concise topic in 2-3 words "
    "that best describes the dataset's primary theme.\n"
    "\n"
    "Title: {title}\n"
    "{description_block}\n"
    "Dataset Sample: {dataset_sample}\n"
    "\n"
    "Topic (2-3 words):\n"
)

TOPIC_SAMPLE_SIZE = 5
TOPIC_SAMPLE_RANDOM_STATE = 0  # match Pneuma ROW_SAMPLE seed (spec §2.3 A7).


def resolve_table_title(filename: str | None, table_id: int) -> str:
    """Table title resolution.

    - ``assays_SEP_table_65.csv`` -> ``"assays"``
    - ``mydataset.csv`` -> ``"mydataset"``
    - empty / None -> ``f"table_{table_id}"``
    """
    if not filename:
        return f"table_{int(table_id)}"
    stem = Path(filename).stem
    if not stem:
        return f"table_{int(table_id)}"
    if "_SEP_" in stem:
        return stem.split("_SEP_", 1)[0]
    return stem


def sample_for_topic(df: pd.DataFrame) -> pd.DataFrame:
    """Return a reproducible 5-row sample, or df unchanged if already ≤5 rows."""
    if TOPIC_SAMPLE_SIZE < len(df):
        return df.sample(TOPIC_SAMPLE_SIZE, random_state=TOPIC_SAMPLE_RANDOM_STATE)
    return df


def sample_for_topic_csv(df: pd.DataFrame) -> str:
    sample = sample_for_topic(df)
    buf = io.StringIO()
    sample.to_csv(buf, index=False)
    return buf.getvalue()


def build_topic_prompt(
    title: str,
    original_description: str | None,
    dataset_sample: str,
) -> str:
    """Build the topic user prompt. AutoDDG topic/generator.py:32-42."""
    description_block = (
        f"Original Description: {original_description}\n" if original_description else ""
    )
    return _TOPIC_USER_PROMPT.format(
        title=title,
        description_block=description_block,
        dataset_sample=dataset_sample,
    )


__all__ = [
    "TOPIC_SYSTEM_MESSAGE",
    "TOPIC_SAMPLE_SIZE",
    "TOPIC_SAMPLE_RANDOM_STATE",
    "resolve_table_title",
    "sample_for_topic",
    "sample_for_topic_csv",
    "build_topic_prompt",
]
