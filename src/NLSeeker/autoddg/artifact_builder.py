from __future__ import annotations

import json
import logging
from typing import Optional, Sequence

import pandas as pd

from src.NLSeeker.autoddg.description import (
    SFD_SYSTEM_MESSAGE,
    UFD_SYSTEM_MESSAGE,
    build_sfd_prompt,
    build_ufd_prompt,
)
from src.NLSeeker.autoddg.narration_profiled import (
    build_profiled_narration_prompt,
    extract_column_stats_line,
    extract_sample_values,
)
from src.NLSeeker.autoddg.profile import (
    _render_profile_summary,
    _render_semantic_notes,
    profile_dataframe_with_metadata,
    serialize_content_profile,
)
from src.NLSeeker.autoddg.semantic import (
    SEMANTIC_SYSTEM_MESSAGE,
    build_semantic_prompt,
    parse_semantic_response,
    render_table_summary,
    sample_for_semantic,
)
from src.NLSeeker.autoddg.topic import (
    TOPIC_SYSTEM_MESSAGE,
    build_topic_prompt,
    resolve_table_title,
    sample_for_topic_csv,
)
from src.NLSeeker.config import NLSeekerConfig
from src.NLSeeker.db_schema import (
    SummaryType,
    fetch_autoddg_profile,
    fetch_autoddg_topic,
    insert_contexts,
    insert_summaries,
    upsert_autoddg_profile,
    upsert_autoddg_topic,
)
from src.NLSeeker.summarize import (
    COLUMN_NARRATION_MAX_NEW_TOKENS,
    ROW_SAMPLE_SIZE,
    ROW_SAMPLE_RANDOM_STATE,
    _build_narration_prompt,
)

LOG = logging.getLogger(__name__)

_RETRY_SEMANTIC = 3
_SEMANTIC_MAX_NEW_TOKENS = 512
_UFD_MAX_NEW_TOKENS = 600
_SFD_MAX_NEW_TOKENS = 1200
_TOPIC_MAX_NEW_TOKENS = 16


def _chat(llm, system_message: str, user_prompt: str, *, max_new_tokens: int) -> str:
    """Single-prompt invocation that prefixes the system message.

    The existing LLMBackend has no chat-template API; we concatenate the
    system message in front of the user prompt with a blank line between.
    """
    joined = f"{system_message}\n\n{user_prompt}"
    out = llm.generate([joined], max_new_tokens=max_new_tokens)
    return out[0]


def _batch_chat(llm, system_message: str, user_prompts: Sequence[str], *, max_new_tokens: int) -> list[str]:
    """Batched form of ``_chat``. Each user prompt is prefixed by the system
    message with a blank line separator, then submitted as a single
    ``llm.generate`` batch.
    """
    joined = [f"{system_message}\n\n{p}" for p in user_prompts]
    return llm.generate(joined, max_new_tokens=max_new_tokens)


class NLArtifactBuilder:
    def __init__(self, *, cfg: NLSeekerConfig, llm, cursor, dbms: str = "duckdb") -> None:
        self.cfg = cfg
        self.llm = llm
        self.cursor = cursor
        self.dbms = dbms

    # ---- per-artifact helpers -------------------------------------------------

    def _has_summary(self, table_id: int, stype: SummaryType) -> bool:
        qmark = "?" if self.dbms == "duckdb" else "%s"
        self.cursor.execute(
            f"SELECT 1 FROM blend_nl_summaries WHERE TableId={qmark} AND SummaryType={qmark} LIMIT 1",
            (int(table_id), stype.value),
        )
        return self.cursor.fetchone() is not None

    def _has_context(self, table_id: int, source: str) -> bool:
        qmark = "?" if self.dbms == "duckdb" else "%s"
        self.cursor.execute(
            f"SELECT 1 FROM blend_nl_contexts WHERE TableId={qmark} AND Source={qmark} LIMIT 1",
            (int(table_id), source),
        )
        return self.cursor.fetchone() is not None

    # ---- main entry -----------------------------------------------------------

    def generate_table(
        self,
        *,
        table_id: int,
        filename: Optional[str],
        df: pd.DataFrame,
        real_contexts: Optional[Sequence[str]] = None,
    ) -> None:
        """Generate all enabled artifacts for one table and upsert into storage.
        Idempotent unless ``cfg.force_regenerate``.
        """
        force = self.cfg.force_regenerate
        cols = [str(c) for c in df.columns]

        # A1 column narrations.
        if force or not self._has_summary(table_id, SummaryType.COLUMN_NARRATION):
            prompts = [_build_narration_prompt(cols, c) for c in cols]
            answers = self.llm.generate(prompts, max_new_tokens=COLUMN_NARRATION_MAX_NEW_TOKENS)
            rows = [
                (int(table_id), i, SummaryType.COLUMN_NARRATION, f"{col}: {ans}".strip())
                for i, (col, ans) in enumerate(zip(cols, answers))
            ]
            insert_summaries(self.cursor, self.dbms, rows)

        # A2 row sample - deterministic, no LLM.
        if force or not self._has_summary(table_id, SummaryType.ROW_SAMPLE):
            if not df.empty:
                n = min(len(df), ROW_SAMPLE_SIZE)
                sample = df.sample(n=n, random_state=ROW_SAMPLE_RANDOM_STATE)
                row_blocks = [
                    " | ".join(f"{c}: {v}" for c, v in r.items())
                    for _, r in sample.iterrows()
                ]
                rows = [
                    (int(table_id), i, SummaryType.ROW_SAMPLE, txt)
                    for i, txt in enumerate(row_blocks)
                ]
                insert_summaries(self.cursor, self.dbms, rows)

        if real_contexts:
            if force or not self._has_context(table_id, "real"):
                rows = [
                    (int(table_id), "real", i, ctx) for i, ctx in enumerate(real_contexts)
                ]
                insert_contexts(self.cursor, self.dbms, rows)

        profile_text: Optional[str] = None
        semantic_notes: Optional[str] = None
        metadata: Optional[dict] = None
        if self.cfg.generate_content_profile:
            cached_profile = fetch_autoddg_profile(self.cursor, self.dbms, table_id, "content")
            if force or cached_profile is None:
                try:
                    profile_text, semantic_notes, metadata = profile_dataframe_with_metadata(df)
                    upsert_autoddg_profile(
                        self.cursor, self.dbms, table_id, "content",
                        json.dumps(metadata, default=str),
                    )
                    if force or not self._has_context(table_id, "syn:content_profile"):
                        serialized = serialize_content_profile(profile_text, semantic_notes)
                        insert_contexts(
                            self.cursor, self.dbms,
                            [(int(table_id), "syn:content_profile", 0, serialized)],
                        )
                except Exception as exc:
                    LOG.warning("A5 content profile failed for TableId=%d: %s", table_id, exc)
            else:
                metadata = json.loads(cached_profile)
                profile_text = _render_profile_summary(metadata)
                semantic_notes = _render_semantic_notes(metadata)

        semantic_results: dict = {}
        if self.cfg.generate_semantic_profile:
            cached = fetch_autoddg_profile(self.cursor, self.dbms, table_id, "semantic")
            if force or cached is None:
                sample_df = sample_for_semantic(df)
                col_to_samples = {
                    c: sample_df[c].astype(str).tolist() for c in cols
                }
                prompts = [build_semantic_prompt(c, col_to_samples[c]) for c in cols]
                answers = _batch_chat(
                    self.llm, SEMANTIC_SYSTEM_MESSAGE, prompts, max_new_tokens=_SEMANTIC_MAX_NEW_TOKENS,
                )
                # 3 LLM attempts per column (1 initial + 2 retries), matching AutoDDG semantic.py:188-192.
                for col, ans in zip(cols, answers):
                    parsed = parse_semantic_response(ans)
                    retries = 0
                    while parsed is None and retries < _RETRY_SEMANTIC - 1:
                        retry_ans = _chat(
                            self.llm, SEMANTIC_SYSTEM_MESSAGE,
                            build_semantic_prompt(col, col_to_samples[col]),
                            max_new_tokens=_SEMANTIC_MAX_NEW_TOKENS,
                        )
                        parsed = parse_semantic_response(retry_ans)
                        retries += 1
                    if parsed is not None:
                        semantic_results[col] = parsed
                # Only persist when at least one column parsed successfully.
                # An empty dict would be written as "{}" and fetched as non-None
                # on the next run, silently skipping the LLM retry.
                if semantic_results:
                    upsert_autoddg_profile(
                        self.cursor, self.dbms, table_id, "semantic",
                        json.dumps(semantic_results, default=str),
                    )
                else:
                    LOG.warning(
                        "A6 semantic profile: all columns failed parse for TableId=%d; not caching",
                        table_id,
                    )
            else:
                semantic_results = json.loads(cached)

            if semantic_results and (force or not self._has_context(table_id, "syn:semantic_profile")):
                text = render_table_summary(cols, semantic_results)
                insert_contexts(
                    self.cursor, self.dbms,
                    [(int(table_id), "syn:semantic_profile", 0, text)],
                )

        topic: str = ""
        if self.cfg.generate_topic:
            cached_topic = fetch_autoddg_topic(self.cursor, self.dbms, table_id)
            if force or cached_topic is None:
                title = resolve_table_title(filename, table_id)
                csv_text = sample_for_topic_csv(df)
                prompt = build_topic_prompt(title=title, original_description=None, dataset_sample=csv_text)
                topic = _chat(
                    self.llm, TOPIC_SYSTEM_MESSAGE, prompt,
                    max_new_tokens=_TOPIC_MAX_NEW_TOKENS,
                ).strip()
                upsert_autoddg_topic(self.cursor, self.dbms, table_id, topic)
            else:
                topic = cached_topic

            if self.cfg.index_topic_standalone and topic:
                if force or not self._has_context(table_id, "syn:topic"):
                    insert_contexts(
                        self.cursor, self.dbms,
                        [(int(table_id), "syn:topic", 0, topic)],
                    )

        ufd_text: str = ""
        if self.cfg.generate_ufd:
            if force or not self._has_context(table_id, "syn:ufd"):
                csv_text = sample_for_topic_csv(df)  # same random_state=0 sample as A7
                semantic_text = (
                    render_table_summary(cols, semantic_results) if semantic_results else ""
                )
                use_topic_in_ufd = self.cfg.topic_into_ufd_sfd
                prompt = build_ufd_prompt(
                    dataset_sample=csv_text,
                    dataset_profile=profile_text or "",
                    semantic_profile=semantic_text,
                    data_topic=topic,
                    use_profile=bool(profile_text),
                    use_semantic_profile=bool(semantic_text),
                    use_topic=use_topic_in_ufd,
                    description_words=100,
                )
                ufd_text = _chat(
                    self.llm, UFD_SYSTEM_MESSAGE, prompt,
                    max_new_tokens=_UFD_MAX_NEW_TOKENS,
                ).strip()
                insert_contexts(
                    self.cursor, self.dbms,
                    [(int(table_id), "syn:ufd", 0, ufd_text)],
                )
            else:
                qmark = "?" if self.dbms == "duckdb" else "%s"
                self.cursor.execute(
                    f"SELECT Text FROM blend_nl_contexts WHERE TableId={qmark} AND Source='syn:ufd' AND ContextIdx=0",
                    (int(table_id),),
                )
                row = self.cursor.fetchone()
                if row is None:
                    LOG.warning(
                        "A8 UFD cache miss for TableId=%d: _has_context returned True "
                        "but SELECT returned no row; ufd_text will be empty",
                        table_id,
                    )
                ufd_text = row[0] if row else ""

        if self.cfg.generate_sfd:
            if not ufd_text:
                # UFD not generated this run; try to fetch from a prior run.
                qmark = "?" if self.dbms == "duckdb" else "%s"
                self.cursor.execute(
                    f"SELECT Text FROM blend_nl_contexts "
                    f"WHERE TableId={qmark} AND Source='syn:ufd' AND ContextIdx=0",
                    (int(table_id),),
                )
                row = self.cursor.fetchone()
                ufd_text = row[0] if row else ""
            if not ufd_text:
                LOG.warning(
                    "A9 SFD skipped for TableId=%d: no UFD text available "
                    "(set cfg.generate_ufd=True or pre-generate UFD)",
                    table_id,
                )
            elif force or not self._has_context(table_id, "syn:sfd"):
                topic_slot = topic if self.cfg.topic_into_ufd_sfd else ""
                prompt = build_sfd_prompt(topic=topic_slot, initial_description=ufd_text)
                sfd_text = _chat(
                    self.llm, SFD_SYSTEM_MESSAGE, prompt,
                    max_new_tokens=_SFD_MAX_NEW_TOKENS,
                ).strip()
                insert_contexts(
                    self.cursor, self.dbms,
                    [(int(table_id), "syn:sfd", 0, sfd_text)],
                )

        if self.cfg.generate_narration_profiled:
            if force or not self._has_summary(table_id, SummaryType.COLUMN_NARRATION_PROFILED):
                stats_lines = {
                    c: extract_column_stats_line(profile_text or "", c) for c in cols
                }
                sem_lines = {
                    c: render_table_summary([c], {c: semantic_results.get(c, {})})
                        .split("\n", 1)[-1]  # drop the header line
                    if c in semantic_results else ""
                    for c in cols
                }
                a4_prompts = []
                for c in cols:
                    a4_prompts.append(
                        build_profiled_narration_prompt(
                            columns=cols,
                            target=c,
                            column_stats_line=stats_lines[c],
                            sample_values=extract_sample_values(df, c),
                            semantic_sentence=sem_lines[c],
                        )
                    )
                answers = self.llm.generate(
                    a4_prompts, max_new_tokens=COLUMN_NARRATION_MAX_NEW_TOKENS,
                )
                rows = [
                    (int(table_id), i, SummaryType.COLUMN_NARRATION_PROFILED, f"{col}: {ans}".strip())
                    for i, (col, ans) in enumerate(zip(cols, answers))
                ]
                insert_summaries(self.cursor, self.dbms, rows)


__all__ = ["NLArtifactBuilder"]
