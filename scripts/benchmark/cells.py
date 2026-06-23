from __future__ import annotations

from src.NLSeeker.db_schema import SummaryType

NARR = SummaryType.COLUMN_NARRATION
NARR_PROF = SummaryType.COLUMN_NARRATION_PROFILED
ROW = SummaryType.ROW_SAMPLE


def _ctx(*srcs):
    return frozenset(srcs)


CELL_PRESETS: dict = {
    "B0":     {"enabled_summary_types": frozenset({NARR, ROW}),
               "context_sources": _ctx("real")},
    "B0nctx": {"enabled_summary_types": frozenset({NARR, ROW}),
               "context_sources": frozenset()},
    "B1":     {"enabled_summary_types": frozenset({NARR, ROW}),
               "context_sources": _ctx("real", "syn:content_profile")},
    "B2":     {"enabled_summary_types": frozenset({NARR_PROF, ROW}),
               "context_sources": _ctx("real")},
    "B3":     {"enabled_summary_types": frozenset({NARR, ROW}),
               "context_sources": _ctx("real", "syn:ufd")},
    "B4":     {"enabled_summary_types": frozenset({NARR, ROW}),
               "context_sources": _ctx("real", "syn:sfd")},
    "B5":     {"enabled_summary_types": frozenset({NARR, ROW}),
               "context_sources": _ctx("real", "syn:ufd", "syn:sfd")},
    "B6":     {"enabled_summary_types": frozenset({NARR_PROF, ROW}),
               "context_sources": _ctx("real", "syn:content_profile")},
    "B7":     {"enabled_summary_types": frozenset({NARR, ROW}),
               "context_sources": _ctx("real", "syn:content_profile",
                                       "syn:semantic_profile", "syn:ufd", "syn:sfd")},
    "B8":     {"enabled_summary_types": frozenset({NARR_PROF, ROW}),
               "context_sources": _ctx("real", "syn:content_profile",
                                       "syn:semantic_profile", "syn:ufd", "syn:sfd")},
    "B8-r1":  {"enabled_summary_types": frozenset({NARR_PROF, ROW}),
               "context_sources": _ctx("syn:content_profile",
                                       "syn:semantic_profile", "syn:ufd", "syn:sfd")},
    "B10-r1": {"enabled_summary_types": frozenset({ROW}),
               "context_sources": _ctx("syn:content_profile",
                                       "syn:semantic_profile", "syn:ufd", "syn:sfd")},
    "B11":    {"enabled_summary_types": frozenset({NARR_PROF, ROW}),
               "context_sources": frozenset()},
    "B12":    {"enabled_summary_types": frozenset({NARR_PROF, ROW}),
               "context_sources": _ctx("real", "syn:content_profile",
                                       "syn:semantic_profile", "syn:ufd", "syn:sfd",
                                       "syn:topic"),
               "index_topic_standalone": True},
}


def preset_for_cell(cell_key: str) -> dict:
    """Return a shallow copy of the preset dict for *cell_key*.

    Raises ``KeyError`` listing the known keys when *cell_key* is absent.
    """
    if cell_key not in CELL_PRESETS:
        raise KeyError(
            f"Unknown cell {cell_key!r}; known cells: {sorted(CELL_PRESETS)}"
        )
    return dict(CELL_PRESETS[cell_key])


__all__ = ["CELL_PRESETS", "preset_for_cell"]
