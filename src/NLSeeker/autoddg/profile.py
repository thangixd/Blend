from __future__ import annotations

from datetime import datetime

import pandas as pd


def _render_profile_summary(metadata: dict) -> str:
    profile_summary: list[str] = []
    for column_meta in metadata.get("columns", []):
        column_summary = f"**{column_meta['name']}**: "
        structural_type = column_meta.get("structural_type", "Unknown")
        column_summary += f"Data is of type {structural_type.split('/')[-1].lower()}. "

        if "num_distinct_values" in column_meta:
            num_distinct_values = column_meta["num_distinct_values"]
            column_summary += f"There are {num_distinct_values} unique values. "

        if "coverage" in column_meta:
            low = 0
            high = 0
            for coverage in column_meta["coverage"]:
                lower_bound = coverage["range"].get("gte", low)
                upper_bound = coverage["range"].get("lte", high)
                low = min(low, lower_bound)
                high = max(high, upper_bound)
            column_summary += f"Coverage spans from {low} to {high}. "

        profile_summary.append(column_summary)

    return (
        "The key data profile information for this dataset includes:\n"
        + "\n".join(profile_summary)
    )


def _render_semantic_notes(metadata: dict) -> str:
    semantic_summary: list[str] = []
    if "temporal_coverage" in metadata:
        for temp_cov in metadata["temporal_coverage"]:
            column_names = ", ".join(temp_cov.get("column_names", []))
            temporal_resolution = temp_cov.get("temporal_resolution", "unknown")
            range_values = [
                entry["range"].get("gte")
                for entry in temp_cov.get("ranges", [])
                if entry.get("range")
            ] + [
                entry["range"].get("lte")
                for entry in temp_cov.get("ranges", [])
                if entry.get("range")
            ]
            range_values = [value for value in range_values if value is not None]
            if range_values:
                min_value = datetime.fromtimestamp(min(range_values)).strftime("%Y-%m-%d")
                max_value = datetime.fromtimestamp(max(range_values)).strftime("%Y-%m-%d")
                date_range = f"from {min_value} to {max_value}"
                semantic_summary.append(
                    f"**Temporal coverage** for columns {column_names} with resolution "
                    f"{temporal_resolution}, covering {date_range}."
                )

    if "spatial_coverage" in metadata:
        for spatial_cov in metadata["spatial_coverage"]:
            column_names = ", ".join(spatial_cov.get("column_names", []))
            spatial_resolution = spatial_cov.get("type", "unknown")
            semantic_summary.append(
                f"**Spatial coverage** for columns {column_names}, "
                f"with type {spatial_resolution}."
            )

    return "\n".join(semantic_summary)


def profile_dataframe(df: pd.DataFrame) -> tuple[str, str]:
    profile_text, semantic_text, _ = profile_dataframe_with_metadata(df)
    return profile_text, semantic_text


def profile_dataframe_with_metadata(df: pd.DataFrame) -> tuple[str, str, dict]:
    import datamart_profiler

    metadata = datamart_profiler.process_dataset(df)
    return _render_profile_summary(metadata), _render_semantic_notes(metadata), metadata


def serialize_content_profile(profile_text: str, semantic_text: str) -> str:
    if not semantic_text:
        return profile_text
    return f"{profile_text}\n\n{semantic_text}"


__all__ = [
    "profile_dataframe",
    "profile_dataframe_with_metadata",
    "serialize_content_profile",
]
