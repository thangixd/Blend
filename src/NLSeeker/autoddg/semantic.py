from __future__ import annotations

import json
import re

# Typing imports
from typing import Iterable, Optional, Sequence

# Verbatim from AutoDDG
SEMANTIC_SYSTEM_MESSAGE = "You are a helpful assistant skilled in dataset semantic analysis."

# Group mode override 
SEMANTIC_GROUP_SYSTEM_MESSAGE = (
    "You are a helpful assistant skilled in dataset semantic analysis. "
    "You analyze multiple columns efficiently in a single response."
)

# 'Data Format' appears in the prompt body but is NOT in the schema or the renderer.
SEMANTIC_SCHEMA_KEYS = frozenset(
    {"Temporal", "Spatial", "Entity Type", "Domain-Specific Types", "Function/Usage Context"}
)

_SCHEMA_TEMPLATE = (
    "{'Temporal':\n"
    "    {\n"
    "        'isTemporal': Does this column contain temporal information? Yes or No,\n"
    "        'resolution': If Yes, specify the resolution (Year, Month, Day, Hour, etc.).\n"
    "    },\n"
    " 'Spatial': {'isSpatial': Does this column contain spatial information? Yes or No,\n"
    "             'resolution': If Yes, specify the resolution (Country, State, City, Coordinates, etc.).},\n"
    " 'Entity Type': What kind of entity does the column describe? (e.g., Person, Location, Organization, Product),\n"
    " 'Domain-Specific Types': What domain is this column from (e.g., Financial, Healthcare, E-commerce, Climate, Demographic),\n"
    " 'Function/Usage Context': How might the data be used (e.g., Aggregation Key, Ranking/Scoring, Interaction Data, Measurement).}\n"
)

_RESPONSE_EXAMPLE = (
    "{\n"
    '"Domain-Specific Types": "General",\n'
    '"Entity Type": "Temporal Entity",\n'
    '"Function/Usage Context": "Aggregation Key",\n'
    '"Spatial": {"isSpatial": false,\n'
    '            "resolution": ""},\n'
    '"Temporal": {"isTemporal": true,\n'
    '            "resolution": "Year"}\n'
    "}\n"
)

_USER_PROMPT_BODY = (
    "You are a dataset semantic analyzer. Based on the column name and sample values, "
    "classify the column into multiple semantic types.\n"
    "Please group the semantic types under the following categories:\n"
    "'Temporal', 'Spatial', 'Entity Type', 'Data Format', 'Domain-Specific Types', 'Function/Usage Context'.\n"
    "Following is the template {template}\n"
    "Please follow these rules:\n"
    "1. The output must be a valid JSON object that can be directly loaded by json.loads. Example response is {response_example}\n"
    "2. All keys from the template must be present in the response.\n"
    "3. All keys and string values must be enclosed in double quotes.\n"
    "4. There must be no trailing commas.\n"
    "5. Use booleans (true/false) and numbers without quotes.\n"
    "6. Do not include any additional information or context in the response.\n"
    "7. If you are unsure about a specific category, you can leave it as an empty string.\n"
    "\n"
    "Column name: {column_name}\n"
    "Sample values: {sample_values}\n"
)


def build_semantic_prompt(column_name: str, sample_values: Iterable[str]) -> str:
    """Build the per-column user prompt."""
    sample_text = ", ".join(str(v) for v in sample_values)
    return _USER_PROMPT_BODY.format(
        template=_SCHEMA_TEMPLATE,
        response_example=_RESPONSE_EXAMPLE,
        column_name=column_name,
        sample_values=sample_text,
    )


def fix_json_response(response_text: str) -> str:
    """Repair malformed LLM JSON: extract first ``{...}`` span, balance ``}``,
    strip trailing commas.
    """
    # Balance unclosed braces before regex extraction so inputs with no closing
    # brace at all still produce a parseable span.
    open_braces = response_text.count("{")
    close_braces = response_text.count("}")
    if open_braces > close_braces:
        response_text = response_text + "}" * (open_braces - close_braces)

    match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if not match:
        return response_text
    response_body = match.group()
    response_body = re.sub(r",\s*}", "}", response_body)
    return response_body


def parse_semantic_response(response_text: str) -> Optional[dict]:
    """Run AutoDDG's repair pass and ``json.loads``. Return None on failure."""
    try:
        return json.loads(fix_json_response(response_text))
    except json.JSONDecodeError:
        return None


def render_column_summary(column: str, semantic_description: dict) -> str:
    column_summary = f"**{column}**: "
    entity_type = semantic_description.get("Entity Type", "Unknown")
    if entity_type and entity_type.lower() not in {"", "unknown"}:
        column_summary += f"Represents {entity_type.lower()}. "

    temporal = semantic_description.get("Temporal", {})
    if isinstance(temporal, dict) and temporal.get("isTemporal"):
        resolution = temporal.get("resolution", "unknown")
        column_summary += f"Contains temporal data (resolution: {resolution}). "

    spatial = semantic_description.get("Spatial", {})
    if isinstance(spatial, dict) and spatial.get("isSpatial"):
        resolution = spatial.get("resolution", "unknown")
        column_summary += f"Contains spatial data (resolution: {resolution}). "

    domain_type = semantic_description.get("Domain-Specific Types", "Unknown")
    if domain_type and domain_type.lower() not in {"", "unknown"}:
        column_summary += f"Domain-specific type: {domain_type.lower()}. "

    function_context = semantic_description.get("Function/Usage Context", "Unknown")
    if function_context and function_context.lower() not in {"", "unknown"}:
        column_summary += f"Function/Usage context: {function_context.lower()}. "

    return column_summary


def render_table_summary(columns: Sequence[str], results: dict) -> str:
    """Per-column lines joined under the AutoDDG header."""
    lines = [
        render_column_summary(col, results[col])
        for col in columns
        if col in results and results[col] is not None
    ]
    return "The key semantic information for this dataset includes:\n" + "\n".join(lines)


SEMANTIC_SAMPLE_SIZE = 5
SEMANTIC_SAMPLE_RANDOM_STATE = 9  # AutoDDG


def sample_for_semantic(df) -> "pd.DataFrame":
    import pandas as pd

    if SEMANTIC_SAMPLE_SIZE < len(df):
        return df.sample(SEMANTIC_SAMPLE_SIZE, random_state=SEMANTIC_SAMPLE_RANDOM_STATE)
    return df


__all__ = [
    "SEMANTIC_SYSTEM_MESSAGE",
    "SEMANTIC_GROUP_SYSTEM_MESSAGE",
    "SEMANTIC_SCHEMA_KEYS",
    "SEMANTIC_SAMPLE_SIZE",
    "SEMANTIC_SAMPLE_RANDOM_STATE",
    "build_semantic_prompt",
    "fix_json_response",
    "parse_semantic_response",
    "render_column_summary",
    "render_table_summary",
    "sample_for_semantic",
]
