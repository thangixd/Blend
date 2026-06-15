from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EVAL_DIR = _PROJECT_ROOT / "EvaluationDataFromPneuma"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    tar_path: Path
    tar_root: str
    content_jsonl: Path
    zip_path: Path
    bx_jsonl_inside_zip: str
    contexts_jsonl_inside_zip: str


DATASETS: dict[str, DatasetSpec] = {
    "adventure_works": DatasetSpec(
        name="adventure_works",
        tar_path=_EVAL_DIR / "pneuma_adventure_works.tar",
        tar_root="pneuma_adventure_works",
        content_jsonl=_EVAL_DIR / "pneuma_adventure_works_questions_annotated.jsonl",
        zip_path=_EVAL_DIR / "adventure-20260604T041942Z-3-001.zip",
        bx_jsonl_inside_zip="adventure/bx_adventure.jsonl",
        contexts_jsonl_inside_zip="adventure/contexts_adventure_merged.jsonl",
    ),
    "chembl": DatasetSpec(
        name="chembl",
        tar_path=_EVAL_DIR / "pneuma_chembl_10K.tar",
        tar_root="pneuma_chembl_10K",
        content_jsonl=_EVAL_DIR / "pneuma_chembl_10K_questions_annotated.jsonl",
        zip_path=_EVAL_DIR / "chembl-20260604T041943Z-3-001.zip",
        bx_jsonl_inside_zip="chembl/bx_chembl.jsonl",
        contexts_jsonl_inside_zip="chembl/contexts_chembl_merged.jsonl",
    ),
    "public_bi": DatasetSpec(
        name="public_bi",
        tar_path=_EVAL_DIR / "pneuma_public_bi.tar",
        tar_root="pneuma_public_bi",
        content_jsonl=_EVAL_DIR / "pneuma_public_bi_questions_annotated.jsonl",
        zip_path=_EVAL_DIR / "public-20260604T041947Z-3-001.zip",
        bx_jsonl_inside_zip="public/bx_public.jsonl",
        contexts_jsonl_inside_zip="public/contexts_public_merged.jsonl",
    ),
    "chicago_open": DatasetSpec(
        name="chicago_open",
        tar_path=_EVAL_DIR / "pneuma_chicago_10K.tar",
        tar_root="pneuma_chicago_10K",
        content_jsonl=_EVAL_DIR / "pneuma_chicago_10K_questions_annotated.jsonl",
        zip_path=_EVAL_DIR / "chicago-20260604T041945Z-3-001.zip",
        bx_jsonl_inside_zip="chicago/bx_chicago.jsonl",
        contexts_jsonl_inside_zip="chicago/contexts_chicago_merged.jsonl",
    ),
    "fetaqa": DatasetSpec(
        name="fetaqa",
        tar_path=_EVAL_DIR / "pneuma_fetaqa.tar",
        tar_root="pneuma_fetaqa",
        content_jsonl=_EVAL_DIR / "pneuma_fetaqa_questions_annotated.jsonl",
        zip_path=_EVAL_DIR / "fetaqa-20260604T041946Z-3-001.zip",
        bx_jsonl_inside_zip="fetaqa/bx_fetaqa.jsonl",
        contexts_jsonl_inside_zip="fetaqa/contexts_fetaqa_merged.jsonl",
    ),
}


__all__ = ["DatasetSpec", "DATASETS"]
