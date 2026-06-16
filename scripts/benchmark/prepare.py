from __future__ import annotations

import csv
import json
import re
import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from glob import glob
from pathlib import Path

from scripts.benchmark.datasets import DatasetSpec


_TABLE_ID_RE = re.compile(r".*_SEP_(table_\d+)\.csv$")
# Bookkeeping CSVs prepare itself writes into the lake dir alongside the data
# tables. Listed by basename so a leading-underscore data table (e.g. FeTaQA's
# ``.338``/``.458``/``.257`` Wikipedia titles, which become ``_338``/``_458``/
# ``_257`` after ``_SEP_table_<n>.csv`` flattening) is NOT excluded.
_BOOKKEEPING_NAMES = frozenset({"_metadata.csv"})


@dataclass(frozen=True)
class PrepareResult:
    lake_dir: Path
    n_tables: int
    n_context_rows: int
    dropped_context_rows: int
    unresolvable_questions: int


def prepare(spec: DatasetSpec, out_root: Path) -> PrepareResult:
    lake_dir = out_root / "lakes" / spec.name
    if lake_dir.exists():
        shutil.rmtree(lake_dir)
    lake_dir.mkdir(parents=True)

    _extract_tar(spec, lake_dir)
    pneuma_id_to_table_id, table_id_to_pneuma_id = _build_manifest(lake_dir)

    n_context_rows, dropped_context_rows = _write_metadata_csv(
        spec, lake_dir, pneuma_id_to_table_id
    )
    _extract_bx_jsonl(spec, lake_dir)

    manifest = {
        "table_id_to_pneuma_id": {str(k): v for k, v in table_id_to_pneuma_id.items()},
        "pneuma_id_to_table_id": pneuma_id_to_table_id,
        "lake_glob": str(lake_dir / "*.csv"),
        "n_tables": len(pneuma_id_to_table_id),
        "health": {
            "n_context_rows": n_context_rows,
            "dropped_context_rows": dropped_context_rows,
        },
    }
    (lake_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2))

    unresolvable = _sweep_unresolvable_questions(spec, lake_dir, pneuma_id_to_table_id)

    manifest["health"]["unresolvable_questions"] = unresolvable
    (lake_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2))

    return PrepareResult(
        lake_dir=lake_dir,
        n_tables=len(pneuma_id_to_table_id),
        n_context_rows=n_context_rows,
        dropped_context_rows=dropped_context_rows,
        unresolvable_questions=unresolvable,
    )


def _extract_tar(spec: DatasetSpec, lake_dir: Path) -> None:
    """Flatten ``<tar_root>/foo.csv`` -> ``<lake_dir>/foo.csv``."""
    with tarfile.open(spec.tar_path, "r") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            name = member.name
            if not name.endswith(".csv"):
                continue
            # Strip the leading "<tar_root>/" prefix.
            prefix = spec.tar_root + "/"
            stripped = name[len(prefix):] if name.startswith(prefix) else name
            target = lake_dir / Path(stripped).name  # flatten any nested dirs
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            target.write_bytes(extracted.read())


def _build_manifest(lake_dir: Path) -> tuple[dict[str, int], dict[int, str]]:
    """Assign integer TableIds via ``sorted(glob(*.csv))`` and parse PNEUMA suffixes.

    Enumerates every ``*.csv`` in the lake; basenames in ``_BOOKKEEPING_NAMES``
    (e.g. ``_metadata.csv``, written into the lake dir by prepare itself) are
    skipped silently and any other non-matching file still raises.
    """
    files = sorted(glob(str(lake_dir / "*.csv")))
    pneuma_id_to_table_id: dict[str, int] = {}
    table_id_to_pneuma_id: dict[int, str] = {}
    table_id = 0
    for path in files:
        name = Path(path).name
        if name in _BOOKKEEPING_NAMES:
            continue
        m = _TABLE_ID_RE.match(name)
        if not m:
            raise RuntimeError(
                f"prepare: {name!r} does not match PNEUMA's "
                "<title>_SEP_table_<n>.csv naming. Refusing to build a "
                "corrupt manifest."
            )
        pneuma_id = m.group(1)
        if pneuma_id in pneuma_id_to_table_id:
            other_tid = pneuma_id_to_table_id[pneuma_id]
            other_path = next(p for p in files
                              if (om := _TABLE_ID_RE.match(Path(p).name))
                              and om.group(1) == pneuma_id and p != path)
            raise RuntimeError(
                f"prepare: duplicate PNEUMA id {pneuma_id!r} in "
                f"{Path(other_path).name!r}, {name!r}. Manifest "
                "cannot be bidirectional."
            )
        pneuma_id_to_table_id[pneuma_id] = table_id
        table_id_to_pneuma_id[table_id] = pneuma_id
        table_id += 1
    return pneuma_id_to_table_id, table_id_to_pneuma_id


def _write_metadata_csv(
    spec: DatasetSpec,
    lake_dir: Path,
    pneuma_id_to_table_id: dict[str, int],
) -> tuple[int, int]:
    """Project contexts_<ds>.jsonl into the (TableId, Context) schema."""
    out = lake_dir / "_metadata.csv"
    n_rows = 0
    n_dropped = 0
    with zipfile.ZipFile(spec.zip_path) as z, out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["TableId", "Context"])
        with z.open(spec.contexts_jsonl_inside_zip) as raw:
            for line in raw:
                if not line.strip():
                    continue
                row = json.loads(line)
                pneuma_id = row.get("table")
                context = row.get("context", "")
                if not context:
                    continue
                tid = pneuma_id_to_table_id.get(pneuma_id)
                if tid is None:
                    n_dropped += 1
                    continue
                writer.writerow([tid, context])
                n_rows += 1
    return n_rows, n_dropped


def _extract_bx_jsonl(spec: DatasetSpec, lake_dir: Path) -> None:
    """Copy bx_<ds>.jsonl out of the zip, alongside the lake."""
    out = lake_dir / "_bx_questions.jsonl"
    with zipfile.ZipFile(spec.zip_path) as z:
        with z.open(spec.bx_jsonl_inside_zip) as raw, out.open("wb") as f:
            f.write(raw.read())


def _sweep_unresolvable_questions(
    spec: DatasetSpec,
    lake_dir: Path,
    pneuma_id_to_table_id: dict[str, int],
) -> int:
    """Count questions whose answer_tables references an unknown PNEUMA id."""
    bx_path = lake_dir / "_bx_questions.jsonl"
    sources = [spec.content_jsonl, bx_path]
    n_bad = 0
    for src in sources:
        if not src.exists():
            continue
        for line in src.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for pid in row.get("answer_tables", []):
                if pid not in pneuma_id_to_table_id:
                    n_bad += 1
                    break
    return n_bad


__all__ = ["prepare", "PrepareResult"]
