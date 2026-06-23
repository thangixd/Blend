"""Drive PNEUMA's public ``Pneuma()`` API to build a vector + BM25 index
under ``pneumaBenchdata/indexes/<dataset>/``.

Determinism + endpoint routing live in
``scripts.benchmark.pneuma_patches.apply_patches()``, which must be called
before this module's ``build_pneuma_index`` runs in production.  ``pneuma_cli``
applies the patches at startup; unit tests mock ``Pneuma`` and skip the patch.

Importing :mod:`scripts.benchmark.pneuma_patches` (without invoking
``apply_patches``) is sufficient to splice the vendored ``pneuma/src``
directory onto ``sys.path`` — the path-splicing happens at module import
time, not inside ``apply_patches``.

PNEUMA's ``add_tables`` walks the target folder with ``os.listdir`` and
ingests every CSV/Parquet it finds.  Our lake also contains a
``_metadata.csv`` bookkeeping file (``TableId,Context``) which PNEUMA
would mistake for a table.  This module therefore mirrors the lake into
a sibling ``_tables_view/`` directory containing only the table files,
points ``add_tables`` at that view, and feeds a translated metadata
file (see ``_write_pneuma_metadata_csv`` below) to ``add_metadata``.

Two metadata-schema details PNEUMA enforces (``registrar.py:408-424``):

  * Column names must be ``table_id, value`` (lowercase, "value", not
    ``TableId, Context`` like Blend writes).
  * Each ``table_id`` must equal the path the table was registered under
    in ``add_tables`` — i.e. the symlink path inside ``_tables_view/``,
    not the integer ``TableId`` from the lake's ``_manifest.json``.

We therefore translate Blend's ``_metadata.csv`` into a sidecar
``_metadata_pneuma.csv`` under the index dir before calling
``add_metadata``.  Blend's file stays untouched so NLSeeker's bench can
still consume it.
"""
from __future__ import annotations

import csv
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Importing pneuma_patches splices ``<repo>/pneuma/src`` onto sys.path
# (see the module-level block at the top of pneuma_patches.py); we don't
# need to call apply_patches() here — pneuma_cli does that at startup
# before invoking us.
from scripts.benchmark import pneuma_patches as _pp  # noqa: F401  (side-effect import)

from pneuma.pneuma import Pneuma  # vendored at Code/Blend/pneuma/src/pneuma

logger = logging.getLogger(__name__)

PNEUMA_ROOT = Path("pneumaBenchdata")
INDEX_ROOT = PNEUMA_ROOT / "indexes"
LAKE_ROOT = PNEUMA_ROOT / "lakes"

_METADATA_NAME = "_metadata.csv"
_PNEUMA_METADATA_NAME = "_metadata_pneuma.csv"
_TABLES_VIEW_NAME = "_tables_view"


@dataclass(frozen=True)
class BuildResult:
    index_name: str
    out_path: Path
    """Relative path to the dataset's index directory (``pneumaBenchdata/indexes/<dataset>``).
    It is relative to the process cwd at call time (i.e. the repo root when
    run via ``pneuma_cli``).  Pass ``out_path.resolve()`` for an absolute path.
    """
    build_wall_clock_s: float
    size_bytes: int


def build_pneuma_index(
    dataset: str,
    *,
    force: bool = False,
    embed_model: Any | None = None,
) -> BuildResult:
    out = INDEX_ROOT / dataset
    lake = LAKE_ROOT / dataset
    wall_clock_path = out / "_build_wall_clock_s.txt"

    if out.exists() and wall_clock_path.exists() and not force:
        return BuildResult(
            index_name=f"pneuma_{dataset}",
            out_path=out,
            build_wall_clock_s=float(wall_clock_path.read_text().strip()),
            size_bytes=_du_sb(out),
        )

    out.mkdir(parents=True, exist_ok=True)

    # Build a sibling view of the lake that excludes _metadata.csv so
    # PNEUMA's add_tables doesn't try to register the bookkeeping file as
    # a table.  The view is rebuilt fresh each call (cheap: symlinks).
    tables_view = _materialize_tables_view(lake, out)
    metadata_src = lake / _METADATA_NAME

    pneuma = Pneuma(
        out_path=str(out),
        use_local_model=False,
        openai_api_key="dummy",
        embed_path="BAAI/bge-base-en-v1.5",
        max_llm_batch_size=50,  # matches Blend NLSeekerConfig default
    )

    if embed_model is None:
        embed_model = _build_sentence_transformer("BAAI/bge-base-en-v1.5")
    pneuma.embed_model = embed_model

    t0 = time.monotonic()
    pneuma.setup()
    pneuma.add_tables(
        str(tables_view), creator="bench", source="file",
        accept_duplicates=True,
    )
    if metadata_src.exists():
        pneuma_metadata = _write_pneuma_metadata_csv(
            blend_metadata=metadata_src,
            lake=lake,
            tables_view=tables_view,
            out=out / _PNEUMA_METADATA_NAME,
        )
        if pneuma_metadata is None:
            logger.warning(
                "Lake %s has _metadata.csv but no rows survived translation "
                "(missing manifest, no matching symlinks, or empty file) — "
                "context strings will be absent from the index.",
                lake,
            )
        else:
            pneuma.add_metadata(str(pneuma_metadata))
    else:
        logger.warning(
            "No _metadata.csv found in lake %s — context strings will be absent "
            "from the index; retrieval quality may be lower than the paper baseline.",
            lake,
        )
    pneuma.summarize()
    pneuma.generate_index(index_name=f"pneuma_{dataset}")
    build_s = time.monotonic() - t0

    wall_clock_path.write_text(f"{build_s:.3f}")
    return BuildResult(
        index_name=f"pneuma_{dataset}",
        out_path=out,
        build_wall_clock_s=build_s,
        size_bytes=_du_sb(out),
    )


def _write_pneuma_metadata_csv(
    *,
    blend_metadata: Path,
    lake: Path,
    tables_view: Path,
    out: Path,
) -> Path | None:

    tid_to_view_path: dict[str, str] = {}
    table_id_int = 0
    for entry in sorted(lake.iterdir()):
        if not entry.is_file() or not entry.name.endswith(".csv"):
            continue
        if entry.name == _METADATA_NAME:
            continue
        view_path = tables_view / entry.name
        if not view_path.exists():
            # Lake has a CSV with no symlink in the view — skip.  Should
            # only happen if the view is stale; _materialize_tables_view
            # rebuilds it fresh each call, so this is defensive.
            table_id_int += 1
            continue
        tid_to_view_path[str(table_id_int)] = str(view_path)
        table_id_int += 1

    n_rows = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with blend_metadata.open() as f_in, out.open("w", newline="") as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.writer(f_out)
        writer.writerow(["table_id", "value"])
        for row in reader:
            blend_tid = row.get("TableId")
            context = row.get("Context", "")
            if blend_tid is None or not context:
                continue
            view_path = tid_to_view_path.get(str(blend_tid))
            if view_path is None:
                continue
            writer.writerow([view_path, context])
            n_rows += 1

    if n_rows == 0:
        out.unlink(missing_ok=True)
        return None
    return out


def _materialize_tables_view(lake: Path, out: Path) -> Path:
    """Create ``out/_tables_view/`` containing one symlink per real table
    file in ``lake`` (everything except ``_metadata.csv``).  Idempotent:
    rebuilds the view from scratch each call.
    """
    view = out / _TABLES_VIEW_NAME
    if view.exists():
        for entry in view.iterdir():
            if entry.is_symlink() or entry.is_file():
                entry.unlink()
            else:
                # Defensive: subdir wandered in; remove it.
                shutil.rmtree(entry)
    else:
        view.mkdir(parents=True)

    if not lake.exists():
        return view

    for entry in sorted(lake.iterdir()):
        if not entry.is_file():
            continue
        if entry.name == _METADATA_NAME:
            continue
        link = view / entry.name
        # Use relative symlink so the view survives lake relocations.
        target = os.path.relpath(entry.resolve(), start=view.resolve())
        link.symlink_to(target)
    return view


def _du_sb(path: Path) -> int:
    """Recursive size in bytes of all files under path."""
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def _build_sentence_transformer(name: str) -> Any:
    import torch  # local import: avoids a torch dep on this module's importers
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading PNEUMA embedder %s on %s", name, device)
    return SentenceTransformer(name, device=device)


__all__ = [
    "build_pneuma_index",
    "BuildResult",
    "PNEUMA_ROOT",
    "INDEX_ROOT",
    "LAKE_ROOT",
    "_build_sentence_transformer",
]
