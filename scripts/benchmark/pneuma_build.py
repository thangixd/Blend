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
points ``add_tables`` at that view, and feeds the original
``_metadata.csv`` to ``add_metadata``.  The view is built with relative
symlinks so it costs nothing on disk and stays in sync if the lake is
regenerated.
"""
from __future__ import annotations

import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

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


def build_pneuma_index(dataset: str, *, force: bool = False) -> BuildResult:
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

    t0 = time.monotonic()
    pneuma.setup()
    pneuma.add_tables(str(tables_view), creator="bench", source="file")
    if metadata_src.exists():
        pneuma.add_metadata(str(metadata_src))
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


__all__ = [
    "build_pneuma_index",
    "BuildResult",
    "PNEUMA_ROOT",
    "INDEX_ROOT",
    "LAKE_ROOT",
]
