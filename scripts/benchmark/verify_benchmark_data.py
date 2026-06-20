"""verify_benchmark_data.py - sanity-check that Blend's benchmark indexes
were built from the same bytes PNEUMA ships in its repo.

Run from the Blend project root:
    python -m scripts.benchmark.verify_benchmark_data [--dataset NAME]

Reports four layers per dataset:
  1. zip vs upstream PNEUMA repo (byte-equal merged.jsonl?)
  2. zip vs _metadata.csv (every row copied?)
  3. _metadata.csv vs corpus.jsonl (every row indexed?)
  4. table_id <-> pneuma_id mapping is bidirectional and consistent

Exits non-zero if any check fails so this can run in CI.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BENCH_DATA = PROJECT_ROOT / "benchmark-data"
PNEUMA_REPO = PROJECT_ROOT.parent / "pneuma"

# Map Blend dataset name -> PNEUMA repo subdirectory name. Blend appends
# `_works`/`_open`/`_bi` to several names; PNEUMA does not.
PNEUMA_DIR = {
    "adventure_works": "adventure",
    "chembl": "chembl",
    "chicago_open": "chicago",
    "fetaqa": "fetaqa",
    "public_bi": "public",
}

_TABLE_ID_RE = re.compile(r"_SEP_(table_\d+)\.csv$")


class Check:
    """One named check. Failures accumulate so the user sees every problem."""

    def __init__(self, name: str):
        self.name = name
        self.failures: list[str] = []
        self.notes: list[str] = []

    def fail(self, msg: str) -> None:
        self.failures.append(msg)

    def note(self, msg: str) -> None:
        self.notes.append(msg)

    @property
    def ok(self) -> bool:
        return not self.failures

    def render(self, indent: str = "  ") -> str:
        head = f"{'OK' if self.ok else 'FAIL'}  {self.name}"
        body = "\n".join(f"{indent}- {n}" for n in self.notes)
        if self.failures:
            body += "\n" + "\n".join(f"{indent}! {f}" for f in self.failures)
        return f"{head}\n{body}" if body else head


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _zip_sha256(zip_path: Path, member: str) -> str:
    h = hashlib.sha256()
    with zipfile.ZipFile(zip_path) as z, z.open(member) as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _spec(dataset: str):
    """Mirror scripts.benchmark.datasets without importing it (this script
    must work even when the benchmark module fails to import)."""
    eval_dir = PROJECT_ROOT / "EvaluationDataFromPneuma"
    suffix_map = {
        "adventure_works": ("adventure-20260604T041942Z-3-001.zip",
                            "adventure/contexts_adventure_merged.jsonl"),
        "chembl":          ("chembl-20260604T041943Z-3-001.zip",
                            "chembl/contexts_chembl_merged.jsonl"),
        "chicago_open":    ("chicago-20260604T041945Z-3-001.zip",
                            "chicago/contexts_chicago_merged.jsonl"),
        "fetaqa":          ("fetaqa-20260604T041946Z-3-001.zip",
                            "fetaqa/contexts_fetaqa_merged.jsonl"),
        "public_bi":       ("public-20260604T041947Z-3-001.zip",
                            "public/contexts_public_merged.jsonl"),
    }
    zip_name, member = suffix_map[dataset]
    return eval_dir / zip_name, member


def check_layer1_zip_vs_pneuma(dataset: str) -> Check:
    """Are the contexts in our local zip byte-equal to the upstream repo?"""
    c = Check(f"L1 zip ↔ upstream PNEUMA repo ({dataset})")
    zip_path, zip_member = _spec(dataset)
    pneuma_path = (PNEUMA_REPO / "data_src" / "benchmarks" / "content"
                   / PNEUMA_DIR[dataset]
                   / f"contexts_{PNEUMA_DIR[dataset]}_merged.jsonl")

    if not zip_path.exists():
        c.fail(f"missing zip: {zip_path}")
        return c
    if not pneuma_path.exists():
        c.note(f"upstream repo not present at {pneuma_path}; skipping byte compare")
        return c

    zip_hash = _zip_sha256(zip_path, zip_member)
    repo_hash = _sha256(pneuma_path)
    if zip_hash == repo_hash:
        c.note(f"sha256 match: {zip_hash[:16]}…")
    else:
        c.fail(f"sha256 mismatch - zip={zip_hash[:16]}… repo={repo_hash[:16]}…")
        # Count rows on each side to scope the divergence.
        with zipfile.ZipFile(zip_path) as z, z.open(zip_member) as f:
            n_zip = sum(1 for line in f if line.strip())
        n_repo = sum(1 for _ in pneuma_path.open() if _.strip())
        c.fail(f"row counts: zip={n_zip}, repo={n_repo}")
    return c


def check_layer2_zip_vs_metadata(dataset: str) -> Check:
    """Did prepare.py copy every (table, context) row from the zip into
    _metadata.csv? Compares per-table counts, not just totals - a swap that
    sums to the same total would slip past a total-only check."""
    c = Check(f"L2 zip ↔ _metadata.csv ({dataset})")
    zip_path, zip_member = _spec(dataset)
    meta_path = BENCH_DATA / "lakes" / dataset / "_metadata.csv"
    manifest_path = BENCH_DATA / "lakes" / dataset / "_manifest.json"

    if not zip_path.exists():
        c.fail(f"missing zip: {zip_path}")
        return c
    if not meta_path.exists():
        c.fail(f"missing _metadata.csv: {meta_path}")
        return c
    if not manifest_path.exists():
        c.fail(f"missing _manifest.json: {manifest_path}")
        return c

    manifest = json.loads(manifest_path.read_text())
    pid2tid: dict[str, int] = manifest["pneuma_id_to_table_id"]

    # Source per-table count from zip.
    src: dict[str, int] = defaultdict(int)
    with zipfile.ZipFile(zip_path) as z, z.open(zip_member) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            src[r["table"]] += 1

    # Indexed per-table count from _metadata.csv (TableId column).
    csv.field_size_limit(2**31 - 1)
    meta_by_tid: dict[int, int] = defaultdict(int)
    with meta_path.open() as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            meta_by_tid[int(row[0])] += 1

    # Compare per-pneuma_id.
    drift: list[str] = []
    src_total = sum(src.values())
    meta_total = sum(meta_by_tid.values())
    for pid, expected in src.items():
        tid = pid2tid.get(pid)
        if tid is None:
            drift.append(f"{pid}: in zip but not in manifest (drops {expected})")
            continue
        got = meta_by_tid.get(tid, 0)
        if got != expected:
            drift.append(f"{pid} (tid={tid}): zip={expected}, _metadata.csv={got}")

    c.note(f"contexts: zip={src_total}, _metadata.csv={meta_total}, "
           f"tables in zip={len(src)}, in manifest={len(pid2tid)}")
    if drift:
        for d in drift[:10]:
            c.fail(d)
        if len(drift) > 10:
            c.fail(f"… and {len(drift) - 10} more mismatches")
    return c


def check_layer3_metadata_vs_corpus(dataset: str) -> Check:
    """Did build_index.py index every row of _metadata.csv as a context doc
    in corpus.jsonl? doc_id format: <tid>_SEP_contexts-<i>."""
    c = Check(f"L3 _metadata.csv ↔ corpus.jsonl ({dataset})")
    meta_path = BENCH_DATA / "lakes" / dataset / "_metadata.csv"
    corpus_path = (BENCH_DATA / "indexes" / dataset / "nl-out" / "indexes"
                   / "fulltext" / f"benchmark_{dataset}" / "corpus.jsonl")

    if not meta_path.exists():
        c.fail(f"missing _metadata.csv: {meta_path}")
        return c
    if not corpus_path.exists():
        c.fail(f"missing corpus.jsonl: {corpus_path}")
        return c

    csv.field_size_limit(2**31 - 1)
    meta_by_tid: dict[int, int] = defaultdict(int)
    with meta_path.open() as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            meta_by_tid[int(row[0])] += 1

    corpus_by_tid: dict[int, int] = defaultdict(int)
    with corpus_path.open() as f:
        for line in f:
            rec = json.loads(line)
            doc_id = rec.get("doc_id", "")
            if "_SEP_contexts" not in doc_id:
                continue
            tid = int(doc_id.split("_SEP_")[0])
            corpus_by_tid[tid] += 1

    meta_total = sum(meta_by_tid.values())
    corpus_total = sum(corpus_by_tid.values())
    c.note(f"contexts: _metadata.csv={meta_total}, corpus.jsonl={corpus_total}")

    drift: list[str] = []
    for tid, expected in meta_by_tid.items():
        got = corpus_by_tid.get(tid, 0)
        if got != expected:
            drift.append(f"tid={tid}: _metadata.csv={expected}, corpus.jsonl={got}")
    if drift:
        for d in drift[:10]:
            c.fail(d)
        if len(drift) > 10:
            c.fail(f"… and {len(drift) - 10} more mismatches")
    return c


def check_layer4_mapping_consistency(dataset: str) -> Check:
    """The pneuma_id <-> table_id mapping must be 1:1 and the value
    extracted from each lake CSV's filename must match what the manifest
    records. A bug here means contexts can land on the wrong tables."""
    c = Check(f"L4 manifest ↔ lake filenames ({dataset})")
    manifest_path = BENCH_DATA / "lakes" / dataset / "_manifest.json"
    lake_dir = BENCH_DATA / "lakes" / dataset

    if not manifest_path.exists():
        c.fail(f"missing _manifest.json: {manifest_path}")
        return c

    manifest = json.loads(manifest_path.read_text())
    pid2tid: dict[str, int] = manifest["pneuma_id_to_table_id"]
    tid2pid: dict[str, str] = manifest["table_id_to_pneuma_id"]

    # Bidirectional?
    bad_fwd = [pid for pid in pid2tid if tid2pid.get(str(pid2tid[pid])) != pid]
    bad_rev = [tid for tid, pid in tid2pid.items() if pid2tid.get(pid) != int(tid)]
    if bad_fwd or bad_rev:
        c.fail(f"manifest not bidirectional: fwd-broken={len(bad_fwd)}, "
               f"rev-broken={len(bad_rev)}")

    # Re-derive the mapping from the lake the same way prepare.py does and
    # diff. This catches "manifest claims X→17 but the file at sorted-index
    # 17 is something else".
    files = sorted(lake_dir.glob("*.csv"))
    derived: dict[str, int] = {}
    tid = 0
    for path in files:
        if path.name == "_metadata.csv":
            continue
        m = _TABLE_ID_RE.search(path.name)
        if not m:
            c.fail(f"lake file {path.name!r} doesn't match _SEP_table_<n>.csv")
            continue
        derived[m.group(1)] = tid
        tid += 1

    diffs = [(pid, manifest_tid, derived.get(pid))
             for pid, manifest_tid in pid2tid.items()
             if derived.get(pid) != manifest_tid]
    if diffs:
        for pid, mtid, dtid in diffs[:10]:
            c.fail(f"{pid}: manifest tid={mtid}, derived tid={dtid}")
        if len(diffs) > 10:
            c.fail(f"… and {len(diffs) - 10} more mismatches")

    c.note(f"manifest tables={len(pid2tid)}, lake CSVs (excluding bookkeeping)"
           f"={len(derived)}")
    return c


def verify_dataset(dataset: str) -> bool:
    """Run all four layers for one dataset; return True iff all pass."""
    print(f"\n=== {dataset} ===")
    checks = [
        check_layer1_zip_vs_pneuma(dataset),
        check_layer2_zip_vs_metadata(dataset),
        check_layer3_metadata_vs_corpus(dataset),
        check_layer4_mapping_consistency(dataset),
    ]
    for c in checks:
        print(c.render())
    return all(c.ok for c in checks)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--dataset",
        choices=sorted(PNEUMA_DIR),
        action="append",
        help="Dataset(s) to verify; repeat the flag for multiple. Default: all.",
    )
    ns = p.parse_args(argv)
    datasets = ns.dataset if ns.dataset else sorted(PNEUMA_DIR)

    all_ok = True
    for ds in datasets:
        if not verify_dataset(ds):
            all_ok = False

    print("\n" + ("ALL CHECKS PASSED" if all_ok else "FAILURES - see above"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
