"""
Fallback table downloader: the original GCS bucket `pneuma_open` is gone (404).
This script pulls each dataset tar from the Google Drive file IDs listed in
README.md, then runs the same extract + `_SEP_` rename post-processing as
the upstream `downloader.py`.
"""
import argparse
import os
import sys
import tarfile

import gdown
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Drive file IDs from data_src/tables/README.md (also benchmarks/content/README.md for BIRD)
DATASETS = {
    "public_bi":       ("pneuma_public_bi.tar",        "1_J532XG7jDBmH4pQ30OHYP8rxBvJtSeE"),
    "chicago":         ("pneuma_chicago_10K.tar",      "1jNd1b8JIL1MLOTQlzMe3T7TIylqAPQ86"),
    "chembl":          ("pneuma_chembl_10K.tar",       "1OKRfd8LMpJJFTF2LX_tlNmWzKG25DXDC"),
    "fetaqa":          ("pneuma_fetaqa.tar",           "1cBREalsBSnnrTg0nWr62ohLXMwC5ffpX"),
    "adventure":       ("pneuma_adventure_works.tar",  "1n7RjFpxsyBlCrXTl0Z9JMMm96q7GJvlc"),
    # BIRD tar isn't in tables/README.md — only the content benchmark IDs are public.
    # BIRD original/annotated tarballs were only on pneuma_open. Skip by default; user
    # can fetch BIRD source separately (https://bird-bench.github.io).
}


def post_process_dataset(dataset_path: str):
    for table in os.listdir(dataset_path):
        if "_SEP_" in table:
            os.rename(
                os.path.join(dataset_path, table),
                os.path.join(dataset_path, table.split("_SEP_")[1]),
            )
    print(f"Processed {dataset_path}")


def extract_tar(tar_path: str, extract_path: str):
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=extract_path)
    print(f"Extracted {tar_path} -> {extract_path}")
    os.remove(tar_path)


def fetch(name: str):
    tar_name, file_id = DATASETS[name]
    out_tar = os.path.join(SCRIPT_DIR, tar_name)
    out_dir = out_tar[:-4]
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        print(f"[skip] {name}: {out_dir} already populated")
        return
    print(f"[fetch] {name}: gdown {file_id} -> {out_tar}")
    gdown.download(id=file_id, output=out_tar, quiet=False)
    extract_tar(out_tar, SCRIPT_DIR)
    post_process_dataset(out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="all",
                        help=f"one of {list(DATASETS)} or 'all'")
    args = parser.parse_args()

    targets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for name in tqdm(targets, desc="datasets"):
        try:
            fetch(name)
        except Exception as exc:
            print(f"[error] {name}: {exc}", file=sys.stderr)
