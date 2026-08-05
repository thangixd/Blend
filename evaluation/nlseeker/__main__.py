import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from evaluation.nlseeker.config import EvalConfig, load_config
from evaluation.nlseeker.datasets import DATASETS, FAMILY_FIELDS
from evaluation.nlseeker.prepare import prepare_dataset, sha256_file
from evaluation.nlseeker.report import aggregate, collect_records, write_summary
from evaluation.nlseeker.rundir import RunDir, write_ini

# Typing imports
from typing import Optional

DATA_ROOT = Path('data/Evaluation/NLSeeker')
DATASETS_DIR = DATA_ROOT / 'datasets'
RESULTS_DIR = DATA_ROOT / 'results'
CACHE_DIR = DATA_ROOT / 'cache'


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='evaluation.nlseeker',
                                     description='Run the NLSeeker evaluation suite.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--config', help='YAML config for a fresh run.')
    group.add_argument('--resume', help='Existing .partial run directory to resume.')
    return parser


def _ping(base_url: str, api_key: str) -> list:
    request = urllib.request.Request(f'{base_url.rstrip("/")}/models',
                                     headers={'Authorization': f'Bearer {api_key}'})
    try:
        response = urllib.request.urlopen(request, timeout=10)
    except urllib.error.HTTPError as error:
        raise ConnectionError(f'endpoint check failed: {base_url} (HTTP {error.code})') from error
    except OSError as error:
        raise ConnectionError(f'endpoint not reachable: {base_url} ({error})') from error
    try:
        body = json.loads(response.read())
    finally:
        response.close()
    return [item['id'] for item in body['data']]


def _count_questions(datasets_dir: Path, name: str, family: str, limit: Optional[int]) -> int:
    spec = DATASETS[name]
    source = FAMILY_FIELDS[family][0]
    if source == 'bc':
        text = (datasets_dir / spec.questions).read_text()
    else:
        with zipfile.ZipFile(datasets_dir / spec.zip) as zf:
            text = zf.read(f'{spec.zip_name}/bx_{spec.zip_name}.jsonl').decode()
    count = sum(1 for line in text.splitlines() if line.strip())
    return count if limit is None else min(count, limit)


def preflight(cfg: EvalConfig, datasets_dir: Path) -> dict:
    for name in cfg.datasets:
        spec = DATASETS[name]
        for filename in (spec.tar, spec.zip, spec.questions):
            if not (datasets_dir / filename).is_file():
                raise FileNotFoundError(f'dataset source missing: {datasets_dir / filename}')
    models = {
        'llm': _ping(cfg.nlseeker['llm_base_url'], cfg.nlseeker['llm_api_key']),
        'embedding': _ping(cfg.nlseeker['embedding_base_url'], cfg.nlseeker['embedding_api_key']),
    }
    combos = cfg.combos()
    total_questions = sum(
        _count_questions(datasets_dir, name, family, cfg.questions_limit) * len(combos)
        for name in cfg.datasets for family in cfg.families)
    print(f'{len(cfg.datasets)} dataset(s) x {len(cfg.families)} family/ies x '
          f'{len(combos)} combo(s): {combos}')
    print(f'{total_questions} query/ies planned in total')
    return models


def _versions() -> dict:
    import bm25s
    import chromadb
    import duckdb
    return {'python': sys.version.split()[0], 'duckdb': duckdb.__version__,
            'chromadb': chromadb.__version__, 'bm25s': bm25s.__version__}


def _run_dataset(cfg: EvalConfig, run: RunDir, name: str) -> None:
    from evaluation.nlseeker.build import build_dataset_artifacts
    from evaluation.nlseeker.datasets import load_questions
    from evaluation.nlseeker.runner import load_id_map, run_family

    started = time.monotonic()
    run.set_dataset(name, status='building')

    prepared = prepare_dataset(DATASETS[name], DATASETS_DIR, CACHE_DIR)
    run.set_dataset(name, tar_sha256=prepared.tar_sha256, zip_sha256=prepared.zip_sha256,
                    questions_sha256=prepared.questions_sha256)

    artifacts = run.artifacts_dir(name)
    ini_path = write_ini(artifacts, name, cfg.nlseeker, alpha=cfg.alpha[0], n=cfg.n[0])
    build_dataset_artifacts(artifacts, prepared, name)
    build_done = time.monotonic()

    run.set_dataset(name, status='running')
    id_map = load_id_map(artifacts / 'eval.duckdb', name)
    out_jsonl = run.dataset_dir(name) / 'per_query.jsonl'
    for family in cfg.families:
        source = FAMILY_FIELDS[family][0]
        questions_path = prepared.questions_jsonl if source == 'bc' else prepared.bx_jsonl
        questions = load_questions(questions_path, family, cfg.questions_limit)
        print(f'{name}/{family}: {len(questions)} questions x {len(cfg.combos())} combo(s)')
        run_family(ini_path, questions, family, cfg.combos(), id_map, out_jsonl)
    finished = time.monotonic()

    run.set_dataset(name, status='complete', duration_s=round(finished - started, 1),
                    build_s=round(build_done - started, 1), run_s=round(finished - build_done, 1))


def run_evaluation(cfg: EvalConfig, run: RunDir) -> bool:
    for name in cfg.datasets:
        if run.dataset_status(name) == 'complete':
            print(f'{name}: already complete, skipping')
            continue
        run.wipe_dataset(name)
        try:
            _run_dataset(cfg, run, name)
        except Exception as error:  # noqa: BLE001 - one dataset must not kill the rest
            run.set_dataset(name, status='failed', error=f'{type(error).__name__}: {error}')
            print(f'{name}: FAILED - {error}', file=sys.stderr)

    complete = [name for name in cfg.datasets if run.dataset_status(name) == 'complete']
    write_summary(aggregate(collect_records(run.path, complete)), run.path)
    run.update_meta(versions=_versions(),
                    config_sha256=sha256_file(run.config_path))
    return len(complete) == len(cfg.datasets)


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.resume:
        run = RunDir.open_partial(Path(args.resume))
        cfg = load_config(run.config_path)
        models = preflight(cfg, DATASETS_DIR)
    else:
        config_path = Path(args.config)
        cfg = load_config(config_path)
        models = preflight(cfg, DATASETS_DIR)
        run = RunDir.create(RESULTS_DIR, cfg.tag, config_path.read_text())

    run.update_meta(endpoint_models=models)

    complete = run_evaluation(cfg, run)
    if complete:
        print(f'run complete: {run.finalize()}')
        return 0
    print(f'run incomplete, resume with: --resume {run.path}', file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(main())
