import hashlib
import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

from evaluation.nlseeker.datasets import DatasetSpec


@dataclass(frozen=True)
class PreparedDataset:
    lake_dir: Path
    bx_jsonl: Path
    contexts_jsonl: Path
    questions_jsonl: Path
    tar_sha256: str
    zip_sha256: str
    questions_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _cached(target_dir: Path, marker_name: str, source_hash: str, extract) -> None:
    marker = target_dir / marker_name
    if marker.is_file() and marker.read_text() == source_hash:
        return
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)
    extract()
    marker.write_text(source_hash)


def prepare_dataset(spec: DatasetSpec, datasets_dir: Path, cache_dir: Path) -> PreparedDataset:
    tar_path = datasets_dir / spec.tar
    zip_path = datasets_dir / spec.zip
    questions_path = datasets_dir / spec.questions
    for path in (tar_path, zip_path, questions_path):
        if not path.is_file():
            raise FileNotFoundError(f'dataset source missing: {path}')

    tar_hash = sha256_file(tar_path)
    zip_hash = sha256_file(zip_path)

    lake_dir = cache_dir / 'lakes' / spec.name
    bench_dir = cache_dir / 'bench' / spec.name

    def extract_lake():
        # Tars carry one top-level directory; flatten it so lake_dir/*.csv globs work.
        with tarfile.open(tar_path) as tar:
            for member in tar.getmembers():
                if not member.name.endswith('.csv'):
                    continue
                member.name = Path(member.name).name
                tar.extract(member, lake_dir, filter='data')

    def extract_bench():
        with zipfile.ZipFile(zip_path) as z:
            for inside, out_name in ((f'{spec.zip_name}/bx_{spec.zip_name}.jsonl', 'bx.jsonl'),
                                     (f'{spec.zip_name}/contexts_{spec.zip_name}.jsonl', 'contexts.jsonl')):
                (bench_dir / out_name).write_bytes(z.read(inside))

    _cached(lake_dir, '.tar.sha256', tar_hash, extract_lake)
    _cached(bench_dir, '.zip.sha256', zip_hash, extract_bench)

    return PreparedDataset(
        lake_dir=lake_dir,
        bx_jsonl=bench_dir / 'bx.jsonl',
        contexts_jsonl=bench_dir / 'contexts.jsonl',
        questions_jsonl=questions_path,
        tar_sha256=tar_hash,
        zip_sha256=zip_hash,
        questions_sha256=sha256_file(questions_path),
    )
