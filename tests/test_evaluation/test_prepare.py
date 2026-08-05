import tarfile
import zipfile

import pytest

from evaluation.nlseeker.datasets import DatasetSpec
from evaluation.nlseeker.prepare import prepare_dataset, sha256_file

SPEC = DatasetSpec('mini', 'mini.tar', 'mini_questions.jsonl', 'mini.zip', 'mini')


def make_sources(datasets_dir):
    datasets_dir.mkdir(parents=True, exist_ok=True)
    lake = datasets_dir / 'stage'
    lake.mkdir(exist_ok=True)
    (lake / 'A_SEP_table_1.csv').write_text('col\n1\n')
    with tarfile.open(datasets_dir / 'mini.tar', 'w') as tar:
        tar.add(lake / 'A_SEP_table_1.csv', arcname='mini/A_SEP_table_1.csv')
    with zipfile.ZipFile(datasets_dir / 'mini.zip', 'w') as z:
        z.writestr('mini/bx_mini.jsonl', '{"context_id": "c", "question_bx1": "q", "question_bx2": "q", "answer_tables": ["table_1"]}\n')
        z.writestr('mini/contexts_mini.jsonl', '{"table": "table_1", "context": "ctx"}\n')
    (datasets_dir / 'mini_questions.jsonl').write_text('{}\n')


def test_prepare_extracts_lake_and_bench_files(tmp_path):
    make_sources(tmp_path / 'datasets')
    prepared = prepare_dataset(SPEC, tmp_path / 'datasets', tmp_path / 'cache')
    assert (prepared.lake_dir / 'A_SEP_table_1.csv').is_file()
    assert prepared.bx_jsonl.read_text().startswith('{"context_id"')
    assert prepared.contexts_jsonl.is_file()
    assert prepared.tar_sha256 == sha256_file(tmp_path / 'datasets' / 'mini.tar')


def test_prepare_skips_reextraction_when_hash_matches(tmp_path):
    make_sources(tmp_path / 'datasets')
    prepared = prepare_dataset(SPEC, tmp_path / 'datasets', tmp_path / 'cache')
    marker = prepared.lake_dir / '.tar.sha256'
    stamp = marker.stat().st_mtime_ns
    prepare_dataset(SPEC, tmp_path / 'datasets', tmp_path / 'cache')
    assert marker.stat().st_mtime_ns == stamp


def test_prepare_reextracts_on_hash_mismatch(tmp_path):
    make_sources(tmp_path / 'datasets')
    prepared = prepare_dataset(SPEC, tmp_path / 'datasets', tmp_path / 'cache')
    (prepared.lake_dir / '.tar.sha256').write_text('stale')
    (prepared.lake_dir / 'junk.csv').write_text('x')
    prepared = prepare_dataset(SPEC, tmp_path / 'datasets', tmp_path / 'cache')
    assert not (prepared.lake_dir / 'junk.csv').exists()


def test_prepare_missing_source_raises(tmp_path):
    (tmp_path / 'datasets').mkdir()
    with pytest.raises(FileNotFoundError):
        prepare_dataset(SPEC, tmp_path / 'datasets', tmp_path / 'cache')
