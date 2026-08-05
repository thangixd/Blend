import configparser

import pytest

from evaluation.nlseeker.rundir import RunDir, write_ini


def test_create_makes_partial_dir_with_config_and_meta(tmp_path):
    run = RunDir.create(tmp_path, 'baseline', 'datasets: [x]\n')
    assert run.path.name.endswith('__baseline.partial')
    assert run.config_path.read_text() == 'datasets: [x]\n'
    assert run.meta()['datasets'] == {}


def test_set_dataset_and_status_roundtrip(tmp_path):
    run = RunDir.create(tmp_path, 't', '')
    run.set_dataset('adventure_works', status='building')
    run.set_dataset('adventure_works', status='complete', duration_s=1.5)
    assert run.dataset_status('adventure_works') == 'complete'
    assert run.meta()['datasets']['adventure_works']['duration_s'] == 1.5
    assert run.dataset_status('never_seen') is None


def test_open_partial_resumes_existing_dir(tmp_path):
    created = RunDir.create(tmp_path, 't', 'cfg')
    reopened = RunDir.open_partial(created.path)
    assert reopened.config_path.read_text() == 'cfg'
    with pytest.raises(FileNotFoundError):
        RunDir.open_partial(tmp_path / 'missing.partial')


def test_wipe_dataset_removes_subdir(tmp_path):
    run = RunDir.create(tmp_path, 't', '')
    marker = run.artifacts_dir('lake_a') / 'x.txt'
    marker.write_text('x')
    run.wipe_dataset('lake_a')
    assert not (run.path / 'lake_a').exists()


def test_finalize_strips_partial_suffix(tmp_path):
    run = RunDir.create(tmp_path, 't', '')
    final = run.finalize()
    assert not final.name.endswith('.partial')
    assert final.is_dir()


def test_finalize_rewrites_partial_paths_in_nlseeker_ini(tmp_path):
    run = RunDir.create(tmp_path, 't', '')
    artifacts = run.artifacts_dir('lake_a')
    write_ini(artifacts, 'lake_a', {'llm_base_url': 'http://x/v1'}, alpha=0.5, n=5)
    final = run.finalize()
    text = (final / 'lake_a' / 'artifacts' / 'nlseeker.ini').read_text()
    assert '.partial' not in text
    parser = configparser.ConfigParser()
    parser.read_string(text)
    assert parser['Database']['path'].startswith(str(final.resolve()))


def test_write_ini_layout(tmp_path):
    ini = write_ini(tmp_path, 'adventure_works',
                    {'llm_base_url': 'http://x/v1', 'llm_api_key': 'k'}, alpha=0.5, n=5)
    parser = configparser.ConfigParser()
    parser.read(ini)
    assert parser['Database']['index_table'] == 'adventure_works'
    assert parser['Database']['path'] == str(tmp_path / 'eval.duckdb')
    assert parser['NLSeeker']['schema'] == 'nl_adventure_works'
    assert parser['NLSeeker']['out_path'] == str(tmp_path / 'nl_index')
    assert parser['NLSeeker']['alpha'] == '0.5'
    assert parser['NLSeeker']['llm_base_url'] == 'http://x/v1'
