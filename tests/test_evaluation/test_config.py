import pytest

from evaluation.nlseeker.config import EvalConfig, NLSEEKER_KEYS, load_config

VALID_NLSEEKER = {
    'llm_base_url': 'http://127.0.0.1:8001/v1', 'llm_api_key': 'k', 'llm_model': 'm',
    'llm_temperature': '0.0', 'llm_max_new_tokens': '512', 'llm_context_length': '32768',
    'llm_concurrency': '1', 'llm_retry_attempts': '5', 'llm_tokenizer': 't',
    'embedding_base_url': 'http://127.0.0.1:8002/v1', 'embedding_api_key': 'k',
    'embedding_model': 'e', 'embedding_max_tokens': '512', 'embedding_batch_size': '256',
    'embedding_tokenizer': 't',
}


def write_config(tmp_path, **overrides):
    import yaml
    doc = {
        'run': {'tag': 'test', 'questions_limit': None},
        'datasets': ['adventure_works'],
        'families': ['BC1'],
        'retrieval': {'k': [1, 5], 'n': [5], 'alpha': [0.5], 'rerank': [True, False]},
        'nlseeker': dict(VALID_NLSEEKER),
    }
    doc.update(overrides)
    path = tmp_path / 'c.yaml'
    path.write_text(yaml.safe_dump(doc))
    return path


def test_loads_valid_config(tmp_path):
    cfg = load_config(write_config(tmp_path))
    assert cfg.tag == 'test'
    assert cfg.datasets == ('adventure_works',)
    assert cfg.k == (1, 5)
    assert cfg.rerank == (True, False)
    assert cfg.combos() == [(1, 5, 0.5, True), (1, 5, 0.5, False),
                            (5, 5, 0.5, True), (5, 5, 0.5, False)]


def test_scalar_promoted_to_list(tmp_path):
    cfg = load_config(write_config(tmp_path, retrieval={'k': 5, 'n': 5, 'alpha': 0.5, 'rerank': True}))
    assert cfg.k == (5,)
    assert cfg.rerank == (True,)


def test_non_bool_rerank_raises(tmp_path):
    with pytest.raises(ValueError, match='rerank'):
        load_config(write_config(tmp_path, retrieval={'k': [1], 'n': [5], 'alpha': [0.5],
                                                      'rerank': ['yes']}))


def test_unknown_top_level_key_raises(tmp_path):
    with pytest.raises(ValueError, match='unknown'):
        load_config(write_config(tmp_path, bogus={'x': 1}))


def test_missing_nlseeker_key_raises(tmp_path):
    broken = dict(VALID_NLSEEKER)
    del broken['llm_model']
    with pytest.raises(ValueError, match='llm_model'):
        load_config(write_config(tmp_path, nlseeker=broken))


def test_datasets_scalar_raises(tmp_path):
    with pytest.raises(ValueError, match='datasets must be a list'):
        load_config(write_config(tmp_path, datasets='adventure_works'))


def test_families_scalar_raises(tmp_path):
    with pytest.raises(ValueError, match='families must be a list'):
        load_config(write_config(tmp_path, families='BC1'))


def test_unknown_dataset_raises(tmp_path):
    with pytest.raises(ValueError, match='no_such_lake'):
        load_config(write_config(tmp_path, datasets=['no_such_lake']))


def test_unknown_family_raises(tmp_path):
    with pytest.raises(ValueError, match='BC9'):
        load_config(write_config(tmp_path, families=['BC9']))


def test_out_path_is_not_configurable(tmp_path):
    bad = dict(VALID_NLSEEKER, out_path='/tmp/x')
    with pytest.raises(ValueError, match='out_path'):
        load_config(write_config(tmp_path, nlseeker=bad))


def test_shipped_configs_parse():
    from pathlib import Path
    for name in ('default.yaml', 'alpha_sweep.yaml', 'n_sweep.yaml', 'smoke.yaml'):
        cfg = load_config(Path('evaluation/nlseeker/configs') / name)
        assert isinstance(cfg, EvalConfig)
