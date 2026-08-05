import json
from pathlib import Path

import pytest

from evaluation.nlseeker.datasets import DATASETS, FAMILIES, Question, load_questions, pneuma_id


def test_registry_has_all_five_datasets():
    assert set(DATASETS) == {'adventure_works', 'chembl_10K', 'chicago_10K', 'fetaqa', 'public_bi'}
    spec = DATASETS['adventure_works']
    assert spec.tar == 'pneuma_adventure_works.tar'
    assert spec.questions == 'pneuma_adventure_works_questions_annotated.jsonl'
    assert spec.zip == 'adventure-20260604T041942Z-3-001.zip'
    assert spec.zip_name == 'adventure'


def test_pneuma_id_strips_title_and_extension():
    assert pneuma_id('JobCandidate_SEP_table_41.csv') == 'table_41'
    assert pneuma_id('Veronica Campbell-Brown , Personal bests_SEP_table_5236.csv') == 'table_5236'


def test_pneuma_id_rejects_malformed_name():
    with pytest.raises(ValueError):
        pneuma_id('no_separator_here.csv')


def _write_jsonl(path, rows):
    path.write_text('\n'.join(json.dumps(r) for r in rows))


def test_load_bc_questions_picks_family_field_and_dedupes_answers(tmp_path):
    p = tmp_path / 'q.jsonl'
    _write_jsonl(p, [
        {'id': 'a', 'question': 'rephrased', 'question_from_sql_1': 'original',
         'answer_tables': ['table_1', 'table_2', 'table_1']},
    ])
    (bc1,) = load_questions(p, 'BC1', None)
    assert bc1 == Question(qid='a', text='original', answer_tables=('table_1', 'table_2'))
    (bc2,) = load_questions(p, 'BC2', None)
    assert bc2.text == 'rephrased'


def test_load_bx_questions_uses_context_id(tmp_path):
    p = tmp_path / 'bx.jsonl'
    _write_jsonl(p, [
        {'context_id': 'table_54_SEP_contexts-0', 'question_bx1': 'one', 'question_bx2': 'two',
         'answer_tables': ['table_54']},
    ])
    (q,) = load_questions(p, 'BX2', None)
    assert q.qid == 'table_54_SEP_contexts-0'
    assert q.text == 'two'


def test_load_questions_limit_takes_first_n(tmp_path):
    p = tmp_path / 'q.jsonl'
    _write_jsonl(p, [{'id': str(i), 'question': f'q{i}', 'question_from_sql_1': f'o{i}',
                      'answer_tables': ['table_1']} for i in range(5)])
    assert [q.qid for q in load_questions(p, 'BC2', 2)] == ['0', '1']


def test_load_questions_unknown_family_raises(tmp_path):
    with pytest.raises(ValueError):
        load_questions(tmp_path / 'q.jsonl', 'BC9', None)
