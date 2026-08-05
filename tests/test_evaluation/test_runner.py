import json

import duckdb

from evaluation.nlseeker.datasets import Question
from evaluation.nlseeker.runner import load_id_map, run_family


class FakePlan:
    def __init__(self, ids):
        self._ids = ids
        self.DB = self
        self.loaded = None

    def load_config(self, path):
        self.loaded = path

    def run(self):
        return self._ids


def test_load_id_map(tmp_path):
    con = duckdb.connect(str(tmp_path / 'eval.duckdb'))
    con.execute('CREATE TABLE "mini_tables" (tableid INTEGER, filename VARCHAR)')
    con.execute("INSERT INTO \"mini_tables\" VALUES (0, 'A_SEP_table_7.csv'), (1, 'B_SEP_table_2.csv')")
    con.close()
    assert load_id_map(tmp_path / 'eval.duckdb', 'mini') == {0: 'table_7', 1: 'table_2'}


def test_run_family_scores_and_streams(tmp_path):
    questions = [Question('q1', 'find candidates', ('table_7',))]
    id_map = {0: 'table_7', 1: 'table_2'}
    out = tmp_path / 'per_query.jsonl'

    def factory(text, k, n, alpha, rerank):
        assert (text, k, n, alpha, rerank) == ('find candidates', 1, 5, 0.5, True)
        return FakePlan([1, 0][:k])

    records = run_family(tmp_path / 'x.ini', questions, 'BC1', [(1, 5, 0.5, True)], id_map, out,
                         plan_factory=factory)
    (record,) = records
    assert record['retrieved'] == ['table_2']
    assert record['hit'] is False and record['rr'] == 0.0
    assert record['family'] == 'BC1' and record['k'] == 1
    assert record['rerank'] == 'on'
    assert json.loads(out.read_text().splitlines()[0])['qid'] == 'q1'


def test_run_family_appends_across_combos(tmp_path):
    questions = [Question('q1', 't', ('table_7',))]
    out = tmp_path / 'per_query.jsonl'
    run_family(tmp_path / 'x.ini', questions, 'BC1', [(1, 5, 0.5, True), (5, 5, 0.5, False)],
               {0: 'table_7'}, out, plan_factory=lambda *a: FakePlan([0]))
    lines = out.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])['k'] == 5
    assert json.loads(lines[1])['hit'] is True
    assert json.loads(lines[1])['rerank'] == 'off'
