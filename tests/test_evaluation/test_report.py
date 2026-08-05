import csv
import json

from evaluation.nlseeker.report import aggregate, collect_records, write_summary


def record(**overrides):
    base = {'qid': 'q', 'family': 'BC1', 'k': 1, 'n': 5, 'alpha': 0.5, 'rerank': 'on',
            'retrieved': [], 'answers': [], 'hit': True, 'recall': 1.0, 'rr': 1.0,
            'latency_ms': 100.0, 'dataset': 'adventure_works'}
    base.update(overrides)
    return base


def test_aggregate_groups_and_averages():
    rows = aggregate([record(), record(hit=False, recall=0.0, rr=0.0, latency_ms=300.0),
                      record(k=5)])
    assert len(rows) == 2
    k1 = next(r for r in rows if r['k'] == 1)
    assert k1['n_questions'] == 2
    assert k1['hit_rate'] == 0.5
    assert k1['mrr'] == 0.5
    assert k1['latency_mean_ms'] == 200.0


def test_aggregate_separates_rerank_modes():
    rows = aggregate([record(), record(rerank='off', hit=False, recall=0.0, rr=0.0)])
    assert len(rows) == 2
    on = next(r for r in rows if r['rerank'] == 'on')
    off = next(r for r in rows if r['rerank'] == 'off')
    assert on['hit_rate'] == 1.0 and off['hit_rate'] == 0.0


def test_collect_records_reads_datasets_and_skips_missing(tmp_path):
    d = tmp_path / 'adventure_works'
    d.mkdir()
    (d / 'per_query.jsonl').write_text(json.dumps({'qid': 'q', 'hit': True}) + '\n')
    records = collect_records(tmp_path, ['adventure_works', 'fetaqa'])
    assert records == [{'qid': 'q', 'hit': True, 'dataset': 'adventure_works'}]


def test_write_summary_produces_csv_and_md(tmp_path):
    write_summary(aggregate([record()]), tmp_path)
    with open(tmp_path / 'summary.csv') as f:
        rows = list(csv.DictReader(f))
    assert rows[0]['hit_rate'] == '1.0'
    md = (tmp_path / 'summary.md').read_text()
    assert 'adventure_works' in md and '| BC1 |' in md
