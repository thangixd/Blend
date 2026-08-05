import csv
import json
from collections import defaultdict
from pathlib import Path

from evaluation.nlseeker.metrics import latency_stats

# Typing imports
from typing import List


SUMMARY_COLUMNS = ['dataset', 'family', 'k', 'n', 'alpha', 'rerank', 'n_questions', 'hit_rate',
                   'recall', 'mrr', 'latency_mean_ms', 'latency_p50_ms', 'latency_p95_ms']


def collect_records(run_path: Path, datasets: List[str]) -> List[dict]:
    records = []
    for dataset in datasets:
        per_query = run_path / dataset / 'per_query.jsonl'
        if not per_query.is_file():
            continue
        with open(per_query) as f:
            for line in f:
                record = json.loads(line)
                record['dataset'] = dataset
                records.append(record)
    return records


def aggregate(records: List[dict]) -> List[dict]:
    groups = defaultdict(list)
    for record in records:
        groups[(record['dataset'], record['family'], record['k'], record['n'],
                record['alpha'], record['rerank'])].append(record)

    rows = []
    for (dataset, family, k, n, alpha, rerank), group in sorted(groups.items()):
        count = len(group)
        mean_ms, p50_ms, p95_ms = latency_stats([r['latency_ms'] for r in group])
        rows.append({
            'dataset': dataset, 'family': family, 'k': k, 'n': n, 'alpha': alpha,
            'rerank': rerank, 'n_questions': count,
            'hit_rate': sum(r['hit'] for r in group) / count,
            'recall': sum(r['recall'] for r in group) / count,
            'mrr': sum(r['rr'] for r in group) / count,
            'latency_mean_ms': mean_ms, 'latency_p50_ms': p50_ms, 'latency_p95_ms': p95_ms,
        })
    return rows


def write_summary(rows: List[dict], run_path: Path) -> None:
    with open(run_path / 'summary.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    lines = ['# NLSeeker evaluation summary', '']
    for dataset in sorted({row['dataset'] for row in rows}):
        lines += [f'## {dataset}', '',
                  '| family | k | n | alpha | rerank | questions | hit rate | recall | MRR | latency p50 (ms) |',
                  '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
        for row in (r for r in rows if r['dataset'] == dataset):
            lines.append(f"| {row['family']} | {row['k']} | {row['n']} | {row['alpha']} "
                         f"| {row['rerank']} | {row['n_questions']} | {row['hit_rate']:.3f} "
                         f"| {row['recall']:.3f} | {row['mrr']:.3f} | {row['latency_p50_ms']:.0f} |")
        lines.append('')
    (run_path / 'summary.md').write_text('\n'.join(lines))
