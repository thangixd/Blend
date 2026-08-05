import json
import time
from pathlib import Path

import duckdb

from evaluation.nlseeker.datasets import Question, pneuma_id
from evaluation.nlseeker.metrics import hit, recall, reciprocal_rank

# Typing imports
from typing import Callable, Dict, List, Optional, Tuple


def load_id_map(db_path: Path, dataset: str) -> Dict[int, str]:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute(f'SELECT tableid, filename FROM "{dataset}_tables"').fetchall()
    finally:
        con.close()
    return {table_id: pneuma_id(filename) for table_id, filename in rows}


def run_family(ini_path: Path, questions: List[Question], family: str,
               combos: List[Tuple[int, int, float, bool]], id_map: Dict[int, str],
               out_jsonl: Path, plan_factory: Optional[Callable] = None) -> List[dict]:
    if plan_factory is None:
        from src.Tasks.NLSearch import NLSearch

        def plan_factory(text, k, n, alpha, rerank):
            return NLSearch(text, k, n, alpha, rerank=rerank)

    records = []
    with open(out_jsonl, 'a') as out:
        for k, n, alpha, rerank in combos:
            for question in questions:
                plan = plan_factory(question.text, k, n, alpha, rerank)
                plan.DB.load_config(ini_path)
                start = time.perf_counter()
                table_ids = plan.run()
                latency_ms = (time.perf_counter() - start) * 1000.0
                retrieved = [id_map[int(table_id)] for table_id in table_ids]
                answers = set(question.answer_tables)
                record = {
                    'qid': question.qid, 'family': family, 'k': k, 'n': n, 'alpha': alpha,
                    'rerank': 'on' if rerank else 'off',
                    'retrieved': retrieved, 'answers': sorted(answers),
                    'hit': hit(retrieved, answers), 'recall': recall(retrieved, answers),
                    'rr': reciprocal_rank(retrieved, answers), 'latency_ms': latency_ms,
                }
                out.write(json.dumps(record) + '\n')
                out.flush()
                records.append(record)
    return records
