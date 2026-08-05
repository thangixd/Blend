"""Trains the per-seeker XGBoost cost models on the configured lake."""
import argparse
import json
import random
import time
from pathlib import Path

import pandas as pd
from xgboost import XGBRegressor

from src.DBHandler import DBHandler

# Typing imports
from typing import Callable, List, Tuple

TOKEN_SEEKERS = ['Keyword', 'SingleColumnOverlap', 'MultiColumnOverlap', 'Correlation']
ALL_SEEKERS = TOKEN_SEEKERS + ['NaturalLanguage']

QUESTION_PROMPT = (
    'Below is a description of one table from a data lake.\n\n{payload}\n\n'
    'Write one short natural-language question that a user could ask a data '
    'discovery system to find this table. Answer with only the question.'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True,
                        help='ini file with [Database] (and [NLSeeker] for NaturalLanguage)')
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--nl-samples', type=int, default=100)
    parser.add_argument('--seekers', nargs='+', default=ALL_SEEKERS, choices=ALL_SEEKERS)
    parser.add_argument('--out', type=Path, default=Path('src/Operators/Seekers'))
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def measure(db: DBHandler, build: Callable, attempts: int) -> Tuple[List[list], List[float]]:
    X, y = [], []
    for _ in range(attempts):
        seeker = build()
        if seeker is None:
            continue
        features = seeker._features(db)
        start = time.perf_counter()
        seeker.run()
        y.append(time.perf_counter() - start)
        X.append(features)
    return X, y


class TableSampler:
    """Caches pivoted tables so repeated samples do not re-read the index."""

    def __init__(self, db: DBHandler, rng: random.Random) -> None:
        self.db = db
        self.rng = rng
        self._tables = {}
        self.table_ids = [row[0] for row in db.execute_and_fetchall('SELECT DISTINCT TableId FROM AllTables')]

    def table(self, table_id: int) -> pd.DataFrame:
        if table_id not in self._tables:
            self._tables[table_id] = self.db.get_table_from_index(table_id)
        return self._tables[table_id]

    def random_table(self) -> pd.DataFrame:
        return self.table(self.rng.choice(self.table_ids))


def build_keyword(db: DBHandler, rng: random.Random):
    from src.Operators.Seekers.Keyword import Keyword
    count = rng.randint(1, 5)
    rows = db.execute_and_fetchall(
        f"SELECT CellValue FROM AllTables WHERE CellValue <> '' USING SAMPLE {count} ROWS")
    if len(rows) == 0:
        return None
    return Keyword([row[0] for row in rows])


def build_sc(sampler: TableSampler):
    from src.Operators.Seekers.SingleColumnOverlap import SingleColumnOverlap
    df = sampler.random_table()
    column = sampler.rng.choice(list(df.columns))
    values = [v for v in df[column].dropna().tolist() if v != ''][:100]
    if len(values) == 0:
        return None
    return SingleColumnOverlap(values)


def build_mc(sampler: TableSampler):
    from src.Operators.Seekers.MultiColumnOverlap import MultiColumnOverlap
    df = sampler.random_table()
    if len(df.columns) < 2:
        return None
    columns = sampler.rng.sample(list(df.columns), sampler.rng.randint(2, min(3, len(df.columns))))
    sub = df[columns].replace('', pd.NA).dropna()
    if len(sub) < 2:
        return None
    sub = sub.sample(min(30, len(sub)), random_state=sampler.rng.randint(0, 2 ** 31))
    return MultiColumnOverlap(sub)


def build_correlation(sampler: TableSampler, numeric_pairs: list):
    from src.Operators.Seekers.Correlation import Correlation
    table_id, numeric_column = sampler.rng.choice(numeric_pairs)
    df = sampler.table(table_id)
    numeric_column = str(numeric_column)
    categorical = [c for c in df.columns if c != numeric_column]
    if numeric_column not in df.columns or len(categorical) == 0:
        return None
    column = sampler.rng.choice(categorical)
    frame = pd.DataFrame({
        'source': df[column],
        'target': pd.to_numeric(df[numeric_column], errors='coerce'),
    }).replace('', pd.NA).dropna()
    if len(frame) < 2:
        return None
    return Correlation(frame['source'].tolist(), frame['target'].tolist())


def collect_token_seeker(name: str, db: DBHandler, rng: random.Random, samples: int):
    sampler = TableSampler(db, rng)
    if name == 'Keyword':
        build = lambda: build_keyword(db, rng)
    elif name == 'SingleColumnOverlap':
        build = lambda: build_sc(sampler)
    elif name == 'MultiColumnOverlap':
        build = lambda: build_mc(sampler)
    else:
        numeric_pairs = db.execute_and_fetchall(
            'SELECT DISTINCT TableId, ColumnId FROM AllTables WHERE Quadrant IS NOT NULL')
        if len(numeric_pairs) == 0:
            raise RuntimeError('No numeric columns in the index - cannot train Correlation')
        build = lambda: build_correlation(sampler, numeric_pairs)
    return measure(db, build, samples)


def generate_questions(config, count: int, rng: random.Random, db: DBHandler) -> List[str]:
    from src.NLSeeker.Clients import LLMClient

    if not config.vector_path.exists():
        raise RuntimeError(f'NL index not found at {config.vector_path} - build it first '
                           '(scripts/create_index_nl_blend_duckdb.py)')

    rows = db.execute_and_fetchall(
        f'SELECT table_id, summary FROM {config.schema}.table_summaries')
    if len(rows) == 0:
        raise RuntimeError(f'{config.schema}.table_summaries is empty - summarize the lake first')

    llm = LLMClient(config)
    try:
        llm.complete([[{'role': 'user', 'content': 'ping'}]], max_new_tokens=4)
    except Exception as e:
        raise RuntimeError(f'LLM endpoint {config.llm_base_url} unreachable') from e

    payloads = [json.loads(row[1])['payload'] for row in rng.choices(rows, k=count)]
    conversations = [[{'role': 'user', 'content': QUESTION_PROMPT.format(payload=p)}] for p in payloads]
    return llm.complete(conversations, desc='generating questions')


def collect_natural_language(db: DBHandler, rng: random.Random, samples: int):
    from src.NLSeeker.Config import NLSeekerConfig
    from src.Operators.Seekers.NaturalLanguage import NaturalLanguage

    config = NLSeekerConfig.load(db.config_path)
    questions = generate_questions(config, samples, rng, db)

    NaturalLanguage(questions[0], k=5).run()  # warm-up: loads Chroma and BM25s once, untimed

    def build():
        return NaturalLanguage(questions.pop(),
                               k=rng.choice([5, 10, 20]),
                               n=rng.choice([3, 5, 10]),
                               rerank=rng.random() < 0.5)

    return measure(db, build, samples - 1)


def main() -> None:
    args = parse_args()

    DBHandler.USE_ML_OPTIMIZER = False  # sampling must not require the very models being trained
    from src.Operators.OperatorBase import Operator
    Operator.DB.load_config(args.config)
    db = Operator.DB
    rng = random.Random(args.seed)

    for name in args.seekers:
        if name == 'NaturalLanguage':
            X, y = collect_natural_language(db, rng, args.nl_samples)
        else:
            X, y = collect_token_seeker(name, db, rng, args.samples)
        if len(y) < 10:
            raise RuntimeError(f'{name}: only {len(y)} usable samples - not enough to train')
        print(f'{name}: {len(y)} samples, fitting...')
        model = XGBRegressor()
        model.fit(X, y)
        model.save_model(args.out / f'{name}_model.json')
        print(f'{name}: saved {args.out / f"{name}_model.json"}')


if __name__ == '__main__':
    main()
