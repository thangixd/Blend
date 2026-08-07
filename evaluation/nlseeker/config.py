from dataclasses import dataclass
from itertools import product
from pathlib import Path

import yaml

from evaluation.nlseeker.datasets import DATASETS, FAMILIES

# Typing imports
from typing import Dict, List, Optional, Tuple


# Exactly the NLSeekerConfig ini keys the YAML must provide. index_name, schema,
# alpha and n are derived per dataset/run and must not appear here.
NLSEEKER_KEYS = frozenset({
    'llm_base_url', 'llm_api_key', 'llm_model', 'llm_temperature', 'llm_max_new_tokens',
    'llm_context_length', 'llm_concurrency', 'llm_retry_attempts', 'llm_tokenizer',
    'embedding_base_url', 'embedding_api_key', 'embedding_model', 'embedding_max_tokens',
    'embedding_batch_size', 'embedding_tokenizer',
})

TOP_LEVEL_KEYS = frozenset({'run', 'datasets', 'families', 'retrieval', 'nlseeker'})


@dataclass(frozen=True)
class EvalConfig:
    tag: str
    questions_limit: Optional[int]
    datasets: Tuple[str, ...]
    families: Tuple[str, ...]
    k: Tuple[int, ...]
    n: Tuple[int, ...]
    alpha: Tuple[float, ...]
    rerank: Tuple[bool, ...]
    nlseeker: Dict[str, str]

    def combos(self) -> List[Tuple[int, int, float, bool]]:
        return list(product(self.k, self.n, self.alpha, self.rerank))


def _as_bool(value) -> bool:
    # YAML already parses on/off/true/false to bool; anything else is a typo.
    if not isinstance(value, bool):
        raise ValueError(f'retrieval.rerank entries must be booleans (on/off), got {value!r}')
    return value


def _as_tuple(value, caster, key: str) -> tuple:
    items = value if isinstance(value, list) else [value]
    if not items:
        raise ValueError(f'retrieval.{key} must not be empty')
    return tuple(caster(item) for item in items)


def _require_keys(section: dict, expected: frozenset, name: str) -> None:
    unknown = set(section) - expected
    missing = expected - set(section)
    if unknown:
        raise ValueError(f'unknown {name} key(s): {sorted(unknown)}')
    if missing:
        raise ValueError(f'missing {name} key(s): {sorted(missing)}')


def load_config(path: Path) -> EvalConfig:
    doc = yaml.safe_load(Path(path).read_text())
    _require_keys(doc, TOP_LEVEL_KEYS, 'top-level')

    run = doc['run']
    _require_keys(run, frozenset({'tag', 'questions_limit'}), 'run')
    limit = run['questions_limit']
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        raise ValueError(f'run.questions_limit must be a positive int or null, got {limit!r}')

    if not isinstance(doc['datasets'], list):
        raise ValueError('datasets must be a list')
    datasets = tuple(doc['datasets'])
    for dataset in datasets:
        if dataset not in DATASETS:
            raise ValueError(f'unknown dataset {dataset!r}; expected one of {sorted(DATASETS)}')

    if not isinstance(doc['families'], list):
        raise ValueError('families must be a list')
    families = tuple(doc['families'])
    for family in families:
        if family not in FAMILIES:
            raise ValueError(f'unknown family {family!r}; expected one of {FAMILIES}')

    retrieval = doc['retrieval']
    _require_keys(retrieval, frozenset({'k', 'n', 'alpha', 'rerank'}), 'retrieval')

    _require_keys(doc['nlseeker'], NLSEEKER_KEYS, 'nlseeker')
    nlseeker = {key: str(value) for key, value in doc['nlseeker'].items()}

    return EvalConfig(
        tag=str(run['tag']),
        questions_limit=limit,
        datasets=datasets,
        families=families,
        k=_as_tuple(retrieval['k'], int, 'k'),
        n=_as_tuple(retrieval['n'], int, 'n'),
        alpha=_as_tuple(retrieval['alpha'], float, 'alpha'),
        rerank=_as_tuple(retrieval['rerank'], _as_bool, 'rerank'),
        nlseeker=nlseeker,
    )
