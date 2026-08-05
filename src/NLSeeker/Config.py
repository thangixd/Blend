import configparser
from dataclasses import dataclass
from pathlib import Path

# Typing imports
from typing import Union


@dataclass(frozen=True)
class NLSeekerConfig:
    out_path: Path
    index_name: str
    schema: str

    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_temperature: float
    llm_max_new_tokens: int
    llm_context_length: int
    llm_concurrency: int
    llm_retry_attempts: int
    llm_tokenizer: str

    embedding_base_url: str
    embedding_api_key: str
    embedding_model: str
    embedding_max_tokens: int
    embedding_batch_size: int
    embedding_tokenizer: str

    alpha: float
    n: int

    @property
    def index_path(self) -> Path:
        return self.out_path / self.index_name

    @property
    def vector_path(self) -> Path:
        return self.index_path / 'vector'

    @property
    def fulltext_path(self) -> Path:
        return self.index_path / 'fulltext'

    @staticmethod
    def load(config_path: Union[str, Path]) -> 'NLSeekerConfig':
        """Reads the [NLSeeker] section of an ini file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f'Config file not found at {config_path}')

        parser = configparser.ConfigParser()
        parser.read(config_path)
        if not parser.has_section('NLSeeker'):
            raise KeyError(f'{config_path} has no [NLSeeker] section')
        section = parser['NLSeeker']

        casts = {'out_path': Path, 'llm_temperature': float, 'llm_max_new_tokens': int,
                 'llm_context_length': int, 'llm_concurrency': int, 'llm_retry_attempts': int,
                 'embedding_max_tokens': int, 'embedding_batch_size': int,
                 'alpha': float, 'n': int}

        values = {}
        for field in NLSeekerConfig.__dataclass_fields__:
            if field not in section:
                raise KeyError(f'[NLSeeker] in {config_path} is missing the key {field!r}')
            values[field] = casts.get(field, str)(section[field])

        return NLSeekerConfig(**values)
