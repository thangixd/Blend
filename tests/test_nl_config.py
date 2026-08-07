import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.NLSeeker.Config import NLSeekerConfig

SECTION = """[Database]
dbms=duckdb
path=x.db
index_table=lake

[NLSeeker]
index_name=lake
schema=nl_lake
llm_base_url=http://localhost:8001/v1
llm_api_key=none
llm_model=m
llm_temperature=0.0
llm_max_new_tokens=64
llm_context_length=4096
llm_concurrency=1
llm_retry_attempts=1
llm_tokenizer=t
embedding_base_url=http://localhost:8002/v1
embedding_api_key=none
embedding_model=e
embedding_max_tokens=512
embedding_batch_size=8
embedding_tokenizer=t
alpha=0.5
n=5
"""


def _config(text: str) -> NLSeekerConfig:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'c.ini'
        path.write_text(text)
        return NLSeekerConfig.load(path)


def test_table_names_derive_from_index_name():
    config = _config(SECTION)
    assert config.documents_table == 'lake_nl_documents'
    assert config.tokens_table == 'lake_nl_tokens'


def test_out_path_is_no_longer_required():
    assert _config(SECTION).index_name == 'lake'


def test_leftover_out_path_key_is_ignored():
    assert _config(SECTION + 'out_path=nl_index\n').index_name == 'lake'


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'{name} ok')
