import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import bm25s
import Stemmer
from bm25s.tokenization import convert_tokenized_to_string_list

from src.NLSeeker import Tokenizer

TEXTS = [
    'The Customer Address of a city',
    'a an the',
    'Sales2020 order-id id',
    'addresses ADDRESS Addressing',
]


def test_matches_bm25s_tokenization():
    expected = convert_tokenized_to_string_list(
        bm25s.tokenize(TEXTS, stopwords='en', stemmer=Stemmer.Stemmer('english'),
                       show_progress=False))
    assert [Tokenizer.tokenize(text) for text in TEXTS] == expected


def test_stopword_only_text_has_no_tokens():
    assert Tokenizer.tokenize('a an the of on or') == []


def test_term_frequencies_counts_repeats():
    assert Tokenizer.term_frequencies('address address city') == {'address': 2, 'citi': 1}


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'{name} ok')
