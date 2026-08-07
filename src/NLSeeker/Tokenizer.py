"""The tokenization contract shared by the NL index build and NL retrieval."""

import re
import Stemmer

# Typing imports
from typing import Dict, List

# Copied from bm25s.stopwords.STOPWORDS_EN. The SQL BM25 scores are only comparable
# with the bm25s-built index while this list and the split pattern match it exactly.
STOPWORDS = frozenset({
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for', 'if', 'in', 'into',
    'is', 'it', 'no', 'not', 'of', 'on', 'or', 'such', 'that', 'the', 'their', 'then',
    'there', 'these', 'they', 'this', 'to', 'was', 'will', 'with',
})

TOKEN_PATTERN = re.compile(r'(?u)\b\w\w+\b')

_stemmer = Stemmer.Stemmer('english')


def tokenize(text: str) -> List[str]:
    """Lowercases, splits, drops stopwords, then stems - stopwords are matched unstemmed."""
    tokens = [token for token in TOKEN_PATTERN.findall(text.lower()) if token not in STOPWORDS]
    return _stemmer.stemWords(tokens)


def term_frequencies(text: str) -> Dict[str, int]:
    """Counts each stemmed token; the total is the BM25 document length."""
    frequencies = {}
    for token in tokenize(text):
        frequencies[token] = frequencies.get(token, 0) + 1
    return frequencies
