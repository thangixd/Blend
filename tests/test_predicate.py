import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.NLSeeker.Predicate import EMPTY_FILTER, TableFilter, parse_additionals


def test_empty_string_is_unconstrained():
    assert parse_additionals('') is EMPTY_FILTER
    assert parse_additionals('   ') is EMPTY_FILTER
    assert EMPTY_FILTER.is_empty()
    assert EMPTY_FILTER.permits(42)


def test_intersection_clause_becomes_allow_list():
    table_filter = parse_additionals(' AND TableId IN (1 , 2 , 3) ')
    assert table_filter == TableFilter(allow=frozenset({1, 2, 3}))
    assert table_filter.permits(2)
    assert not table_filter.permits(4)


def test_difference_clause_becomes_deny_list():
    table_filter = parse_additionals(' AND TableId NOT IN (4 , 5) ')
    assert table_filter == TableFilter(deny=frozenset({4, 5}))
    assert table_filter.permits(1)
    assert not table_filter.permits(4)


def test_chained_clauses_combine():
    table_filter = parse_additionals(' AND TableId NOT IN (4 , 5)  AND TableId IN (1 , 2 , 4) ')
    assert table_filter == TableFilter(allow=frozenset({1, 2, 4}), deny=frozenset({4, 5}))
    assert table_filter.permits(1)
    assert not table_filter.permits(4)      # denied wins over allowed
    assert not table_filter.permits(3)


def test_repeated_clauses_intersect_and_union():
    table_filter = parse_additionals('TableId IN (1, 2, 3) AND TableId IN (2, 3, 4) '
                                     'AND TableId NOT IN (7) AND TableId NOT IN (8)')
    assert table_filter == TableFilter(allow=frozenset({2, 3}), deny=frozenset({7, 8}))


def test_keywords_are_case_insensitive():
    table_filter = parse_additionals('and tableid not in (9)')
    assert table_filter == TableFilter(deny=frozenset({9}))


def test_single_id_without_spaces():
    assert parse_additionals('TableId IN (7)') == TableFilter(allow=frozenset({7}))


def test_unknown_clause_raises():
    for additionals in ['ColumnId IN (1)', 'TableId IN ()', 'TableId = 3',
                        'TableId IN (1) OR TableId IN (2)']:
        try:
            parse_additionals(additionals)
        except ValueError:
            pass
        else:
            raise AssertionError(f'expected ValueError for {additionals!r}')


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'{name} ok')
