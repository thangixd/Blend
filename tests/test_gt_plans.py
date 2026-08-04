import ast
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS = ROOT / 'tests' / 'tasks'
INDEX_DB = ROOT / 'tests' / 'test_duckdb.db'

EXPECTED = {
    "aug_by_example_addresses": [1, 71, 72, 81, 83, 86],
    "correlation_address_postal": [1, 1, 60, 60, 68, 68, 69, 71, 71, 72, 72, 72, 72, 72, 72, 73, 74, 75, 75, 75, 81,
                                   81, 81, 81, 81, 81, 81, 81, 81, 82, 82, 83, 83, 84, 84, 84, 86, 86, 86, 86, 87, 87],
    "data_imputation_addresses": [1, 71, 72, 81, 83, 86],
    "dependent_data_addresses": [1, 9, 26, 71, 72, 75, 81, 83, 84, 85, 86],
    "keyword_addresses": [1, 61, 71, 72, 81, 83, 84, 85, 86],
    "mc_join_city_postal": [1, 71, 72, 75, 81, 83, 86],
    "multi_column_colin_state": [26, 69, 70, 71, 81, 84],
    "negative_example_addresses": [9, 60, 74, 82],
    "sc_join_city": [1, 1, 8, 9, 26, 26, 60, 68, 69, 69, 70, 70, 71, 71, 71, 71, 72, 72, 72, 72, 73, 74, 75, 81, 81,
                     81, 82, 83, 83, 83, 84, 84, 86, 86, 87],
    "union_address": [1, 2, 3, 4, 5, 8, 9, 11, 13, 16, 23, 24, 25, 26, 30, 35, 41, 43, 44, 45, 46, 47, 60, 61, 62,
                      63, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 81, 82, 83, 84, 85, 86, 87],
}

RETURNED = re.compile(r'returned \d+ TableIds: (\[.*\])')


def run_plan(case):
    plan = TASKS / case / 'gt_plan.py'
    result = subprocess.run([sys.executable, str(plan)], cwd=ROOT, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        raise AssertionError(f'{case} exited {result.returncode}:\n{result.stderr[-2000:]}')

    match = RETURNED.search(result.stdout)
    if match is None:
        raise AssertionError(f'{case} printed no TableId line:\n{result.stdout[-2000:]}')

    return ast.literal_eval(match.group(1))


def check(case):
    ids = run_plan(case)
    expected = EXPECTED[case]
    if Counter(ids) != Counter(expected):
        surplus = sorted((Counter(ids) - Counter(expected)).elements())
        missing = sorted((Counter(expected) - Counter(ids)).elements())
        raise AssertionError(f'{case}: unexpected {surplus}, missing {missing}')

    return ids


def _requirements():
    if not INDEX_DB.exists():
        return f'{INDEX_DB} missing, build it with scripts/create_index_duckdb.py'
    return None


try:
    import pytest
except ImportError:
    pass
else:
    @pytest.mark.parametrize('case', sorted(EXPECTED))
    def test_gt_plan(case):
        reason = _requirements()
        if reason:
            pytest.skip(reason)
        check(case)


def main():
    reason = _requirements()
    if reason:
        print(f'skipped: {reason}')
        return 0

    failures = 0
    for case in sorted(EXPECTED):
        try:
            ids = check(case)
        except AssertionError as error:
            failures += 1
            print(f'FAIL {case}\n     {error}')
        else:
            print(f'ok   {case:<30} {len(ids)} TableIds')

    print(f'\n{len(EXPECTED) - failures}/{len(EXPECTED)} plans match the report')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
