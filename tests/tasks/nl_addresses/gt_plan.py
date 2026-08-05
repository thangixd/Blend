import sys
from pathlib import Path

_ROOT = Path(__file__).resolve()
for _p in [_ROOT, *_ROOT.parents]:
    if (_p / 'src' / 'Plan.py').exists() and (_p / 'tests' / 'tasks').exists():
        sys.path.insert(0, str(_p))
        import os as _os
        _os.chdir(_p)
        break
else:
    raise RuntimeError('could not locate Blend repo root above ' + str(_ROOT))

from src.Plan import Plan
from src.Operators import Seekers


def main() -> None:
    LAKE_CONFIG = "tests/tasks/adventure_works.ini"
    question = "Which tables contain customer mailing addresses?"

    k = 50
    plan = Plan()
    plan.add("nl", Seekers.NaturalLanguage(question, k))

    plan.DB.load_config(Path(LAKE_CONFIG))

    ids = plan.run() or []
    print(f"returned {len(ids)} TableIds: {ids}")


if __name__ == "__main__":
    main()
