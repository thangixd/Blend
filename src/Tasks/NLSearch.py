from src.Plan import Plan
from src.Operators.Seekers import NLSeeker

# Typing imports
from typing import Optional


def NLSearch(
    query: str,
    k: int = 10,
    n: Optional[int] = None,
    alpha: Optional[float] = None,
    index_name: Optional[str] = None,
    config_overrides: Optional[dict] = None,
) -> Plan:
    plan = Plan()
    plan.add(
        "nlseeker",
        NLSeeker(
            query=query,
            k=k,
            n=n,
            alpha=alpha,
            index_name=index_name,
            config_overrides=config_overrides,
        ),
    )
    return plan
