from src.Plan import Plan
from src.Operators.Seekers import NaturalLanguage

# typing imports
from typing import Optional


def NLSearch(query: str, k: int = 10, n: Optional[int] = None, alpha: Optional[float] = None,
             index_name: Optional[str] = None) -> Plan:
    plan = Plan()
    plan.add("nl", NaturalLanguage(query, k, n, alpha, index_name))
    return plan
