from src.Plan import Plan
from src.Operators.Seekers import NaturalLanguage, SingleColumnOverlap
from src.Operators.Combiners import Intersection

# typing imports
from typing import List, Optional


def HybridNLSearch(query_values: List[str], question: str, k: int = 10, seeker_k: int = 50,
                   n: Optional[int] = None, alpha: Optional[float] = None,
                   index_name: Optional[str] = None, rerank: bool = True) -> Plan:
    plan = Plan()
    plan.add("values", SingleColumnOverlap(query_values, seeker_k))
    plan.add("nl", NaturalLanguage(question, seeker_k, n, alpha, index_name, rerank))
    plan.add("intersection", Intersection(k), inputs=["values", "nl"])
    return plan
