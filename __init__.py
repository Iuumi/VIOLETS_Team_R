"""
VIOLETS Evaluation  — RQ1: Accuracy & Hallucination Prevention.

"""
from .rq1_pipeline import RQ1Results, run_rq1_evaluation
from .violets_interface import StubVioletsBackend, VioletsBackend, set_backend

__all__ = [
    "run_rq1_evaluation",
    "RQ1Results",
    "set_backend",
    "VioletsBackend",
    "StubVioletsBackend",
]

